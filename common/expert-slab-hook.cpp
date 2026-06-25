// expert-slab-hook.cpp — Story S4 implementation. See expert-slab-hook.h.

#include "expert-slab-hook.h"
#include "common.h"

#include "llama-expert-slab.h"   // src/ (PRIVATE include via common/CMakeLists.txt)
#include "gguf.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <regex>
#include <thread>
#include <atomic>

// ---- per-layer expert tensor layout (read from the GGUF) ----
// (layer_expert_layout now defined in the header so unique_ptr is complete
// at the cli.cpp call site.)

// (llama_expert_slab_hook_state is also defined in the header.)

// Extract layer index from a tensor name like "blk.42.ffn_gate_exps.weight" -> 42.
static int extract_layer_from_name(const std::string & name) {
    std::smatch m;
    if (std::regex_search(name, m, std::regex("blk\\.(\\d+)\\."))) {
        return std::stoi(m[1].str());
    }
    return -1;
}

// Scan one GGUF shard file for blk.N.ffn_{gate,up,down}_exps.weight tensors
// and record each layer's layout (file path, tensor data offset, per-expert
// stride). Calls cb(tensor_name, layer, fragment, ne_outer_dim_bytes_per_expert).
// Returns the set of layer indices found.
static void scan_shard_for_expert_tensors(const std::string & path,
                                          std::unordered_map<uint32_t, layer_expert_layout> & out) {
    gguf_init_params ip{};
    ip.no_alloc = true;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), ip);
    if (!gctx) return;
    size_t data_off = gguf_get_data_offset(gctx);
    int64_t n_t = gguf_get_n_tensors(gctx);
    for (int64_t i = 0; i < n_t; ++i) {
        const char * tname = gguf_get_tensor_name(gctx, i);
        if (!tname) continue;
        std::string name(tname);
        // match blk.N.ffn_(gate|up|down)_exps.weight
        std::smatch m;
        if (!std::regex_match(name, m, std::regex("blk\\.(\\d+)\\.ffn_(gate|up|down)_exps\\.weight"))) continue;
        int layer = std::stoi(m[1].str());
        std::string frag = m[2].str();
        size_t toff = data_off + gguf_get_tensor_offset(gctx, i);
        // The tensor is [d_intermediate * n_experts]? No: for concatenated
        // experts, ne[0]=d_intermediate_dim, ne[1]=n_experts (row-major), so
        // per-expert row stride = nb[1] / n_experts == ggml_row_size(type, ne[0]).
        // We don't have the tensor type/dims here via gguf API directly; but the
        // per-expert byte stride = tensor_nbytes / n_experts. gguf doesn't expose
        // nbytes directly either. We read the GGUF KV for n_experts separately,
        // but simplest: defer stride compute to the caller which knows the
        // type. Here we just record the fragment's file path + tensor data offset;
        // the per-expert stride is computed at load time from ggml_nbytes(tensor)
        // which needs the loaded model. To keep S4 self-contained, we compute
        // stride from the hotlist's expert count + the tensor's GGUF type/shape.
        // For now record path + offset; resolve stride at pre-warm via gguf
        // tensor info lookup below (see resolve_strides_in_shard).
        auto & lay = out[(uint32_t) layer];
        if      (frag == "gate") { lay.gate_path = path; lay.gate_off = toff; }
        else if (frag == "up")   { lay.up_path   = path; lay.up_off   = toff; }
        else if (frag == "down") { lay.down_path = path; lay.down_off = toff; }
    }
    gguf_free(gctx);
}

// For a given layer+fragment, open its shard and read the tensor's type + dims
// to compute the per-expert byte stride = tensor_nbytes / n_experts. We get
// n_experts from the GGUF KV "<arch>.expert_count" (global) like trace-moe.
static size_t fragment_expert_bytes(const std::string & path, const std::string & tname,
                                    size_t n_experts) {
    if (n_experts == 0) return 0;
    gguf_init_params ip{}; ip.no_alloc = true;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), ip);
    if (!gctx) return 0;
    size_t nb = 0;
    int64_t n_t = gguf_get_n_tensors(gctx);
    for (int64_t i = 0; i < n_t; ++i) {
        const char * n = gguf_get_tensor_name(gctx, i);
        if (!n) continue;
        if (std::string(n) != tname) continue;
        nb = gguf_get_tensor_size(gctx, i);
        break;
    }
    gguf_free(gctx);
    return nb ? nb / n_experts : 0;
}

// Read n_experts from the GGUF KV (trace-moe uses "<arch>.expert_count").
static size_t read_n_experts_from_gguf(const std::string & path) {
    gguf_init_params ip{}; ip.no_alloc = true;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), ip);
    if (!gctx) return 0;
    size_t n = 0;
    // Try common KV names; the arch prefix varies, so scan all string KVs for
    // one ending in ".expert_count".
    int64_t n_kv = gguf_get_n_kv(gctx);
    for (int64_t i = 0; i < n_kv; ++i) {
        const char * key = gguf_get_key(gctx, i);
        if (!key) continue;
        std::string k(key);
        if (k.size() >= 13 && k.substr(k.size()-13) == ".expert_count") {
            if (gguf_get_kv_type(gctx, i) == GGUF_TYPE_UINT32) {
                n = (size_t) gguf_get_val_u32(gctx, i);
            } else if (gguf_get_kv_type(gctx, i) == GGUF_TYPE_INT32) {
                n = (size_t) gguf_get_val_i32(gctx, i);
            }
            if (n) break;
        }
    }
    gguf_free(gctx);
    return n;
}

// Pread helper: read `len` bytes at `off` from a file into `dst`.
static bool pread_bytes(const std::string & path, size_t off, size_t len, void * dst) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    if (std::fseek(f, (long) off, SEEK_SET) != 0) { std::fclose(f); return false; }
    bool ok = (std::fread(dst, 1, len, f) == len);
    std::fclose(f);
    return ok;
}

bool llama_expert_slab_hook_install(common_params & params,
                                    std::unique_ptr<llama_expert_slab_hook_state> & state_out,
                                    std::string & err) {
    state_out.reset();
    if (params.streaming_cache_experts == 0) {
        // Default-off: no-op, beast-mode parity preserved.
        return true;
    }

    auto st = std::make_unique<llama_expert_slab_hook_state>();
    st->slab = std::make_unique<llama_expert_slab_cache>();

    // ---- 1. scan model GGUF shard(s) for the expert-tensor layout ----
    const std::string model_path = params.model.path;
    if (model_path.empty()) { err = "streaming-cache: model.path is empty"; return false; }
    // If split, the path points at shard 1; scan all shards matching the
    // -NNNN-of-MMMM.gguf pattern. For simplicity, scan the single shard path;
    // if it's shard 1 of a split, also scan the sibling shards.
    std::vector<std::string> shards;
    shards.push_back(model_path);
    // detect split siblings: GLM-5.2-...-00001-of-00009.gguf
    std::smatch sm;
    if (std::regex_search(model_path, sm, std::regex("^(.*?)-(\\d+)-of-(\\d+)\\.gguf$"))) {
        std::string base = sm[1].str();
        int nn = std::stoi(sm[3].str());
        for (int i = 1; i <= nn; ++i) {
            char buf[64]; std::snprintf(buf, sizeof(buf), "-%05d-of-%05d.gguf", i, nn);
            std::string p = base + buf;
            if (p != model_path) shards.push_back(p);
        }
    }
    size_t n_experts = 0;
    for (const auto & sp : shards) {
        if (n_experts == 0) n_experts = read_n_experts_from_gguf(sp);
        scan_shard_for_expert_tensors(sp, st->layers);
    }
    if (n_experts == 0) n_experts = 256; // GLM-5.2 default fallback
    // resolve per-expert byte strides for each layer's fragments
    for (auto & kv : st->layers) {
        auto & lay = kv.second;
        lay.n_experts = n_experts;
        lay.gate_stride = fragment_expert_bytes(lay.gate_path,
            "blk." + std::to_string(kv.first) + ".ffn_gate_exps.weight", n_experts);
        lay.up_stride   = fragment_expert_bytes(lay.up_path,
            "blk." + std::to_string(kv.first) + ".ffn_up_exps.weight",   n_experts);
        lay.down_stride = fragment_expert_bytes(lay.down_path,
            "blk." + std::to_string(kv.first) + ".ffn_down_exps.weight", n_experts);
    }
    // Use the first layer with a complete layout to fix the slab geometry.
    bool geom_set = false;
    for (auto & kv : st->layers) {
        auto & lay = kv.second;
        if (lay.gate_stride && lay.up_stride && lay.down_stride) {
            st->slab->gate_expert_bytes = lay.gate_stride;
            st->slab->up_expert_bytes   = lay.up_stride;
            st->slab->down_expert_bytes = lay.down_stride;
            st->slab->per_expert_bytes  = lay.gate_stride + lay.up_stride + lay.down_stride;
            geom_set = true;
            break;
        }
    }
    if (!geom_set) {
        err = "streaming-cache: could not resolve expert tensor geometry from GGUF";
        return false;
    }
    st->slab->cache_experts = params.streaming_cache_experts;

    // ---- 2. plan + alloc_locked ----
    size_t budget = params.streaming_cache_bytes;
    if (budget == 0) {
        size_t ws = llama_expert_slab_recommended_working_set_bytes();
        size_t non_routed = 0; // unknown without the full tensor dir; use 0 so
                              // the plan = recommended*4/5 (cap applied by budget rule)
        (void) non_routed;
        budget = (size_t)((double) ws * LLAMA_EXPERT_SLAB_BUDGET_FRACTION);
    }
    if (!st->slab->alloc_locked(budget, err)) {
        return false;
    }
    fprintf(stderr, "%s: expert-slab hook installed — ", __func__);
    st->slab->print_summary("hook");

    // ---- 3. ingest hotlist + pre-warm ----
    std::vector<llama_expert_slab_hotlist_entry> hl;
    if (!params.streaming_hotlist.empty()) {
        hl = llama_expert_slab_hotlist_load(params.streaming_hotlist, err);
        if (!err.empty()) {
            fprintf(stderr, "expert-slab hook: hotlist load warning: %s (continuing cold)\n", err.c_str());
            err.clear();
        }
    }
    // Pre-warm (synchronous in this S4 cut; the background-thread refinement is
    // noted in the plan). Pre-warm the top min(N_hot, cache_experts) entries by
    // preading their gate/up/down bytes from the GGUF shard into slab slots.
    size_t n_warm = std::min(hl.size(), st->slab->cache_experts);
    for (size_t i = 0; i < n_warm; ++i) {
        const auto & e = hl[i];
        auto it = st->layers.find(e.layer);
        if (it == st->layers.end()) continue;
        const auto & lay = it->second;
        // Find a free slot (or recycle slot 0).
        size_t slot = SIZE_MAX;
        for (size_t s = 0; s < st->slab->slots.size(); ++s) {
            if (!st->slab->slots[s].valid) { slot = s; break; }
        }
        if (slot == SIZE_MAX) slot = 0;
        uint8_t * dst = st->slab->slot_ptr(slot);
        if (!dst) continue;
        bool ok_g = pread_bytes(lay.gate_path, lay.gate_off + e.expert_id*lay.gate_stride,
                                lay.gate_stride, dst + 0);
        bool ok_u = pread_bytes(lay.up_path,   lay.up_off   + e.expert_id*lay.up_stride,
                                lay.up_stride,   dst + st->slab->gate_expert_bytes);
        bool ok_d = pread_bytes(lay.down_path, lay.down_off + e.expert_id*lay.down_stride,
                                lay.down_stride, dst + st->slab->gate_expert_bytes + st->slab->up_expert_bytes);
        if (ok_g && ok_u && ok_d) {
            st->slab->slots[slot].layer     = e.layer;
            st->slab->slots[slot].expert_id = e.expert_id;
            st->slab->slots[slot].valid     = true;
            st->slab->slot_ready[slot]      = true;
            st->slab->map[((uint64_t)e.layer << 32) | e.expert_id] = slot;
            st->slab->hotlist_entries_loaded++;
            st->slab->n_prewarm_loaded++;
        }
    }
    fprintf(stderr, "expert-slab hook: pre-warmed %u of %zu hotlist entries into %zu-slot slab\n",
            st->slab->hotlist_entries_loaded, hl.size(), st->slab->cache_experts);

    // ---- 4. install cb_eval ----
    auto * slab_ptr = st->slab.get();
    params.cb_eval = [](struct ggml_tensor * t, bool ask, void * ud) -> bool {
        if (ask) return true;
        if (!t) return true;
        const char * nm = t->name;
        if (!nm || !nm[0]) return true;
        std::string name(nm);
        // match "ffn_moe_topk" or "ffn_moe_topk-<layer>"
        if (name.rfind("ffn_moe_topk", 0) != 0) return true;
        int layer = extract_layer_from_name(name);
        if (layer < 0) {
            // name might be exactly "ffn_moe_topk" with layer in metadata; try
            // parsing a trailing "-N"
            std::smatch m;
            if (std::regex_search(name, m, std::regex("-(\\d+)$"))) layer = std::stoi(m[1].str());
            if (layer < 0) return true;
        }
        if (t->type != GGML_TYPE_I32) return true;
        int n_used   = (int) t->ne[0];
        int n_tokens = (int) t->ne[1];
        if (n_used <= 0 || n_tokens <= 0) return true;
        size_t nbytes = ggml_nbytes(t);
        std::vector<uint8_t> raw(nbytes);
        ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
        const int32_t * src = reinterpret_cast<const int32_t *>(raw.data());
        auto * slab = static_cast<llama_expert_slab_cache *>(ud);
        if (!slab) return true;
        for (int tok = 0; tok < n_tokens; ++tok) {
            for (int e = 0; e < n_used; ++e) {
                int32_t expert_id = src[tok * n_used + e];
                if (expert_id >= 0) {
                    slab->note_moe_selection((uint32_t) layer, (uint32_t) expert_id);
                }
            }
        }
        return true;
    };
    params.cb_eval_user_data = slab_ptr;

    state_out = std::move(st);
    return true;
}

void llama_expert_slab_hook_finalize(const std::unique_ptr<llama_expert_slab_hook_state> & state) {
    if (!state || !state->slab) return;
    fprintf(stderr, "%s: ", __func__);
    state->slab->print_summary("final");
}
