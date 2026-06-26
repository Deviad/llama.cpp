// expert-slab-hook.cpp — Story S4 implementation. See expert-slab-hook.h.

#include "expert-slab-hook.h"
#include "common.h"

#include "llama-expert-slab.h"   // src/ (PRIVATE include via common/CMakeLists.txt)
#include "gguf.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <map>
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
        // Story S7: capture the fragment's qtype so the uniform predicate can
        // detect spliced boosted layers (gate/up at IQ3_S, down at IQ4_NL, etc.).
        int32_t qtype = (int32_t) gguf_get_tensor_type(gctx, i);
        auto & lay = out[(uint32_t) layer];
        if      (frag == "gate") { lay.gate_path = path; lay.gate_off = toff; lay.gate_qtype = qtype; }
        else if (frag == "up")   { lay.up_path   = path; lay.up_off   = toff; lay.up_qtype   = qtype; }
        else if (frag == "down") { lay.down_path = path; lay.down_off = toff; lay.down_qtype = qtype; }
    }
    gguf_free(gctx);
}

// For a given layer+fragment, open its shard and read the tensor's type + dims
// to compute the per-expert byte stride = tensor_nbytes / n_experts. We get
// n_experts from the GGUF KV "<arch>.expert_count" (global) like trace-moe.
// Story S7: also returns the fragment's qtype via *out_qtype (may be null).
static size_t fragment_expert_bytes(const std::string & path, const std::string & tname,
                                    size_t n_experts, int32_t * out_qtype = nullptr) {
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
        if (out_qtype) *out_qtype = (int32_t) gguf_get_tensor_type(gctx, i);
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
    // ---- Story S5: size + enable the runtime-writeback table ----
    st->slab->n_experts_table = (uint32_t) n_experts;
    st->hotlist_out_path      = params.streaming_hotlist_out;
    st->streaming_report      = params.streaming_report;
    st->slab->writeback_enabled = !params.streaming_hotlist_out.empty();
    if (st->slab->writeback_enabled) {
        // size to (max_scanned_layer + 1) * n_experts; layers without selections
        // contribute zero rows at save time. cap max_layer at a sane value
        // (e.g. 9999) so a malformed GGUF can't blow up memory.
        uint32_t max_layer = 0;
        for (const auto & kv : st->layers) max_layer = std::max(max_layer, kv.first);
        if (max_layer < 9999) {
            st->slab->selection_counts.assign((size_t)(max_layer + 1) * n_experts, 0);
        } else {
            st->slab->writeback_enabled = false;
        }
    }
    // resolve per-expert byte strides + qtypes for each layer's fragments
    for (auto & kv : st->layers) {
        auto & lay = kv.second;
        lay.n_experts = n_experts;
        lay.gate_stride = fragment_expert_bytes(lay.gate_path,
            "blk." + std::to_string(kv.first) + ".ffn_gate_exps.weight", n_experts, &lay.gate_qtype);
        lay.up_stride   = fragment_expert_bytes(lay.up_path,
            "blk." + std::to_string(kv.first) + ".ffn_up_exps.weight",   n_experts, &lay.up_qtype);
        lay.down_stride = fragment_expert_bytes(lay.down_path,
            "blk." + std::to_string(kv.first) + ".ffn_down_exps.weight", n_experts, &lay.down_qtype);
    }
    // Use the MAJORITY qtype triple across complete layers to fix the slab
    // geometry + class. (First-scanned would be wrong for the baseline, where
    // blk.78 is the rare IQ4_NL MTP exception among 75 IQ2_S layers — picking
    // blk.78's class would invert the classification and mark all real routed
    // layers as "boosted".) Majority vote makes the baseline uniform: slab class
    // = IQ2_S, blk.78 = the one boosted MTP layer (slab-ineligible, mmap-served).
    bool geom_set = false;
    int32_t slab_gate_q = 0, slab_up_q = 0, slab_down_q = 0;
    size_t slab_class_per_expert_bytes = 0;
    {
        std::map<std::tuple<int32_t,int32_t,int32_t,size_t>, uint32_t> tally;
        for (const auto & kv : st->layers) {
            const auto & lay = kv.second;
            if (!lay.gate_stride || !lay.up_stride || !lay.down_stride) continue;
            auto key = std::make_tuple(lay.gate_qtype, lay.up_qtype, lay.down_qtype,
                                       lay.gate_stride + lay.up_stride + lay.down_stride);
            tally[key]++;
        }
        uint32_t best = 0;
        for (const auto & kv2 : tally) {
            if (kv2.second > best) {
                best = kv2.second;
                slab_gate_q = std::get<0>(kv2.first);
                slab_up_q   = std::get<1>(kv2.first);
                slab_down_q = std::get<2>(kv2.first);
                slab_class_per_expert_bytes = std::get<3>(kv2.first);
            }
        }
        if (best > 0) {
            // find strides matching the elected class
            for (const auto & kv : st->layers) {
                const auto & lay = kv.second;
                if (lay.gate_qtype == slab_gate_q && lay.up_qtype == slab_up_q &&
                    lay.down_qtype == slab_down_q &&
                    (lay.gate_stride + lay.up_stride + lay.down_stride) == slab_class_per_expert_bytes) {
                    st->slab->gate_expert_bytes = lay.gate_stride;
                    st->slab->up_expert_bytes   = lay.up_stride;
                    st->slab->down_expert_bytes = lay.down_stride;
                    st->slab->per_expert_bytes  = slab_class_per_expert_bytes;
                    geom_set = true;
                    break;
                }
            }
        }
    }
    if (!geom_set) {
        err = "streaming-cache: could not resolve expert tensor geometry from GGUF";
        return false;
    }
    // Story S7: per-layer uniform predicate. A layer is uniform (slab-eligible)
    // iff all three fragments' qtypes match the slab class. Spliced boosted
    // layers (e.g. S8's IQ3_S gate/up + IQ4_NL down among IQ2_S layers) are
    // marked non-uniform and skipped at pre-warm (they'd waste slab slots on
    // bytes that can't be served by this single-class slab).
    // Today's baseline (uniform IQ2_S experts + IQ4_NL blk.78 MTP exception)
    // is detected as uniform for slab purposes because blk.78 is MTP, not a
    // normal routed-expert dispatch layer (so it doesn't appear in `layers`
    // as a ffn_*_exps tensor) — single-class fast path, no fallback.
    uint32_t n_uniform = 0, n_boosted = 0;
    for (auto & kv : st->layers) {
        auto & lay = kv.second;
        lay.uniform = (lay.gate_qtype == slab_gate_q &&
                       lay.up_qtype   == slab_up_q   &&
                       lay.down_qtype == slab_down_q);
        if (lay.uniform) n_uniform++; else n_boosted++;
    }
    if (params.streaming_report) {
        fprintf(stderr, "expert-slab S7: slab class gate=%s up=%s down=%s; "
                "%u uniform layers, %u boosted (mmap-served) layers\n",
                ggml_type_name((ggml_type) slab_gate_q),
                ggml_type_name((ggml_type) slab_up_q),
                ggml_type_name((ggml_type) slab_down_q),
                n_uniform, n_boosted);
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

    // ---- 3. ingest hotlist + (background-threaded) pre-warm ----
    // Story S4 background-thread cut: the pre-warm loop runs on
    // state->prewarm_thread so install does not block model load / first
    // decode. The slab's reserve/publish protocol (prewarm_reserve -> pread ->
    // prewarm_publish) publishes each (layer,expert)->slot map entry with
    // slot_ready=false first, then marks ready; a concurrent decode-time
    // lookup() that hits an in-flight slot blocks on slab_cv (diagnosed by
    // prewarm_blocks) instead of missing. pread happens outside the lock
    // (disjoint slots). finalize() joins the thread before writeback/summary.
    std::vector<llama_expert_slab_hotlist_entry> hl;
    if (!params.streaming_hotlist.empty()) {
        hl = llama_expert_slab_hotlist_load(params.streaming_hotlist, err);
        if (!err.empty()) {
            fprintf(stderr, "expert-slab hook: hotlist load warning: %s (continuing cold)\n", err.c_str());
            err.clear();
        }
    }
    state_out = std::move(st);   // state now lives at its final location
    auto * state_ptr = state_out.get();

    if (!hl.empty()) {
        // Capture by value: the slab pointer, the layers map (shared_ptr to avoid copy),
        // the hotlist (moved in), and the entry count to pre-warm.
        size_t n_warm = std::min(hl.size(), state_ptr->slab->cache_experts);
        std::vector<llama_expert_slab_hotlist_entry> hl_mv = std::move(hl);
        // layers map is read-only during the worker's lifetime (finalize joins
        // before any teardown), so a raw pointer to it is safe.
        auto * layers_ptr = &state_ptr->layers;
        auto * slab_p    = state_ptr->slab.get();

        state_ptr->prewarm_thread = std::thread([state_ptr, slab_p, layers_ptr,
                                                  hl_mv = std::move(hl_mv), n_warm]() {
            uint32_t n_skipped_boosted = 0;
            for (size_t i = 0; i < n_warm; ++i) {
                const auto & e = hl_mv[i];
                auto it = layers_ptr->find(e.layer);
                if (it == layers_ptr->end()) continue;
                const auto & lay = it->second;
                // Story S7: skip boosted (spliced, non-uniform) layers — their
                // bytes are a different qtype and can't be served by this
                // single-class slab. They fall back to direct mmap reads.
                if (!lay.uniform) { n_skipped_boosted++; continue; }

                // Reserve: publish the map entry with slot_ready=false so a
                // concurrent lookup blocks on the cv rather than missing.
                size_t slot = slab_p->prewarm_reserve(e.layer, e.expert_id);
                if (slot == SIZE_MAX) continue;
                uint8_t * dst = slab_p->slot_ptr(slot);
                if (!dst) {
                    slab_p->prewarm_publish(slot, false, e.layer, e.expert_id);
                    continue;
                }
                // Load bytes OUTSIDE the lock (disjoint slots).
                bool ok_g = pread_bytes(lay.gate_path, lay.gate_off + (size_t)e.expert_id*lay.gate_stride,
                                        lay.gate_stride, dst + 0);
                bool ok_u = pread_bytes(lay.up_path,   lay.up_off   + (size_t)e.expert_id*lay.up_stride,
                                        lay.up_stride,   dst + slab_p->gate_expert_bytes);
                bool ok_d = pread_bytes(lay.down_path, lay.down_off + (size_t)e.expert_id*lay.down_stride,
                                        lay.down_stride, dst + slab_p->gate_expert_bytes + slab_p->up_expert_bytes);
                bool ok = ok_g && ok_u && ok_d;
                slab_p->prewarm_publish(slot, ok, e.layer, e.expert_id);
                if (ok) {
                    slab_p->hotlist_entries_loaded++;
                    slab_p->n_prewarm_loaded++;
                }
            }
            fprintf(stderr, "expert-slab hook: pre-warmed %u of %zu hotlist entries into %zu-slot slab"
                    " (skipped %u boosted-layer entries, S7 mmap-served)\n",
                    slab_p->hotlist_entries_loaded, hl_mv.size(), slab_p->cache_experts, n_skipped_boosted);
            state_ptr->prewarm_done.store(true, std::memory_order_release);
        });
        fprintf(stderr, "expert-slab hook: pre-warming %zu hotlist entries in background"
                " (%zu-slot slab)\n", n_warm, state_ptr->slab->cache_experts);
    } else {
        state_ptr->prewarm_done.store(true, std::memory_order_release);
    }

    // ---- 4. install cb_eval ----
    auto * slab_ptr = state_ptr->slab.get();
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

    return true;
}

void llama_expert_slab_hook_finalize(const std::unique_ptr<llama_expert_slab_hook_state> & state) {
    if (!state || !state->slab) return;
    // Story S4 background-thread: join the pre-warm worker before writeback /
    // summary so hotlist_entries_loaded / n_prewarm_loaded / selection_counts
    // are stable. The worker iterates a bounded hotlist and exits on its own; if
    // finalize is reached early (e.g. short run), join waits for the remaining
    // pre-warm preads to finish (bounded by hotlist size). Detach would risk
    // use-after-free since the worker reads state->slab / state->layers.
    if (state->prewarm_thread.joinable()) {
        state->prewarm_thread.join();
    }
    // Story S5: write the measured hotlist before printing the summary.
    if (state->slab->writeback_enabled && !state->hotlist_out_path.empty()) {
        std::string err;
        if (state->slab->hotlist_save(state->hotlist_out_path, err)) {
            fprintf(stderr, "expert-slab hook: wrote measured hotlist to %s\n",
                    state->hotlist_out_path.c_str());
        } else {
            fprintf(stderr, "expert-slab hook: hotlist writeback FAILED: %s\n", err.c_str());
        }
    }
    // Story S7: per-layer uniform/boosted + slab/mmap path report.
    if (state->streaming_report) {
        fprintf(stderr, "expert-slab S7 report: per-layer routed-expert path\n");
        fprintf(stderr, "  layer | class   | gate(q,stride)        up(q,stride)          down(q,stride)         | per_expert_bytes\n");
        for (const auto & kv : state->layers) {
            const auto & lay = kv.second;
            const char * cls = lay.uniform ? "uniform" : "boosted";
            const char * path = lay.uniform ? "slab" : "mmap";
            fprintf(stderr, "  %5u | %-7s | %-8s/%-8zu %-8s/%-8zu %-8s/%-8zu | %zu  [%s-served]\n",
                    kv.first, cls,
                    ggml_type_name((ggml_type) lay.gate_qtype), lay.gate_stride,
                    ggml_type_name((ggml_type) lay.up_qtype),   lay.up_stride,
                    ggml_type_name((ggml_type) lay.down_qtype), lay.down_stride,
                    lay.gate_stride + lay.up_stride + lay.down_stride, path);
        }
    }
    fprintf(stderr, "%s: ", __func__);
    state->slab->print_summary("final");
}
