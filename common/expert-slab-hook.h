// expert-slab-hook.h — Story S4: hotlist ingestion + slab pre-warm + cb_eval hook.
//
// This module wires the routed-expert slab (src/llama-expert-slab.h) into the
// live inference path. It is part of common/ (so cli.cpp / server.cpp can use
// it) and includes the slab's private header via the ../src include path added
// in common/CMakeLists.txt.
//
// What S4 wires (end-to-end on llama-cli):
//   1. Reads the model GGUF shard(s) to build a per-layer expert-tensor layout
//      map (gate/up/down file offset + per-expert stride). Used for pre-warm
//      byte loads and (later, S7) for MISS-driven loads.
//   2. Constructs + alloc_locked's the slab using the plan_cache sizing (or the
//      explicit --streaming-cache-bytes budget).
//   3. Ingests --hotlist (ds4 v1 format) and pre-warms the top
//      min(N_hot, cache_experts) entries by preading their gate/up/down expert
//      bytes directly from the GGUF shard files into slab slots, in priority
//      order, before the first token.
//   4. Installs params.cb_eval, which fires for every graph tensor node; the
//      hook detects `ffn_moe_topk-<layer>` tensors, reads the selected expert
//      ids via ggml_backend_tensor_get (the same synced-read trace-moe uses),
//      and calls slab.note_moe_selection(layer, expert) for each — this is the
//      real HIT/MISS measurement against the pre-warmed slab.
//   5. At shutdown, prints the slab summary (hit rate, pre-warm coverage).
//
// Default-off: when params.streaming_cache_experts == 0, install() is a no-op
// and inference is byte-identical to today (beast-mode parity is structural).
//
// The actual byte-serving substitution (slab replacing mmap reads inside
// ggml_mul_mat_id) is the deeper invasive change tracked as S7 (coexistence
// with spliced boosted layers, which fall back to direct mmap). S4 delivers
// pre-warm + live hit-rate measurement; S7 delivers the serving。

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <atomic>
#include <thread>

// Include the slab type definition (in src/, llama's tree) via an explicit
// relative path so unique_ptr<llama_expert_slab_cache> is complete at the
// cli.cpp call site without requiring CMake include-path propagation.
#include "../src/llama-expert-slab.h"

struct common_params;

// (llama_expert_slab_cache is now fully defined by the include above.)

// Per-layer expert-tensor layout read from the model GGUF (file offsets +
// per-expert byte strides for the gate/up/down concatenated-experts tensors).
// Story S7: also records per-fragment qtype so a uniform predicate can decide
// whether the layer's experts match the slab's single class (slab-eligible) or
// are "boosted" (spliced to a different qtype; fall back to direct mmap).
struct layer_expert_layout {
    std::string gate_path, up_path, down_path;
    size_t gate_off = 0, up_off = 0, down_off = 0;
    size_t gate_stride = 0, up_stride = 0, down_stride = 0;
    size_t n_experts = 0;
    // Story S7: per-fragment qtype (GGML_TYPE_*). A layer is "uniform" iff all
    // three fragments share the same qtype AND that qtype matches the slab's
    // class (the first complete layer's qtypes). Spliced boosted layers have a
    // different qtype and are slab-ineligible (skip pre-warm; mmap-served).
    int32_t gate_qtype = 0; // enum ggml_type
    int32_t up_qtype   = 0;
    int32_t down_qtype = 0;
    // Resolved at install: is this layer slab-eligible (uniform w/ slab class)?
    bool uniform = false;
};

// Hook state. Defined here (not opaque) so unique_ptr's destructor is complete
// at the cli.cpp call site.
struct llama_expert_slab_hook_state {
    std::unique_ptr<llama_expert_slab_cache> slab;
    std::unordered_map<uint32_t, layer_expert_layout> layers;
    std::atomic<bool> prewarm_done{false};
    std::thread prewarm_thread;
    // Story S5: path to write the measured hotlist at shutdown ("" = no write).
    std::string hotlist_out_path;
    // Story S7: emit the per-layer uniform/boosted report at finalize.
    bool streaming_report = false;

    // Story S4 background-thread: safety-net destructor. finalize() joins the
    // thread; if it is NOT called (abnormal exit / unique_ptr destroyed without
    // finalize), std::thread's destructor would call std::terminate while still
    // joinable. Join here so teardown is always clean.
    ~llama_expert_slab_hook_state() {
        if (prewarm_thread.joinable()) {
            prewarm_thread.join();
        }
    }
};

// Opaque hook state (defined in the .cpp).
struct llama_expert_slab_hook_state_legacy;

// Install the expert-slab hook on `params` if streaming_cache_experts > 0.
// Reads the model GGUF to build the per-layer expert layout, constructs +
// alloc_locked's the slab, ingests --hotlist, pre-warms the top entries.
// Sets params.cb_eval / params.cb_eval_user_data so the hook fires during
// decode. Returns true on success (or when disabled, which is a clean no-op),
// false on a fatal setup error (err set). When disabled, *state_out is null.
//
// Must be called AFTER params.model.path is resolved and BEFORE
// common_init_from_params / llama_new_context_with_model (so cb_eval is set
// on the context cparams at creation time). The model file need only be
// readable; it does not need to be loaded yet.
bool llama_expert_slab_hook_install(common_params & params,
                                    std::unique_ptr<llama_expert_slab_hook_state> & state_out,
                                    std::string & err);

// Finalize: print the slab summary (hit rate, pre-warm coverage, mlock state)
// to stderr. Safe to call with state==nullptr (no-op).
void llama_expert_slab_hook_finalize(const std::unique_ptr<llama_expert_slab_hook_state> & state);
