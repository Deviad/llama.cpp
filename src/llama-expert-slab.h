#pragma once

// llama-expert-slab.h — routed-expert SSD streaming slab cache (Story S1).
//
// A single-size-class cache of N expert slots, each slot holding one expert's
// full (gate, up, down) weight triple, served from an SSD-backed mmap. This is
// the data-structure + I/O layer for the SSD-streaming endeavour tracked in
// LLAMACPP_GLM52_ENHANCEMENT_PLAN.md (Stories S1-S7). The design mirrors ds4's
// `ds4_ssd_auto_cache_plan` / `streaming_layer_routed_expert_bytes` arithmetic.
//
// Scope of Story S1 (this file): the slab allocator + auto cache-plan sizing +
// (layer, expert) -> slot|MISS lookup + a read-from-mmap helper. Live
// interception of the MoE forward path is deferred to S4 (pre-warm) and S7
// (coexistence). Default-off: when no `--streaming-cache-*` flag is passed,
// nothing here is constructed and inference is byte-identical to today.
//
// The slab lives next to llama-mmap (src/llama-mmap.{h,cpp}) because its
// backing store for MISS reads is the model's existing weight mmap. See
// SSD_STREAMING notes in LLAMACPP_GLM52_ENHANCEMENT_PLAN.md §1 (llama.cpp
// already mmaps the whole GGUF; this slab is a *hot subset* of the routed-
// expert region, mlock'd in bounded chunks — the chunked-mlock discipline is
// Story S2).

#include <cstdint>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

// --- Story S2: chunked mlock state ---
//
// ds4 locks the slab in 256 MiB chunks (not one huge mlock) and touches every
// page with a write pattern `p[pos] = pos/page` before each chunk's mlock to
// force commitment. On mid-chunk mlock failure, the already-locked prefix is
// munlock'd and the slab falls back to non-locked malloc with a warning.
//
// The chunked approach avoids kernel-panic-class VM accounting failures on
// macOS when a single huge mlock would exceed the wired-memory limit.
static constexpr size_t LLAMA_EXPERT_SLAB_MLOCK_CHUNK = 256ull * 1024 * 1024; // 256 MiB

// ds4's budget rule: explicit-cache bytes <= 7/10 * recommended_working_set.
// On Metal this maps to MTLDevice.recommendedMaxWorkingSetSize; elsewhere a
// caller-supplied default or the largest GPU device's memory_total.
static constexpr double LLAMA_EXPERT_SLAB_BUDGET_FRACTION = 7.0 / 10.0;

// Resolve the host's recommended working-set size in bytes. On Apple Silicon
// this returns MTLDevice.recommendedMaxWorkingSetSize (via the Metal backend
// device's memory_total). On non-Metal hosts, returns the largest GPU-class
// backend device's memory_total, or a 16 GiB fallback if no GPU device found.
// Exposed so callers (CLI flag handler, slab planning) can apply the 7/10 rule.
size_t llama_expert_slab_recommended_working_set_bytes();

// --- Story S3: per-region madvise(WILLNEED) at prefill granularity ---
//
// llama-mmap.cpp already hints the *whole* weight file with POSIX_MADV_WILLNEED
// at mmap time (line 463) and then POSIX_MADV_RANDOM. That's a coarse prefetch.
// S3 adds a scoped helper that hints a *single layer's routed-expert span*
// ahead of that layer's FFN compute, so SSD read latency overlaps with the
// preceding layer's compute rather than stalling it.
//
// The helper is intentionally freestanding (does not touch the slab cache) —
// it operates on the model's weight mmap address directly. Live invocation
// from the prefill path is wired in Story S4 (the dispatch hook installation);
// S3 lands the helper + the --trace-prefetch debug flag + a unit test, matching
// the S1/S2 pattern of landing the data-structure layer before the live hook.
//
// `mmap_base` is the model weight mmap address (same pointer llama-mmap
// exposes via `llama_mmap::addr()`). `offset`/`len` describe the byte span of
// one layer's routed-expert region within that mmap. `layer` is the layer
// index (only used for the --trace-prefetch log line). `tag` is a short label
// for the log (e.g. "prefill" / "decode-warmup"). Returns 0 on success, errno
// on posix_madvise failure (caller decides whether to warn).
//
// Compile-guarded: on platforms without POSIX_MADV_WILLNEED this is a no-op
// returning 0, so callers can invoke it unconditionally.
int llama_expert_slab_prefetch_willneed(const void * mmap_base, size_t offset,
                                        size_t len, uint32_t layer,
                                        const char * tag, bool trace);

// --- Story S4: hotlist ingestion + pre-warm + dispatch-hook counting ---
//
// ds4 expert hotlist v1 format (produced by Story S6's
// common/scripts/derive_glm52_hotlist.py from Phase 2b traces):
//   # ds4 expert hotlist v1
//   # model ...
//   # layers <N>
//   # experts <N>
//   # layer_records <N>
//   # selections <N>
//   # columns: layer expert hits weight
//   <layer> <expert> <hits> <weight>        (sorted by descending weight)
//
// Returns a vector of {layer, expert_id, weight} sorted by descending weight.
// Tolerant: skips blank lines, # comments, and parses the four numeric columns.
struct llama_expert_slab_hotlist_entry {
    uint32_t layer     = 0;
    uint32_t expert_id = 0;
    uint64_t hits      = 0;
    double   weight    = 0.0;
};
std::vector<llama_expert_slab_hotlist_entry>
llama_expert_slab_hotlist_load(const std::string & path, std::string & err);

struct llama_expert_slab_cache {
    // Per-expert byte geometry. For a layer L, the routed-expert tensors are
    //   blk.L.ffn_gate_exps.weight  (concatenation of n_experts gate blocks)
    //   blk.L.ffn_up_exps.weight    (concatenation of n_experts up blocks)
    //   blk.L.ffn_down_exps.weight  (concatenation of n_experts down blocks)
    // Each fragment may have a different qtype/dims, so its per-expert byte
    // stride differs. The slab stores the full triple per expert, so
    // per_expert_bytes = gate_stride + up_stride + down_stride (matches ds4's
    // `streaming_layer_routed_expert_bytes`).
    size_t gate_expert_bytes  = 0;
    size_t up_expert_bytes    = 0;
    size_t down_expert_bytes  = 0;
    size_t per_expert_bytes   = 0; // sum of the three

    // Slab geometry. cache_experts slots, each per_expert_bytes wide.
    size_t cache_experts = 0; // slot count (0 => disabled)
    size_t slab_bytes    = 0; // cache_experts * per_expert_bytes

    // The slab itself: a resident, (optionally, in S2) mlock'd buffer. Owned.
    // Layout: slot i occupies bytes [i*per_expert_bytes, (i+1)*per_expert_bytes).
    // Within a slot, the triple is laid out as [gate | up | down] in that order
    // (matching the natural read order of the dispatch hook).
    uint8_t * slab = nullptr; // heap-allocated in S1; S2 will make it mlock'd.

    // --- Story S2: chunked mlock state ---
    bool     mlocked        = false; // true iff the slab was successfully mlock'd
    size_t   mlocked_bytes  = 0;     // how many bytes (prefix) are currently locked
    size_t   budget_bytes  = 0;     // max bytes the caller is willing to mlock (7/10 rule)

    // Slot bookkeeping. slots[i] = {layer, expert_id, valid}. A free slot has
    // valid==false. The map gives O(1) (layer, expert_id) -> slot index on HIT.
    struct slot_state {
        uint32_t layer     = 0;
        uint32_t expert_id = 0;
        bool     valid     = false;
    };
    std::vector<slot_state>                 slots;
    std::unordered_map<uint64_t, size_t>    map; // key = (uint64_t(layer)<<32) | expert_id

    // --- counters (Story S5 will wire writeback; S1 just counts) ---
    uint64_t n_hit  = 0;
    uint64_t n_miss = 0;
    uint64_t n_prewarm_loaded = 0;

    // --- Story S4: pre-warm + dispatch-hook counting ---
    uint32_t hotlist_entries_loaded = 0; // count of hotlist entries actually pre-warmed
    uint64_t prewarm_hits           = 0; // lookups that HIT a pre-warmed slot
    uint64_t prewarm_misses         = 0; // lookups that MISSed a pre-warmed slot
    // Per-slot readiness flag (for the background-thread blocking discipline):
    // a slot is "ready" once its pre-warm load completed. The main thread blocks
    // on lookup() only if the slot is requested before ready==true (rare).
    std::vector<bool> slot_ready;

    // Record a MoE expert-selection event (one expert chosen for one token on
    // one layer). Counts HIT/MISS against the slab's current contents.
    // Returns true if the selected expert was a HIT (already resident).
    // This is the cb_eval hook entry point: the caller (installed as
    // params.cb_eval) reads the ffn_moe_topk tensor, extracts the selected
    // expert ids, and calls this for each (layer, expert_id) pair.
    bool note_moe_selection(uint32_t layer, uint32_t expert_id);

    // --- Story S3: prefill-madvise de-dup ---
    // Tracks which (layer, prefill-chunk) pairs have already been hinted so
    // the WILLNEED advisory is issued exactly once per pair (no re-hinting on
    // decode tokens, which read from the slab, not the mmap). Key is
    // (uint64_t(layer) << 32) | chunk_index. S4 wires the live invocation;
    // S3 just provides the bookkeeping + the prefetch_one() driver.
    std::unordered_map<uint64_t, bool> prefetched_chunks;
    bool trace_prefetch = false; // set from --trace-prefetch

    // Issue posix_madvise(WILLNEED) for one layer's expert span, but only the
    // first time this (layer, chunk) pair is seen. Returns true if a hint was
    // actually issued, false if it was de-duped (already hinted) or no-op'd.
    bool prefetch_one(const void * mmap_base, size_t offset, size_t len,
                      uint32_t layer, uint32_t chunk_index, const char * tag);

    // --- construction / teardown ---
    llama_expert_slab_cache() = default;
    ~llama_expert_slab_cache();

    llama_expert_slab_cache(const llama_expert_slab_cache &) = delete;
    llama_expert_slab_cache & operator=(const llama_expert_slab_cache &) = delete;

    // Allocate the slab. Returns true on success, false (and sets err) on
    // allocation failure. After this, slots are all free and the map is empty.
    // per_expert_bytes and cache_experts must already be set by the caller
    // (see `plan_cache`). The slab is zero-initialized.
    bool alloc(std::string & err);

    // --- Story S2: chunked mlock ---
    //
    // Allocates the slab (if not already) and mlocks it in 256 MiB chunks,
    // touching every page with the write pattern p[pos]=pos/page before each
    // chunk's mlock to force commitment. budget_bytes caps the locked prefix
    // (7/10 of recommended_working_set, per ds4). On mid-chunk mlock failure,
    // the already-locked prefix is munlock'd, the slab is freed, and the cache
    // falls back to non-locked allocation with a warning (caller can retry with
    // a smaller budget). Returns true if the full slab (or the budgeted prefix)
    // was mlock'd; false (err set) if allocation itself failed.
    //
    // If the slab fits within budget_bytes, the entire slab is mlock'd.
    // Otherwise, only the first floor(budget_bytes/per_expert_bytes) experts'
    // worth of bytes are mlock'd (the rest stays ordinary heap and may be
    // paged out under pressure — S4's pre-warm plan avoids loading cold experts
    // into the unlocked tail).
    bool alloc_locked(size_t budget_bytes_in, std::string & err);

    // --- core I/O API (validated by test-expert-slab.cpp) ---
    //
    // Look up (layer, expert_id). Returns the slot index on HIT (and bumps
    // n_hit), or SIZE_MAX on MISS (and bumps n_miss). Does not read or copy.
    size_t lookup(uint32_t layer, uint32_t expert_id);

    // On MISS, load expert `expert_id` of layer `layer` into a free slot
    // (eviction is LRU-ish: picks the first free slot, or if none, slot 0 —
    // S2's mlock discipline + S5's LRU writeback will refine this). Reads
    // `per_expert_bytes` bytes from the model mmap at
    //   gate_off + expert_id*gate_stride  (gate_expert_bytes)
    //   up_off   + expert_id*up_stride    (up_expert_bytes)
    //   down_off + expert_id*down_stride  (down_expert_bytes)
    // and writes them into the chosen slot's [gate|up|down] regions. Returns
    // the slot index, or SIZE_MAX on failure (no free slot AND mmap is null).
    //
    // `mmap_base` is the model weight mmap address (the same pointer
    // llama-mmap exposes). The three offsets/strides describe where layer L's
    // gate/up/down expert blocks live within that mmap.
    struct expert_layout {
        size_t gate_off, gate_stride;
        size_t up_off,   up_stride;
        size_t down_off, down_stride;
    };
    size_t load_on_miss(uint32_t layer, uint32_t expert_id,
                        const uint8_t * mmap_base, const expert_layout & lay);

    // Direct slot pointer accessor (for the dispatch hook to read weights from
    // the slab on a confirmed HIT). slot_idx must be < cache_experts.
    uint8_t * slot_ptr(size_t slot_idx);

    // True iff this slab is enabled (cache_experts > 0 and slab != null).
    bool enabled() const { return cache_experts > 0 && slab != nullptr; }

    // Print a one-line summary to stderr (cache plan + counters).
    void print_summary(const char * label) const;

    // Free the slab (called by dtor; also callable to reset).
    void reset();
};

// Auto cache-plan sizing (Story S1 AC). Mirrors ds4's plan:
//   model_target       = recommended_working_set_bytes * 4 / 5
//   cache_bytes        = model_target - non_routed_bytes
//   cache_experts      = min(model_expert_count, cache_bytes / per_expert_bytes)
// `recommended_working_set_bytes` is the host's recommended resident budget
// (on Metal, Metal's device recommendation; elsewhere a caller-supplied
// default). non_routed_bytes is the sum of all non-routed-expert tensor bytes
// (attention, shared experts, norms, embeddings, output) — i.e. everything
// that stays resident regardless of the slab.
//
// Returns the computed cache_experts (clamped to [0, model_expert_count]).
// If cache_bytes is too small to hold even one expert, returns 0 (slab
// disabled with a warning printed).
size_t llama_expert_slab_plan_cache(
    size_t recommended_working_set_bytes,
    size_t non_routed_bytes,
    size_t per_expert_bytes,
    size_t model_expert_count);

// Parse a human byte-size string like "16GiB", "8gib", "4gb", "2048m", "512"
// into bytes. Accepts K/M/G/T prefixes (binary, /1024) with optional trailing
// 'B'. Returns 0 on parse failure (and sets err). Used by the
// --streaming-cache-bytes flag.
size_t llama_expert_slab_parse_bytes(const std::string & s, std::string & err);
