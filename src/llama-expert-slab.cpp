// llama-expert-slab.cpp — routed-expert SSD streaming slab cache (Story S1).
// See llama-expert-slab.h for scope and design.

#include "llama-expert-slab.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#if defined(__unix__) || defined(__APPLE__)
#  include <sys/mman.h>   // mlock, munlock
#  include <unistd.h>    // sysconf(_SC_PAGESIZE)
#endif

#include "ggml-backend.h" // ggml_backend_dev_get/count, ggml_backend_dev_memory


size_t llama_expert_slab_recommended_working_set_bytes() {
    // Iterate registered backend devices; take the largest GPU-class device's
    // memory_total. On Apple Silicon the Metal device reports MTLDevice
    // .recommendedMaxWorkingSetSize here (see ggml-metal-device.m:853). On
    // non-Metal hosts the GPU device's memory_total is the VRAM ceiling.
    size_t best = 0;
    const int n_dev = (int) ggml_backend_dev_count();
    for (int i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        size_t free_b = 0, total_b = 0;
        ggml_backend_dev_memory(dev, &free_b, &total_b);
        if (total_b > best) best = total_b;
    }
    if (best == 0) {
        // No GPU-class device found; fall back to a conservative 16 GiB host
        // default so the 7/10 rule still produces a nonzero plan.
        best = 16ULL * 1024 * 1024 * 1024;
        fprintf(stderr,
            "expert-slab: no GPU backend device found; using 16 GiB fallback for recommended_working_set\n");
    }
    return best;
}


static uint64_t make_key(uint32_t layer, uint32_t expert_id) {
    return (uint64_t(layer) << 32) | uint32_t(expert_id);
}

llama_expert_slab_cache::~llama_expert_slab_cache() {
    reset();
}

void llama_expert_slab_cache::reset() {
    // Story S2: if the slab was chunked-mlock'd, release the lock first.
    if (slab && mlocked && mlocked_bytes > 0) {
        size_t chunk = LLAMA_EXPERT_SLAB_MLOCK_CHUNK;
        for (size_t off = 0; off < mlocked_bytes; off += chunk) {
            size_t n = std::min(chunk, mlocked_bytes - off);
            // munlock is best-effort; ignore failures during teardown.
            (void) munlock(slab + off, n);
        }
    }
    mlocked = false;
    mlocked_bytes = 0;
    prefetched_chunks.clear();
    if (slab) {
        std::free(slab);
        slab = nullptr;
    }
    slots.clear();
    map.clear();
    cache_experts = 0;
    slab_bytes    = 0;
    // per-expert geometry is preserved across reset() so a re-alloc can reuse it.
}

bool llama_expert_slab_cache::alloc(std::string & err) {
    if (per_expert_bytes == 0) {
        err = "per_expert_bytes is 0 (caller must set fragment strides first)";
        return false;
    }
    if (cache_experts == 0) {
        err = "cache_experts is 0 (slab disabled)";
        return false;
    }
    slab_bytes = per_expert_bytes * cache_experts;
    // Use aligned_alloc so the slab is page-friendly (S2 mlock needs this).
    // posix_memalign is universally available on the platforms llama.cpp targets.
    void * p = nullptr;
    if (posix_memalign(&p, 4096, slab_bytes) != 0 || p == nullptr) {
        err = "posix_memalign failed for slab_bytes=" + std::to_string(slab_bytes);
        slab = nullptr;
        return false;
    }
    slab = static_cast<uint8_t *>(p);
    std::memset(slab, 0, slab_bytes);
    slots.assign(cache_experts, slot_state{});
    map.clear();
    map.reserve(cache_experts * 2);
    n_hit = n_miss = n_prewarm_loaded = 0;
    return true;
}

size_t llama_expert_slab_cache::lookup(uint32_t layer, uint32_t expert_id) {
    if (!enabled()) {
        return SIZE_MAX;
    }
    auto it = map.find(make_key(layer, expert_id));
    if (it != map.end()) {
        n_hit++;
        return it->second;
    }
    n_miss++;
    return SIZE_MAX;
}

size_t llama_expert_slab_cache::load_on_miss(uint32_t layer, uint32_t expert_id,
                                             const uint8_t * mmap_base,
                                             const expert_layout & lay) {
    if (!enabled()) {
        return SIZE_MAX;
    }
    // Already present? short-circuit (counts as a hit, not a miss-load).
    size_t slot = lookup(layer, expert_id);
    if (slot != SIZE_MAX) {
        return slot;
    }
    // Find a free slot; if none, recycle slot 0 (S2/S5 will replace with LRU).
    size_t target = SIZE_MAX;
    for (size_t i = 0; i < slots.size(); ++i) {
        if (!slots[i].valid) { target = i; break; }
    }
    if (target == SIZE_MAX) {
        target = 0;
        // Evict the current occupant of slot 0 from the map.
        if (slots[0].valid) {
            map.erase(make_key(slots[0].layer, slots[0].expert_id));
        }
    }

    if (mmap_base == nullptr) {
        return SIZE_MAX; // nothing to read from
    }

    uint8_t * dst = slab + target * per_expert_bytes;
    // gate | up | down  (in that order within the slot)
    if (per_expert_bytes != gate_expert_bytes + up_expert_bytes + down_expert_bytes) {
        // geometry inconsistent — refuse rather than corrupt the slot
        return SIZE_MAX;
    }
    if (lay.gate_stride < gate_expert_bytes ||
        lay.up_stride   < up_expert_bytes   ||
        lay.down_stride < down_expert_bytes) {
        return SIZE_MAX;
    }
    std::memcpy(dst + 0,
                mmap_base + lay.gate_off + size_t(expert_id) * lay.gate_stride,
                gate_expert_bytes);
    std::memcpy(dst + gate_expert_bytes,
                mmap_base + lay.up_off + size_t(expert_id) * lay.up_stride,
                up_expert_bytes);
    std::memcpy(dst + gate_expert_bytes + up_expert_bytes,
                mmap_base + lay.down_off + size_t(expert_id) * lay.down_stride,
                down_expert_bytes);

    slots[target].layer     = layer;
    slots[target].expert_id = expert_id;
    slots[target].valid     = true;
    map[make_key(layer, expert_id)] = target;
    return target;
}

uint8_t * llama_expert_slab_cache::slot_ptr(size_t slot_idx) {
    if (!enabled() || slot_idx >= cache_experts) {
        return nullptr;
    }
    return slab + slot_idx * per_expert_bytes;
}

bool llama_expert_slab_cache::prefetch_one(const void * mmap_base, size_t offset,
                                          size_t len, uint32_t layer,
                                          uint32_t chunk_index, const char * tag) {
    uint64_t key = make_key(layer, chunk_index);
    auto it = prefetched_chunks.find(key);
    if (it != prefetched_chunks.end() && it->second) {
        // Already hinted this (layer, chunk) pair — de-dup (decode tokens don't
        // re-hint; decode reads go through the slab, not the mmap).
        return false;
    }
    int rc = llama_expert_slab_prefetch_willneed(
        mmap_base, offset, len, layer, tag, trace_prefetch);
    if (rc == 0) {
        prefetched_chunks[key] = true;
        return true;
    }
    return false;
}

void llama_expert_slab_cache::print_summary(const char * label) const {
    if (label) {
        fprintf(stderr, "%s: ", label);
    }
    fprintf(stderr,
        "expert-slab: per_expert_bytes=%zu (gate=%zu up=%zu down=%zu) "
        "cache_experts=%zu slab_bytes=%zu hits=%llu misses=%llu prewarm=%llu "
        "mlocked=%s mlocked_bytes=%zu budget_bytes=%zu (chunk=%zuMiB)\n",
        per_expert_bytes, gate_expert_bytes, up_expert_bytes, down_expert_bytes,
        cache_experts, slab_bytes,
        (unsigned long long) n_hit, (unsigned long long) n_miss,
        (unsigned long long) n_prewarm_loaded,
        mlocked ? "yes" : "no", mlocked_bytes, budget_bytes,
        LLAMA_EXPERT_SLAB_MLOCK_CHUNK / (1024*1024));
}

bool llama_expert_slab_cache::alloc_locked(size_t budget_bytes_in, std::string & err) {
    budget_bytes = budget_bytes_in;
    // First do the plain allocation (posix_memalign + zero).
    if (!alloc(err)) {
        return false;
    }
    if (budget_bytes == 0) {
        // No mlock requested — leave as a normal heap allocation.
        fprintf(stderr, "expert-slab: mlock disabled (budget_bytes=0); slab is ordinary heap\n");
        return true;
    }
    // Cap the locked prefix at min(slab_bytes, budget_bytes).
    size_t want = std::min(slab_bytes, budget_bytes);
    // Touch every page and mlock in 256 MiB chunks.
    size_t page = 4096; // posix_memalign already gave us 4096-aligned base
    size_t chunk = LLAMA_EXPERT_SLAB_MLOCK_CHUNK;
    for (size_t off = 0; off < want; off += chunk) {
        size_t n = std::min(chunk, want - off);
        uint8_t * p = slab + off;
        // Touch every page in this chunk with the ds4 write pattern p[pos]=pos/page.
        // This forces physical commitment before mlock so the kernel doesn't fail
        // on a COW zero page that hasn't been faulted in.
        for (size_t pos = 0; pos < n; pos += page) {
            p[pos] = (uint8_t)((pos / page) & 0xff);
        }
        if (mlock(p, n) != 0) {
            // Mid-chunk failure: roll back the already-locked prefix and fall back.
            int saved = errno;
            fprintf(stderr,
                "expert-slab: mlock failed at offset %zu (chunk %zu, errno=%d: %s); "
                "rolling back %zu bytes and falling back to non-locked heap\n",
                off, n, saved, std::strerror(saved), mlocked_bytes);
            for (size_t o2 = 0; o2 < mlocked_bytes; o2 += chunk) {
                size_t n2 = std::min(chunk, mlocked_bytes - o2);
                (void) munlock(slab + o2, n2);
            }
            mlocked = false;
            mlocked_bytes = 0;
            // Keep the heap slab (caller still has a working cache, just unlocked).
            fprintf(stderr,
                "expert-slab: fallback to non-locked heap (slab may be paged out under pressure)\n");
            return true; // allocation itself succeeded; mlock is best-effort
        }
        mlocked_bytes += n;
    }
    mlocked = (mlocked_bytes > 0);
    if (mlocked && mlocked_bytes < slab_bytes) {
        fprintf(stderr,
            "expert-slab: locked %zu of %zu slab bytes (budget cap); tail %zu bytes unlocked\n",
            mlocked_bytes, slab_bytes, slab_bytes - mlocked_bytes);
    }
    return true;
}

int llama_expert_slab_prefetch_willneed(const void * mmap_base, size_t offset,
                                        size_t len, uint32_t layer,
                                        const char * tag, bool trace) {
    if (mmap_base == nullptr || len == 0) {
        return 0; // nothing to hint
    }
#if defined(POSIX_MADV_WILLNEED) && (defined(__unix__) || defined(__APPLE__))
    const uint8_t * p = static_cast<const uint8_t *>(mmap_base) + offset;
    int rc = posix_madvise(const_cast<uint8_t *>(p), len, POSIX_MADV_WILLNEED);
    if (trace) {
        fprintf(stderr,
            "expert-slab prefetch: %s layer=%u offset=%zu len=%zu addr=%p rc=%d %s\n",
            tag ? tag : "?", (unsigned) layer, offset, len, (const void *) p,
            rc, rc == 0 ? "ok" : std::strerror(rc));
    }
    return rc;
#else
    // Platform without POSIX_MADV_WILLNEED — silent no-op (compile-guard).
    if (trace) {
        fprintf(stderr,
            "expert-slab prefetch: %s layer=%u offset=%zu len=%zu (no-op: POSIX_MADV_WILLNEED unavailable)\n",
            tag ? tag : "?", (unsigned) layer, offset, len);
    }
    return 0;
#endif
}


size_t llama_expert_slab_plan_cache(
    size_t recommended_working_set_bytes,
    size_t non_routed_bytes,
    size_t per_expert_bytes,
    size_t model_expert_count) {
    if (per_expert_bytes == 0 || model_expert_count == 0) {
        return 0;
    }
    // model_target = recommended * 4/5
    size_t model_target = (recommended_working_set_bytes * 4) / 5;
    if (model_target <= non_routed_bytes) {
        fprintf(stderr,
            "expert-slab: recommended_working_set=%zu * 4/5 = %zu is smaller than "
            "non_routed_bytes=%zu; slab disabled\n",
            recommended_working_set_bytes, model_target, non_routed_bytes);
        return 0;
    }
    size_t cache_bytes = model_target - non_routed_bytes;
    size_t cache_experts = cache_bytes / per_expert_bytes;
    if (cache_experts == 0) {
        fprintf(stderr,
            "expert-slab: cache_bytes=%zu < per_expert_bytes=%zu; slab disabled\n",
            cache_bytes, per_expert_bytes);
        return 0;
    }
    if (cache_experts > model_expert_count) {
        cache_experts = model_expert_count;
    }
    fprintf(stderr,
        "expert-slab plan: recommended=%zu model_target=%zu non_routed=%zu "
        "cache_bytes=%zu per_expert=%zu -> cache_experts=%zu (model has %zu)\n",
        recommended_working_set_bytes, model_target, non_routed_bytes,
        cache_bytes, per_expert_bytes, cache_experts, model_expert_count);
    return cache_experts;
}

size_t llama_expert_slab_parse_bytes(const std::string & s, std::string & err) {
    if (s.empty()) { err = "empty byte-size string"; return 0; }
    char * end = nullptr;
    errno = 0;
    double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str()) { err = "not a number: " + s; return 0; }
    if (errno == ERANGE) { err = "out of range: " + s; return 0; }
    // Suffix grammar (case-insensitive), binary (/1024):
    //   <num>            -> bytes
    //   <num>{k|m|g|t}[i][b]   e.g. 16g, 16gb, 16gib, 16G, 16384m, 2048
    // After the first suffix letter we optionally consume one 'i'/'I' then
    // optionally one 'b'/'B'. Anything else is an error.
    double mul = 1.0;
    if (*end != '\0') {
        char c = *end | 0x20; // lowercase
        switch (c) {
            case 'k': mul = 1024.0;                              break;
            case 'm': mul = 1024.0 * 1024.0;                    break;
            case 'g': mul = 1024.0 * 1024.0 * 1024.0;           break;
            case 't': mul = 1024.0 * 1024.0 * 1024.0 * 1024.0; break;
            default:
                err = std::string("unknown size suffix '") + *end + "' in " + s;
                return 0;
        }
        end++;
        if (*end == 'i' || *end == 'I') end++;
        if (*end == 'b' || *end == 'B') end++;
    }
    if (*end != '\0') {
        err = std::string("trailing characters '") + end + "' in " + s;
        return 0;
    }
    double bytes = v * mul;
    if (bytes < 0) { err = "negative: " + s; return 0; }
    return (size_t) bytes;
}
