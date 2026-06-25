// test-expert-slab.cpp — unit test for the routed-expert slab cache (Story S1).
//
// Validates the ACs in LLAMACPP_GLM52_ENHANCEMENT_PLAN.md Story S1 that do not
// require live inference: alloc, (layer,expert)->slot|MISS lookup, load_on_miss
// reads the right bytes from a synthetic mmap at the right offsets, hit returns
// slab bytes, and auto cache-plan sizing math is correct.
//
// Build: part of the tests/ CMake target via llama_build(test-expert-slab.cpp).
// Run:   build-metal/bin/test-expert-slab
//
// The synthetic mmap is a small byte buffer where expert i's gate/up/down
// regions are filled with distinguishable patterns so a copy error is obvious.

#include "../src/llama-expert-slab.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__); \
        failures++; \
    } else { \
        fprintf(stderr, "ok:   %s (line %d)\n", (msg), __LINE__); \
    } \
} while (0)

int main() {
    // --------------------------------------------------------------------
    // Test 1: plan_cache math (ds4 formula: target=recommended*4/5,
    //         cache_bytes = target - non_routed, cache_experts = cache_bytes /
    //         per_expert, clamped to model_expert_count).
    // --------------------------------------------------------------------
    {
        // recommended=100GB, non_routed=20GB, per_expert=1GB, model has 256 experts
        // -> target=80GB, cache_bytes=60GB, cache_experts=60
        size_t r = llama_expert_slab_plan_cache(
            100ULL * 1024 * 1024 * 1024,
            20ULL * 1024 * 1024 * 1024,
            1ULL  * 1024 * 1024 * 1024,
            256);
        CHECK(r == 60, "plan_cache: 100GB rec, 20GB non-routed, 1GB/expert -> 60 experts");

        // clamp to model expert count
        size_t r2 = llama_expert_slab_plan_cache(
            100ULL * 1024 * 1024 * 1024,
            20ULL * 1024 * 1024 * 1024,
            1ULL  * 1024 * 1024 * 1024,
            10); // only 10 experts in the model
        CHECK(r2 == 10, "plan_cache: clamps to model_expert_count=10");

        // too small -> 0 (disabled)
        size_t r3 = llama_expert_slab_plan_cache(
            1024, // 1KB recommended, absurdly small
            0,
            4096,
            256);
        CHECK(r3 == 0, "plan_cache: too-small budget -> 0 (disabled)");

        // non_routed exceeds target -> 0
        size_t r4 = llama_expert_slab_plan_cache(
            100ULL * 1024 * 1024 * 1024,
            200ULL * 1024 * 1024 * 1024, // bigger than target
            1ULL * 1024 * 1024 * 1024,
            256);
        CHECK(r4 == 0, "plan_cache: non_routed > target -> 0 (disabled)");
    }

    // --------------------------------------------------------------------
    // Test 2: byte-size parser
    // --------------------------------------------------------------------
    {
        std::string err;
        CHECK(llama_expert_slab_parse_bytes("2048", err)   == 2048, "parse: bare number");
        CHECK(llama_expert_slab_parse_bytes("16g", err)    == 16ULL*1024*1024*1024, "parse: 16g");
        CHECK(llama_expert_slab_parse_bytes("16gb", err)   == 16ULL*1024*1024*1024, "parse: 16gb");
        CHECK(llama_expert_slab_parse_bytes("16GiB", err)  == 16ULL*1024*1024*1024, "parse: 16GiB");
        CHECK(llama_expert_slab_parse_bytes("8M", err)     == 8ULL*1024*1024, "parse: 8M");
        CHECK(llama_expert_slab_parse_bytes("512m", err)   == 512ULL*1024*1024, "parse: 512m");
        CHECK(llama_expert_slab_parse_bytes("1T", err)     == 1ULL*1024*1024*1024*1024, "parse: 1T");
        CHECK(llama_expert_slab_parse_bytes("garbage", err) == 0, "parse: garbage -> 0");
        CHECK(llama_expert_slab_parse_bytes("16q", err)    == 0, "parse: bad suffix -> 0");
    }

    // --------------------------------------------------------------------
    // Test 3: AC — "constructs a tiny synthetic expert tensor, fills a 2-slot
    //         slab, verifies HIT returns slab bytes and MISS reads from the
    //         mmap at the correct offset."
    // --------------------------------------------------------------------
    {
        // Synthetic model: 2 layers (0,1), 4 experts each, each fragment = 64
        // bytes per expert (tiny). Layout in the synthetic mmap:
        //   layer 0 gate: [0,   256)   up: [256, 512)   down: [512, 768)
        //   layer 1 gate: [768, 1024)  up: [1024,1280)  down: [1280,1536)
        // per_expert stride per fragment = 64 (4 experts * 64 = 256 per fragment).
        const size_t n_experts = 4;
        const size_t frag_bytes = n_experts * 64; // 256 per fragment
        const size_t per_layer = frag_bytes * 3;  // 768
        const size_t n_layers = 2;
        std::vector<uint8_t> fake_mmap(n_layers * per_layer, 0);
        // Fill each expert region with a unique byte = layer*100 + expert_id*10 + frag_code
        // frag_code: gate=1, up=2, down=3. So expert (L,E) gate region is byte (L*100+E*10+1).
        auto fill = [&](uint32_t L, uint32_t E, int frag_code, size_t off) {
            uint8_t b = (uint8_t)(L * 100 + E * 10 + frag_code);
            std::memset(fake_mmap.data() + off, b, 64);
        };
        for (uint32_t L = 0; L < n_layers; ++L) {
            size_t base = L * per_layer;
            for (uint32_t E = 0; E < n_experts; ++E) {
                fill(L, E, 1, base + 0 * frag_bytes + E * 64);              // gate
                fill(L, E, 2, base + 1 * frag_bytes + E * 64);               // up
                fill(L, E, 3, base + 2 * frag_bytes + E * 64);               // down
            }
        }

        // Build a 2-slot slab. per_expert_bytes = 64+64+64 = 192.
        llama_expert_slab_cache slab;
        slab.gate_expert_bytes = 64;
        slab.up_expert_bytes   = 64;
        slab.down_expert_bytes = 64;
        slab.per_expert_bytes  = 192;
        slab.cache_experts     = 2;
        std::string err;
        CHECK(slab.alloc(err), "slab alloc (2 slots, 192 bytes/expert)");

        // Lookups on an empty slab should MISS.
        CHECK(slab.lookup(0, 0) == SIZE_MAX, "lookup on empty slab: MISS");
        CHECK(slab.n_miss == 1, "MISS counter incremented");
        CHECK(slab.n_hit  == 0, "HIT counter still 0");

        // Load expert (0,1) on miss. Its mmap layout:
        //   gate_off = 0*frag_bytes + 0 = 0;     gate_stride = 64
        //   up_off   = 1*frag_bytes;            up_stride   = 64
        //   down_off = 2*frag_bytes;            down_stride = 64
        llama_expert_slab_cache::expert_layout lay0;
        lay0.gate_off = 0;            lay0.gate_stride = 64;
        lay0.up_off   = frag_bytes;   lay0.up_stride   = 64;
        lay0.down_off = 2*frag_bytes; lay0.down_stride = 64;
        size_t s0 = slab.load_on_miss(0, 1, fake_mmap.data(), lay0);
        CHECK(s0 != SIZE_MAX, "load_on_miss (0,1) succeeds");
        CHECK(slab.slots[s0].valid, "slot marked valid after load");
        CHECK(slab.slots[s0].layer == 0 && slab.slots[s0].expert_id == 1,
              "slot records (layer=0, expert=1)");

        // Verify the slot bytes match what we wrote into the synthetic mmap.
        uint8_t * p = slab.slot_ptr(s0);
        CHECK(p != nullptr, "slot_ptr returns non-null");
        // gate region (64 bytes) should be byte (0*100 + 1*10 + 1) = 11
        CHECK(p[0] == 11 && p[63] == 11, "slot gate bytes = 11 (layer0 expert1 gate)");
        // up region = 12
        CHECK(p[64] == 12 && p[127] == 12, "slot up bytes = 12");
        // down region = 13
        CHECK(p[128] == 13 && p[191] == 13, "slot down bytes = 13");

        // HIT now: lookup(0,1) returns s0, bumps hit counter.
        CHECK(slab.lookup(0, 1) == s0, "lookup (0,1) now HITs the loaded slot");
        CHECK(slab.n_hit == 1, "HIT counter incremented");

        // Load a second expert (1,3) -> goes into a different slot.
        llama_expert_slab_cache::expert_layout lay1;
        lay1.gate_off = per_layer + 0;            lay1.gate_stride = 64;
        lay1.up_off   = per_layer + frag_bytes;   lay1.up_stride   = 64;
        lay1.down_off = per_layer + 2*frag_bytes; lay1.down_stride = 64;
        size_t s1 = slab.load_on_miss(1, 3, fake_mmap.data(), lay1);
        CHECK(s1 != SIZE_MAX && s1 != s0, "load_on_miss (1,3) -> different slot");
        uint8_t * q = slab.slot_ptr(s1);
        // gate byte for (1,3) = 1*100 + 3*10 + 1 = 131
        CHECK(q[0] == 131, "slot (1,3) gate byte = 131");
        CHECK(q[64] == 132, "slot (1,3) up byte = 132");
        CHECK(q[128] == 133, "slot (1,3) down byte = 133");

        // MISS on an unloaded expert still returns SIZE_MAX.
        CHECK(slab.lookup(0, 2) == SIZE_MAX, "lookup (0,2) MISS (not loaded)");

        // Loading a third expert evicts (the slab has 2 slots). S1 eviction is
        // "pick first free, else slot 0" — both slots are full, so slot 0 is
        // recycled. The (0,1) entry should be gone from the map, (1,3) preserved.
        size_t s2 = slab.load_on_miss(0, 2, fake_mmap.data(), lay0);
        CHECK(s2 == 0, "third load evicts slot 0 (LRU-ish fallback)");
        CHECK(slab.lookup(0, 1) == SIZE_MAX, "(0,1) evicted from map");
        CHECK(slab.lookup(0, 2) == 0, "(0,2) now resident in slot 0");
        CHECK(slab.lookup(1, 3) == s1, "(1,3) still resident");

        // load_on_miss with null mmap_base returns SIZE_MAX (nothing to read).
        CHECK(slab.load_on_miss(0, 0, nullptr, lay0) == SIZE_MAX,
              "load_on_miss with null mmap returns SIZE_MAX");

        slab.print_summary("test-expert-slab");
    }

    // --------------------------------------------------------------------
    // Test 4: enabled() / reset() lifecycle
    // --------------------------------------------------------------------
    {
        llama_expert_slab_cache slab;
        CHECK(!slab.enabled(), "fresh slab is disabled");
        slab.per_expert_bytes = 64; slab.gate_expert_bytes = 64;
        slab.up_expert_bytes = 0; slab.down_expert_bytes = 0;
        slab.cache_experts = 1;
        std::string err;
        CHECK(slab.alloc(err), "alloc 1-slot slab");
        CHECK(slab.enabled(), "slab enabled after alloc");
        slab.reset();
        CHECK(!slab.enabled(), "slab disabled after reset");
        CHECK(slab.slab == nullptr, "reset frees slab pointer");
    }

    // --------------------------------------------------------------------
    // Test 5 (Story S2): chunked mlock alloc_locked + budget rule
    // --------------------------------------------------------------------
    {
        // recommended_working_set resolver should return a nonzero value on
        // any real host (Metal device or 16 GiB fallback).
        size_t ws = llama_expert_slab_recommended_working_set_bytes();
        CHECK(ws > 0, "recommended_working_set_bytes > 0");

        // ds4 budget rule: budget = 7/10 * recommended_working_set.
        double budget = (double)ws * LLAMA_EXPERT_SLAB_BUDGET_FRACTION;
        CHECK(budget > 0, "7/10 budget is positive");

        // Build a slab large enough to exercise multi-chunk locking (2 MiB slab,
        // budget = 1 MiB so the locked prefix is capped below the full slab).
        llama_expert_slab_cache slab;
        slab.gate_expert_bytes = 1024 * 1024;       // 1 MiB gate per expert
        slab.up_expert_bytes   = 512 * 1024;        // 512 KiB up
        slab.down_expert_bytes = 512 * 1024;        // 512 KiB down
        slab.per_expert_bytes  = 2 * 1024 * 1024;   // 2 MiB per expert
        slab.cache_experts     = 1;                 // 2 MiB total slab
        std::string err;
        // budget 1 MiB < slab 2 MiB -> locked prefix capped, tail unlocked.
        CHECK(slab.alloc_locked(1024 * 1024, err), "alloc_locked (1 MiB budget, 2 MiB slab)");
        CHECK(slab.enabled(), "slab enabled after alloc_locked");
        // locked prefix should be <= budget and a multiple of the chunk size
        // (the last partial chunk is locked at its actual byte count).
        CHECK(slab.mlocked_bytes <= 1024 * 1024, "mlocked_bytes <= budget");
        CHECK(slab.mlocked_bytes > 0, "some bytes were locked");
        CHECK(slab.mlocked_bytes <= slab.slab_bytes, "mlocked prefix within slab");
        if (slab.mlocked_bytes < slab.slab_bytes) {
            // tail should be unlocked
            CHECK(slab.mlocked_bytes == 1024 * 1024,
                  "locked prefix == budget when slab > budget");
        }
        slab.reset();
        CHECK(!slab.mlocked, "reset clears mlocked flag");
        CHECK(slab.mlocked_bytes == 0, "reset clears mlocked_bytes");

        // alloc_locked with budget=0 -> normal heap, no mlock.
        llama_expert_slab_cache slab2;
        slab2.per_expert_bytes = 64; slab2.gate_expert_bytes = 64;
        slab2.up_expert_bytes = 0; slab2.down_expert_bytes = 0;
        slab2.cache_experts = 1;
        CHECK(slab2.alloc_locked(0, err), "alloc_locked budget=0 (no mlock)");
        CHECK(!slab2.mlocked, "budget=0 => not mlocked");
        CHECK(slab2.enabled(), "budget=0 slab still enabled");

        // S2 AC literal: "merge-sort baseline runs with the slab enabled at a
        // 16 GiB cache budget without mlock failure." The slab is not yet
        // wired into the model loader (that's S4), so this validates the
        // 16 GiB budget against the real Metal recommended_working_set
        // (>= 16 GiB / 0.7 here) and that alloc_locked(16 GiB) succeeds.
        llama_expert_slab_cache slab3;
        slab3.per_expert_bytes = 64; slab3.gate_expert_bytes = 64;
        slab3.up_expert_bytes = 0; slab3.down_expert_bytes = 0;
        slab3.cache_experts = 1;
        size_t budget_16g = 16ULL * 1024 * 1024 * 1024;
        // 7/10 rule: budget must be <= 0.7 * recommended_working_set
        CHECK(budget_16g <= (size_t)((double)ws * LLAMA_EXPERT_SLAB_BUDGET_FRACTION),
              "16 GiB budget respects 7/10 of recommended_working_set");
        CHECK(slab3.alloc_locked(budget_16g, err), "alloc_locked 16 GiB budget on Metal");
        // Slab is only 64 bytes, so mlocked_bytes == 64 (whole slab fits).
        CHECK(slab3.mlocked, "16 GiB budget mlocks the (tiny) slab");
        slab3.reset();
    }

    if (failures == 0) {
        fprintf(stderr, "\nALL TESTS PASSED\n");
        return 0;
    }
    fprintf(stderr, "\n%d CHECK(s) FAILED\n", failures);
    return 1;
}
