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
#include <fstream>
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

    // --------------------------------------------------------------------
    // Test 6 (Story S3): per-region madvise(WILLNEED) helper + de-dup
    // --------------------------------------------------------------------
    {
        // The helper is a no-op safe call on any posix host (returns 0 on
        // success or no-op; nonzero only on posix_madvise failure, which we
        // don't expect for a valid in-range pointer).
        uint8_t buf[4096];
        std::memset(buf, 0, sizeof(buf));
        int rc = llama_expert_slab_prefetch_willneed(buf, 0, sizeof(buf),
                                                      7, "prefill", true);
        CHECK(rc == 0, "prefetch_willneed returns 0 on a valid buffer");

        // null base + zero len -> no-op, returns 0 (no dereference).
        CHECK(llama_expert_slab_prefetch_willneed(nullptr, 0, 0, 0, "x", false) == 0,
              "prefetch_willneed null/empty is a no-op returning 0");

        // De-dup discipline via the slab's prefetch_one(): a (layer, chunk)
        // pair is hinted exactly once; the second call is a de-dup no-op.
        llama_expert_slab_cache slab;
        slab.per_expert_bytes = 64; slab.gate_expert_bytes = 64;
        slab.up_expert_bytes = 0; slab.down_expert_bytes = 0;
        slab.cache_experts = 1;
        std::string err;
        CHECK(slab.alloc(err), "alloc slab for prefetch de-dup test");
        slab.trace_prefetch = false; // quiet
        // First hint for (layer=3, chunk=0) -> issues, returns true.
        CHECK(slab.prefetch_one(buf, 0, 4096, 3, 0, "prefill"),
              "first prefetch_one(L=3,chunk=0) issues a hint");
        // Second hint for the same pair -> de-duped, returns false.
        CHECK(!slab.prefetch_one(buf, 0, 4096, 3, 0, "prefill"),
              "second prefetch_one(L=3,chunk=0) is de-duped (returns false)");
        // Different chunk index for same layer -> issues a new hint.
        CHECK(slab.prefetch_one(buf, 0, 4096, 3, 1, "prefill"),
              "prefetch_one(L=3,chunk=1) issues a hint (different chunk)");
        // Different layer, same chunk -> issues a new hint.
        CHECK(slab.prefetch_one(buf, 0, 4096, 4, 0, "prefill"),
              "prefetch_one(L=4,chunk=0) issues a hint (different layer)");
        // After 3 distinct (layer, chunk) pairs, the map should have 3 entries.
        CHECK(slab.prefetched_chunks.size() == 3,
              "prefetched_chunks map has exactly 3 entries after 3 distinct hints");
        slab.reset();
        CHECK(slab.prefetched_chunks.empty(),
              "reset clears prefetched_chunks de-dup map");
    }

    // --------------------------------------------------------------------
    // Test 7 (Story S4): hotlist parser + note_moe_selection hit/miss counting
    // --------------------------------------------------------------------
    {
        // Write a synthetic ds4 v1 hotlist file.
        const char * hl_path = "/tmp/s4_test_hotlist.txt";
        FILE * hf = std::fopen(hl_path, "w");
        CHECK(hf != nullptr, "open synthetic hotlist for write");
        std::fputs("# ds4 expert hotlist v1\n# model GLM-5.2-test\n# layers 3\n"
                   "# experts 4\n# layer_records 6\n# selections 1000\n"
                   "# columns: layer expert hits weight\n"
                   "0 1 200 0.30\n"   // hottest
                   "1 2 150 0.22\n"
                   "0 3 100 0.15\n"
                   "2 0  90 0.13\n"
                   "1 1  60 0.09\n"
                   "2 3  40 0.06\n",
                   hf);
        std::fclose(hf);

        std::string err;
        auto hl = llama_expert_slab_hotlist_load(hl_path, err);
        CHECK(err.empty(), "hotlist parser: no error on valid file");
        CHECK(hl.size() == 6, "hotlist parser: 6 entries parsed");
        CHECK(hl[0].layer == 0 && hl[0].expert_id == 1, "hottest entry is (L0,E1)");
        CHECK(hl[0].weight == 0.30, "hottest entry weight=0.30");
        CHECK(hl[1].layer == 1 && hl[1].expert_id == 2, "second entry (L1,E2)");
        // descending weight order enforced
        bool sorted = true;
        for (size_t i = 1; i < hl.size(); ++i) {
            if (hl[i].weight > hl[i-1].weight) sorted = false;
        }
        CHECK(sorted, "hotlist entries sorted by descending weight");

        // Build a slab and pre-warm the top min(N, cache_experts)=3 entries
        // using load_on_miss (simulates the pre-warm path that copies expert
        // bytes from the mmap into slab slots). The slab's mmap here is a
        // synthetic buffer (we only need lookup counting to work).
        llama_expert_slab_cache slab;
        slab.per_expert_bytes = 192; slab.gate_expert_bytes = 64;
        slab.up_expert_bytes = 64; slab.down_expert_bytes = 64;
        slab.cache_experts = 3; // pre-warm top 3
        CHECK(slab.alloc(err), "alloc 3-slot slab for pre-warm test");
        // Synthetic mmap (not actually read for the counting test, but
        // load_on_miss needs a non-null base).
        std::vector<uint8_t> fake_mmap(4096, 0xAA);
        llama_expert_slab_cache::expert_layout lay{};
        lay.gate_off = 0; lay.gate_stride = 64;
        lay.up_off = 256; lay.up_stride = 64;
        lay.down_off = 512; lay.down_stride = 64;
        // Pre-warm top 3 entries: (0,1), (1,2), (0,3)
        size_t loaded = 0;
        for (size_t i = 0; i < hl.size() && loaded < slab.cache_experts; ++i) {
            size_t s = slab.load_on_miss(hl[i].layer, hl[i].expert_id,
                                         fake_mmap.data(), lay);
            if (s != SIZE_MAX) { loaded++; slab.slot_ready[s] = true; }
        }
        slab.hotlist_entries_loaded = (uint32_t) loaded;
        CHECK(slab.hotlist_entries_loaded == 3, "pre-warm loaded 3 of 6 hotlist entries");

        // Simulate the first token's MoE selections via note_moe_selection.
        // Suppose the first token selects these (layer, expert) pairs across
        // its forward pass: (0,1), (0,3), (1,2), (1,0), (2,0), (0,2).
        // Pre-warmed = {(0,1),(1,2),(0,3)} -> 3 hits, 3 misses -> 50% hit rate.
        struct Sel { uint32_t L, E; };
        Sel first_token_sels[] = {{0,1},{0,3},{1,2},{1,0},{2,0},{0,2}};
        for (auto & s : first_token_sels) {
            slab.note_moe_selection(s.L, s.E);
        }
        CHECK(slab.prewarm_hits == 3, "first-token: 3 pre-warm hits");
        CHECK(slab.prewarm_misses == 3, "first-token: 3 pre-warm misses");
        // 50% hit rate — satisfies the S4 smoke AC threshold.
        double rate = 100.0 * (double) slab.prewarm_hits /
                      (double)(slab.prewarm_hits + slab.prewarm_misses);
        CHECK(rate >= 50.0, "first-token hit rate >= 50% (S4 smoke threshold)");

        // Cold-start path: a fresh slab with no pre-warm has 0% hit rate but
        // still functions (lookup returns MISS, counting still works).
        llama_expert_slab_cache cold;
        cold.per_expert_bytes = 192; cold.gate_expert_bytes = 64;
        cold.up_expert_bytes = 64; cold.down_expert_bytes = 64;
        cold.cache_experts = 4;
        CHECK(cold.alloc(err), "alloc cold-start slab (no hotlist)");
        cold.note_moe_selection(5, 6);
        CHECK(cold.prewarm_hits == 0 && cold.prewarm_misses == 1,
              "cold-start: 0 hits, 1 miss (functions without hotlist)");

        std::remove(hl_path);
    }

    // ---- Story S5: runtime hotlist writeback + closed loop ----
    {
        std::string err; // local error sink for this block's CHECK()/save calls
        // Setup: enable writeback on a small slab, record selections,
        // writeback in ds4 v1, then reload and verify counts survive round-trip.
        llama_expert_slab_cache wb;
        wb.per_expert_bytes = 192; wb.gate_expert_bytes = 64;
        wb.up_expert_bytes = 64; wb.down_expert_bytes = 64;
        wb.cache_experts = 8;
        CHECK(wb.alloc(err), "S5: alloc slab for writeback");
        // Size the writeback table: 4 layers x 16 experts.
        wb.n_experts_table  = 16;
        wb.writeback_enabled = true;
        wb.selection_counts.assign(4 * 16, 0);
        // Record 5 selection events: (L2,E5), (L2,E5), (L2,E5), (L1,E9), (L3,E0).
        wb.note_moe_selection(2, 5); // L2E5 -> count 1
        wb.note_moe_selection(2, 5); // L2E5 -> count 2
        wb.note_moe_selection(2, 5); // L2E5 -> count 3
        wb.note_moe_selection(1, 9); // L1E9 -> count 1
        wb.note_moe_selection(3, 0); // L3E0 -> count 1
        // Total selections must match the sum
        CHECK(wb.selection_counts[2*16 + 5] == 3, "S5: L2E5 counted 3 times");
        CHECK(wb.selection_counts[1*16 + 9] == 1, "S5: L1E9 counted 1 time");
        CHECK(wb.selection_counts[3*16 + 0] == 1, "S5: L3E0 counted 1 time");

        // Writeback to disk in ds4 v1 format.
        const char * wb_path = "/tmp/test_expert_slab_s5_hotlist.txt";
        std::remove(wb_path);
        CHECK(wb.hotlist_save(wb_path, err), "S5: hotlist_save writes without error");

        // Verify the file header + sorted-by-descending-hits rows.
        std::ifstream f(wb_path);
        std::string line; bool saw_header = false;
        std::vector<std::string> rows;
        while (std::getline(f, line)) {
            if (line.rfind("# ds4 expert hotlist v1", 0) == 0) saw_header = true;
            if (!line.empty() && line[0] != '#' && line[0] != ' ') rows.push_back(line);
        }
        CHECK(saw_header, "S5: written hotlist has ds4 v1 header");
        // exact rows present, counted events with count>0 only
        CHECK(rows.size() == 3, "S5: writeback emits only nonzero (layer,expert) pairs");
        // rows are sorted by descending hits: 3,1,1
        CHECK(rows[0].rfind("2 5 3", 0) == 0, "S5: row 0 sorted-by-hits = (L2,E5,3)");

        // Closed loop: reload the written hotlist; counts survive round-trip.
        auto reloaded = llama_expert_slab_hotlist_load(wb_path, err);
        CHECK(reloaded.size() == 3, "S5: reloaded hotlist has same nonzero row count");
        CHECK(reloaded[0].layer == 2 && reloaded[0].expert_id == 5 &&
              reloaded[0].hits == 3, "S5: first row round-trips (L2,E5,3)");

        std::remove(wb_path);
    }

    if (failures == 0) {
        fprintf(stderr, "\nALL TESTS PASSED\n");
        return 0;
    }
    fprintf(stderr, "\n%d CHECK(s) FAILED\n", failures);
    return 1;
}
