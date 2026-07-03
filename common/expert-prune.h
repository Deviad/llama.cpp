// expert-prune.h — Story S25: per-expert pruning via router-logit mask.
//
// Loads a prune list (JSON written by the kitchen's
// derive_glm52_prune_list.py) and applies a router-logit mask before the
// top-K expert selection in build_moe_ffd. Pruned experts get
// selection_probs = -INFINITY, so the router's top-K never selects them.
//
// No weights are physically removed — the mask is a runtime router constraint.
// This preserves all model weights (so the experiment is reversible: drop
// the flag, inference is byte-identical again) and lets us sweep prune
// fraction with a single binary rather than re-quantizing per slice.
//
// The prune list JSON schema (subset we care about):
//   {
//     "per_layer_count": 9,
//     "n_pruned": 675,
//     "by_layer": {"3": [46, 55, ...], "4": [...], ...}
//   }
//
// CLI: --prune-experts PATH   (env: LLAMA_PRUNE_EXPERTS)
//
// This module is process-global: load once at cli.cpp startup, then
// build_moe_ffd reads the mask via expert_prune_layer_mask(ctx, il, n_expert).

#pragma once

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;

// Parse the JSON prune-list file and populate the static global prune state.
// Returns true on success; logs to stderr and returns false on failure
// (with `err` set on parse error).
// Idempotent: re-loading replaces the prior state. Safe to call multiple times.
bool expert_prune_load(const std::string & path, std::string & err);

// True if no prune list is loaded (so build_moe_ffd can skip the mask step).
bool expert_prune_empty();

// Number of pruned experts for `layer`, or 0 if the layer has no prune entries
// or the prune state is empty.
size_t expert_prune_count_for_layer(int32_t layer);

// Build a constant tensor of shape [n_expert] (or [n_expert, 1]) whose entries
// are -INFINITY for pruned expert IDs and 0 for unpruned. Returns nullptr if
// the layer has no pruned experts (or no prune state is loaded) — callers
// should treat nullptr as "no masking needed".
//
// The result is allocated in `ctx` and is intended to be added to
// selection_probs (broadcast across n_tokens) before top-K selection.
//
// DEPRECATED: host writes to mask->data at graph-build time do not propagate
// to the Metal backend (data is NULL pre-allocation). Use
// expert_prune_layer_ids() + build_inp_prune_mask() instead.
ggml_tensor * expert_prune_layer_mask(ggml_context * ctx, int32_t layer, int64_t n_expert);

// Return the list of pruned expert IDs for `layer` (empty if none / not loaded).
// The vector is a snapshot under the lock; callers can use it to fill a mask
// tensor's data in a graph-input set_input() callback (after backend alloc).
std::vector<int32_t> expert_prune_layer_ids(int32_t layer);
