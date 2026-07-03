// expert-prune.cpp — Story S25: per-expert pruning via router-logit mask.
//
// See expert-prune.h for design. Implementation: a process-global prune map
// loaded once at CLI startup, queried per-layer from build_moe_ffd.

#include "expert-prune.h"

#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <mutex>
#include <cmath>

// Tiny JSON parsing: we only need the "by_layer" object mapping
//  "LAYER_ID_INT" -> [EXPERT_ID_INT, ...]
// Use a hand-rolled strict parser to avoid adding a JSON dep to common/.
// Format of the prune list is well-defined; if it deviates, fail loud.

namespace {

struct PruneState {
    std::unordered_map<int32_t, std::vector<int32_t>> by_layer;
    bool loaded = false;
};

PruneState & state() {
    static PruneState s;
    return s;
}

std::mutex & mtx() {
    static std::mutex m;
    return m;
}

// Skip whitespace and C/C++-style comments inside a JSON stream parser.
// We only parse the by_layer object; everything else is skipped.
struct Cursor {
    const std::string & s;
    size_t i = 0;
    explicit Cursor(const std::string & src) : s(src) {}
    char peek() const { return i < s.size() ? s[i] : '\0'; }
    char next() { return i < s.size() ? s[i++] : '\0'; }
    void skip_ws() {
        while (i < s.size()) {
            char c = s[i];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { ++i; continue; }
            if (c == '/' && i + 1 < s.size()) {
                if (s[i+1] == '/') { while (i < s.size() && s[i] != '\n') ++i; continue; }
                if (s[i+1] == '*') {
                    i += 2;
                    while (i + 1 < s.size() && !(s[i] == '*' && s[i+1] == '/')) ++i;
                    if (i + 1 < s.size()) i += 2;
                    continue;
                }
            }
            break;
        }
    }
    bool expect(char c) {
        skip_ws();
        if (peek() == c) { ++i; return true; }
        return false;
    }
};

bool skip_number(Cursor & c) {
    // Skip a JSON number (int or float, with optional +/- sign and exponent)
    c.skip_ws();
    size_t start = c.i;
    if (c.peek() == '-' || c.peek() == '+') c.next();
    while (c.i < c.s.size()) {
        char ch = c.peek();
        if (std::isdigit((unsigned char)ch) || ch == '.' || ch == 'e' || ch == 'E'
                || ch == '+' || ch == '-') {
            c.next();
        } else {
            break;
        }
    }
    return c.i > start;
}

bool parse_int(Cursor & c, int & out) {
    c.skip_ws();
    size_t start = c.i;
    if (c.peek() == '-') c.next();
    while (c.i < c.s.size() && std::isdigit((unsigned char)c.peek())) c.next();
    if (c.i == start) return false;
    try { out = std::stoi(c.s.substr(start, c.i - start)); }
    catch (...) { return false; }
    return true;
}

bool parse_string(Cursor & c, std::string & out) {
    c.skip_ws();
    if (c.peek() != '"') return false;
    c.next();
    std::ostringstream oss;
    while (c.i < c.s.size() && c.peek() != '"') {
        char ch = c.next();
        if (ch == '\\' && c.i < c.s.size()) {
            char e = c.next();
            switch (e) {
                case 'n': oss << '\n'; break;
                case 't': oss << '\t'; break;
                case '"': oss << '"'; break;
                case '\\': oss << '\\'; break;
                default: oss << e; break;
            }
        } else {
            oss << ch;
        }
    }
    if (c.peek() != '"') return false;
    c.next();
    out = oss.str();
    return true;
}

bool skip_value(Cursor & c) {
    // Skip any JSON value (number, string, true, false, null, object, array)
    c.skip_ws();
    char ch = c.peek();
    if (ch == '"') { std::string s; return parse_string(c, s); }
    if (ch == '{' || ch == '[') {
        char close = (ch == '{') ? '}' : ']';
        c.next();
        c.skip_ws();
        if (c.peek() == close) { c.next(); return true; }
        while (true) {
            if (ch == '{') {
                std::string k;
                if (!parse_string(c, k)) return false;
                if (!c.expect(':')) return false;
                if (!skip_value(c)) return false;
            } else {
                if (!skip_value(c)) return false;
            }
            c.skip_ws();
            char nxt = c.peek();
            if (nxt == ',') { c.next(); continue; }
            if (nxt == close) { c.next(); return true; }
            return false;
        }
    }
    if (ch == 't' || ch == 'f' || ch == 'n') {
        // consume alphabetic run (true/false/null)
        while (c.i < c.s.size() && std::isalpha((unsigned char)c.peek())) c.next();
        return true;
    }
    return skip_number(c);
}

}  // namespace

bool expert_prune_load(const std::string & path, std::string & err) {
    std::ifstream f(path);
    if (!f) {
        err = "cannot open: " + path;
        return false;
    }
    std::ostringstream oss;
    oss << f.rdbuf();
    std::string s = oss.str();

    Cursor c(s);
    PruneState next;
    if (!c.expect('{')) {
        err = "expected top-level object";
        return false;
    }
    c.skip_ws();
    if (c.peek() == '}') {
        // empty object; load nothing
        std::lock_guard<std::mutex> lk(mtx());
        state() = next;
        state().loaded = true;
        return true;
    }
    while (true) {
        std::string key;
        if (!parse_string(c, key)) { err = "bad key"; return false; }
        if (!c.expect(':')) { err = "expected ':'"; return false; }
        if (key == "by_layer") {
            // parse { "LID": [ints], ... }
            if (!c.expect('{')) { err = "by_layer not object"; return false; }
            c.skip_ws();
            if (c.peek() == '}') { c.next(); }
            else while (true) {
                std::string lid;
                if (!parse_string(c, lid)) { err = "bad by_layer key"; return false; }
                if (!c.expect(':')) { err = "expected ':' in by_layer"; return false; }
                if (!c.expect('[')) { err = "by_layer value not array"; return false; }
                std::vector<int32_t> ids;
                c.skip_ws();
                if (c.peek() == ']') { c.next(); }
                else while (true) {
                    int v;
                    if (!parse_int(c, v)) { err = "bad expert int"; return false; }
                    ids.push_back(v);
                    c.skip_ws();
                    if (c.peek() == ',') { c.next(); continue; }
                    if (c.peek() == ']') { c.next(); break; }
                    err = "expected , or ] in by_layer array";
                    return false;
                }
                int layer_id;
                try { layer_id = std::stoi(lid); }
                catch (...) { err = "layer id not int: " + lid; return false; }
                next.by_layer[layer_id] = std::move(ids);
                c.skip_ws();
                if (c.peek() == ',') { c.next(); continue; }
                if (c.peek() == '}') { c.next(); break; }
                err = "expected , or } in by_layer";
                return false;
            }
        } else {
            // skip unknown value
            if (!skip_value(c)) { err = "bad value for " + key; return false; }
        }
        c.skip_ws();
        if (c.peek() == ',') { c.next(); continue; }
        if (c.peek() == '}') { c.next(); break; }
        err = "expected , or } at top level";
        return false;
    }

    size_t total = 0;
    for (auto & kv : next.by_layer) total += kv.second.size();
    std::lock_guard<std::mutex> lk(mtx());
    state() = next;
    state().loaded = true;
    fprintf(stderr, "%s: loaded prune list %s — %zu layers, %zu pruned experts total\n",
            __func__, path.c_str(), state().by_layer.size(), total);
    return true;
}

bool expert_prune_empty() {
    std::lock_guard<std::mutex> lk(mtx());
    return !state().loaded || state().by_layer.empty();
}

size_t expert_prune_count_for_layer(int32_t layer) {
    std::lock_guard<std::mutex> lk(mtx());
    if (!state().loaded) return 0;
    auto it = state().by_layer.find(layer);
    return it == state().by_layer.end() ? 0 : it->second.size();
}

std::vector<int32_t> expert_prune_layer_ids(int32_t layer) {
    std::lock_guard<std::mutex> lk(mtx());
    if (!state().loaded) return {};
    auto it = state().by_layer.find(layer);
    if (it == state().by_layer.end()) return {};
    return it->second;
}

ggml_tensor * expert_prune_layer_mask(ggml_context * ctx, int32_t layer, int64_t n_expert) {
    std::vector<int32_t> pruned;
    {
        std::lock_guard<std::mutex> lk(mtx());
        if (!state().loaded) return nullptr;
        auto it = state().by_layer.find(layer);
        if (it == state().by_layer.end() || it->second.empty()) return nullptr;
        pruned = it->second;
    }
    if (pruned.empty()) return nullptr;

    // Strategy: allocate a [n_expert] mask tensor in ctx0's pool, mark it as a
    // graph input (so its data buffer is host-allocated by the scheduler),
    // then write -INF for pruned experts and 0 for unpruned directly into the
    // tensor's data buffer. set_input() tags the tensor as a graph input so
    // the backend allocates a host-side buffer; direct host writes are safe
    // (mirrors how the kq_mask tensor is populated in llama-graph.cpp).
    ggml_tensor * mask = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_expert);
    if (!mask) {
        fprintf(stderr, "expert_prune_layer_mask: ggml_new_tensor_1d returned null!\n");
        return nullptr;
    }
    ggml_set_input(mask);
    float * data = (float *) mask->data;
    fprintf(stderr, "%s: layer=%d n_expert=%lld pruned=%zu mask_data=%p\n",
            __func__, layer, (long long) n_expert, pruned.size(), (void *) data);
    if (!data) {
        // The data buffer may be lazily allocated by the scheduler after
        // graph-building. Fall back to no-mask for this layer.
        return nullptr;
    }
    for (int64_t i = 0; i < n_expert; ++i) data[i] = 0.0f;
    for (int32_t e : pruned) {
        if (e >= 0 && e < (int32_t) n_expert) data[e] = -std::numeric_limits<float>::infinity();
    }
    return mask;
}
