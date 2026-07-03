// llama-dspark-bridge.cpp — Story S29 (MTP drafter target adapter)
//
// A small long-lived subprocess that wraps a patched llama.cpp model
// (vendor/llama.cpp) and exposes per-token TARGET distributions + (optional)
// hidden-state embeddings to the Python `glm52_dspark` decoder over a
// length-prefixed stdin/stdout protocol. This is the bridge that lets
// DSparkDecoder talk to a real GLM-5.2 target without depending on
// llama-cpp-python.
//
// Protocol (one JSON request per line on stdin; response = JSON header line
// followed by raw little-endian binary bytes on stdout):
//
//   Request: {"op":"feed","tokens":[t1,t2,...]}             (int32 token IDs)
//   Response header: {"ok":true,"n":N,"v":V,"embd":E}\n
//     followed by N*V float32 logits, then (if E>0) N*E float32 embeddings.
//     Logits row i (V floats) is the model's next-token distribution given
//     the prefix [..session tokens.., t1, ..., ti]. (Standard causal LM.)
//
//   Request: {"op":"truncate","n":K}
//     Keeps KV positions [0, K) for seq 0, removes [K, end). Used on draft
//     rejection to roll back the un-accepted suffix.
//   Response header: {"ok":true,"pos_max":P}\n
//
//   Request: {"op":"reset"}
//     Clears the entire KV cache.
//   Response header: {"ok":true}\n
//
//   Request: {"op":"tokenize","text":"...","add_bos":true,"special":true}
//   Response header: {"ok":true,"n":N}\n  followed by N int32 token IDs.
//
//   Request: {"op":"info"}
//   Response header: {"ok":true,"v":V,"embd":E,"n_ctx":C}\n
//
//   Request: {"op":"exit"}
//     Clean shutdown.
//
// On any parse / decode error the bridge prints a header
//   {"ok":false,"err":"..."}\n
// with NO binary body, and continues serving.
//
// Build: see CMakeLists.txt (mirrors examples/trace-moe). Links llama-common
// + llama. Run from the kitchen root:
//   vendor/llama.cpp/build-metal/bin/llama-dspark-bridge -m <gguf> -ngl 999 -c 32768
// Optional flags: -t N (--threads), --embedding (expose hidden states; needed
// for the capture pipeline / trained backbone, not for correctness smoke).

#include "common.h"
#include "arg.h"
#include "chat.h"
#include "log.h"
#include "llama.h"
#include "../src/llama-ext.h" // Story S31: llama_set_embeddings_layer_inp / llama_get_embeddings_layer_inp (Option B multi-layer capture)

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <nlohmann/json.hpp>
#ifdef USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

// Binary write helpers (little-endian; x86_64 + arm64 are LE).
static void write_bytes(const void * p, size_t n) {
    std::fwrite(p, 1, n, stdout);
    std::fflush(stdout);
}
static void write_int32(int32_t v) { write_bytes(&v, sizeof(v)); }
static void write_floats(const float * p, size_t n) { write_bytes(p, n * sizeof(float)); }

// Story S31 AC4: read a binary payload from stdin (used by dspark_cycle_split
// to receive Python's p_d drafter distributions). Called AFTER std::getline
// consumed the JSON header line, so the stream pointer is right at the
// binary tail.
static bool read_bytes_into(void * dst, size_t n) {
    std::cin.read(static_cast<char *>(dst), (std::streamsize) n);
    return (size_t) std::cin.gcount() == n;
}

static void write_header_ok(int n_tokens, int n_vocab, int n_embd) {
    nlohmann::json h = {
        {"ok", true},
        {"n",  n_tokens},
        {"v",  n_vocab},
        {"embd", n_embd},
    };
    std::string s = h.dump();
    std::fwrite(s.data(), 1, s.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

static void write_header_ok_kv(const std::map<std::string, int64_t> & kv) {
    nlohmann::json h = {{"ok", true}};
    for (auto & [k, v] : kv) h[k] = v;
    std::string s = h.dump();
    std::fwrite(s.data(), 1, s.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

static void write_header_err(const std::string & msg) {
    nlohmann::json h = {{"ok", false}, {"err", msg}};
    std::string s = h.dump();
    std::fwrite(s.data(), 1, s.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

struct BridgeArgs {
    std::string model_path;
    int  ngl       = 999;
    int  n_ctx     = 32768;
    int  n_threads = -1;
    bool embedding = false;
    // Story S31 Option B: CSV list of target-layer indices to extract hidden
    // states from (DSpark Eq. 2: m layer hiddens at the anchor). When non-empty,
    // the feed/next_greedy ops append m*n_embd float32 values (the concatenated
    // [H(l_1);...;H(l_m)] for the LAST fed token) to the response body, and
    // the header carries "embd_layers": m. Default empty = current behavior.
    std::vector<int32_t> extract_layers;
};

// ---------------------------------------------------------------------------
// Story S30: trained DSpark drafter weights, loaded from a flat .bin file
// produced by `glm52_dspark.export_checkpoint`. All weights are float32,
// row-major contiguous. Pointers are into the mmap'd region (or NULL if no
// drafter has been loaded yet).
//
// The math mirrors `glm52_dspark/training.py::forward` EXACTLY (same equation
// order, same biases). The correctness test `test_cpp_drafter_forward.py`
// pins this against the Python path.
// ---------------------------------------------------------------------------
struct DrafterState {
    bool loaded = false;
    // Dimensions (copied from the .bin.json sidecar at load time).
    int E = 0;       // target hidden dim (6144 for GLM-5.2)
    int d = 0;       // drafter hidden dim (2048)
    int V = 0;       // vocab size (154880)
    int gamma = 0;   // block size (5)
    int r_markov = 0;// Markov head rank (256)
    int r_lm = 0;    // LM head low-rank (256)
    // mmap'd .bin region (kept alive for the process lifetime).
    void *  bin_base = nullptr;
    size_t  bin_size = 0;
    // Weight pointers into bin_base (offsets from the sidecar).
    const float * W_proj  = nullptr;  // (E, d)
    const float * b_proj  = nullptr;  // (d,)
    const float * pos_emb = nullptr;  // (gamma, d)
    const float * W1_mlp  = nullptr;  // (d, d)
    const float * b1_mlp  = nullptr;  // (d,)
    const float * W2_mlp  = nullptr;  // (d, d)
    const float * b2_mlp  = nullptr;  // (d,)
    const float * A       = nullptr;  // (d, r_lm)
    const float * B       = nullptr;  // (r_lm, V)
    const float * W1m     = nullptr;  // (V, r_markov)
    const float * W2m     = nullptr;  // (r_markov, V)
    const float * w_conf  = nullptr;  // (d + r_markov,)
    // Scratch buffers for the forward pass (avoid per-call allocation).
    std::vector<float> z0, z_k, m_k, h_k, h_proj, U_k, bias_k, p_logit;
    // p_d scratch (gamma positions of V-dim draft distributions) for the
    // fused dspark_cycle op. Reused across cycles.
    std::vector<float> p_d;            // (gamma, V)
    // RNG for the C++ rejection sampler (seeded at drafter_load).
    uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
};

static void print_usage(const char * argv0) {
    LOG_INF("usage: %s -m MODEL.gguf [-ngl N] [-c NCTX] [-t THREADS] [--embedding] [--extract-layers L1,L2,...,Lm]\n", argv0);
}

static BridgeArgs parse_args(int argc, char ** argv) {
    BridgeArgs a;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto next = [&](const char * flag) -> std::string {
            if (i + 1 >= argc) {
                LOG_ERR("missing value for %s\n", flag);
                std::exit(1);
            }
            return argv[++i];
        };
        if      (s == "-m"  || s == "--model")     a.model_path = next("-m");
        else if (s == "-ngl"|| s == "--n-gpu-layers") a.ngl = std::stoi(next("-ngl"));
        else if (s == "-c"  || s == "--ctx-size")  a.n_ctx = std::stoi(next("-c"));
        else if (s == "-t"  || s == "--threads")   a.n_threads = std::stoi(next("-t"));
        else if (s == "--embedding" || s == "--embeddings") a.embedding = true;
        else if (s == "--extract-layers") {
            // CSV list of target-layer indices (e.g. "9,39,68").
            std::string csv = next("--extract-layers");
            std::stringstream ss(csv);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) a.extract_layers.push_back((int32_t) std::stoi(item));
            }
            if (a.extract_layers.empty()) {
                LOG_ERR("--extract-layers requires at least one layer id\n");
                std::exit(1);
            }
        }
        else if (s == "-h"  || s == "--help")      { print_usage(argv[0]); std::exit(0); }
        else LOG_WRN("ignoring unknown arg: %s\n", s.c_str());
    }
    if (a.model_path.empty()) {
        LOG_ERR("error: -m MODEL is required\n");
        print_usage(argv[0]);
        std::exit(1);
    }
    return a;
}

// ---------------------------------------------------------------------------
// Story S30: drafter math helpers.
// ---------------------------------------------------------------------------

// Parse the .bin.json sidecar to find a tensor's byte_offset and shape.
struct TensorMeta {
    long byte_offset = 0;
    long byte_length = 0;
    std::vector<int> shape;
};
static TensorMeta find_tensor(const nlohmann::json & sidecar, const std::string & name) {
    TensorMeta m;
    for (const auto & t : sidecar["tensors"]) {
        if (t["name"] == name) {
            m.byte_offset = t["byte_offset"].get<long>();
            m.byte_length = t["byte_length"].get<long>();
            for (const auto & s : t["shape"]) m.shape.push_back(s.get<int>());
            return m;
        }
    }
    LOG_ERR("drafter tensor not found in sidecar: %s\n", name.c_str());
    return m;
}

// Map a tensor by name from the mmap'd .bin region.
static const float * map_tensor(const nlohmann::json & sidecar,
                                const char * base, const std::string & name) {
    TensorMeta m = find_tensor(sidecar, name);
    return reinterpret_cast<const float *>(base + m.byte_offset);
}

// y[B x M] = x[B x K] @ W[K x M], x row-major (B rows of K), W row-major
// (K rows of M). Output y row-major (B rows of M).
// CACHE-FRIENDLY gemv-style: for each output row b, walk W by rows (each
// W[k,:] is a contiguous M-vector) and accumulate x[b,k]*W[k,:] into y[b,:].
// This inner SAXPY loop is what BLAS optimizes and what the compiler
// auto-vectorizes; the naive column-walking alternative thrashes cache
// when M is large (V=154,880) and runs 10-100x slower.
#ifdef USE_ACCELERATE
// Use Accelerate's cblas_sgemm (column-major, but CblasRowMajor makes it
// row-major-transparent). This hits ~40-80 GFLOP/s on Apple Silicon vs
// ~5-10 GFLOP/s for the auto-vectorized naive loop below.
static void gemm(const float * x, const float * W, float * y, int B, int K, int M) {
    // y[B,M] = x[B,K] @ W[K,M], row-major. cblas row-major: M=B, N=Mout, K=K;
    // A=x (B,K) lda=K, B=W (K,Mout) ldb=Mout, C=y (B,Mout) ldc=Mout.
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
               B, M, K, 1.0f, x, K, W, M, 0.0f, y, M);
}
static void gemm_bias(const float * x, const float * W, const float * bias,
                      float * y, int B, int K, int M) {
    // y = x @ W + bias_row. Use beta=0 then add bias in a vectorized pass;
    // bias broadcasts across the B rows.
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
               B, M, K, 1.0f, x, K, W, M, 0.0f, y, M);
    if (bias) {
        for (int b = 0; b < B; b++) {
            float * yb = y + (size_t) b * M;
            cblas_saxpy(M, 1.0f, bias, 1, yb, 1);
        }
    }
}
#else
static void gemm(const float * x, const float * W, float * y, int B, int K, int M) {
    #pragma omp parallel for
    for (int b = 0; b < B; b++) {
        const float * xb = x + (size_t) b * K;
        float * yb = y + (size_t) b * M;
        for (int m = 0; m < M; m++) yb[m] = 0.0f;
        for (int k = 0; k < K; k++) {
            float xbk = xb[k];
            const float * Wk = W + (size_t) k * M;
            for (int m = 0; m < M; m++) yb[m] += xbk * Wk[m];
        }
    }
}
static void gemm_bias(const float * x, const float * W, const float * bias,
                      float * y, int B, int K, int M) {
    #pragma omp parallel for
    for (int b = 0; b < B; b++) {
        const float * xb = x + (size_t) b * K;
        float * yb = y + (size_t) b * M;
        if (bias) {
            for (int m = 0; m < M; m++) yb[m] = bias[m];
        } else {
            for (int m = 0; m < M; m++) yb[m] = 0.0f;
        }
        for (int k = 0; k < K; k++) {
            float xbk = xb[k];
            const float * Wk = W + (size_t) k * M;
            for (int m = 0; m < M; m++) yb[m] += xbk * Wk[m];
        }
    }
}
#endif

// argmax over a float array of length N (returns the index of the max).
static int argmax(const float * p, int N) {
    int best = 0;
    float bestv = p[0];
    for (int i = 1; i < N; i++) {
        if (p[i] > bestv) { bestv = p[i]; best = i; }
    }
    return best;
}

// numerically stable softmax in-place over the last axis. p has length N.
// also returns log(sum(exp)) for log-prob recovery.
static float softmax_inplace(float * p, int N) {
    float mx = p[0];
    for (int i = 1; i < N; i++) if (p[i] > mx) mx = p[i];
    float sum = 0.0f;
    for (int i = 0; i < N; i++) { float e = expf(p[i] - mx); p[i] = e; sum += e; }
    float inv = 1.0f / sum;
    for (int i = 0; i < N; i++) p[i] *= inv;
    return logf(sum) + mx;  // log partition function
}

static float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// xorshift64* — fast deterministic RNG for the C++ rejection sampler.
// NOTE: this is NOT numerically identical to numpy's RNG; the rejection
// sampler's output distribution is preserved regardless of the RNG stream
// (Leviathan-2023), so statistical equivalence (KL divergence over many
// samples) is the right correctness gate, not byte-equality.
static float rng_uniform(DrafterState & dr) {
    uint64_t x = dr.rng_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    dr.rng_state = x;
    uint64_t r = (x * 0x2545F4914F6CDD1DULL) >> 11;  // top 53 bits
    return (float) ((double) r / (double) (1ULL << 53));
}

// Sample from a probability distribution over V tokens via cumulative sum
// + binary search. Returns the sampled token id in [0, V).
static int sample_from_dist(const float * p, int V, DrafterState & dr) {
    // Build cumulative sum in a scratch buffer (reusing p_logit is risky if
    // concurrency; allocate a small one on the stack for binary search).
    // For V=154880 this is ~600KB — too big for stack. Use a thread-local.
    thread_local std::vector<float> cum;
    if ((int) cum.size() < V) cum.assign(V, 0.0f);
    float s = 0.0f;
    for (int i = 0; i < V; i++) { s += p[i]; cum[i] = s; }
    float r = rng_uniform(dr) * s;  // scale by the total mass (in case of drift)
    // Binary search for the first index where cum[i] >= r.
    int lo = 0, hi = V - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cum[mid] < r) lo = mid + 1; else hi = mid;
    }
    return lo;
}

int main(int argc, char ** argv) {

    std::setlocale(LC_NUMERIC, "C");
    common_init();

    BridgeArgs ba = parse_args(argc, argv);

    common_params params;
    params.model.path       = ba.model_path;
    params.n_gpu_layers     = ba.ngl;
    params.n_ctx            = ba.n_ctx;
    params.cpuparams.n_threads       = ba.n_threads;
    params.cpuparams_batch.n_threads = ba.n_threads;
    params.embedding        = ba.embedding;
    // Minimal sampler config — bridge emits RAW logits, caller does sampling.
    params.warmup           = false;  // skip warmup; the first feed is the real prompt

    llama_backend_init();
    llama_numa_init(params.numa);

    LOG_INF("loading model: %s (ngl=%d, n_ctx=%d, embedding=%d)\n",
            ba.model_path.c_str(), ba.ngl, ba.n_ctx, ba.embedding ? 1 : 0);

    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx  = init->context();
    if (!model || !ctx) {
        LOG_ERR("failed to load model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t V       = llama_vocab_n_tokens(vocab);
    const int32_t n_embd  = ba.embedding ? (int32_t) llama_model_n_embd(model) : 0;
    const int32_t n_extract = (int32_t) ba.extract_layers.size();
    const int32_t n_layer   = (int32_t) llama_model_n_layer(model);
    // Story S31 Option B: enable per-layer hidden-state extraction. Each lid
    // makes llama_get_embeddings_layer_inp(ctx, lid) return a valid pointer
    // (n_tokens × n_embd row-major) after the next llama_decode. Validate
    // each lid is in [0, n_layer) so a typo doesn't silently abort decode.
    for (int32_t lid : ba.extract_layers) {
        if (lid < 0 || lid >= n_layer) {
            LOG_ERR("--extract-layers id %d out of range [0, %d)\n", lid, n_layer);
            return 1;
        }
        llama_set_embeddings_layer_inp(ctx, (uint32_t) lid, true);
        LOG_INF("extract-layer enabled: lid=%d (input to that layer = output of layer %d)\n",
                lid, lid - 1);
    }
    const llama_seq_id seq_id = 0;

    // Story S29 AC1 (chat template): build a chat-template handle from the
    // loaded model so the bridge can apply the GLM Jinja template (and its
    // enable_thinking checkbox) internally — equivalent to
    // `llama-cli --jinja -cnv -st`. This is the path the DSpark training-
    // data capture pipeline uses to produce non-thinking-mode responses.
    common_chat_templates_ptr chat_tmpls = common_chat_templates_init(model, /*override*/ "", "", "");
    if (!chat_tmpls) {
        LOG_ERR("failed to init chat templates from model\n");
        return 1;
    }

    LOG_INF("ready: V=%d, n_embd=%d\n", V, n_embd);

    // Track the session's KV length so we can build batches with correct pos.
    llama_pos n_past = 0;
    const int n_batch = llama_n_batch(ctx);
    std::vector<llama_token> tok_buf;
    tok_buf.reserve(n_batch);

    // Story S30: drafter state (loaded lazily via the drafter_load op).
    DrafterState drafter;
    // Scratch for the hidden state collected from the most recent feed
    // (the target's h_anchor for the next draft cycle).
    std::vector<float> last_hidden;  // length n_embd, or empty if !out.embedding

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        if (line.back() == '\r') line.pop_back();
        nlohmann::json req;
        try {
            req = nlohmann::json::parse(line);
        } catch (const std::exception & e) {
            write_header_err(std::string("json parse: ") + e.what());
            continue;
        }
        if (!req.contains("op")) {
            write_header_err("missing 'op'");
            continue;
        }
        std::string op = req["op"];

        if (op == "exit" || op == "quit") {
            write_header_ok_kv({});
            break;
        } else if (op == "info") {
            write_header_ok_kv({{"v", V}, {"embd", n_embd}, {"n_ctx", (int64_t) ba.n_ctx}});
            continue;
        } else if (op == "reset") {
            llama_memory_clear(llama_get_memory(ctx), true);
            n_past = 0;
            write_header_ok_kv({});
            continue;
        } else if (op == "truncate") {
            int64_t keep = req.value("n", -1LL);
            if (keep < 0) { write_header_err("truncate requires n>=0"); continue; }
            // Remove KV positions [keep, end) for seq 0.
            llama_memory_seq_rm(llama_get_memory(ctx), seq_id, (llama_pos) keep, -1);
            n_past = (llama_pos) keep;
            llama_pos pmax = llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id);
            write_header_ok_kv({{"pos_max", (int64_t) pmax}});
            continue;
        } else if (op == "detokenize") {
            if (!req.contains("tokens") || !req["tokens"].is_array()) {
                write_header_err("detokenize requires tokens[]");
                continue;
            }
            std::string out;
            char buf[128];
            for (auto & t : req["tokens"]) {
                if (!t.is_number_integer()) {
                    write_header_err("tokens[] must be ints");
                    break;
                }
                llama_token tok = (llama_token) t.get<int32_t>();
                int32_t n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
                if (n > 0) out.append(buf, n);
            }
            write_header_ok_kv({{"n", (int64_t) out.size()}});
            if (!out.empty()) write_bytes(out.data(), out.size());
            continue;
        } else if (op == "chat_apply") {
            // Apply the GLM chat template (Jinja) to a messages array, then
            // tokenize the resulting prompt. Equivalent to
            // `llama-cli --jinja -cnv -st` prepended inputs.
            //
            // Request:
            //   {"op":"chat_apply",
            //    "messages":[{"role":"user","content":"..."}, ...],
            //    "add_generation_prompt": true,    // default true
            //    "enable_thinking": false,           // default true
            //    "use_jinja": true,                  // default true
            //    "add_bos": -1,                       // default -1 = use template default
            //    "add_eos": -1,                       // default -1 = use template default
            //    "kwargs":{"reasoning_effort":"null", ...}} // extra Jinja kwargs
            // Response:
            //   {"ok":true,"n":N}\n  + N*int32 binary
            if (!req.contains("messages") || !req["messages"].is_array()) {
                write_header_err("chat_apply requires messages[]");
                continue;
            }
            common_chat_templates_inputs inputs;
            inputs.add_generation_prompt = req.value("add_generation_prompt", true);
            inputs.use_jinja             = req.value("use_jinja", true);
            inputs.enable_thinking       = req.value("enable_thinking", true);
            if (req.contains("kwargs") && req["kwargs"].is_object()) {
                for (auto it = req["kwargs"].begin(); it != req["kwargs"].end(); ++it) {
                    if (it.value().is_string()) {
                        inputs.chat_template_kwargs[it.key()] = it.value().get<std::string>();
                    }
                }
            }
            // The common_chat_templates struct is opaque outside chat.cpp, so
            // we default add_bos/add_eos to false; the Jinja template itself
            // controls BOS/EOS placement. Callers can force via add_bos/add_eos.
            inputs.add_bos = req.value("add_bos", false);
            inputs.add_eos = req.value("add_eos", false);

            try {
                for (auto & m : req["messages"]) {
                    common_chat_msg msg;
                    msg.role    = m.value("role", "user");
                    msg.content = m.value("content", "");
                    inputs.messages.push_back(msg);
                }
            } catch (const std::exception & e) {
                write_header_err(std::string("messages parse: ") + e.what());
                continue;
            }

            std::string prompt = common_chat_templates_apply(chat_tmpls.get(), inputs).prompt;

            std::vector<llama_token> toks = common_tokenize(ctx, prompt, /*add_bos*/ false, /*special*/ true);
            write_header_ok_kv({{"n", (int64_t) toks.size()}});
            if (!toks.empty()) {
                write_bytes(toks.data(), toks.size() * sizeof(llama_token));
            }
            continue;
        } else if (op == "tokenize") {
            std::string text = req.value("text", std::string());
            bool add_bos = req.value("add_bos", false);
            bool special = req.value("special", true);
            std::vector<llama_token> toks = common_tokenize(ctx, text, add_bos, special);
            write_header_ok_kv({{"n", (int64_t) toks.size()}});
            if (!toks.empty()) write_bytes(toks.data(), toks.size() * sizeof(int32_t));
            continue;
        } else if (op == "feed") {
            if (!req.contains("tokens") || !req["tokens"].is_array()) {
                write_header_err("feed requires tokens[]");
                continue;
            }
            auto & arr = req["tokens"];
            tok_buf.clear();
            bool bad = false;
            for (auto & t : arr) {
                if (!t.is_number_integer()) {
                    write_header_err("tokens[] must be ints");
                    bad = true;
                    break;
                }
                tok_buf.push_back((llama_token) t.get<int32_t>());
            }
            if (bad) continue;
            // Decode in n_batch-sized chunks.
            bool ok = true;
            size_t idx = 0;
            // We'll collect logits+embeddings for ALL fed positions, in order.
            std::vector<float> all_logits;
            std::vector<float> all_embd;
            all_logits.reserve(tok_buf.size() * (size_t) V);
            if (ba.embedding) all_embd.reserve(tok_buf.size() * (size_t) n_embd);

            // Build batch of at most n_batch tokens at a time; positions must
            // continue from n_past.
            while (idx < tok_buf.size()) {
                size_t chunk = std::min((size_t) n_batch, tok_buf.size() - idx);
                llama_batch batch = llama_batch_init((int) chunk, 0, 1);
                for (size_t i = 0; i < chunk; i++) {
                    batch.token[i]    = tok_buf[idx + i];
                    batch.pos[i]      = n_past + (llama_pos) i;
                    batch.n_seq_id[i] = 1;
                    batch.seq_id[i][0] = seq_id;
                    batch.logits[i]   = 1;  // emit logits for every position
                }
                batch.n_tokens = (int) chunk;
                if (llama_decode(ctx, batch) != 0) {
                    LOG_ERR("llama_decode failed at idx=%zu n_past=%d\n", idx, (int) n_past);
                    ok = false;
                    llama_batch_free(batch);
                    break;
                }
                // Collect logits + embeddings for each position in this chunk.
                for (size_t i = 0; i < chunk; i++) {
                    const float * lg = llama_get_logits_ith(ctx, (int32_t) i);
                    if (!lg) { LOG_ERR("get_logits_ith(%zu) null\n", i); ok = false; break; }
                    all_logits.insert(all_logits.end(), lg, lg + V);
                    if (ba.embedding) {
                        const float * em = llama_get_embeddings_ith(ctx, (int32_t) i);
                        if (em) all_embd.insert(all_embd.end(), em, em + n_embd);
                    }
                }
                n_past += (llama_pos) chunk;
                idx += chunk;
                llama_batch_free(batch);
            }

            if (!ok) { write_header_err("llama_decode failed (see stderr)"); continue; }
            // Cache the last fed token's hidden state for the trained drafter
            // (h_anchor for the next draft cycle). Mirrors the Python adapter.
            if (ba.embedding && !all_embd.empty()) {
                last_hidden.assign(all_embd.end() - n_embd, all_embd.end());
            }
            write_header_ok((int) tok_buf.size(), V, n_embd);
            write_floats(all_logits.data(), all_logits.size());
            if (ba.embedding) write_floats(all_embd.data(), all_embd.size());
            continue;
        } else if (op == "next_greedy") {
            // Fast single-token greedy advance for the capture sampling pass.
            // Returns ONLY the argmax token id (int32) + hidden (E float32) —
            // NOT the full V-dim logits. This cuts the per-token pipe transfer
            // from ~600KB (full V) to ~24KB (just hidden), eliminating the
            // pipe I/O bottleneck during the 1024-token greedy response pass.
            if (!req.contains("token") || !req["token"].is_number_integer()) {
                write_header_err("next_greedy requires token:int");
                continue;
            }
            llama_token tok = (llama_token) req["token"].get<int32_t>();
            llama_batch batch = llama_batch_init(1, 0, 1);
            batch.token[0]    = tok;
            batch.pos[0]       = n_past;
            batch.n_seq_id[0]  = 1;
            batch.seq_id[0][0] = seq_id;
            batch.logits[0]    = 1;
            batch.n_tokens     = 1;
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("next_greedy llama_decode failed\n");
                llama_batch_free(batch);
                write_header_err("next_greedy decode failed");
                continue;
            }
            const float * lg = llama_get_logits_ith(ctx, 0);
            int best = lg ? argmax(lg, V) : 0;
            if (ba.embedding) {
                const float * em = llama_get_embeddings_ith(ctx, 0);
                if (em) last_hidden.assign(em, em + n_embd);
            }
            // Story S31 Option B: gather m-layer context for this single token.
            std::vector<float> h_ctx_layers;
            if (n_extract > 0) {
                h_ctx_layers.reserve((size_t) n_extract * n_embd);
                for (int32_t k = 0; k < n_extract; k++) {
                    const float * layer = llama_get_embeddings_layer_inp(ctx, (uint32_t) ba.extract_layers[k]);
                    if (!layer) { LOG_ERR("next_greedy: get_embeddings_layer_inp(lid=%d) null\n", ba.extract_layers[k]); break; }
                    h_ctx_layers.insert(h_ctx_layers.end(), layer, layer + n_embd);
                }
            }
            llama_batch_free(batch);
            n_past += 1;
            // Header: ok=1, n=1, v=V, embd=n_embd, embd_layers=m (if extract).
            // Body: argmax token id (int32) + hidden (E float32) if embedding
            //       + m*E float32 (concat layer hiddens) if extract-layers.
            if (n_extract > 0) {
                write_header_ok_kv({
                    {"n", 1}, {"v", V}, {"embd", n_embd},
                    {"embd_layers", (int64_t) n_extract},
                });
            } else {
                write_header_ok(1, V, n_embd);
            }
            int32_t best_tok = best;
            write_bytes(&best_tok, sizeof(int32_t));
            if (ba.embedding) write_floats(last_hidden.data(), (size_t) n_embd);
            if (n_extract > 0) write_floats(h_ctx_layers.data(), h_ctx_layers.size());
            continue;
        } else if (op == "drafter_load") {
            // Story S30 AC1: load the trained drafter weights from a flat
            // .bin file produced by glm52_dspark.export_checkpoint.
            if (!req.contains("bin") || !req["bin"].is_string()) {
                write_header_err("drafter_load requires bin=<path>");
                continue;
            }
            std::string bin_path = req["bin"].get<std::string>();
            std::string json_path = bin_path + ".json";
            // Read the .bin.json sidecar.
            std::ifstream jf(json_path);
            if (!jf) { write_header_err("cannot open sidecar: " + json_path); continue; }
            nlohmann::json sidecar;
            jf >> sidecar;
            // mmap the .bin.
            int fd = open(bin_path.c_str(), O_RDONLY);
            if (fd < 0) { write_header_err("cannot open bin: " + bin_path); continue; }
            struct stat st;
            if (fstat(fd, &st) != 0) { close(fd); write_header_err("fstat failed"); continue; }
            size_t sz = (size_t) st.st_size;
            void * base = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
            if (base == MAP_FAILED) { close(fd); write_header_err("mmap failed"); continue; }
            // Close fd — the mapping stays alive (MAP_PRIVATE keeps it).
            close(fd);
            // Bind weight pointers from the sidecar offsets.
            const char * b = static_cast<const char *>(base);
            drafter.bin_base = base;
            drafter.bin_size = sz;
            drafter.E        = sidecar.value("E", 0);
            drafter.d        = sidecar.value("d", 0);
            drafter.V        = sidecar.value("V", 0);
            drafter.gamma    = sidecar.value("gamma", 0);
            drafter.r_markov = sidecar.value("r_markov", 0);
            drafter.r_lm     = sidecar.value("r_lm", 0);
            drafter.W_proj  = map_tensor(sidecar, b, "W_proj");
            drafter.b_proj  = map_tensor(sidecar, b, "b_proj");
            drafter.pos_emb = map_tensor(sidecar, b, "pos_emb");
            drafter.W1_mlp  = map_tensor(sidecar, b, "W1_mlp");
            drafter.b1_mlp  = map_tensor(sidecar, b, "b1_mlp");
            drafter.W2_mlp  = map_tensor(sidecar, b, "W2_mlp");
            drafter.b2_mlp  = map_tensor(sidecar, b, "b2_mlp");
            drafter.A       = map_tensor(sidecar, b, "A");
            drafter.B       = map_tensor(sidecar, b, "B");
            drafter.W1m     = map_tensor(sidecar, b, "W1m");
            drafter.W2m     = map_tensor(sidecar, b, "W2m");
            drafter.w_conf = map_tensor(sidecar, b, "w_conf");
            // Dim check vs the loaded model.
            if (drafter.V != V) {
                write_header_err("drafter V mismatch: ckpt V=" + std::to_string(drafter.V)
                                 + " model V=" + std::to_string(V));
                drafter.loaded = false;
                continue;
            }
            if (ba.embedding && drafter.E != n_embd) {
                write_header_err("drafter E mismatch: ckpt E=" + std::to_string(drafter.E)
                                 + " model n_embd=" + std::to_string(n_embd));
                drafter.loaded = false;
                continue;
            }
            // Allocate scratch buffers sized for this checkpoint.
            int d = drafter.d, g = drafter.gamma, Vd = drafter.V;
            int rlm = drafter.r_lm, rmk = drafter.r_markov, E = drafter.E;
            drafter.z0.assign(d, 0.0f);
            drafter.z_k.assign((size_t) g * d, 0.0f);
            drafter.m_k.assign((size_t) g * d, 0.0f);
            drafter.h_k.assign((size_t) g * d, 0.0f);
            drafter.h_proj.assign((size_t) g * rlm, 0.0f);
            drafter.U_k.assign((size_t) g * Vd, 0.0f);
            drafter.bias_k.assign((size_t) g * Vd, 0.0f);
            drafter.p_d.assign((size_t) g * Vd, 0.0f);
            drafter.p_logit.assign((size_t) Vd, 0.0f);
            drafter.loaded = true;
            LOG_INF("drafter loaded: d=%d gamma=%d r_markov=%d r_lm=%d V=%d E=%d\n",
                    d, g, rmk, rlm, Vd, E);
            write_header_ok_kv({
                {"loaded", 1}, {"d", d}, {"gamma", g},
                {"r_markov", rmk}, {"r_lm", rlm}, {"V", Vd}, {"E", E},
            });
            continue;
        } else if (op == "drafter_forward") {
            // Story S30 AC2: run the trained drafter in C++ and return
            // draft_tokens[gamma] (greedy argmax) + draft_logprobs[gamma] +
            // confidences[gamma]. Uses cached last_hidden as h_anchor.
            if (!drafter.loaded) {
                write_header_err("drafter not loaded; call drafter_load first");
                continue;
            }
            if (!req.contains("anchor_token") || !req["anchor_token"].is_number_integer()) {
                write_header_err("drafter_forward requires anchor_token:int");
                continue;
            }
            int anchor_tok = req["anchor_token"].get<int>();
            if (anchor_tok < 0 || anchor_tok >= drafter.V) {
                write_header_err("anchor_token out of range [0, V)");
                continue;
            }
            if (last_hidden.empty()) {
                write_header_err("no last_hidden cached; run a feed with --embedding first");
                continue;
            }
            const int d = drafter.d, g = drafter.gamma, Vd = drafter.V;
            const int rlm = drafter.r_lm, rmk = drafter.r_markov;
            const float * h_anchor = last_hidden.data();
            // 1. z0 = h_anchor @ W_proj + b_proj   (E,) x (E,d) -> (d,)
            gemm_bias(h_anchor, drafter.W_proj, drafter.b_proj,
                      drafter.z0.data(), 1, drafter.E, d);
            // 2. z_k[k] = z0 + pos_emb[k]  (gamma positions, each d-dim)
            #pragma omp parallel for collapse(2)
            for (int k = 0; k < g; k++) {
                for (int i = 0; i < d; i++) {
                    drafter.z_k[(size_t) k * d + i] = drafter.z0[i] + drafter.pos_emb[(size_t) k * d + i];
                }
            }
            // 3. m_k = relu(z_k @ W1_mlp + b1_mlp)   (g,d) x (d,d) -> (g,d)
            gemm_bias(drafter.z_k.data(), drafter.W1_mlp, drafter.b1_mlp,
                      drafter.m_k.data(), g, d, d);
            #pragma omp parallel for collapse(2)
            for (int k = 0; k < g; k++) {
                for (int i = 0; i < d; i++) {
                    float v = drafter.m_k[(size_t) k * d + i];
                    drafter.m_k[(size_t) k * d + i] = v > 0.0f ? v : 0.0f;
                }
            }
            // 4. h_k = m_k @ W2_mlp + b2_mlp   (g,d) x (d,d) -> (g,d)
            gemm_bias(drafter.m_k.data(), drafter.W2_mlp, drafter.b2_mlp,
                      drafter.h_k.data(), g, d, d);
            // 5. h_proj = h_k @ A   (g,d) x (d,rlm) -> (g,rlm)
            gemm_bias(drafter.h_k.data(), drafter.A, nullptr,
                      drafter.h_proj.data(), g, d, rlm);
            // 6. U_k = h_proj @ B   (g,rlm) x (rlm,V) -> (g,V) — the big one.
            gemm(drafter.h_proj.data(), drafter.B, drafter.U_k.data(), g, rlm, Vd);
            // 7. Sequential sampling: for k=0..gamma-1, add the Markov head
            //    bias_k = W1m[prev_tok] @ W2m, softmax, argmax -> draft token.
            //    The first position uses anchor_tok as prev. Each subsequent
            //    position uses the previously sampled draft token.
            std::vector<int32_t> draft_tokens(g);
            std::vector<float>   draft_logprobs(g);
            std::vector<float>   confidences(g);
            int prev_tok = anchor_tok;
            for (int k = 0; k < g; k++) {
                // bias_k = W1m[prev_tok] @ W2m   (rmk,) x (rmk,V) -> (V,)
                const float * emb_prev = drafter.W1m + (size_t) prev_tok * rmk;
                float * bias_k = drafter.bias_k.data() + (size_t) k * Vd;
                gemm(emb_prev, drafter.W2m, bias_k, 1, rmk, Vd);
                // p_logit = U_k[k] + bias_k  (we only need one position at
                // a time; reuse the single-V scratch for the softmax).
                const float * U_row = drafter.U_k.data() + (size_t) k * Vd;
                #pragma omp parallel for
                for (int v = 0; v < Vd; v++) {
                    drafter.p_logit[v] = U_row[v] + bias_k[v];
                }
                // softmax over V (the log partition is for log-prob recovery)
                (void) softmax_inplace(drafter.p_logit.data(), Vd);
                int chosen = argmax(drafter.p_logit.data(), Vd);
                draft_tokens[k] = chosen;
                draft_logprobs[k] = logf(drafter.p_logit[chosen]);
                prev_tok = chosen;
                // Confidence head: c_k = sigmoid(w_conf . [h_k ; W1m[prev_tok]])
                //  where prev_tok here is the prev DRAFT token (the token
                //  we just sampled), matching training.py's semantics.
                float conf_logit = 0.0f;
                const float * hk = drafter.h_k.data() + (size_t) k * d;
                for (int i = 0; i < d; i++) conf_logit += drafter.w_conf[i] * hk[i];
                for (int j = 0; j < rmk; j++) {
                    conf_logit += drafter.w_conf[d + j] * emb_prev[j];
                }
                confidences[k] = sigmoidf(conf_logit);
            }
            // Return: header + draft_tokens (g int32) + logprobs (g f32)
            // + confidences (g f32). (Distributions are NOT shipped — the
            // verifier will run in C++ in the dspark_cycle op.)
            write_header_ok_kv({
                {"n_draft", g}, {"V", Vd},
            });
            write_bytes(draft_tokens.data(), g * sizeof(int32_t));
            write_floats(draft_logprobs.data(), g);
            write_floats(confidences.data(), g);
            continue;
        } else if (op == "dspark_cycle") {
            // Story S30 AC3: the fused zero-overhead cycle. Runs the C++
            // drafter + C++ target feed + C++ rejection verifier + KV
            // rollback all in one op. Returns ONLY accepted_tokens +
            // n_accepted (no full-V distributions cross the boundary).
            if (!drafter.loaded) {
                write_header_err("drafter not loaded; call drafter_load first");
                continue;
            }
            if (!req.contains("anchor_token") || !req["anchor_token"].is_number_integer()) {
                write_header_err("dspark_cycle requires anchor_token:int");
                continue;
            }
            int anchor_tok = req["anchor_token"].get<int>();
            if (anchor_tok < 0 || anchor_tok >= drafter.V) {
                write_header_err("anchor_token out of range [0, V)");
                continue;
            }
            if (last_hidden.empty()) {
                write_header_err("no last_hidden cached; run a feed with --embedding first");
                continue;
            }
            // Optional seed override (else the session RNG continues).
            if (req.contains("seed") && req["seed"].is_number_integer()) {
                drafter.rng_state = (uint64_t) req["seed"].get<int64_t>() | 0x9E3779B97F4A7C15ULL;
            }
            const int d = drafter.d, g = drafter.gamma, Vd = drafter.V;
            const int rlm = drafter.r_lm, rmk = drafter.r_markov;
            const float * h_anchor = last_hidden.data();

            // --- 1. C++ drafter forward (same as drafter_forward), keeping
            // the per-position draft distributions in p_d for the verifier. ---
            gemm_bias(h_anchor, drafter.W_proj, drafter.b_proj, drafter.z0.data(), 1, drafter.E, d);
            #pragma omp parallel for collapse(2)
            for (int k = 0; k < g; k++)
                for (int i = 0; i < d; i++)
                    drafter.z_k[(size_t) k * d + i] = drafter.z0[i] + drafter.pos_emb[(size_t) k * d + i];
            gemm_bias(drafter.z_k.data(), drafter.W1_mlp, drafter.b1_mlp, drafter.m_k.data(), g, d, d);
            #pragma omp parallel for collapse(2)
            for (int k = 0; k < g; k++)
                for (int i = 0; i < d; i++) {
                    float v = drafter.m_k[(size_t) k * d + i];
                    drafter.m_k[(size_t) k * d + i] = v > 0.0f ? v : 0.0f;
                }
            gemm_bias(drafter.m_k.data(), drafter.W2_mlp, drafter.b2_mlp, drafter.h_k.data(), g, d, d);
            gemm_bias(drafter.h_k.data(), drafter.A, nullptr, drafter.h_proj.data(), g, d, rlm);
            gemm(drafter.h_proj.data(), drafter.B, drafter.U_k.data(), g, rlm, Vd);
            // Sequential greedy sampling; store each position's softmax in p_d.
            std::vector<int32_t> draft_tokens(g);
            std::vector<float>   confidences(g);
            int prev_tok = anchor_tok;
            for (int k = 0; k < g; k++) {
                const float * emb_prev = drafter.W1m + (size_t) prev_tok * rmk;
                float * bias_k = drafter.bias_k.data() + (size_t) k * Vd;
                gemm(emb_prev, drafter.W2m, bias_k, 1, rmk, Vd);
                const float * U_row = drafter.U_k.data() + (size_t) k * Vd;
                float * p_d_k = drafter.p_d.data() + (size_t) k * Vd;
                #pragma omp parallel for
                for (int v = 0; v < Vd; v++) p_d_k[v] = U_row[v] + bias_k[v];
                (void) softmax_inplace(p_d_k, Vd);
                draft_tokens[k] = (int32_t) argmax(p_d_k, Vd);
                prev_tok = draft_tokens[k];
                // Confidence head.
                float conf_logit = 0.0f;
                const float * hk = drafter.h_k.data() + (size_t) k * d;
                for (int i = 0; i < d; i++) conf_logit += drafter.w_conf[i] * hk[i];
                for (int j = 0; j < rmk; j++) conf_logit += drafter.w_conf[d + j] * emb_prev[j];
                confidences[k] = sigmoidf(conf_logit);
            }

            // --- 2. C++ target feed of [anchor, *draft_tokens]. ---
            // Build the feed batch (reuse the existing decode path inline).
            std::vector<llama_token> feed_tokens;
            feed_tokens.push_back((llama_token) anchor_tok);
            for (int k = 0; k < g; k++) feed_tokens.push_back((llama_token) draft_tokens[k]);
            int n_feed = (int) feed_tokens.size();   // gamma+1
            std::vector<float> p_t((size_t) n_feed * Vd, 0.0f);
            bool feed_ok = true;
            size_t f_idx = 0;
            while (f_idx < (size_t) n_feed) {
                size_t chunk = std::min((size_t) n_batch, (size_t) n_feed - f_idx);
                llama_batch batch = llama_batch_init((int) chunk, 0, 1);
                for (size_t i = 0; i < chunk; i++) {
                    batch.token[i]    = feed_tokens[f_idx + i];
                    batch.pos[i]      = n_past + (llama_pos) i;
                    batch.n_seq_id[i] = 1;
                    batch.seq_id[i][0] = seq_id;
                    batch.logits[i]   = 1;
                }
                batch.n_tokens = (int) chunk;
                if (llama_decode(ctx, batch) != 0) {
                    LOG_ERR("dspark_cycle feed decode failed\n");
                    feed_ok = false;
                    llama_batch_free(batch);
                    break;
                }
                for (size_t i = 0; i < chunk; i++) {
                    const float * lg = llama_get_logits_ith(ctx, (int32_t) i);
                    if (!lg) { feed_ok = false; break; }
                    float * dst = p_t.data() + (size_t) (f_idx + i) * Vd;
                    #pragma omp parallel for
                    for (int v = 0; v < Vd; v++) dst[v] = lg[v];
                }
                n_past += (llama_pos) chunk;
                f_idx += chunk;
                llama_batch_free(batch);
            }
            if (!feed_ok) { write_header_err("dspark_cycle target feed failed"); continue; }
            // Update last_hidden to the LAST fed token's hidden state (the
            // last accepted draft's position, or anchor if all rejected) for
            // the NEXT cycle. We pin it to position n_feed-1 (the last draft)
            // BEFORE rollback; the sampler will fix the true committed anchor.
            if (ba.embedding) {
                const float * em_last = llama_get_embeddings_ith(ctx, (int32_t) (n_feed - 1));
                if (em_last) {
                    last_hidden.assign(em_last, em_last + n_embd);
                }
            }
            // Softmax each target distribution row in place.
            for (int k = 0; k < n_feed; k++) {
                (void) softmax_inplace(p_t.data() + (size_t) k * Vd, Vd);
            }

            // --- 3. C++ rejection sampler (Leviathan-2023). ---
            std::vector<int32_t> accepted_tokens;
            accepted_tokens.reserve(g + 1);
            int n_accepted = 0;
            int reject_position = -1;
            // Position k's verifier uses p_t row k and p_d row k.
            for (int k = 0; k < g; k++) {
                int x_k = draft_tokens[k];
                const float * q = drafter.p_d.data() + (size_t) k * Vd;
                const float * p = p_t.data() + (size_t) k * Vd;
                float q_x = q[x_k];
                float p_x = p[x_k];
                float ratio = (q_x <= 0.0f) ? 1.0f : std::min(1.0f, p_x / q_x);
                float r = rng_uniform(drafter);
                if (r < ratio) {
                    accepted_tokens.push_back((int32_t) x_k);
                    n_accepted += 1;
                    continue;
                }
                // Reject: resample x_k* from normalized(max(0, p - q)).
                float * resample_scratch = drafter.bias_k.data() + (size_t) k * Vd;  // reuse
                #pragma omp parallel for
                for (int v = 0; v < Vd; v++) {
                    float diff = p[v] - q[v];
                    resample_scratch[v] = diff > 0.0f ? diff : 0.0f;
                }
                int x_star = sample_from_dist(resample_scratch, Vd, drafter);
                accepted_tokens.push_back((int32_t) x_star);
                reject_position = k;
                break;
            }
            int bonus_token = -1;
            if (reject_position < 0) {
                // All accepted: sample bonus from p_t row gamma.
                const float * p_bonus = p_t.data() + (size_t) g * Vd;
                bonus_token = sample_from_dist(p_bonus, Vd, drafter);
                accepted_tokens.push_back((int32_t) bonus_token);
            }

            // --- 4. KV-cache rollback for the un-verified suffix. ---
            // We fed n_feed = gamma+1 tokens this cycle (anchor + all
            // drafts). The committed suffix is:
            //   all-accepted:  anchor + gamma drafts        = n_pre + gamma + 1
            //   rejected at k: anchor + (k accepted drafts) = n_pre + k + 1
            //                  (resampled token is sampled, NOT fed to KV,
            //                   so it stays OUT of committed; rejected
            //                   draft[k] MUST be rolled back)
            // The session length after the feed is n_pre + n_feed.
            size_t committed = (size_t) (n_past - n_feed);  // n_pre
            if (reject_position < 0) {
                committed += (size_t) (g + 1);          // anchor + all drafts (bonus not fed)
            } else {
                committed += (size_t) (reject_position + 1);  // anchor + reject_pos accepted drafts (rejected draft[k] rolled back)
            }
            if ((size_t) n_past > committed) {
                llama_memory_seq_rm(llama_get_memory(ctx), seq_id, (llama_pos) committed, -1);
                n_past = (llama_pos) committed;
            }
            // Update last_hidden to the new anchor (the last committed token).
            int new_anchor = accepted_tokens.back();
            // Cache the hidden state at the committed position for the next
            // cycle's drafter. The committed position in KV is committed-1
            // (0-indexed). Embeddings are stored at decode time; we fetch the
            // one at the new anchor's position.
            if (ba.embedding && committed > 0) {
                const float * em_new = llama_get_embeddings_ith(ctx, (int32_t) (committed - 1));
                if (em_new) last_hidden.assign(em_new, em_new + n_embd);
            }
            (void) new_anchor;

            // --- 5. Return. ---
            int n_out = (int) accepted_tokens.size();
            write_header_ok_kv({
                {"n_accepted", (int64_t) n_accepted},
                {"n_committed", (int64_t) n_out},
                {"reject_position", (int64_t) reject_position},
                {"bonus", (int64_t) bonus_token},
                {"n_anchor", (int64_t) new_anchor},
            });
            write_bytes(accepted_tokens.data(), (size_t) n_out * sizeof(int32_t));
            write_floats(confidences.data(), g);
            continue;
        } else if (op == "dspark_cycle_split") {
            // Story S31 AC4: split cycle — drafter runs in PYTHON (DFlash via
            // PyTorch), C++ does target feed + Leviathan rejection + KV
            // rollback. Takes draft_tokens + p_d (draft distributions) +
            // confidences as input. Returns the same fields as dspark_cycle.
            //
            // Request: {"op":"dspark_cycle_split",
            //          "anchor_token":int,
            //          "draft_tokens":[int; gamma],
            //          "confidences":[float; gamma],
            //          "p_d":"<gamma*V float32 binary>"}
            //         (p_d is sent as raw bytes following the JSON header,
            //          same framing as `feed`'s logits / `drafter_load`'s .bin).
            if (!req.contains("anchor_token") || !req["anchor_token"].is_number_integer()) {
                write_header_err("dspark_cycle_split requires anchor_token:int");
                continue;
            }
            int anchor_tok = req["anchor_token"].get<int>();
            if (anchor_tok < 0 || anchor_tok >= (int) V) {
                write_header_err("anchor_token out of range [0, V)");
                continue;
            }
            if (!req.contains("draft_tokens") || !req["draft_tokens"].is_array()) {
                write_header_err("dspark_cycle_split requires draft_tokens:int[gamma]");
                continue;
            }
            auto j_drafts = req["draft_tokens"];
            int g = (int) j_drafts.size();
            if (g <= 0 || g > 64) {
                write_header_err("draft_tokens length out of range (1..64)");
                continue;
            }
            // Read p_d (gamma*V float32 = 4*gamma*V bytes) from the binary tail.
            // The JSON header MUST carry n_pd_bytes so we know how much to read.
            int64_t n_pd_bytes = 0;
            if (req.contains("n_pd_bytes") && req["n_pd_bytes"].is_number_integer()) {
                n_pd_bytes = req["n_pd_bytes"].get<int64_t>();
            }
            const size_t expect_pd = (size_t) g * V * sizeof(float);
            if ((size_t) n_pd_bytes != expect_pd) {
                write_header_err("p_d byte length mismatch: got " + std::to_string(n_pd_bytes)
                                 + ", expected " + std::to_string(expect_pd));
                continue;
            }
            // --- 0. Read the p_d binary payload (gamma*V float32 row-major). ---
            std::vector<float> p_d((size_t) g * V, 0.0f);
            if (!read_bytes_into(p_d.data(), expect_pd)) {
                write_header_err("dspark_cycle_split truncated p_d payload");
                continue;
            }
            // Copy draft_tokens into a C++ vector.
            std::vector<int32_t> draft_tokens(g);
            for (int k = 0; k < g; k++) {
                if (!j_drafts[k].is_number_integer()) {
                    write_header_err("draft_tokens[" + std::to_string(k) + "] not int");
                    continue;
                }
                int t = j_drafts[k].get<int>();
                if (t < 0 || t >= (int) V) {
                    write_header_err("draft_tokens[" + std::to_string(k) + "] out of range");
                    continue;
                }
                draft_tokens[k] = (int32_t) t;
            }
            // confidences (gamma float32) — optional in the split op.
            std::vector<float> confidences(g, 0.0f);
            if (req.contains("confidences") && req["confidences"].is_array()) {
                auto j_conf = req["confidences"];
                if ((int) j_conf.size() == g) {
                    for (int k = 0; k < g; k++) confidences[k] = (float) j_conf[k].get<double>();
                }
            }
            // Optional seed override (else session RNG continues).
            if (req.contains("seed") && req["seed"].is_number_integer()) {
                drafter.rng_state = (uint64_t) req["seed"].get<int64_t>() | 0x9E3779B97F4A7C15ULL;
            }
            const int Vd = (int) V;
            const int n_feed = g + 1;  // anchor + gamma drafts

            // --- 1. C++ target feed of [anchor, *draft_tokens]. ---
            std::vector<llama_token> feed_tokens;
            feed_tokens.push_back((llama_token) anchor_tok);
            for (int k = 0; k < g; k++) feed_tokens.push_back((llama_token) draft_tokens[k]);
            std::vector<float> p_t((size_t) n_feed * Vd, 0.0f);
            bool feed_ok = true;
            size_t f_idx = 0;
            while (f_idx < (size_t) n_feed) {
                size_t chunk = std::min((size_t) n_batch, (size_t) n_feed - f_idx);
                llama_batch batch = llama_batch_init((int) chunk, 0, 1);
                for (size_t i = 0; i < chunk; i++) {
                    batch.token[i]    = feed_tokens[f_idx + i];
                    batch.pos[i]      = n_past + (llama_pos) i;
                    batch.n_seq_id[i] = 1;
                    batch.seq_id[i][0] = seq_id;
                    batch.logits[i]   = 1;
                }
                batch.n_tokens = (int) chunk;
                if (llama_decode(ctx, batch) != 0) {
                    LOG_ERR("dspark_cycle_split feed decode failed\n");
                    feed_ok = false;
                    llama_batch_free(batch);
                    break;
                }
                for (size_t i = 0; i < chunk; i++) {
                    const float * lg = llama_get_logits_ith(ctx, (int32_t) i);
                    if (!lg) { feed_ok = false; break; }
                    float * dst = p_t.data() + (size_t) (f_idx + i) * Vd;
                    #pragma omp parallel for
                    for (int v = 0; v < Vd; v++) dst[v] = lg[v];
                }
                n_past += (llama_pos) chunk;
                f_idx += chunk;
                llama_batch_free(batch);
            }
            if (!feed_ok) { write_header_err("dspark_cycle_split target feed failed"); continue; }
            // Pin last_hidden to the last fed token's position (BEFORE rollback).
            if (ba.embedding) {
                const float * em_last = llama_get_embeddings_ith(ctx, (int32_t) (n_feed - 1));
                if (em_last) last_hidden.assign(em_last, em_last + n_embd);
            }
            // Softmax each target distribution row in place.
            for (int k = 0; k < n_feed; k++) {
                (void) softmax_inplace(p_t.data() + (size_t) k * Vd, Vd);
            }

            // --- 2. C++ rejection sampler (Leviathan-2023). ---
            // Uses the EXTERNAL p_d (Python drafter) and our just-computed p_t.
            std::vector<int32_t> accepted_tokens;
            accepted_tokens.reserve(g + 1);
            int n_accepted = 0;
            int reject_position = -1;
            for (int k = 0; k < g; k++) {
                int x_k = draft_tokens[k];
                const float * q = p_d.data() + (size_t) k * Vd;       // drafter dist (Python)
                const float * p = p_t.data() + (size_t) k * Vd;       // target dist (C++)
                float q_x = q[x_k];
                float p_x = p[x_k];
                float ratio = (q_x <= 0.0f) ? 1.0f : std::min(1.0f, p_x / q_x);
                float r = rng_uniform(drafter);
                if (r < ratio) {
                    accepted_tokens.push_back((int32_t) x_k);
                    n_accepted += 1;
                    continue;
                }
                // Reject: resample x_k* from normalized(max(0, p - q)).
                std::vector<float> resample_scratch((size_t) Vd, 0.0f);
                #pragma omp parallel for
                for (int v = 0; v < Vd; v++) {
                    float diff = p[v] - q[v];
                    resample_scratch[v] = diff > 0.0f ? diff : 0.0f;
                }
                int x_star = sample_from_dist(resample_scratch.data(), Vd, drafter);
                accepted_tokens.push_back((int32_t) x_star);
                reject_position = k;
                break;
            }
            int bonus_token = -1;
            if (reject_position < 0) {
                // All accepted: sample bonus from p_t row gamma (target dist at the bonus position).
                const float * p_bonus = p_t.data() + (size_t) g * Vd;
                bonus_token = sample_from_dist(p_bonus, Vd, drafter);
                accepted_tokens.push_back((int32_t) bonus_token);
            }

            // --- 3. KV-cache rollback for the un-verified suffix. ---
            // Same arithmetic as dspark_cycle (see comment there for proof).
            size_t committed = (size_t) (n_past - n_feed);  // n_pre
            if (reject_position < 0) {
                committed += (size_t) (g + 1);          // anchor + all drafts (bonus not fed)
            } else {
                committed += (size_t) (reject_position + 1);  // anchor + reject_pos accepted drafts
            }
            if ((size_t) n_past > committed) {
                llama_memory_seq_rm(llama_get_memory(ctx), seq_id, (llama_pos) committed, -1);
                n_past = (llama_pos) committed;
            }
            // Update last_hidden to the new anchor's position.
            int new_anchor = accepted_tokens.back();
            if (ba.embedding && committed > 0) {
                const float * em_new = llama_get_embeddings_ith(ctx, (int32_t) (committed - 1));
                if (em_new) last_hidden.assign(em_new, em_new + n_embd);
            }

            // --- 4. Return. ---
            int n_out = (int) accepted_tokens.size();
            write_header_ok_kv({
                {"n_accepted", (int64_t) n_accepted},
                {"n_committed", (int64_t) n_out},
                {"reject_position", (int64_t) reject_position},
                {"bonus", (int64_t) bonus_token},
                {"n_anchor", (int64_t) new_anchor},
            });
            write_bytes(accepted_tokens.data(), (size_t) n_out * sizeof(int32_t));
            write_floats(confidences.data(), (size_t) g);
            continue;
        } else {
            write_header_err("unknown op: " + op);
            continue;
        }
    }

    LOG_INF("bridge exit\n");
    return 0;
}
