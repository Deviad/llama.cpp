#include "arg.h"
#include "common.h"
#include "download.h"
#include "speculative.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sstream>
#include <unordered_set>

#undef NDEBUG
#include <cassert>

int main(void) {
    common_params params;

    printf("test-arg-parser: make sure there is no duplicated arguments in any examples\n\n");
    for (int ex = 0; ex < LLAMA_EXAMPLE_COUNT; ex++) {
        try {
            auto ctx_arg = common_params_parser_init(params, (enum llama_example)ex);
            common_params_add_preset_options(ctx_arg.options);
            std::unordered_set<std::string> seen_args;
            std::unordered_set<std::string> seen_env_vars;
            for (const auto & opt : ctx_arg.options) {
                // check for args duplications
                for (const auto & arg : opt.get_args()) {
                    if (seen_args.find(arg) == seen_args.end()) {
                        seen_args.insert(arg);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same argument: %s", arg.c_str());
                        exit(1);
                    }
                }
                // check for env var duplications
                for (const auto & env : opt.get_env()) {
                    if (seen_env_vars.find(env) == seen_env_vars.end()) {
                        seen_env_vars.insert(env);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same env var: %s", env.c_str());
                        exit(1);
                    }
                }

                // exclude spec args from this check
                // ref: https://github.com/ggml-org/llama.cpp/pull/22397
                const bool skip = opt.is_spec;

                // ensure shorter argument precedes longer argument
                if (!skip && opt.args.size() > 1) {
                    const std::string first(opt.args.front());
                    const std::string last(opt.args.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }

                // same check for negated arguments
                if (opt.args_neg.size() > 1) {
                    const std::string first(opt.args_neg.front());
                    const std::string last(opt.args_neg.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter negated argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }
            }
        } catch (std::exception & e) {
            printf("%s\n", e.what());
            assert(false);
        }
    }

    auto list_str_to_char = [](std::vector<std::string> & argv) -> std::vector<char *> {
        std::vector<char *> res;
        for (auto & arg : argv) {
            res.push_back(const_cast<char *>(arg.data()));
        }
        return res;
    };

    std::vector<std::string> argv;

    printf("test-arg-parser: test invalid usage\n\n");

    // missing value
    argv = {"binary_name", "-m"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (int)
    argv = {"binary_name", "-ngl", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (enum)
    argv = {"binary_name", "-sm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // non-existence arg in specific example (--draft cannot be used outside llama-speculative)
    argv = {"binary_name", "--draft", "123"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_EMBEDDING));

    // negated arg
    argv = {"binary_name", "--no-mmap"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));


    printf("test-arg-parser: test valid usage\n\n");

    argv = {"binary_name", "-m", "model_file.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "model_file.gguf");

    argv = {"binary_name", "-t", "1234"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.cpuparams.n_threads == 1234);

    argv = {"binary_name", "--verbose"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.verbosity > 1);

    argv = {"binary_name", "-m", "abc.gguf", "--predict", "6789", "--batch-size", "9090"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "abc.gguf");
    assert(params.n_predict == 6789);
    assert(params.n_batch == 9090);

    // --draft cannot be used outside llama-speculative
    argv = {"binary_name", "--spec-draft-n-max", "123"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(params.speculative.draft.n_max == 123);

    common_params external_params;
    argv = {"binary_name", "--spec-type", "draft-external", "--spec-external-trace", "trace.json", "--spec-draft-n-max", "7"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), external_params, LLAMA_EXAMPLE_SERVER));
    assert(common_speculative_type_from_name("draft-external") == COMMON_SPECULATIVE_TYPE_DRAFT_EXTERNAL);
    assert(common_speculative_type_to_str(COMMON_SPECULATIVE_TYPE_DRAFT_EXTERNAL) == "draft-external");
    assert(external_params.speculative.external.trace_path == "trace.json");
    assert(common_speculative_n_max(&external_params.speculative) == 7);
    assert(external_params.speculative.need_n_rs_seq() == 7);

    {
        common_params_speculative replay_params;
        replay_params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_EXTERNAL };
        replay_params.draft.n_max = 2;

        replay_params.external.trace_path = "test-arg-parser-missing-external-trace.json";
        std::remove(replay_params.external.trace_path.c_str());
        try {
            common_speculative_free(common_speculative_init(replay_params, 1));
            assert(false && "missing external trace must fail");
        } catch (const std::runtime_error &) {
        }

        const char * trace_path = "test-arg-parser-external-trace.json";
        {
            std::ofstream output(trace_path);
            output << "{}";
        }
        replay_params.external.trace_path = trace_path;
        try {
            common_speculative_free(common_speculative_init(replay_params, 1));
            assert(false && "malformed external trace must fail");
        } catch (const std::runtime_error &) {
        }

        const std::vector<std::string> invalid_traces = {
            R"({"schema":"glm52_external_draft_replay.v1","provenance":{},"max_draft_tokens":1,"cases":[{"id":"a","prefix_tokens":[1],"anchor_token":2,"draft_tokens":[3]}]})",
            R"({"schema":"glm52_external_draft_replay.v1","provenance":{},"case_count":1,"max_draft_tokens":1,"unknown":0,"cases":[{"id":"a","prefix_tokens":[1],"anchor_token":2,"draft_tokens":[3]}]})",
            R"({"schema":"glm52_external_draft_replay.v1","provenance":{},"case_count":2,"max_draft_tokens":1,"cases":[{"id":"a","prefix_tokens":[1],"anchor_token":2,"draft_tokens":[3]},{"id":"a","prefix_tokens":[2],"anchor_token":3,"draft_tokens":[4]}]})",
            R"({"schema":"glm52_external_draft_replay.v1","provenance":{},"case_count":2,"max_draft_tokens":1,"cases":[{"id":"a","prefix_tokens":[1],"anchor_token":2,"draft_tokens":[3]},{"id":"b","prefix_tokens":[1],"anchor_token":4,"draft_tokens":[5]}]})",
            R"({"schema":"glm52_external_draft_replay.v1","provenance":{},"case_count":2,"max_draft_tokens":1,"cases":[{"id":"a","prefix_tokens":[1],"anchor_token":2,"draft_tokens":[3]}]})",
            R"({"schema":"glm52_external_draft_replay.v1","provenance":{},"case_count":1,"max_draft_tokens":2,"cases":[{"id":"a","prefix_tokens":[1],"anchor_token":2,"draft_tokens":[3]}]})",
            R"({"schema":"glm52_external_draft_replay.v1","provenance":{},"case_count":1,"max_draft_tokens":1,"cases":[{"id":"a","prefix_tokens":[2147483648],"anchor_token":2,"draft_tokens":[3]}]})",
            R"({"schema":"glm52_external_draft_replay.v1","provenance":{},"case_count":1,"max_draft_tokens":1,"cases":[{"id":"a","prefix_tokens":[1],"anchor_token":2,"draft_tokens":[3],"unknown":0}]})",
        };
        for (const auto & contents : invalid_traces) {
            {
                std::ofstream output(trace_path);
                output << contents;
            }
            try {
                common_speculative_free(common_speculative_init(replay_params, 1));
                assert(false && "invalid external trace must fail");
            } catch (const std::runtime_error &) {
            }
        }

        {
            std::ofstream output(trace_path);
            output << R"({"schema":"glm52_external_draft_replay.v1","provenance":{},"case_count":1,"max_draft_tokens":3,"cases":[{"id":"case-a","prefix_tokens":[1,2],"anchor_token":3,"draft_tokens":[4,5,6]}]})";
        }
        common_speculative * replay = common_speculative_init(replay_params, 1);
        llama_tokens prompt = { 1, 2 };
        llama_token forced_anchor = LLAMA_TOKEN_NULL;
        assert(common_speculative_get_external_anchor(replay, prompt, forced_anchor));
        assert(forced_anchor == 3);
        assert(!common_speculative_get_external_anchor(replay, llama_tokens({ 1 }), forced_anchor));
        llama_tokens result;
        common_speculative_get_draft_params(replay, 0) = {
            /* .drafting = */ true,
            /* .n_max    = */ 1,
            /* .n_past   = */ 2,
            /* .id_last  = */ 3,
            /* .prompt   = */ &prompt,
            /* .result   = */ &result,
        };
        common_speculative_draft(replay);
        assert(result == llama_tokens({ 4 }));

        result.clear();
        auto & draft_params = common_speculative_get_draft_params(replay, 0);
        draft_params.drafting = true;
        draft_params.n_max = 3;
        common_speculative_draft(replay);
        assert(result == llama_tokens({ 4, 5 }));

        result.clear();
        draft_params.drafting = true;
        draft_params.id_last = 99;
        common_speculative_draft(replay);
        assert(result.empty());

        auto incompatible = replay_params;
        incompatible.types = {
            COMMON_SPECULATIVE_TYPE_DRAFT_EXTERNAL,
            COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,
        };
        try {
            common_speculative_free(common_speculative_init(incompatible, 1));
            assert(false && "combined external speculative types must fail");
        } catch (const std::invalid_argument &) {
        }
        incompatible = replay_params;
        incompatible.draft.mparams.path = "draft.gguf";
        try {
            common_speculative_free(common_speculative_init(incompatible, 1));
            assert(false && "external replay with draft model must fail");
        } catch (const std::invalid_argument &) {
        }

        common_speculative_free(replay);
        std::remove(trace_path);
    }

    // multi-value args (CSV)
    argv = {"binary_name", "--lora", "file1.gguf,\"file2,2.gguf\",\"file3\"\"3\"\".gguf\",file4\".gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.lora_adapters.size() == 4);
    assert(params.lora_adapters[0].path == "file1.gguf");
    assert(params.lora_adapters[1].path == "file2,2.gguf");
    assert(params.lora_adapters[2].path == "file3\"3\".gguf");
    assert(params.lora_adapters[3].path == "file4\".gguf");

// skip this part on windows, because setenv is not supported
#ifdef _WIN32
    printf("test-arg-parser: skip on windows build\n");
#else
    printf("test-arg-parser: test environment variables (valid + invalid usages)\n\n");

    setenv("LLAMA_ARG_THREADS", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "blah.gguf");
    assert(params.cpuparams.n_threads == 1010);

    printf("test-arg-parser: test negated environment variables\n\n");

    setenv("LLAMA_ARG_MMAP", "0", true);
    setenv("LLAMA_ARG_NO_PERF", "1", true); // legacy format
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.use_mmap == false);
    assert(params.no_perf == true);

    printf("test-arg-parser: test environment variables being overwritten\n\n");

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name", "-m", "overwritten.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "overwritten.gguf");
    assert(params.cpuparams.n_threads == 1010);
#endif // _WIN32

    printf("test-arg-parser: test download functions\n\n");
    const char * GOOD_URL = "http://ggml.ai/";
    const char * BAD_URL  = "http://ggml.ai/404";

    {
        printf("test-arg-parser: test good URL\n\n");
        auto res = common_remote_get_content(GOOD_URL, {});
        assert(res.first == 200);
        assert(res.second.size() > 0);
        std::string str(res.second.data(), res.second.size());
        assert(str.find("llama.cpp") != std::string::npos);
    }

    {
        printf("test-arg-parser: test bad URL\n\n");
        auto res = common_remote_get_content(BAD_URL, {});
        assert(res.first == 404);
    }

    {
        printf("test-arg-parser: test max size error\n");
        common_remote_params params;
        params.max_size = 1;
        try {
            common_remote_get_content(GOOD_URL, params);
            assert(false && "it should throw an error");
        } catch (std::exception & e) {
            printf("  expected error: %s\n\n", e.what());
        }
    }

    printf("test-arg-parser: all tests OK\n\n");
}
