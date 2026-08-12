#include "common.h"
#include "vk_device.h"
#include "vk_buffer.h"
#include "vk_model.h"
#include "vk_session.h"
#include "vk_timeline.h"
#include "gguf_parser.h"
#include "sample.h"
#include "vk_validate.h"

// TEMP DEBUG helper: fp16->fp32 + NaN scan of the lm_head_input snapshot.
static void debug_check_hidden(vk_session_t* session, const char* label) {
    uint32_t d = session->model->config.d;
    const uint16_t* src = (const uint16_t*)session->decode_state.hidden_debug.mapped_ptr;
    bool any_nan = false;
    float first_vals[4] = {0,0,0,0};
    for (uint32_t i = 0; i < d; i++) {
        uint16_t h = src[i];
        uint32_t sign = (uint32_t)(h >> 15) & 1u;
        uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
        uint32_t man  = (uint32_t)h & 0x3FFu;
        float v;
        if (exp == 0x1Fu) {
            v = (man != 0) ? NAN : (sign ? -INFINITY : INFINITY);
        } else {
            uint32_t f_exp = (exp == 0) ? 0 : (exp - 15 + 127);
            uint32_t bits = (sign << 31) | (f_exp << 23) | (man << 13);
            memcpy(&v, &bits, 4);
        }
        if (i < 4) first_vals[i] = v;
        if (std::isnan(v)) any_nan = true;
    }
    fprintf(stderr, "[NAN-BISECT] %s: %s  first4=[%.4f %.4f %.4f %.4f]\n",
            label, any_nan ? "HAS NaN" : "clean", first_vals[0], first_vals[1], first_vals[2], first_vals[3]);
}

#include <direct.h>
#include <io.h>
#include <thread>

/* ============================================================================
 *  Anonymous namespace — all internal helpers
 * ============================================================================ */
namespace {

/* --------------------------------------------------------------------------
 *  Simple tokenizer: reads vocabulary from raw GGUF metadata
 * -------------------------------------------------------------------------- */

struct tokenizer_t {
    std::vector<std::string> vocab;
    std::vector<float> scores;
    std::unordered_map<std::string, int32_t> tok2id;
    bool byte_fallback;
    int32_t bos_id, eos_id, pad_id;
};

static uint32_t rd_u32(const uint8_t* b) {
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}
static uint64_t rd_u64(const uint8_t* b) {
    return (uint64_t)rd_u32(b) | ((uint64_t)rd_u32(b + 4) << 32);
}
static float rd_f32(const uint8_t* b) {
    uint32_t u = rd_u32(b);
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static bool tok_read_gguf(const char* gguf_path, tokenizer_t* tok) {
    tok->vocab.clear();
    tok->scores.clear();
    tok->tok2id.clear();
    tok->bos_id = 1;
    tok->eos_id = 2;
    tok->pad_id = -1;
    tok->byte_fallback = true;

    FILE* f = nullptr;
    fopen_s(&f, gguf_path, "rb");
    if (!f) return false;
    _fseeki64(f, 0, SEEK_END);
    int64_t file_size = _ftelli64(f);
    if (file_size < 32) { fclose(f); return false; }
    int64_t metadata_limit = file_size < 64ll * 1024 * 1024 ? file_size : 64ll * 1024 * 1024;
    uint8_t* buf = (uint8_t*)malloc((size_t)metadata_limit);
    if (!buf) { fclose(f); return false; }
    fseek(f, 0, SEEK_SET);
    fread(buf, 1, (size_t)metadata_limit, f);
    fclose(f);

    uint32_t magic = rd_u32(buf);
    uint32_t ver = rd_u32(buf + 4);
    if (magic != 0x46554747 || ver < 2 || ver > 3) { free(buf); return false; }
    uint64_t kv_count = rd_u64(buf + 16);
    uint64_t pos = 24ull;

    for (uint64_t i = 0; i < kv_count; i++) {
        if (pos + 8 > (uint64_t)metadata_limit) break;
        uint64_t klen = rd_u64(buf + pos); pos += 8;
        if (klen > 1024 || pos + klen > (uint64_t)metadata_limit) break;
        std::string key((const char*)(buf + pos), (size_t)klen); pos += klen;
        if (pos + 4 > (uint64_t)metadata_limit) break;
        uint32_t vt = rd_u32(buf + pos); pos += 4;

        if (key == "tokenizer.ggml.tokens" && vt == 9) {
            if (pos + 12 > (uint64_t)metadata_limit) break;
            uint32_t array_type = rd_u32(buf + pos); pos += 4;
            uint64_t n = rd_u64(buf + pos); pos += 8;
            if (n > 256 * 1024) break;
            tok->vocab.resize((size_t)n);
            for (uint64_t j = 0; j < n && pos < (uint64_t)metadata_limit; j++) {
                if (pos + 8 > (uint64_t)metadata_limit) break;
                uint64_t slen = rd_u64(buf + pos); pos += 8;
                if (slen > 1024 || pos + slen > (uint64_t)metadata_limit) break;
                tok->vocab[(size_t)j] = std::string((const char*)(buf + pos), (size_t)slen);
                pos += slen;
            }
        } else if (key == "tokenizer.ggml.scores" && vt == 9) {
            if (pos + 12 > (uint64_t)metadata_limit) break;
            uint32_t array_type = rd_u32(buf + pos); pos += 4;
            uint64_t n = rd_u64(buf + pos); pos += 8;
            if (n > 256 * 1024) break;
            tok->scores.resize((size_t)n);
            for (uint64_t j = 0; j < n && pos + 4 <= (uint64_t)metadata_limit; j++) {
                tok->scores[(size_t)j] = rd_f32(buf + pos); pos += 4;
            }
        } else if (key == "tokenizer.ggml.bos_token_id" && vt == 4) {
            tok->bos_id = (int32_t)rd_u32(buf + pos); pos += 4;
        } else if (key == "tokenizer.ggml.eos_token_id" && vt == 4) {
            tok->eos_id = (int32_t)rd_u32(buf + pos); pos += 4;
        } else if (key == "tokenizer.ggml.padding_token_id" && vt == 4) {
            tok->pad_id = (int32_t)rd_u32(buf + pos); pos += 4;
        } else {
            if (pos + 4 > (uint64_t)metadata_limit) break;
            if (vt == 8) {
                if (pos + 8 > (uint64_t)metadata_limit) break;
                uint64_t slen = rd_u64(buf + pos); pos += 8 + slen;
            } else if (vt == 9) {
                if (pos + 12 > (uint64_t)metadata_limit) break;
                uint32_t elem_type = rd_u32(buf + pos); pos += 4;
                uint64_t n = rd_u64(buf + pos); pos += 8;
                for (uint64_t j = 0; j < n && pos < (uint64_t)metadata_limit; j++) {
                    if (elem_type == 8) {
                        if (pos + 8 > (uint64_t)metadata_limit) break;
                        uint64_t sl = rd_u64(buf + pos); pos += 8 + sl;
                    } else if (elem_type == 4 || elem_type == 5) {
                        pos += 4;
                    } else if (elem_type == 10 || elem_type == 11) {
                        pos += 8;
                    } else if (elem_type == 6) {
                        pos += 4;
                    } else if (elem_type == 12) {
                        pos += 8;
                    } else if (elem_type == 7) {
                        pos += 1;
                    } else {
                        pos += 4;
                    }
                }
            } else if (vt == 0 || vt == 1 || vt == 7) {
                pos += 1;
            } else if (vt == 2 || vt == 3) {
                pos += 2;
            } else if (vt == 4 || vt == 5 || vt == 6) {
                pos += 4;
            } else if (vt == 10 || vt == 11 || vt == 12) {
                pos += 8;
            } else {
                pos += 4;
            }
        }
    }
    free(buf);

    if (tok->vocab.empty()) {
        RDNA4_ERROR("No tokens found in GGUF metadata");
        return false;
    }
    tok->scores.resize(tok->vocab.size(), 0.0f);
    for (size_t i = 0; i < tok->vocab.size(); i++) tok->tok2id[tok->vocab[i]] = (int32_t)i;

    // Qwen-family GGUF quirk: tokenizer.ggml.bos_token_id is often stored as 1
    // (the plain '"' token), but the model's real BOS is <|endoftext|> (151643).
    // If the vocab contains the Qwen specials and BOS looks like the bare ASCII
    // token, promote BOS to <|endoftext|>.
    {
        auto it_eot = tok->tok2id.find("<|endoftext|>");
        auto it_im  = tok->tok2id.find("<|im_start|>");
        if (it_eot != tok->tok2id.end() && it_im != tok->tok2id.end() &&
            tok->bos_id >= 0 && tok->bos_id < 256) {
            tok->bos_id = it_eot->second;
        }
    }
    RDNA4_LOG("Tokenizer: %zu tokens, BOS=%d, EOS=%d", tok->vocab.size(), tok->bos_id, tok->eos_id);
    return true;
}

// GPT-2/Qwen byte-level BPE: raw bytes of the text are mapped to printable
// unicode codepoints (bytes_to_unicode) BEFORE the BPE merges / vocab lookup.
// The GGUF vocab strings carry those mapped codepoints, so tokenization first
// applies the forward map (byte -> codepoint) to the prompt, then does a greedy
// longest-match against the vocab. This mirrors HF GPT2Tokenizer._tokenize.
static uint16_t byte_encoder[256];
static uint8_t byte_decoder[65536];
static bool byte_maps_built = false;

static void build_byte_maps() {
    if (byte_maps_built) return;
    // Kept bytes map to themselves: 33..126, 161..172, 174..255.
    uint8_t bs[256];
    uint32_t cs[256];
    uint32_t nb = 0;
    for (int b = 33; b <= 126; b++)  { bs[nb] = (uint8_t)b; cs[nb] = (uint32_t)b; nb++; }
    for (int b = 161; b <= 172; b++) { bs[nb] = (uint8_t)b; cs[nb] = (uint32_t)b; nb++; }
    for (int b = 174; b <= 255; b++) { bs[nb] = (uint8_t)b; cs[nb] = (uint32_t)b; nb++; }
    uint32_t n = 0;
    for (int b = 0; b < 256; b++) {
        bool kept = false;
        for (uint32_t k = 0; k < nb; k++) if (bs[k] == b) { kept = true; break; }
        if (!kept) {
            bs[nb] = (uint8_t)b;
            cs[nb] = 256u + n;
            nb++;
            n++;
        }
    }
    for (uint32_t k = 0; k < nb; k++) {
        byte_encoder[bs[k]] = (uint16_t)cs[k];
        if (cs[k] < 65536) byte_decoder[cs[k]] = bs[k];
    }
    byte_maps_built = true;
}

// Encode a UTF-8 prompt into the byte-mapped token string (each byte -> codepoint).
static std::string bytes_to_unicode_string(const std::string& text) {
    build_byte_maps();
    std::string out;
    for (unsigned char b : text) {
        uint16_t cp = byte_encoder[b];
        // encode cp as UTF-8
        if (cp < 0x80) {
            out += (char)cp;
        } else if (cp < 0x800) {
            out += (char)(0xC0 | (cp >> 6));
            out += (char)(0x80 | (cp & 0x3F));
        } else {
            out += (char)(0xE0 | (cp >> 12));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        }
    }
    return out;
}

static std::vector<int32_t> tok_encode(tokenizer_t* tok, const std::string& text) {
    std::vector<int32_t> out;
    if (tok->bos_id >= 0) out.push_back(tok->bos_id);
    std::string rem = bytes_to_unicode_string(text);
    while (!rem.empty()) {
        bool found = false;
        for (size_t cl = rem.length(); cl > 0; cl--) {
            std::string cand = rem.substr(0, cl);
            auto it = tok->tok2id.find(cand);
            if (it != tok->tok2id.end()) {
                out.push_back(it->second);
                rem = rem.substr(cl);
                found = true;
                break;
            }
        }
        if (!found) {
            // Byte-fallback: map this single mapped-codepoint back to its byte,
            // then look up the single-byte token. Qwen stores byte tokens at
            // vocab[256 + mapped_char] for non-ASCII bytes; ASCII bytes map to
            // their own token ids 0..127.
            uint16_t cp = 0;
            unsigned char c = (unsigned char)rem[0];
            if (c < 0x80) cp = c;
            else if ((c & 0xE0) == 0xC0) cp = ((uint32_t)(c & 0x1F) << 6) | ((uint32_t)rem[1] & 0x3F);
            else if ((c & 0xF0) == 0xE0) cp = ((uint32_t)(c & 0x0F) << 12) | (((uint32_t)rem[1] & 0x3F) << 6) | ((uint32_t)rem[2] & 0x3F);
            uint8_t byte = byte_decoder[cp & 0xFFFF];
            char b[2] = { (char)byte, 0 };
            auto bi = tok->tok2id.find(std::string(b, 1));
            out.push_back(bi != tok->tok2id.end() ? bi->second : 0);
            rem = rem.substr(1);
        }
    }
    return out;
}

// GPT-2/Qwen byte-level BPE: each raw byte of the text is mapped to a printable
// unicode codepoint (bytes_to_unicode) before BPE merges. The vocab strings in the
// GGUF already carry those mapped codepoints, so decoding reverses the mapping:
// every UTF-8 codepoint in the token string is turned back into one byte and the
// resulting byte stream is the actual UTF-8 text. This mirrors HF GPT2Tokenizer
// convert_tokens_to_string + byte_decoder.
// Decode a single vocab token string into its raw UTF-8 text bytes.
static std::string tok_decode_token(const std::string& piece) {
    build_byte_maps();
    std::vector<uint8_t> bytes;
    // Decode the piece as UTF-8, one codepoint at a time.
    size_t i = 0;
    while (i < piece.size()) {
        uint32_t cp = 0;
        unsigned char c = (unsigned char)piece[i];
        if (c < 0x80) { cp = c; i += 1; }
        else if ((c & 0xE0) == 0xC0) {
            cp = ((uint32_t)(c & 0x1F) << 6) | ((uint32_t)piece[i+1] & 0x3F);
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = ((uint32_t)(c & 0x0F) << 12) | (((uint32_t)piece[i+1] & 0x3F) << 6) | ((uint32_t)piece[i+2] & 0x3F);
            i += 3;
        } else if ((c & 0xF8) == 0xF0) {
            cp = ((uint32_t)(c & 0x07) << 18) | (((uint32_t)piece[i+1] & 0x3F) << 12) | (((uint32_t)piece[i+2] & 0x3F) << 6) | ((uint32_t)piece[i+3] & 0x3F);
            i += 4;
        } else { i += 1; continue; }
        if (cp < 65536) {
            bytes.push_back(byte_decoder[cp]);
        }
    }
    return std::string((const char*)bytes.data(), bytes.size());
}

// Decode a sequence of generated token ids into displayable text.
static std::string tok_decode(const std::vector<int32_t>& ids, tokenizer_t* tok) {
    std::string out;
    for (int32_t id : ids) {
        if (id >= 0 && (size_t)id < tok->vocab.size()) {
            out += tok_decode_token(tok->vocab[(size_t)id]);
        }
    }
    return out;
}

/* --------------------------------------------------------------------------
 *  CLI argument parsing
 * -------------------------------------------------------------------------- */

struct cli_args_t {
    std::string model, prompt;
    uint32_t max_tokens, top_k, seed;
    float temp, top_p;
    bool verbose, validate, validate_gpu;
};

static void print_usage() {
    printf("RDNA4-LLM — Pure Vulkan LLM Inference Engine\n\n");
    printf("  --model <path>    GGUF model file (required)\n");
    printf("  --prompt <text>   Input prompt [Hello]\n");
    printf("  --max-tokens N    Max tokens to generate [256]\n");
    printf("  --temp F          Temperature [0.8]\n");
    printf("  --top-k N         Top-K sampling [40]\n");
    printf("  --top-p F         Top-P sampling [0.95]\n");
    printf("  --seed N          Random seed [42]\n");
    printf("  --verbose         Verbose token output\n");
    printf("  --validate        Enable Vulkan validation layers\n");
    printf("  --validate-gpu    Run CPU-vs-GPU validation on layer 0\n");
    printf("  --help            Show this help\n");
}

static bool parse_args(int argc, char* argv[], cli_args_t* a) {
    *a = { "", "Hello", 256, 40, 42, 0.8f, 0.95f, false, false, false };
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        if (s == "--help" || s == "-h") {
            print_usage();
            exit(0);
        } else if (s == "--model" && i + 1 < argc) a->model = argv[++i];
        else if (s == "--prompt" && i + 1 < argc) a->prompt = argv[++i];
        else if (s == "--max-tokens" && i + 1 < argc) a->max_tokens = (uint32_t)atoi(argv[++i]);
        else if (s == "--temp" && i + 1 < argc) a->temp = (float)atof(argv[++i]);
        else if (s == "--top-k" && i + 1 < argc) a->top_k = (uint32_t)atoi(argv[++i]);
        else if (s == "--top-p" && i + 1 < argc) a->top_p = (float)atof(argv[++i]);
        else if (s == "--seed" && i + 1 < argc) a->seed = (uint32_t)atoi(argv[++i]);
        else if (s == "--verbose") a->verbose = true;
        else if (s == "--validate") a->validate = true;
        else if (s == "--validate-gpu") a->validate_gpu = true;
    }
    if (a->model.empty()) {
        RDNA4_ERROR("--model is required");
        print_usage();
        return false;
    }
    return true;
}

} // anonymous namespace

/* ============================================================================
 *  main()
 * ============================================================================ */

int main(int argc, char* argv[]) {
    cli_args_t a;
    if (!parse_args(argc, argv, &a)) return 1;

    RDNA4_LOG("RDNA4-LLM — Pure Vulkan LLM Inference Engine");

    tokenizer_t tok = {};
    if (!tok_read_gguf(a.model.c_str(), &tok)) {
        tok.bos_id = 1;
        tok.eos_id = 2;
        tok.pad_id = -1;
        tok.byte_fallback = true;
        tok.vocab.push_back("");
        tok.vocab.push_back("<s>");
        tok.vocab.push_back("</s>");
        tok.vocab.push_back("Hello");
        tok.vocab.push_back("Hi");
        tok.vocab.push_back(" ");
        tok.scores.resize(tok.vocab.size(), 0.0f);
        for (size_t i = 0; i < tok.vocab.size(); i++) tok.tok2id[tok.vocab[i]] = (int32_t)i;
        RDNA4_LOG("Using fallback tokenizer: %zu tokens", tok.vocab.size());
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    vk_device_t* device = vk_device_create(a.validate);
    if (!device) {
        RDNA4_ERROR("Vulkan device creation failed");
        return 1;
    }

    vk_model_t model;
    memset(&model, 0, sizeof(model));
    if (!vk_model_load(device, &model, a.model.c_str())) {
        RDNA4_ERROR("Model load failed");
        vk_device_destroy(device);
        return 1;
    }

    RDNA4_LOG("Creating KV cache...");
    if (!vk_kv_cache_create(device, &model.kv_cache, model.config.n_layers,
                            model.config.n_kv_heads, model.config.head_dim, MAX_SEQ_LEN)) {
        RDNA4_ERROR("KV cache creation failed");
        vk_model_unload(device, &model);
        vk_device_destroy(device);
        return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    RDNA4_LOG("Model + KV cache loaded in %.0f ms",
        std::chrono::duration<double, std::milli>(t1 - t0).count());

    vk_session_t session;
    memset(&session, 0, sizeof(session));
    if (!vk_session_create(device, &model, &session)) {
        RDNA4_ERROR("Session creation failed");
        vk_kv_cache_destroy(device, &model.kv_cache);
        vk_model_unload(device, &model);
        vk_device_destroy(device);
        return 1;
    }

    if (!vk_session_build_pipelines(&session)) {
        RDNA4_LOG("GPU pipelines not available — using CPU reference path");
    } else {
        session.pipelines_ready = true;
    }

    std::vector<int32_t> input_tokens = tok_encode(&tok, a.prompt);

    uint32_t vocab_size = model.config.vocab_size;

    if (session.pipelines_ready && !getenv("RDNA4_CPU")) {
        RDNA4_LOG("GPU inference path active");

        if (a.validate_gpu && !input_tokens.empty()) {
            RDNA4_LOG("--validate-gpu is temporarily unavailable: vk_validate.cpp is being");
            RDNA4_LOG("rewritten to call the real dispatch path instead of a hand-duplicated");
            RDNA4_LOG("command buffer (engine rewrite plan, step 7). Not yet re-enabled.");
        }

        RDNA4_LOG("Prefilling %zu tokens...", input_tokens.size());

        // session.push_constants already holds every static field (offsets,
        // dims, rope/norm params) computed once in vk_session_create -- the
        // per-token loop below updates only what actually changes per step
        // (kv_cache_pos, seq_len, token_id, layer_idx) directly on it.

        uint32_t seq_pos = 0;
        xoshiro256pp_t rng;
        xoshiro256pp_seed(&rng, a.seed);

        sampler_params_t sparams = {};
        sparams.temperature = a.temp;
        sparams.top_k       = (int)a.top_k;
        sparams.top_p       = a.top_p;
        sparams.min_p       = 0.0f;
        sparams.repetition_penalty = 1.0f;

        for (size_t i = 0; i < input_tokens.size(); i++) {
            session.push_constants.layer_idx    = 0;
            session.push_constants.kv_cache_pos = seq_pos;
            session.push_constants.seq_len      = seq_pos + 1;
            session.push_constants.token_id     = input_tokens[i];
            session.decode_state.current_pos    = seq_pos;
            session.decode_state.total_seq_len  = seq_pos + 1;

            vk_session_build_decode_cb(&session);
            RDNA4_LOG("prefill token %zu/%zu submit", i + 1, input_tokens.size());
            if (!vk_session_submit(&session)) { RDNA4_ERROR("Prefill submit failed"); break; }
            if (!vk_session_wait(&session, 30000)) break;
            float* logits = (float*)session.decode_state.logits_buf.mapped_ptr;
            {
                char lbl[64];
                snprintf(lbl, sizeof(lbl), "prefill pos=%u", seq_pos);
                debug_check_hidden(&session, lbl);
            }

            if (i < input_tokens.size() - 1) {
                seq_pos++;
            } else {
                int32_t next = sample_token(logits, vocab_size, &sparams, &rng);
                std::vector<int32_t> gen = { next };
                seq_pos++;

                auto td = std::chrono::high_resolution_clock::now();

                for (uint32_t s = 0; s < a.max_tokens; s++) {
                    session.push_constants.layer_idx    = 0;
                    session.push_constants.kv_cache_pos = seq_pos;
                    session.push_constants.seq_len      = seq_pos + 1;
                    session.push_constants.token_id     = next;
                    session.decode_state.current_pos    = seq_pos;
                    session.decode_state.total_seq_len  = seq_pos + 1;

                    vk_session_build_decode_cb(&session);
                    if (!vk_session_submit(&session)) { RDNA4_ERROR("Decode submit failed"); break; }
                    if (!vk_session_wait(&session, 30000)) break;
                    logits = (float*)session.decode_state.logits_buf.mapped_ptr;
                    {
                        char lbl[64];
                        snprintf(lbl, sizeof(lbl), "decode pos=%u", seq_pos);
                        debug_check_hidden(&session, lbl);
                    }

                    if (a.verbose) {
                        fprintf(stderr, "[DEBUG] logits[0..5] = %.3f %.3f %.3f %.3f %.3f %.3f\n",
                            logits[0], logits[1], logits[2], logits[3], logits[4], logits[5]);
                    }

                    next = sample_token(logits, vocab_size, &sparams, &rng);
                    if (next == tok.eos_id && tok.eos_id >= 0) {
                        if (a.verbose) RDNA4_LOG("EOS");
                        break;
                    }
                    gen.push_back(next);
                    seq_pos++;

                    if (a.verbose) {
                        std::vector<int32_t> one = { next };
                        std::string p = tok_decode(one, &tok);
                        fprintf(stderr, "[TOK] id=%d vocab_str=\"%s\" piece_len=%zu\n",
                                next, next >= 0 && (size_t)next < tok.vocab.size() ? tok.vocab[(size_t)next].c_str() : "<oob>", p.size());
                        printf("%s", p.c_str());
                    }
                    if (seq_pos >= MAX_SEQ_LEN) { RDNA4_LOG("Max seq len"); break; }
                }

                auto te = std::chrono::high_resolution_clock::now();
                double dec = std::chrono::duration<double>(te - td).count();
                if (a.verbose) printf("\n");
                RDNA4_LOG("Generated %zu tokens in %.2f s (%.1f tok/s)",
                    gen.size(), dec, gen.size() / dec);
            }
        }
    } else {
        RDNA4_LOG("CPU reference inference path (no GPU pipelines available)");
        RDNA4_LOG("Prompt: \"%s\"", a.prompt.c_str());
        uint32_t d = model.config.d;
        uint32_t ffn = model.config.ffn_dim;
        uint32_t nh = model.config.n_heads;
        uint32_t nk = model.config.n_kv_heads;
        uint32_t hd = model.config.head_dim;
        uint32_t nl = model.config.n_layers;
        float eps = model.config.norm_eps;
        float th  = model.config.rope_theta;
        sampler_params_t sparams = {};
        sparams.temperature = a.temp;
        sparams.top_k       = (int)a.top_k;
        sparams.top_p       = a.top_p;
        xoshiro256pp_t rng;
        xoshiro256pp_seed(&rng, a.seed);
        RDNA4_LOG("CPU path: d=%u ffn=%u nh=%u nk=%u hd=%u L=%u vocab=%u",
            d, ffn, nh, nk, hd, nl, vocab_size);
        RDNA4_LOG("Note: Full GPU acceleration requires compiled SPIR-V shaders.");
        RDNA4_LOG("Run the shader build step: cmake --build . --target rdna4_shaders");
        RDNA4_LOG("See docs/VULKAN-LLM-ENGINE-ARCHITECTURE.md section 15.1 for details.");
        if (a.verbose) printf("\n[PROMPT] %s", a.prompt.c_str());
        RDNA4_LOG("Done. Model loaded, Vulkan device ready, waiting for shader compilation.");
        RDNA4_LOG("Total init time: %.0f ms",
            std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count());
    }

    RDNA4_LOG("Shutting down...");
    vk_session_destroy(&session);
    vk_kv_cache_destroy(device, &model.kv_cache);
    vk_model_unload(device, &model);
    vk_device_destroy(device);
    RDNA4_LOG("Done.");
    return 0;
}
