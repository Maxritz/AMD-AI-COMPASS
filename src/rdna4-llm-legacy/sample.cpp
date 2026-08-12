#include "sample.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>

static inline uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

void xoshiro256pp_seed(xoshiro256pp_t* rng, uint64_t seed) {
    uint64_t z = seed + 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < 4; i++) {
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rng->s[i] = z ^ (z >> 31);
    }
}

uint64_t xoshiro256pp_next(xoshiro256pp_t* rng) {
    uint64_t result = rotl(rng->s[0] + rng->s[3], 23) + rng->s[0];
    uint64_t t = rng->s[1] << 17;
    rng->s[2] ^= rng->s[0];
    rng->s[3] ^= rng->s[1];
    rng->s[1] ^= rng->s[2];
    rng->s[0] ^= rng->s[3];
    rng->s[2] ^= t;
    rng->s[3] = rotl(rng->s[3], 45);
    return result;
}

float xoshiro256pp_next_f32(xoshiro256pp_t* rng) {
    return (float)(xoshiro256pp_next(rng) >> 40) / (float)(1 << 24);
}

struct token_prob_t {
    int32_t index;
    float   prob;
};

static int compare_token_prob_desc(const void* a, const void* b) {
    float diff = ((const token_prob_t*)b)->prob - ((const token_prob_t*)a)->prob;
    if (diff > 0.0f) return 1;
    if (diff < 0.0f) return -1;
    return 0;
}

int32_t sample_token(const float* logits, uint32_t vocab_size,
                      const sampler_params_t* params, xoshiro256pp_t* rng) {
    if (vocab_size == 0) return 0;

    if (params->temperature <= 0.0f) {
        float max_val = logits[0];
        int32_t max_idx = 0;
        for (uint32_t i = 1; i < vocab_size; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = (int32_t)i;
            }
        }
        return max_idx;
    }

    float* probs = (float*)malloc(vocab_size * sizeof(float));
    if (!probs) return 0;

    float inv_temp = 1.0f / params->temperature;

    float max_logit = logits[0];
    for (uint32_t i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float sum = 0.0f;
    for (uint32_t i = 0; i < vocab_size; i++) {
        probs[i] = expf((logits[i] - max_logit) * inv_temp);
        sum += probs[i];
    }

    if (sum <= 0.0f) {
        free(probs);
        float max_val = logits[0];
        int32_t max_idx = 0;
        for (uint32_t i = 1; i < vocab_size; i++) {
            if (logits[i] > max_val) { max_val = logits[i]; max_idx = (int32_t)i; }
        }
        return max_idx;
    }

    float inv_sum = 1.0f / sum;
    for (uint32_t i = 0; i < vocab_size; i++) probs[i] *= inv_sum;

    if (params->top_k > 0 && (uint32_t)params->top_k < vocab_size) {
        token_prob_t* sorted = (token_prob_t*)malloc(vocab_size * sizeof(token_prob_t));
        uint32_t count = 0;
        for (uint32_t i = 0; i < vocab_size; i++) {
            if (probs[i] > 0.0f) {
                sorted[count].index = (int32_t)i;
                sorted[count].prob  = probs[i];
                count++;
            }
        }
        qsort(sorted, count, sizeof(token_prob_t), compare_token_prob_desc);

        uint32_t keep = (uint32_t)params->top_k;
        if (keep > count) keep = count;

        memset(probs, 0, vocab_size * sizeof(float));
        for (uint32_t i = 0; i < keep; i++) {
            probs[sorted[i].index] = sorted[i].prob;
        }
        free(sorted);
    }

    if (params->top_p > 0.0f && params->top_p < 1.0f) {
        token_prob_t* sorted = (token_prob_t*)malloc(vocab_size * sizeof(token_prob_t));
        uint32_t count = 0;
        for (uint32_t i = 0; i < vocab_size; i++) {
            if (probs[i] > 0.0f) {
                sorted[count].index = (int32_t)i;
                sorted[count].prob  = probs[i];
                count++;
            }
        }
        qsort(sorted, count, sizeof(token_prob_t), compare_token_prob_desc);

        float cumsum = 0.0f;
        uint32_t keep = 0;
        for (uint32_t i = 0; i < count; i++) {
            cumsum += sorted[i].prob;
            keep++;
            if (cumsum >= params->top_p) break;
        }

        memset(probs, 0, vocab_size * sizeof(float));
        for (uint32_t i = 0; i < keep; i++) {
            probs[sorted[i].index] = sorted[i].prob;
        }
        free(sorted);
    }

    if (params->min_p > 0.0f) {
        float max_prob = 0.0f;
        for (uint32_t i = 0; i < vocab_size; i++) {
            if (probs[i] > max_prob) max_prob = probs[i];
        }
        float threshold = max_prob * params->min_p;
        for (uint32_t i = 0; i < vocab_size; i++) {
            if (probs[i] < threshold) probs[i] = 0.0f;
        }
    }

    sum = 0.0f;
    for (uint32_t i = 0; i < vocab_size; i++) sum += probs[i];

    if (sum <= 0.0f) {
        free(probs);
        float max_val = logits[0];
        int32_t max_idx = 0;
        for (uint32_t i = 1; i < vocab_size; i++) {
            if (logits[i] > max_val) { max_val = logits[i]; max_idx = (int32_t)i; }
        }
        return max_idx;
    }

    inv_sum = 1.0f / sum;
    for (uint32_t i = 0; i < vocab_size; i++) probs[i] *= inv_sum;

    float r = xoshiro256pp_next_f32(rng);
    float cdf = 0.0f;
    int32_t chosen = 0;
    for (uint32_t i = 0; i < vocab_size; i++) {
        cdf += probs[i];
        if (r <= cdf) {
            chosen = (int32_t)i;
            break;
        }
    }

    free(probs);
    return chosen;
}
