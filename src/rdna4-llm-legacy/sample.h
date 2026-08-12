#pragma once

#include <cstdint>

struct xoshiro256pp_t {
    uint64_t s[4];
};

struct sampler_params_t {
    float temperature;
    int   top_k;
    float top_p;
    float min_p;
    float repetition_penalty;
};

void    xoshiro256pp_seed(xoshiro256pp_t* rng, uint64_t seed);
uint64_t xoshiro256pp_next(xoshiro256pp_t* rng);
float   xoshiro256pp_next_f32(xoshiro256pp_t* rng);
int32_t sample_token(const float* logits, uint32_t vocab_size,
                      const sampler_params_t* params, xoshiro256pp_t* rng);
