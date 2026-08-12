#pragma once

#include "common.h"
#include "vk_device.h"
#include "vk_model.h"
#include "vk_session.h"

#include <vector>
#include <cstdio>

std::vector<float> download_buffer(vk_device_t* dev, vk_buffer_t* buf, uint32_t num_elements);

std::vector<float> cpu_rms_norm(const std::vector<float>& input,
                                 const std::vector<float>& norm_weight,
                                 float eps, uint32_t d);

std::vector<float> cpu_dequant_q8_0(const uint8_t* packed_data,
                                     uint32_t n_rows, uint32_t n_cols);

std::vector<float> cpu_matmul(const std::vector<float>& A,
                               const std::vector<float>& B,
                               uint32_t M, uint32_t N, uint32_t K);

bool validate_layer(vk_device_t* dev, vk_model_t* model, vk_session_t* session,
                    uint32_t layer_idx, int32_t token_id,
                    const std::vector<float>& cpu_hidden,
                    FILE* log_file);

bool vk_validate_run(vk_device_t* dev, vk_model_t* model, vk_session_t* session,
                     int32_t token_id, uint32_t kv_pos, FILE* log_file);
