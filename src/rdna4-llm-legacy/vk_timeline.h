#pragma once

#include "common.h"

struct vk_timeline_t {
    VkSemaphore semaphore;
    uint64_t    current_value;
    uint64_t    completed_value;
};

bool     vk_timeline_create(VkDevice device, vk_timeline_t* tl);
void     vk_timeline_destroy(VkDevice device, vk_timeline_t* tl);
uint64_t vk_timeline_advance(vk_timeline_t* tl);
bool     vk_timeline_wait(VkDevice device, vk_timeline_t* tl, uint64_t value, uint64_t timeout_ns);
uint64_t vk_timeline_poll(VkDevice device, vk_timeline_t* tl);
