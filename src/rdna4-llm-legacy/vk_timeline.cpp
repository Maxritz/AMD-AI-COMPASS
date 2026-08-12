#include "vk_timeline.h"

bool vk_timeline_create(VkDevice device, vk_timeline_t* tl) {
    memset(tl, 0, sizeof(*tl));

    VkSemaphoreTypeCreateInfo type_info = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    type_info.initialValue  = 0;

    VkSemaphoreCreateInfo ci = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    ci.pNext = &type_info;

    VkResult result = vkCreateSemaphore(device, &ci, nullptr, &tl->semaphore);
    if (result != VK_SUCCESS) {
        RDNA4_ERROR("vkCreateSemaphore (timeline) failed: %d", (int)result);
        return false;
    }

    tl->current_value   = 0;
    tl->completed_value = 0;
    return true;
}

void vk_timeline_destroy(VkDevice device, vk_timeline_t* tl) {
    if (tl->semaphore) {
        vkDestroySemaphore(device, tl->semaphore, nullptr);
        tl->semaphore = VK_NULL_HANDLE;
    }
    memset(tl, 0, sizeof(*tl));
}

uint64_t vk_timeline_advance(vk_timeline_t* tl) {
    tl->current_value++;
    return tl->current_value;
}

bool vk_timeline_wait(VkDevice device, vk_timeline_t* tl, uint64_t value, uint64_t timeout_ns) {
    VkSemaphoreWaitInfo wait_info = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wait_info.semaphoreCount = 1;
    wait_info.pSemaphores    = &tl->semaphore;
    wait_info.pValues        = &value;

    VkResult result = vkWaitSemaphores(device, &wait_info, timeout_ns);
    if (result == VK_TIMEOUT) return false;
    if (result == VK_SUCCESS) {
        tl->completed_value = value;
        return true;
    }
    RDNA4_ERROR("vkWaitSemaphores failed: %d", (int)result);
    return false;
}

uint64_t vk_timeline_poll(VkDevice device, vk_timeline_t* tl) {
    uint64_t val = 0;
    VkResult result = vkGetSemaphoreCounterValue(device, tl->semaphore, &val);
    if (result == VK_SUCCESS) {
        tl->completed_value = val;
    }
    return val;
}
