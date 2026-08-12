#include "vk_buffer.h"

bool vk_buffer_create(vk_device_t* dev, VkDeviceSize size, VkBufferUsageFlags usage,
                      bool device_local, vk_buffer_t* out) {
    memset(out, 0, sizeof(*out));
    out->size = size;

    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = size;
    bci.usage = usage | (device_local ? VK_BUFFER_USAGE_TRANSFER_DST_BIT : 0);
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci = {};
    if (device_local) {
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        aci.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        out->is_host_visible = false;
    } else {
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        aci.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        out->is_host_visible = true;
    }

    VkResult r = vmaCreateBuffer(dev->allocator, &bci, &aci, &out->buffer, &out->allocation, &out->alloc_info);
    if (r != VK_SUCCESS) {
        RDNA4_ERROR("vmaCreateBuffer failed: %d (size=%llu device_local=%d)", (int)r,
                    (unsigned long long)size, device_local);
        return false;
    }
    return true;
}

void vk_buffer_destroy(vk_device_t* dev, vk_buffer_t* buf) {
    if (buf->buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(dev->allocator, buf->buffer, buf->allocation);
    }
    memset(buf, 0, sizeof(*buf));
}

// Waits on a fence with a FINITE timeout and returns false (not a hang) on
// VK_TIMEOUT or a device-lost error. An earlier version of this engine used
// vkWaitForFences(..., UINT64_MAX) for load-time transfers with no escape
// hatch; when the GPU stalled (two Vulkan compute workloads contending for
// the same physical/display GPU), the process blocked forever and Windows'
// TDR did not cleanly recover, requiring a hard reboot. Every fence wait in
// this engine must go through this function.
static bool wait_fence_bounded(vk_device_t* dev, VkFence fence, uint64_t timeout_ms, const char* what) {
    VkResult r = vkWaitForFences(dev->device, 1, &fence, VK_TRUE, timeout_ms * 1000000ull);
    if (r == VK_SUCCESS) return true;
    if (r == VK_TIMEOUT) {
        RDNA4_ERROR("%s: fence timeout after %llu ms", what, (unsigned long long)timeout_ms);
        return false;
    }
    RDNA4_ERROR("%s: vkWaitForFences failed: %d (device lost?)", what, (int)r);
    return false;
}

bool vk_buffer_upload(vk_device_t* dev, vk_buffer_t* dst, const void* data, VkDeviceSize bytes,
                      uint64_t timeout_ms) {
    if (bytes == 0) return true;
    if (bytes > dst->size) {
        RDNA4_ERROR("vk_buffer_upload: %llu bytes exceeds buffer size %llu",
                    (unsigned long long)bytes, (unsigned long long)dst->size);
        return false;
    }

    if (dst->is_host_visible) {
        memcpy(dst->alloc_info.pMappedData, data, (size_t)bytes);
        return true;
    }

    vk_buffer_t staging;
    if (!vk_buffer_create(dev, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false, &staging)) return false;
    memcpy(staging.alloc_info.pMappedData, data, (size_t)bytes);

    VkCommandBufferAllocateInfo cbai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbai.commandPool = dev->transfer_cmd_pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(dev->device, &cbai, &cb));

    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    VkBufferCopy region = { 0, 0, bytes };
    vkCmdCopyBuffer(cb, staging.buffer, dst->buffer, 1, &region);
    vkEndCommandBuffer(cb);

    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence;
    VK_CHECK(vkCreateFence(dev->device, &fci, nullptr, &fence));

    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VK_CHECK(vkQueueSubmit(dev->transfer_queue, 1, &si, fence));

    bool ok = wait_fence_bounded(dev, fence, timeout_ms, "vk_buffer_upload");

    vkDestroyFence(dev->device, fence, nullptr);
    vkFreeCommandBuffers(dev->device, dev->transfer_cmd_pool, 1, &cb);
    vk_buffer_destroy(dev, &staging);
    return ok;
}

bool vk_buffer_download(vk_device_t* dev, vk_buffer_t* src, void* out_data, VkDeviceSize bytes,
                        uint64_t timeout_ms) {
    if (bytes == 0) return true;
    if (bytes > src->size) {
        RDNA4_ERROR("vk_buffer_download: %llu bytes exceeds buffer size %llu",
                    (unsigned long long)bytes, (unsigned long long)src->size);
        return false;
    }

    if (src->is_host_visible) {
        memcpy(out_data, src->alloc_info.pMappedData, (size_t)bytes);
        return true;
    }

    vk_buffer_t staging;
    if (!vk_buffer_create(dev, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, &staging)) return false;

    VkCommandBufferAllocateInfo cbai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbai.commandPool = dev->transfer_cmd_pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(dev->device, &cbai, &cb));

    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    VkBufferCopy region = { 0, 0, bytes };
    vkCmdCopyBuffer(cb, src->buffer, staging.buffer, 1, &region);
    vkEndCommandBuffer(cb);

    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence;
    VK_CHECK(vkCreateFence(dev->device, &fci, nullptr, &fence));

    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VK_CHECK(vkQueueSubmit(dev->transfer_queue, 1, &si, fence));

    bool ok = wait_fence_bounded(dev, fence, timeout_ms, "vk_buffer_download");
    if (ok) {
        memcpy(out_data, staging.alloc_info.pMappedData, (size_t)bytes);
    }

    vkDestroyFence(dev->device, fence, nullptr);
    vkFreeCommandBuffers(dev->device, dev->transfer_cmd_pool, 1, &cb);
    vk_buffer_destroy(dev, &staging);
    return ok;
}
