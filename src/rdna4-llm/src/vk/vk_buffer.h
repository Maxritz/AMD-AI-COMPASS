#pragma once

#include "vk_device.h"

struct vk_buffer_t {
    VkBuffer           buffer;
    VmaAllocation      allocation;
    VmaAllocationInfo  alloc_info;
    VkDeviceSize       size;
    bool               is_host_visible;
};

// device_local=true -> fast VRAM, upload via staging; false -> host-visible/coherent, mapped directly.
bool vk_buffer_create(vk_device_t* dev, VkDeviceSize size, VkBufferUsageFlags usage,
                      bool device_local, vk_buffer_t* out);
void vk_buffer_destroy(vk_device_t* dev, vk_buffer_t* buf);

// Blocking upload via a one-shot staging buffer + fence wait (finite timeout --
// see the plan's fence-safety note: infinite waits with no escape hatch caused
// a real system hang last session).
bool vk_buffer_upload(vk_device_t* dev, vk_buffer_t* dst, const void* data, VkDeviceSize bytes,
                      uint64_t timeout_ms = 30000);

// Blocking download, same fence-safety rule.
bool vk_buffer_download(vk_device_t* dev, vk_buffer_t* src, void* out_data, VkDeviceSize bytes,
                        uint64_t timeout_ms = 30000);
