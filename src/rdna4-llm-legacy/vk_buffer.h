#pragma once

#include "vk_device.h"

bool vk_buffer_create(
    VmaAllocator     allocator,
    VkDeviceSize     size,
    VkBufferUsageFlags usage,
    VmaMemoryUsage   memory_usage,
    VmaAllocationCreateFlags vma_flags,
    const char*      name,
    vk_device_t*     device,
    vk_buffer_t*     out_buf);

bool vk_buffer_create_device_local(
    VmaAllocator     allocator,
    VkDeviceSize     size,
    VkBufferUsageFlags usage,
    const char*      name,
    vk_device_t*     device,
    vk_buffer_t*     out_buf);

bool vk_buffer_create_host_visible(
    VmaAllocator     allocator,
    VkDeviceSize     size,
    VkBufferUsageFlags usage,
    bool             coherent,
    const char*      name,
    vk_device_t*     device,
    vk_buffer_t*     out_buf);

bool vk_buffer_create_staging(
    VmaAllocator     allocator,
    VkDeviceSize     size,
    const char*      name,
    vk_device_t*     device,
    vk_buffer_t*     out_buf);

void vk_buffer_destroy(
    VmaAllocator     allocator,
    vk_buffer_t*     buf);

void vk_buffer_upload(
    VmaAllocator     allocator,
    vk_buffer_t*     buf,
    const void*      data,
    VkDeviceSize     size);

VkDeviceAddress vk_buffer_get_device_address(
    VkDevice         device,
    vk_buffer_t*     buf);
