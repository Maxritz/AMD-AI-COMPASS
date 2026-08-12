#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

#include "vk_buffer.h"

#include <cstdio>
#include <cstring>

bool vk_buffer_create(
    VmaAllocator     allocator,
    VkDeviceSize     size,
    VkBufferUsageFlags usage,
    VmaMemoryUsage   memory_usage,
    VmaAllocationCreateFlags vma_flags,
    const char*      name,
    vk_device_t*     device,
    vk_buffer_t*     out_buf)
{
    memset(out_buf, 0, sizeof(vk_buffer_t));

    VkBufferCreateInfo buf_ci = {};
    buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_ci.size = size;
    buf_ci.usage = usage;
    buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo alloc_ci = {};
    alloc_ci.usage = memory_usage;
    alloc_ci.flags = vma_flags;

    VkResult result = vmaCreateBuffer(allocator, &buf_ci, &alloc_ci,
        &out_buf->buffer, &out_buf->allocation, &out_buf->alloc_info);

    if (result != VK_SUCCESS) {
        RDNA4_ERROR("Failed to allocate buffer '%s': %zu bytes, VkResult=%d",
            name, (size_t)size, (int)result);
        return false;
    }

    out_buf->size = size;
    out_buf->usage = usage;
    out_buf->device_address = 0;
    out_buf->mapped_ptr = nullptr;
    out_buf->is_host_visible = false;
    out_buf->is_host_coherent = false;

    if (vma_flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) {
        out_buf->mapped_ptr = out_buf->alloc_info.pMappedData;
        out_buf->is_host_visible = true;
        VkMemoryPropertyFlags mem_flags;
        vmaGetAllocationMemoryProperties(allocator, out_buf->allocation, &mem_flags);
        out_buf->is_host_coherent = (mem_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    } else {
        VkMemoryPropertyFlags mem_flags;
        vmaGetAllocationMemoryProperties(allocator, out_buf->allocation, &mem_flags);
        out_buf->is_host_visible = (mem_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
        out_buf->is_host_coherent = (mem_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    }

    if (device && name) {
        PFN_vkSetDebugUtilsObjectNameEXT set_name = (PFN_vkSetDebugUtilsObjectNameEXT)
            vkGetInstanceProcAddr(device->instance, "vkSetDebugUtilsObjectNameEXT");
        if (set_name) {
            VkDebugUtilsObjectNameInfoEXT name_info = {};
            name_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
            name_info.objectType = VK_OBJECT_TYPE_BUFFER;
            name_info.objectHandle = (uint64_t)out_buf->buffer;
            name_info.pObjectName = name;
            set_name(device->device, &name_info);
        }
    }

    return true;
}

bool vk_buffer_create_device_local(
    VmaAllocator     allocator,
    VkDeviceSize     size,
    VkBufferUsageFlags usage,
    const char*      name,
    vk_device_t*     device,
    vk_buffer_t*     out_buf)
{
    VkBufferUsageFlags extra = 0;
    if (device && device->has_bda) {
        extra = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    return vk_buffer_create(allocator, size, usage | extra,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0, name, device, out_buf);
}

bool vk_buffer_create_host_visible(
    VmaAllocator     allocator,
    VkDeviceSize     size,
    VkBufferUsageFlags usage,
    bool             coherent,
    const char*      name,
    vk_device_t*     device,
    vk_buffer_t*     out_buf)
{
    VmaAllocationCreateFlags flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    VmaMemoryUsage mem_usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    if (coherent) {
        mem_usage = VMA_MEMORY_USAGE_AUTO;
    }

    VkBufferUsageFlags extra = 0;
    if (device && device->has_bda) {
        extra = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }

    return vk_buffer_create(allocator, size, usage | extra,
        mem_usage, flags, name, device, out_buf);
}

bool vk_buffer_create_staging(
    VmaAllocator     allocator,
    VkDeviceSize     size,
    const char*      name,
    vk_device_t*     device,
    vk_buffer_t*     out_buf)
{
    return vk_buffer_create(allocator, size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_MAPPED_BIT |
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        name, device, out_buf);
}

void vk_buffer_destroy(
    VmaAllocator     allocator,
    vk_buffer_t*     buf)
{
    if (!buf || buf->buffer == VK_NULL_HANDLE) return;

    if (buf->mapped_ptr) {
        vmaUnmapMemory(allocator, buf->allocation);
        buf->mapped_ptr = nullptr;
    }

    vmaDestroyBuffer(allocator, buf->buffer, buf->allocation);
    memset(buf, 0, sizeof(vk_buffer_t));
}

void vk_buffer_upload(
    VmaAllocator     allocator,
    vk_buffer_t*     buf,
    const void*      data,
    VkDeviceSize     size)
{
    if (!buf || !buf->mapped_ptr) {
        RDNA4_ERROR("vk_buffer_upload: buffer not mapped");
        return;
    }
    if (size > buf->size) {
        RDNA4_ERROR("vk_buffer_upload: size %zu exceeds buffer size %zu",
            (size_t)size, (size_t)buf->size);
        return;
    }

    memcpy(buf->mapped_ptr, data, (size_t)size);

    if (!buf->is_host_coherent) {
        vmaFlushAllocation(allocator, buf->allocation, 0, size);
    }
}

VkDeviceAddress vk_buffer_get_device_address(
    VkDevice         device,
    vk_buffer_t*     buf)
{
    if (buf->device_address != 0) {
        return buf->device_address;
    }

    VkBufferDeviceAddressInfo addr_info = {};
    addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addr_info.buffer = buf->buffer;
    buf->device_address = vkGetBufferDeviceAddress(device, &addr_info);
    return buf->device_address;
}
