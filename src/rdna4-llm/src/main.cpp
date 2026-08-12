// Phase 0 harness: prove the full Vulkan compute round-trip (device init,
// buffer upload via staging, pipeline dispatch, buffer download) works before
// any GGUF/model logic exists. Later phases replace this main() with the real
// CLI; this file is deliberately small and self-contained.

#include "common.h"
#include "vk/vk_device.h"
#include "vk/vk_buffer.h"

#include <fstream>

static std::vector<uint32_t> read_spv(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        RDNA4_ERROR("Failed to open SPIR-V file: %s", path);
        return {};
    }
    size_t size = (size_t)f.tellg();
    f.seekg(0);
    std::vector<uint32_t> code(size / sizeof(uint32_t));
    f.read(reinterpret_cast<char*>(code.data()), (std::streamsize)size);
    return code;
}

int main() {
    vk_device_t* dev = vk_device_create(/*enable_validation=*/true);
    if (!dev) {
        RDNA4_ERROR("Device creation failed");
        return 1;
    }

    // ---- Load shader, build pipeline ----
    std::string spv_path = std::string(RDNA4_LLM_SPV_DIR) + "/harness_add.spv";
    std::vector<uint32_t> spv = read_spv(spv_path.c_str());
    if (spv.empty()) {
        vk_device_destroy(dev);
        return 1;
    }

    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = spv.size() * sizeof(uint32_t);
    smci.pCode = spv.data();
    VkShaderModule shader_module;
    VK_CHECK(vkCreateShaderModule(dev->device, &smci, nullptr, &shader_module));

    VkDescriptorSetLayoutBinding bindings[3] = {};
    for (uint32_t i = 0; i < 3; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dslci.bindingCount = 3;
    dslci.pBindings = bindings;
    VkDescriptorSetLayout set_layout;
    VK_CHECK(vkCreateDescriptorSetLayout(dev->device, &dslci, nullptr, &set_layout));

    VkPushConstantRange pcr = {};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(uint32_t);

    VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &set_layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VkPipelineLayout pipeline_layout;
    VK_CHECK(vkCreatePipelineLayout(dev->device, &plci, nullptr, &pipeline_layout));

    VkPipelineShaderStageCreateInfo stage_ci = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    stage_ci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_ci.module = shader_module;
    stage_ci.pName = "main";

    VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage = stage_ci;
    cpci.layout = pipeline_layout;
    VkPipeline pipeline;
    VK_CHECK(vkCreateComputePipelines(dev->device, dev->pipeline_cache, 1, &cpci, nullptr, &pipeline));

    // ---- Buffers ----
    const uint32_t N = 1024;
    const VkDeviceSize bytes = N * sizeof(float);
    VkBufferUsageFlags storage_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    vk_buffer_t buf_a, buf_b, buf_c;
    vk_buffer_create(dev, bytes, storage_usage, /*device_local=*/true, &buf_a);
    vk_buffer_create(dev, bytes, storage_usage, /*device_local=*/true, &buf_b);
    vk_buffer_create(dev, bytes, storage_usage, /*device_local=*/true, &buf_c);

    std::vector<float> host_a(N), host_b(N), expected(N);
    for (uint32_t i = 0; i < N; i++) {
        host_a[i] = (float)i;
        host_b[i] = (float)(N - i) * 0.5f;
        expected[i] = host_a[i] + host_b[i];
    }

    bool up_ok = vk_buffer_upload(dev, &buf_a, host_a.data(), bytes)
              && vk_buffer_upload(dev, &buf_b, host_b.data(), bytes);
    if (!up_ok) {
        RDNA4_ERROR("Upload failed");
        vk_device_destroy(dev);
        return 1;
    }

    // ---- Descriptor set ----
    VkDescriptorPoolSize pool_size = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
    VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &pool_size;
    VkDescriptorPool desc_pool;
    VK_CHECK(vkCreateDescriptorPool(dev->device, &dpci, nullptr, &desc_pool));

    VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool = desc_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &set_layout;
    VkDescriptorSet desc_set;
    VK_CHECK(vkAllocateDescriptorSets(dev->device, &dsai, &desc_set));

    VkDescriptorBufferInfo buf_infos[3] = {
        { buf_a.buffer, 0, VK_WHOLE_SIZE },
        { buf_b.buffer, 0, VK_WHOLE_SIZE },
        { buf_c.buffer, 0, VK_WHOLE_SIZE },
    };
    VkWriteDescriptorSet writes[3] = {};
    for (uint32_t i = 0; i < 3; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = desc_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buf_infos[i];
    }
    vkUpdateDescriptorSets(dev->device, 3, writes, 0, nullptr);

    // ---- Dispatch ----
    VkCommandBufferAllocateInfo cbai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbai.commandPool = dev->compute_cmd_pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(dev->device, &cbai, &cb));

    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &desc_set, 0, nullptr);
    vkCmdPushConstants(cb, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &N);
    vkCmdDispatch(cb, (N + 63) / 64, 1, 1);
    vkEndCommandBuffer(cb);

    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence;
    VK_CHECK(vkCreateFence(dev->device, &fci, nullptr, &fence));

    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VK_CHECK(vkQueueSubmit(dev->compute_queue, 1, &si, fence));

    VkResult wait_result = vkWaitForFences(dev->device, 1, &fence, VK_TRUE, 30000ull * 1000000ull);
    if (wait_result != VK_SUCCESS) {
        RDNA4_ERROR("Dispatch fence wait failed/timed out: %d", (int)wait_result);
        vk_device_destroy(dev);
        return 1;
    }

    // ---- Download + verify ----
    std::vector<float> host_c(N);
    if (!vk_buffer_download(dev, &buf_c, host_c.data(), bytes)) {
        RDNA4_ERROR("Download failed");
        vk_device_destroy(dev);
        return 1;
    }

    bool all_match = true;
    for (uint32_t i = 0; i < N; i++) {
        if (std::fabs(host_c[i] - expected[i]) > 1e-4f) {
            RDNA4_ERROR("Mismatch at i=%u: got %f expected %f", i, host_c[i], expected[i]);
            all_match = false;
            if (i > 10) break; // don't spam
        }
    }

    if (all_match) {
        RDNA4_LOG("PHASE 0 HARNESS: PASS (%u elements, GPU compute round-trip verified)", N);
    } else {
        RDNA4_LOG("PHASE 0 HARNESS: FAIL");
    }

    vkDestroyFence(dev->device, fence, nullptr);
    vkFreeCommandBuffers(dev->device, dev->compute_cmd_pool, 1, &cb);
    vkDestroyDescriptorPool(dev->device, desc_pool, nullptr);
    vk_buffer_destroy(dev, &buf_a);
    vk_buffer_destroy(dev, &buf_b);
    vk_buffer_destroy(dev, &buf_c);
    vkDestroyPipeline(dev->device, pipeline, nullptr);
    vkDestroyPipelineLayout(dev->device, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(dev->device, set_layout, nullptr);
    vkDestroyShaderModule(dev->device, shader_module, nullptr);
    vk_device_destroy(dev);

    return all_match ? 0 : 1;
}
