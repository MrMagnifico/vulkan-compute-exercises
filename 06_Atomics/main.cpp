#include <vulkan/vulkan.hpp>
#include <framework.h>
#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
	try
	{
		vk::UniqueInstance instance;
		vk::PhysicalDevice physicalDevice;
		vk::UniqueDevice device;
		vk::UniqueCommandPool commandPool;
		vk::Queue queue;

		Framework::setupBasicCompute("Task 7", VK_API_VERSION_1_3, {}, {}, instance, physicalDevice, device, queue, commandPool);

		const uint32_t numWorkGroups 	= 25;   							// How many work groups we will start
		const uint32_t sizeWorkGroup	= 128;  							// How many threads we expect in each work group
		const uint32_t numsTested 		= numWorkGroups * sizeWorkGroup;	// Total number of threads
		const size_t primesSize			= numsTested * sizeof(int);			// Size of buffer holding primes output

		VmaAllocator allocator = Framework::createAllocator(*instance, VK_API_VERSION_1_3, physicalDevice, *device);

		// TODO: The description for the resources that go into our pipeline (and descriptor set):
		// - One storage buffer for the counter (size of a single int)
		// - One storage buffer for the output array with primes (size of numTested * size of (int)).
		// - Both of them should be host-visible, host-coherent AND device-local.
		VkBuffer counterBuff, primesBuff;
		VkBufferCreateInfo counterBuffCreateInfo	= vk::BufferCreateInfo({}, 1ULL * sizeof(int), vk::BufferUsageFlagBits::eStorageBuffer);
		VkBufferCreateInfo primesBuffCreateInfo 	= vk::BufferCreateInfo({}, primesSize, vk::BufferUsageFlagBits::eStorageBuffer);
		VmaAllocationCreateInfo counterBuffAllocInfo = {}, primesBuffAllocInfo = {};
		counterBuffAllocInfo.usage = primesBuffAllocInfo.usage	= VMA_MEMORY_USAGE_AUTO;
		auto memReqFlags					= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		counterBuffAllocInfo.requiredFlags	= memReqFlags;
		primesBuffAllocInfo.requiredFlags 	= memReqFlags;
		counterBuffAllocInfo.flags = primesBuffAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT; // This allows memory to be mappable in random order when when usage is set to VMA_MEMORY_USAGE_AUTO
		VmaAllocation counterAlloc, primesAlloc;
		vmaCreateBuffer(allocator, &counterBuffCreateInfo, &counterBuffAllocInfo, &counterBuff, &counterAlloc, nullptr);
		vmaCreateBuffer(allocator, &primesBuffCreateInfo, &primesBuffAllocInfo, &primesBuff, &primesAlloc, nullptr);

		// TODO: As before, take all the necessary steps to have a descriptor set that references
		// the above created buffers. Make sure the bindings match the shader "primes.comp".
		// =======================================================================================
		// Define descriptor pool to provide two needed bindings
		std::vector<vk::DescriptorPoolSize> poolSizes	= { {vk::DescriptorType::eStorageBuffer, 2U}};
		vk::DescriptorPoolCreateInfo poolCreateInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1U, poolSizes);
		vk::UniqueDescriptorPool descriptorPool 		= device->createDescriptorPoolUnique(poolCreateInfo);
		// Define bindings for two needed buffers
		vk::DescriptorSetLayoutBinding counterBinding(0U, vk::DescriptorType::eStorageBuffer, 1U, vk::ShaderStageFlagBits::eCompute);
		vk::DescriptorSetLayoutBinding primesBinding(1U, vk::DescriptorType::eStorageBuffer, 1U, vk::ShaderStageFlagBits::eCompute);
		std::array<vk::DescriptorSetLayoutBinding, 2UL> allBindings	= { counterBinding, primesBinding };
		vk::DescriptorSetLayoutCreateInfo descLayoutCreateInfo({}, allBindings);
		vk::UniqueDescriptorSetLayout descriptorSetLayout = device->createDescriptorSetLayoutUnique(descLayoutCreateInfo);
		// Create descriptor set
		vk::DescriptorSetAllocateInfo setAllocateInfo(*descriptorPool, *descriptorSetLayout);
		std::vector<vk::UniqueDescriptorSet> descriptorSets	= device->allocateDescriptorSetsUnique(setAllocateInfo);
		vk::UniqueDescriptorSet& descriptorSet 				= descriptorSets[0];
		// Connect bindings in descriptor set to actual buffers
		vk::DescriptorBufferInfo counterBuffDescriptorInfo(counterBuff, 0UL, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo primesBuffDescriptorInfo(primesBuff, 0UL, VK_WHOLE_SIZE);
		vk::WriteDescriptorSet counterDescriptorSetWrite(*descriptorSet, 0U, 0U, 1U, vk::DescriptorType::eStorageBuffer, {}, &counterBuffDescriptorInfo);
		vk::WriteDescriptorSet primesDescriptorSetWrite(*descriptorSet, 1U, 0U, 1U, vk::DescriptorType::eStorageBuffer, {}, &primesBuffDescriptorInfo);
		device->updateDescriptorSets({counterDescriptorSetWrite, primesDescriptorSetWrite}, {});

		// TODO: Map and initialize buffer memories. Set the counter, a single integer, to 0. Next,
		// map the output array to the above variable "outputMapped". Initialize the entries in the 
		// array all to UINT_MAX, or 0xFFFFFFFF (equivalent). Hint: for this, you can actually just 
		// use std::memset to do it in one line. 
		uint32_t *counterMapped, *primesMapped = nullptr;
		vmaMapMemory(allocator, counterAlloc, (void**) &counterMapped);
		vmaMapMemory(allocator, primesAlloc, (void**) &primesMapped);
		*counterMapped = 0U;
		std::memset(primesMapped, UINT_MAX, primesSize);
		vmaUnmapMemory(allocator, counterAlloc);
		vmaUnmapMemory(allocator, primesAlloc);
		vmaFlushAllocation(allocator, counterAlloc, 0ULL, VK_WHOLE_SIZE);
		vmaFlushAllocation(allocator, primesAlloc, 0ULL, VK_WHOLE_SIZE);

		vk::UniqueShaderModule shaderModule;
		vk::UniquePipelineLayout layout;
		vk::UniquePipelineCache cache;
		vk::UniquePipeline pipeline;
		Framework::setupComputePipeline("primes.comp.spv", { *descriptorSetLayout }, *device, shaderModule, layout, cache, pipeline);

		vk::CommandBufferAllocateInfo allocateInfo(*commandPool, vk::CommandBufferLevel::ePrimary, 1);
		auto cmdBuffers = device->allocateCommandBuffersUnique(allocateInfo);
		vk::MemoryBarrier memoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead);
		cmdBuffers[0]->begin(vk::CommandBufferBeginInfo{});
		cmdBuffers[0]->bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
		cmdBuffers[0]->bindDescriptorSets(vk::PipelineBindPoint::eCompute, *layout, 0, *descriptorSet, {});
		cmdBuffers[0]->dispatch(numWorkGroups, 1, 1);
		cmdBuffers[0]->pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost, {}, memoryBarrier, {}, {});
		cmdBuffers[0]->end();

		Framework::initDebugging();
		Framework::beginCapture();
		queue.submit(vk::SubmitInfo({}, {}, *cmdBuffers[0]));
		Framework::endCapture();

		device->waitIdle();

		vmaMapMemory(allocator, primesAlloc, (void**) &primesMapped);
		std::sort(primesMapped, primesMapped + numsTested);
		std::cout << "All prime numbers from 0 to " << numsTested << ": ";
		for (int i = 0; i < numsTested; i++) {
			if (primesMapped && primesMapped[i] != 0xffffffff) { std::cout << (i == 0 ? "" : ", ") << primesMapped[i]; }
		}
		vmaUnmapMemory(allocator, primesAlloc);

		// Free allocated resources
		vmaDestroyBuffer(allocator, counterBuff, counterAlloc);
		vmaDestroyBuffer(allocator, primesBuff, primesAlloc);
		vmaDestroyAllocator(allocator);
	}
	catch (Framework::NotImplemented e)
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}

/*
==================================== Task 6 ====================================
1) Create all the resources you need to fill a descriptor set for the shader.
2) Map and initialize the buffer memories in main.cpp.
3) Complete the shader file primes.comp to compute and store prime numbers.
4) Optional: Why are the numbers not in order? Sort them on the CPU and print.
*/