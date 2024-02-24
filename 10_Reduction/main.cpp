#include <vulkan/vulkan.hpp>
#include <framework.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>



int main(int argc, char* argv[]) {
	// Fill out array to be reduced and compute reference sum
	constexpr int N = 50'000'000;
	std::default_random_engine eng(42);
	std::uniform_real_distribution<float> dist;
	std::vector<float> input(N);
	double full_sum = 0.0;
	std::cout << "Computing reference value..." << std::endl;
	for (int i = 0; i < N; i++) {
		input[i] 	= dist(eng);
		full_sum	+= input[i];
	}
	double reference_sum = 0;
	for (int i = 0; i < N; i++) {
		input[i] 		= (float)((input[i] * 42.0) / full_sum);
		reference_sum	+= input[i];
	}
	std::cout << "CPU reference (double): " << reference_sum << std::endl;

	try {
		vk::UniqueInstance instance;
		vk::PhysicalDevice physicalDevice;
		vk::UniqueDevice device;
		vk::UniqueCommandPool commandPool;
		vk::Queue queue;
		Framework::setupBasicCompute("Task 10", VK_API_VERSION_1_3, {}, { VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME }, instance, physicalDevice, device, queue, commandPool);

		VmaAllocator allocator = Framework::createAllocator(*instance, VK_API_VERSION_1_3, physicalDevice, *device);

		// TODO: Set up all necessary resources and prepare them to do a reduction
		// of the input array as fast as you can manage! Pick the proper resources
		// and memory and initialize their contents. Prepare a descriptor set layout,
		// descriptor set and record a command buffer.
		//
		// TODO: Make sure to copy all the data from the CPU input vector to your
		// GPU-side buffers, as well as necessary parameters (check the shader source).

		// Create buffer and allocate memory for its storage
		constexpr size_t arrBuffSize = N * sizeof(float);
		VkBuffer arrBuff, arrLenBuff;
		VkBufferCreateInfo arrBuffCreateInfo 	= vk::BufferCreateInfo({}, arrBuffSize, vk::BufferUsageFlagBits::eStorageBuffer);
		VkBufferCreateInfo arrLenBuffCreateInfo	= vk::BufferCreateInfo({}, sizeof(uint32_t), vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst);
		VmaAllocationCreateInfo arrBuffAllocInfo = {}, arrLenBuffAllocInfo = {};
		arrBuffAllocInfo.usage = arrLenBuffAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
		arrBuffAllocInfo.requiredFlags	= arrLenBuffAllocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT 	|
																			  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT	|
																			  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		arrBuffAllocInfo.flags = arrLenBuffAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
		VmaAllocation arrAlloc, arrLenAlloc;
		vmaCreateBuffer(allocator, &arrBuffCreateInfo, &arrBuffAllocInfo, &arrBuff, &arrAlloc, nullptr);
		vmaCreateBuffer(allocator, &arrLenBuffCreateInfo, &arrLenBuffAllocInfo, &arrLenBuff, &arrLenAlloc, nullptr);

		// Define descriptor pool to provide needed bindings
		std::vector<vk::DescriptorPoolSize> poolSizes	= {{vk::DescriptorType::eUniformBuffer, 1U},
														   {vk::DescriptorType::eStorageBuffer, 1U}};
		vk::DescriptorPoolCreateInfo poolCreateInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1U, poolSizes);
		vk::UniqueDescriptorPool descriptorPool 		= device->createDescriptorPoolUnique(poolCreateInfo);
		// Define bindings for needed buffers
		vk::DescriptorSetLayoutBinding arrLenBinding(0U, vk::DescriptorType::eUniformBuffer, 1U, vk::ShaderStageFlagBits::eCompute);
		vk::DescriptorSetLayoutBinding arrBinding(1U, vk::DescriptorType::eStorageBuffer, 1U, vk::ShaderStageFlagBits::eCompute);
		std::array<vk::DescriptorSetLayoutBinding, 2UL> allBindings	= { arrBinding, arrLenBinding };
		vk::DescriptorSetLayoutCreateInfo descLayoutCreateInfo({}, allBindings);
		vk::UniqueDescriptorSetLayout descriptorSetLayout = device->createDescriptorSetLayoutUnique(descLayoutCreateInfo);
		// Create descriptor set
		vk::DescriptorSetAllocateInfo setAllocateInfo(*descriptorPool, *descriptorSetLayout);
		std::vector<vk::UniqueDescriptorSet> descriptorSets	= device->allocateDescriptorSetsUnique(setAllocateInfo);
		vk::UniqueDescriptorSet& descriptorSet 				= descriptorSets[0];
		// Connect bindings in descriptor set to actual buffers
		vk::DescriptorBufferInfo arrLenBuffDescriptorInfo(arrLenBuff, 0UL, VK_WHOLE_SIZE);
		vk::WriteDescriptorSet arrLenDescriptorSetWrite(*descriptorSet, 0U, 0U, 1U, vk::DescriptorType::eUniformBuffer, {}, &arrLenBuffDescriptorInfo);
		vk::DescriptorBufferInfo arrBuffDescriptorInfo(arrBuff, 0UL, VK_WHOLE_SIZE);
		vk::WriteDescriptorSet arrDescriptorSetWrite(*descriptorSet, 1U, 0U, 1U, vk::DescriptorType::eStorageBuffer, {}, &arrBuffDescriptorInfo);
		device->updateDescriptorSets({arrLenDescriptorSetWrite, arrDescriptorSetWrite}, {});

		// Transfer array data to GPU memory
		float* arrMapped;
		vmaMapMemory(allocator, arrAlloc, (void**) &arrMapped);
		std::memcpy(arrMapped, input.data(), arrBuffSize);
		vmaUnmapMemory(allocator, arrAlloc);
		vmaFlushAllocation(allocator, arrAlloc, 0U, VK_WHOLE_SIZE);

		// Set up compute pipeline
		vk::UniqueShaderModule shaderModule;
		vk::UniquePipelineLayout layout;
		vk::UniquePipelineCache cache;
		vk::UniquePipeline pipeline;
		Framework::setupComputePipeline("reduce.comp.spv", { *descriptorSetLayout }, *device, shaderModule, layout, cache, pipeline);

		// TODO: Set up your submission and submit your command buffer! 
		vk::CommandBufferAllocateInfo allocateInfo(*commandPool, vk::CommandBufferLevel::ePrimary, 1);
		auto cmdBuffers = device->allocateCommandBuffersUnique(allocateInfo);
		vk::MemoryBarrier hostRead(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead);
		vk::MemoryBarrier shaderRecursion(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
		vk::MemoryBarrier lenUpdateWrite(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead);
		vk::MemoryBarrier lenUpdatePreviousRead(vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eTransferWrite);
		cmdBuffers[0]->begin(vk::CommandBufferBeginInfo{});
		cmdBuffers[0]->bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
		cmdBuffers[0]->bindDescriptorSets(vk::PipelineBindPoint::eCompute, *layout, 0, *descriptorSet, {});
		
		// Recursion to reduce in chunks of workGroupSize
		constexpr uint32_t workGroupSize	= 1024U; // Must be the same as LOCAL_SIZE in shaders/reduce.comp
		uint32_t elemsToReduce				= N;
		do {
			uint32_t numWorkGroups = (elemsToReduce / workGroupSize) + 1U;
			cmdBuffers[0]->pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
										   {}, lenUpdatePreviousRead, {}, {});
			cmdBuffers[0]->updateBuffer(arrLenBuff, 0, sizeof(uint32_t), &elemsToReduce);
			cmdBuffers[0]->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
										   {}, lenUpdateWrite, {}, {});
			cmdBuffers[0]->dispatch(numWorkGroups, 1, 1);
			cmdBuffers[0]->pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
										   {}, shaderRecursion, {}, {});
			elemsToReduce = static_cast<uint32_t>(std::ceil(elemsToReduce / workGroupSize));
		} while (elemsToReduce > 1U);

		cmdBuffers[0]->pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost, {}, hostRead, {}, {});
		cmdBuffers[0]->end();

		// Time GPU execution
		auto beforeGPU = std::chrono::system_clock::now();
		Framework::initDebugging();
		Framework::beginCapture();
		queue.submit(vk::SubmitInfo({}, {}, * cmdBuffers[0]));	// Execute command buffer with optional debugging
		Framework::endCapture();
		device->waitIdle();										// Wait until device is idle before proceeding
		auto afterGPU = std::chrono::system_clock::now();
		std::cout << "GPU time: " << std::chrono::duration_cast<std::chrono::milliseconds>(afterGPU - beforeGPU).count() << " ms\n";

		// TODO: read back the result of the reduction from the GPU.
		// Store it in "input_sum_gpu".
		float input_sum_gpu = 0.0f;
		vmaMapMemory(allocator, arrAlloc, (void**) &arrMapped);
		input_sum_gpu = arrMapped[0];
		vmaUnmapMemory(allocator, arrAlloc);
		std::cout << "GPU reduction (float): " << input_sum_gpu << std::endl;

		// Compute single-precision float CPU reference
		auto beforeCPU = std::chrono::system_clock::now();
		float input_sum = 0.0f;
		for (int i = 0; i < N; i++) { input_sum += input[i]; }
		auto afterCPU = std::chrono::system_clock::now();
		std::cout << "CPU time: " << std::chrono::duration_cast<std::chrono::milliseconds>(afterCPU - beforeCPU).count() << " ms\n";
		std::cout << "CPU reduction (float): " << input_sum << std::endl;

		// Free allocated resources
		vmaDestroyBuffer(allocator, arrBuff, arrAlloc);
		vmaDestroyBuffer(allocator, arrLenBuff, arrLenAlloc);
		vmaDestroyAllocator(allocator);
	}
	catch (Framework::NotImplemented e) { std::cerr << e.what() << std::endl; }

	return EXIT_SUCCESS;
}

/*
==================================== Task 10 ====================================
1) Prepare the environment and necessary GPU resources for reduction (addition).
2) Implement the corresponding GLSL shader code in reduce.comp.
3) Make sure that your solution is as fast as you can manage. Be sure to pick the
proper memory types for your buffers and that your shader uses fast techniques
such as optimal algorithms, shared memory and perhaps even subgroups! As you 
prepare your solution, take note whether your optimizations are actually making
things faster, and by how much!
4) Optional: Compare the accuracy of the CPU (float) reduction to your GPU (float)
reduction. Which one comes closer to the CPU (double) reference? Why?
*/