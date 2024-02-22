#include <vulkan/vulkan.hpp>
#include <framework.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>


struct MatrixInfo {
	uint32_t width;
    uint32_t height;
};

template<size_t W, size_t H>
struct Matrix {
	MatrixInfo info;
	float data[W][H];
};

int main(int argc, char* argv[]) {
	// Generate random NxN matrix
	constexpr int N = 1024;
	std::default_random_engine eng(42);
	std::uniform_real_distribution<float> dist;
	std::vector<float> A(N * N), B(N * N), C(N * N);
	for (int i = 0; i < N * N; i++) {
		A[i] = dist(eng);
		B[i] = dist(eng);
		C[i] = 0;
	}

	try {
		// Set up Vulkan instance
		vk::UniqueInstance instance;
		vk::PhysicalDevice physicalDevice;
		vk::UniqueDevice device;
		vk::UniqueCommandPool commandPool;
		vk::Queue queue;
		Framework::setupBasicCompute("Task 9", VK_API_VERSION_1_3, {}, {}, instance, physicalDevice, device, queue, commandPool);

		VmaAllocator allocator = Framework::createAllocator(*instance, VK_API_VERSION_1_3, physicalDevice, *device);

		std::vector<float> C_computed_by_GPU(N * N, 0);

		// TODO: Set up all necessary resources and prepare them to do NxN matrix
		// mutliplication C = A * B on the GPU. Compare timing to CPU baseline. To 
		// (somewhat) accurately measure GPU time, use chrono to capture a timestamp 
		// before you submit the command buffer and then again AFTER you synchronized 
		// CPU and GPU, and compute the difference. Copy your result from the GPU into
		// C_computed_by_GPU. The error to the CPU result should be low (< 0.001).
		// Please note that he CPU computation is slooow, especially in Debug mode.

		// Create buffers and allocate memory for their storage
		constexpr size_t singleSize = sizeof(MatrixInfo) + (N * N * sizeof(float));
		constexpr size_t aSize = singleSize, bSize = singleSize, cSize = singleSize;
		VkBuffer aBuff, bBuff, cBuff;
		VkBufferCreateInfo aBuffCreateInfo	= vk::BufferCreateInfo({}, aSize, vk::BufferUsageFlagBits::eStorageBuffer);
		VkBufferCreateInfo bBuffCreateInfo	= vk::BufferCreateInfo({}, bSize, vk::BufferUsageFlagBits::eStorageBuffer);
		VkBufferCreateInfo cBuffCreateInfo	= vk::BufferCreateInfo({}, cSize, vk::BufferUsageFlagBits::eStorageBuffer);
		VmaAllocationCreateInfo aBuffAllocInfo = {}, bBuffAllocInfo = {}, cBuffAllocInfo = {};
		aBuffAllocInfo.usage = bBuffAllocInfo.usage	= cBuffAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
		auto memReqFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		aBuffAllocInfo.requiredFlags = bBuffAllocInfo.requiredFlags = cBuffAllocInfo.requiredFlags = memReqFlags;
		aBuffAllocInfo.flags = bBuffAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;	// A and B matrix memory will only be written to sequentially
		cBuffAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT; 									// C matrix will be read from and written to
		VmaAllocation aAlloc, bAlloc, cAlloc;
		vmaCreateBuffer(allocator, &aBuffCreateInfo, &aBuffAllocInfo, &aBuff, &aAlloc, nullptr);
		vmaCreateBuffer(allocator, &bBuffCreateInfo, &bBuffAllocInfo, &bBuff, &bAlloc, nullptr);
		vmaCreateBuffer(allocator, &cBuffCreateInfo, &cBuffAllocInfo, &cBuff, &cAlloc, nullptr);

		// Define descriptor pool to provide needed bindings
		std::vector<vk::DescriptorPoolSize> poolSizes	= {{vk::DescriptorType::eStorageBuffer, 3U}};
		vk::DescriptorPoolCreateInfo poolCreateInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1U, poolSizes);
		vk::UniqueDescriptorPool descriptorPool 		= device->createDescriptorPoolUnique(poolCreateInfo);
		// Define bindings for needed buffers
		vk::DescriptorSetLayoutBinding aBinding(0U, vk::DescriptorType::eStorageBuffer, 1U, vk::ShaderStageFlagBits::eCompute);
		vk::DescriptorSetLayoutBinding bBinding(1U, vk::DescriptorType::eStorageBuffer, 1U, vk::ShaderStageFlagBits::eCompute);
		vk::DescriptorSetLayoutBinding cBinding(2U, vk::DescriptorType::eStorageBuffer, 1U, vk::ShaderStageFlagBits::eCompute);
		std::array<vk::DescriptorSetLayoutBinding, 3UL> allBindings	= { aBinding, bBinding, cBinding };
		vk::DescriptorSetLayoutCreateInfo descLayoutCreateInfo({}, allBindings);
		vk::UniqueDescriptorSetLayout descriptorSetLayout = device->createDescriptorSetLayoutUnique(descLayoutCreateInfo);
		// Create descriptor set
		vk::DescriptorSetAllocateInfo setAllocateInfo(*descriptorPool, *descriptorSetLayout);
		std::vector<vk::UniqueDescriptorSet> descriptorSets	= device->allocateDescriptorSetsUnique(setAllocateInfo);
		vk::UniqueDescriptorSet& descriptorSet 				= descriptorSets[0];
		// Connect bindings in descriptor set to actual buffers
		vk::DescriptorBufferInfo aBuffDescriptorInfo(aBuff, 0UL, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo bBuffDescriptorInfo(bBuff, 0UL, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo cBuffDescriptorInfo(cBuff, 0UL, VK_WHOLE_SIZE);
		vk::WriteDescriptorSet aDescriptorSetWrite(*descriptorSet, 0U, 0U, 1U, vk::DescriptorType::eStorageBuffer, {}, &aBuffDescriptorInfo);
		vk::WriteDescriptorSet bDescriptorSetWrite(*descriptorSet, 1U, 0U, 1U, vk::DescriptorType::eStorageBuffer, {}, &bBuffDescriptorInfo);
		vk::WriteDescriptorSet cDescriptorSetWrite(*descriptorSet, 2U, 0U, 1U, vk::DescriptorType::eStorageBuffer, {}, &cBuffDescriptorInfo);
		device->updateDescriptorSets({aDescriptorSetWrite, bDescriptorSetWrite, cDescriptorSetWrite}, {});

		// Transfer matrix data to GPU
		Matrix<N, N> *aMapped, *bMapped, *cMapped;
		MatrixInfo singleInfo 			= { N, N };
		constexpr size_t singleMatSize	= N * N * sizeof(float);
		vmaMapMemory(allocator, aAlloc, (void**) &aMapped);
		vmaMapMemory(allocator, bAlloc, (void**) &bMapped);
		vmaMapMemory(allocator, cAlloc, (void**) &cMapped);
		aMapped->info = bMapped->info = cMapped->info = singleInfo;
		std::memcpy(aMapped->data, A.data(), singleMatSize);
		std::memcpy(bMapped->data, B.data(), singleMatSize);
		std::memcpy(cMapped->data, C.data(), singleMatSize);
		vmaUnmapMemory(allocator, aAlloc);
		vmaUnmapMemory(allocator, bAlloc);
		vmaUnmapMemory(allocator, cAlloc);
		vmaFlushAllocation(allocator, aAlloc, 0U, VK_WHOLE_SIZE);
		vmaFlushAllocation(allocator, bAlloc, 0U, VK_WHOLE_SIZE);
		vmaFlushAllocation(allocator, cAlloc, 0U, VK_WHOLE_SIZE);

		// Set up compute pipeline
		vk::UniqueShaderModule shaderModule;
		vk::UniquePipelineLayout layout;
		vk::UniquePipelineCache cache;
		vk::UniquePipeline pipeline;
		Framework::setupComputePipeline("matrixmult.comp.spv", { *descriptorSetLayout }, *device, shaderModule, layout, cache, pipeline);

		// Define work group size and number
		// We assume equal sizes in both dimensions
		constexpr uint32_t workGroupSize 	= 32U; // Equal number of threads in X and Y axes; Must be the same as LOCAL_SIZE in shaders/matrixmult.comp
		constexpr uint32_t numWorkGroupsX	= (N / workGroupSize) + 1U;
		constexpr uint32_t numWorkGroupsY	= (N / workGroupSize) + 1U;

		// Create and define command buffer
		vk::CommandBufferAllocateInfo allocateInfo(*commandPool, vk::CommandBufferLevel::ePrimary, 1);
		auto cmdBuffers = device->allocateCommandBuffersUnique(allocateInfo);
		vk::MemoryBarrier memoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead);
		cmdBuffers[0]->begin(vk::CommandBufferBeginInfo{});
		cmdBuffers[0]->bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
		cmdBuffers[0]->bindDescriptorSets(vk::PipelineBindPoint::eCompute, *layout, 0, *descriptorSet, {});
		cmdBuffers[0]->dispatch(numWorkGroupsX, numWorkGroupsY, 1);
		cmdBuffers[0]->pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost, {}, memoryBarrier, {}, {});
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

		// Copy result data from GPU
		vmaMapMemory(allocator, cAlloc, (void**) cMapped);
		std::memcpy(C_computed_by_GPU.data(), cMapped->data, singleMatSize);
		vmaUnmapMemory(allocator, cAlloc);
		
		// CPU computation
		auto beforeCPU = std::chrono::system_clock::now();
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				for (int k = 0; k < N; k++)
					C[i * N + j] += A[i * N + k] * B[k * N + j];
		auto afterCPU = std::chrono::system_clock::now();
		std::cout << "CPU time: " << std::chrono::duration_cast<std::chrono::milliseconds>(afterCPU - beforeCPU).count() << " ms\n";
		
		// CPU-GPU difference
		float error = 0.0f;
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				error += std::abs(C_computed_by_GPU[i * N + j] - C[i * N + j]);
		std::cout << "Avg. CPU-GPU Difference: " << error / (N * N) << std::endl;

		// Free allocated resources
		vmaDestroyBuffer(allocator, aBuff, aAlloc);
		vmaDestroyBuffer(allocator, bBuff, bAlloc);
		vmaDestroyBuffer(allocator, cBuff, cAlloc);
		vmaDestroyAllocator(allocator);
	}
	catch (Framework::NotImplemented e) { std::cerr << e.what() << std::endl; }

	return EXIT_SUCCESS;
}

/*
==================================== Task 9 ====================================
0) Read up on "efficient" matrix multiplication on the GPU in the CUDA guide:
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
1) Prepare the environment and necessary GPU resources for matrix multiplication.
2) Implement the corresponding GLSL shader code in matrixmult.comp.
3) Optional: Compare against a version that does not exploit shared memory. Can
you see a clear difference? HINT: You might want to start with this simpler 
version anyway before trying matrix multiplication with shared memory. Also:
if you want fast performance, matrix memory should probably be device-local!
*/