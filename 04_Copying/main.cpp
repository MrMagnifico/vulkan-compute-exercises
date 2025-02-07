#include <vulkan/vulkan.hpp>
#include <framework.h>
#include <iostream>

int main()
{
	try
	{
		// Variables that will hold the objects we need to run Vulkan applications
		vk::UniqueInstance instance;
		vk::PhysicalDevice physicalDevice;
		vk::UniqueDevice device;
		vk::UniqueCommandPool commandPool;
		vk::Queue queue;

		Framework::setupBasicCompute("Task 4", VK_API_VERSION_1_3, {}, {}, instance, physicalDevice, device, queue, commandPool);
		
		size_t sizeA 					= sizeof(int) * 60, sizeX = sizeof(int) * 20;
		auto bufferUsage				= vk::BufferUsageFlagBits::eTransferDst	| vk::BufferUsageFlagBits::eTransferSrc;
		VkMemoryPropertyFlags propsA 	= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		VkMemoryPropertyFlags propsX	= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		
		// TODO: Let's use the VMA from here on out... to this end, go ahead and implement
		// createAllocator in framework.cpp.
		VmaAllocator allocator = Framework::createAllocator(*instance, VK_API_VERSION_1_3, physicalDevice, *device);

		// TODO: Create four buffers using the Vulkan memory allocator. Buffer A should be of size sizeA;
		// all others should be sizeX. All buffers should have usage "bufferUsage" (all of them should be 
		// able to be copied from and copied to. Each buffer also needs a separate allocation (read: memory). 
		// Buffer A's allocation should have required flags "propsA" (reachable from host, device-local), all 
		// others should have the required flags "propsX" (device-local). Note: you can easily convert vk::
		// structs to standard Vk structs, simply by assignment. E.g., you can do:
		// VkBufferCreateInfo info = vk::BufferCreateInfo({}, ... ), 
		// which lets you benefit from vulkan.hpp's sensible default constructor parameters. Isn't that neat?
		VkBuffer bufferA, bufferB, bufferC, bufferD;
		VkBufferCreateInfo aBufferCreateInfo = vk::BufferCreateInfo({}, sizeA, bufferUsage);
		VkBufferCreateInfo xBufferCreateInfo = vk::BufferCreateInfo({}, sizeX, bufferUsage);
		VmaAllocationCreateInfo aBufferAllocInfo = {}, xBufferAllocInfo = {};
		aBufferAllocInfo.usage = xBufferAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
		aBufferAllocInfo.requiredFlags = propsA;
		xBufferAllocInfo.requiredFlags = propsX;
		aBufferAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT; // This allows memory to be mappable when when usage is set to VMA_MEMORY_USAGE_AUTO
		VmaAllocation allocationA, allocationB, allocationC, allocationD;
		vmaCreateBuffer(allocator, &aBufferCreateInfo, &aBufferAllocInfo, &bufferA, &allocationA, nullptr);
		vmaCreateBuffer(allocator, &xBufferCreateInfo, &xBufferAllocInfo, &bufferB, &allocationB, nullptr);
		vmaCreateBuffer(allocator, &xBufferCreateInfo, &xBufferAllocInfo, &bufferC, &allocationC, nullptr);
		vmaCreateBuffer(allocator, &xBufferCreateInfo, &xBufferAllocInfo, &bufferD, &allocationD, nullptr);

		// The content of this array will be copied across our different buffers
		int magic[60] = { 
			1, 0, 11, 13101, 103, 22, 44, 2828, 7, 7, 0, 13, 21, 34, 55, 0, 77, 99, 12, 21,
			0, 1, 2, 3, 5, 6, 7, 8, 9, 4, 10, 11, 12, 13, 610, 987, 1597, 2584, 4181, 6765,
			89, 144, 233, 377, 22, 44, 10101, 30, 0, 0, 7, 7, 12, 144, 2568, 1, 303, 0, 2, 377
		};

		// TODO: Map the entire host-visible memory of buffer A and write all 60 magic numbers there
		// ==========================================================================================
		int* mappedA;
		vmaMapMemory(allocator, allocationA, (void**) &mappedA);
		memcpy(mappedA, magic, sizeA);
		vmaUnmapMemory(allocator, allocationA);
		vmaFlushAllocation(allocator, allocationA, 0ULL, sizeA);

		// Create our basic command buffer
		vk::CommandBufferAllocateInfo allocateInfo{ *commandPool, vk::CommandBufferLevel::ePrimary, 1 };
		auto cmdBuffers = device->allocateCommandBuffersUnique(allocateInfo);

		// Distribute the contents of buffer A to the different device-local buffers
		cmdBuffers[0]->begin(vk::CommandBufferBeginInfo{});
		cmdBuffers[0]->copyBuffer(bufferA, bufferB, vk::BufferCopy(0, 0, 20 * sizeof(int)));
		cmdBuffers[0]->copyBuffer(bufferA, bufferC, vk::BufferCopy(20 * sizeof(int), 0, 20 * sizeof(int)));
		cmdBuffers[0]->copyBuffer(bufferA, bufferD, vk::BufferCopy(40 * sizeof(int), 0, 20 * sizeof(int)));
		
		// TODO: We will need barriers to ensure that a copy becomes visible before the next
		// copy starts. Let's create a general memory barrier, which can enforce visibility
		// for certain accesses across ALL device memory. We will use this to enforce that 
		// the copies can be seen before we initiate any further copying: transfer write 
		// accesses (source) should become visible to all later transfer writes and transfer 
		// reads (destination). This basically says, written data must be visible when we read 
		// again, but also that writes must be published before we write again! We need the latter 
		// because the order of the copies matters for getting the correct result.
		vk::MemoryBarrier deviceMemoryBarrier(vk::AccessFlagBits::eTransferWrite,
											  vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite);

		// TODO: Prepare another general memory barrier. This one, we use to ensure that all copies
		// are available before we read buffer A from the host. The source should thus be transfer 
		// writes, and the destination host reads.
		vk::MemoryBarrier hostMemoryBarrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead);

		// TODO: Record the following commands to the command buffer in this exact order:
		// - Apply the first memory barrier. This ensures that the above copies can be seen by later copies.
		//   To apply it, you must invoke pipelineBarrier on the command buffer, describe the relevant stages 
		//   and which memory barrier to enforce. Pipeline barriers make sure that execution of one stage finishes 
		//   before another stage begins (one stage SYNCHRONIZES WITH another). Since all we do is copy, the only 
		//   stages we care about here are the transfer stages. With a pipeline barrier from transfer to transfer, 
		//   a copy will definitely finish executing before the next one starts. But we know that execution alone 
		//   is not everything, memory accesses must also become AVAILABLE, so later operations can see them. Since 
		//   making memory available is expensive, Vulkan asks you to provide explicit memory barriers to enforce
		//   availability/visibility as part of a pipeline barrier. 
		// - Copy 19 integers from buffer C to buffer A. Use an offset of 1 integer for both reading and writing.
		// - Apply the first memory barrier again. This ensures that copying the 19 integers above has finished.
		// - Copy 4 integers from buffer D to buffer A. Use offsets of 0 for reading and 10 integers for writing.
		// - Copy 5 integers from buffer C to buffer A. Use an offset of 15 integers for both reading and writing.
		// - Copy the integer at location 7 in buffer A to location 5 in the same buffer.
		// - Apply the first memory barrier again. This ensures that all copies are done before the final step.
		// - Copy 4 integers from buffer B to buffer A. Use offsets of 11 / 6 integers for reading / writing.
		// - Finally, apply the second memory barrier for reading buffer A from the host safely. In contrast to 
		//   the other pipeline barriers, now use the host stage as destination stage. 
		cmdBuffers[0]->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, deviceMemoryBarrier, {}, {});
		cmdBuffers[0]->copyBuffer(bufferC, bufferA, vk::BufferCopy(1 * sizeof(int), 1 * sizeof(int), 19 * sizeof(int)));
		cmdBuffers[0]->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, deviceMemoryBarrier, {}, {});
		cmdBuffers[0]->copyBuffer(bufferD, bufferA, vk::BufferCopy(0, 10 * sizeof(int), 4 * sizeof(int)));
		cmdBuffers[0]->copyBuffer(bufferC, bufferA, vk::BufferCopy(15 * sizeof(int), 15 * sizeof(int), 5 * sizeof(int)));
		cmdBuffers[0]->copyBuffer(bufferA, bufferA, vk::BufferCopy(7 * sizeof(int), 5 * sizeof(int), 1 * sizeof(int)));
		cmdBuffers[0]->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, deviceMemoryBarrier, {}, {});
		cmdBuffers[0]->copyBuffer(bufferB, bufferA, vk::BufferCopy(11 * sizeof(int), 6 * sizeof(int), 4 * sizeof(int)));
		cmdBuffers[0]->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, hostMemoryBarrier, {}, {});
		
		cmdBuffers[0]->end();

		Framework::initDebugging();
		Framework::beginCapture();
		queue.submit(vk::SubmitInfo({}, {}, *cmdBuffers[0]));
		Framework::endCapture();

		device->waitIdle();

		vmaMapMemory(allocator, allocationA, (void**) &mappedA);
		for (int i = 0; i < 20; i++) { std::cout << mappedA[i] << " "; }
		vmaUnmapMemory(allocator, allocationA);

		// TODO: Don't forget to clean up the VMA when you are done. This includes unmapping
		// all still mapped memory, destroying all buffers and eventually the allocator itself. 
		vmaDestroyBuffer(allocator, bufferA, allocationA);
		vmaDestroyBuffer(allocator, bufferB, allocationB);
		vmaDestroyBuffer(allocator, bufferC, allocationC);
		vmaDestroyBuffer(allocator, bufferD, allocationD);
		vmaDestroyAllocator(allocator);
	}
	catch (Framework::NotImplemented e)
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}

/*
==================================== Task 4 ====================================
1) Implement "createAllocator" in framework.cpp.
2) Each buffer must be created, find suitable memory and have it bound. Use VMA!
3) Create the required memory barriers in main.cpp.
4) Follow the outlined steps to retrieve the secret output.
5) The resulting sequence is rather interesting! Do you recognize it?
6) Optional: What happens if you DON'T use the barriers as described? Why?
*/
