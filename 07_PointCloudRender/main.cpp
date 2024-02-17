#include <vulkan/vulkan.hpp>
#include <framework.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>
#include <glm/glm.hpp>
#include <glm/matrix.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

int main(int argc, char* argv[])
{
	try
	{
		if (argc < 2)
		{
			std::cout << "No image point cloud file name provided!" << std::endl;
			return -1;
		}

		vk::UniqueInstance instance;
		vk::PhysicalDevice physicalDevice;
		vk::UniqueDevice device;
		vk::UniqueCommandPool commandPool;
		vk::Queue queue;

		Framework::setupBasicCompute("Task 9", VK_API_VERSION_1_3, {}, {}, instance, physicalDevice, device, queue, commandPool);

		// TODO: This is the point cloud struct that you hold on the CPU. The method below
		// will read a file and store its contents into the variables named "position" and
		// "color". Afterwards, it is copied to the GPU and read in the shader pointcloud.comp. 
		// The shader defines its own version of this struct, also with two 3D vectors.
		// However, in its current form, this will most likely not work! Take a look at 
		// the shader. Where is the discrepancy? Find a solution so the GPU can easily read
		// the data you provide in your struct on the GPU. HINT: It will be helpful to consult
		// the GLSL specification of the std430 layout that the shader code expects. HINT:
		// To debug your program, just start a single block with a single thread and let it
		// fetch and print a few points. Compare with outputs on the CPU. This is an easy way
		// to make sure they match.
		// =====================================================================================
		// Section 7.6.2.2 (https://registry.khronos.org/OpenGL/specs/gl/glspec46.core.pdf) and (https://www.reddit.com/r/opengl/comments/12inqn3/how_does_glsl_alignment_work/jfvlhvc/)
		// 3-component vectors in both std430 and std140 are aligned in the same manner as vec4 (i.e. to 16 bytes)
		struct Point {
			alignas(16) glm::vec3 position;
			alignas(16) glm::vec3 color;
		};

		// TODO: Same as above, the shader has an equivalent of the below struct. There is
		// a little twist: the .width and .height are packed into a vector resolution[2] in
		// the shader code. They will be stored in a uniform buffer, which does not have the
		// std430 layout, but instead uses std140. What do you need to change to make sure 
		// the CPU data can be accessed without problems on the GPU? HINT: for an "elegant"
		// solution, you can check out the "alignas" C++ keyword.
		// ===================================================================================
		// In std140, arrays' alignments are rounded up to vec4 alignment requirements
		// In this example, the combined .resolution member will take up 16 bytes hence we align the following member to a 16 byte boundary
		struct Parameters {
			alignas(16) uint32_t width;
			alignas(16) uint32_t height;
			alignas(16) uint32_t numPoints;
			alignas(16) glm::mat4 mvp;
		};

		glm::vec3 minimum, maximum;
		std::vector<Point> pointcloud;
		Framework::readPointCloud(argv[1], pointcloud, minimum, maximum);

		constexpr uint32_t width = 800, height = 800;
		const size_t pointsSize 		= pointcloud.size() * sizeof(Point);
		constexpr size_t imageSize		= width * height * sizeof(uint32_t); 
		const uint32_t workGroupSize	= (static_cast<uint32_t>(pointcloud.size()) / 128U) + 1U; // Local groups are of size 128

		VmaAllocator allocator = Framework::createAllocator(*instance, VK_API_VERSION_1_3, physicalDevice, *device);

		// TODO: Description for the resources that go into our pipeline (and descriptor set):
		// - One uniform buffer for the rendering parameters (size of Parameters struct)
		// - One storage buffer for the point cloud (pointcloud.size() * size of Point struct)
		// - One storage buffer for the output image (width * height * size of uint32).
		// - All of them should be host-visible, host-coherent and device-local.
		// Allocate needed buffers
		VkBuffer paramsBuff, pointsBuff, imageBuff;
		VkBufferCreateInfo paramsBuffCreateInfo	= vk::BufferCreateInfo({}, sizeof(Parameters), vk::BufferUsageFlagBits::eUniformBuffer);
		VkBufferCreateInfo pointsBuffCreateInfo	= vk::BufferCreateInfo({}, pointsSize, vk::BufferUsageFlagBits::eStorageBuffer);
		VkBufferCreateInfo imageBuffCreateInfo	= vk::BufferCreateInfo({}, imageSize, vk::BufferUsageFlagBits::eStorageBuffer);
		VmaAllocationCreateInfo paramsBuffAllocInfo = {}, pointsBuffAllocInfo = {}, imageBuffAllocInfo = {};
		paramsBuffAllocInfo.usage = pointsBuffAllocInfo.usage	= VMA_MEMORY_USAGE_AUTO;
		auto memReqFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		paramsBuffAllocInfo.requiredFlags = pointsBuffAllocInfo.requiredFlags = imageBuffAllocInfo.requiredFlags = memReqFlags;
		paramsBuffAllocInfo.flags = pointsBuffAllocInfo.flags = imageBuffAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT; // This allows memory to be mappable in random order when when usage is set to VMA_MEMORY_USAGE_AUTO
		VmaAllocation paramsAlloc, pointsAlloc, imageAlloc;
		vmaCreateBuffer(allocator, &paramsBuffCreateInfo, &paramsBuffAllocInfo, &paramsBuff, &paramsAlloc, nullptr);
		vmaCreateBuffer(allocator, &pointsBuffCreateInfo, &pointsBuffAllocInfo, &pointsBuff, &pointsAlloc, nullptr);
		vmaCreateBuffer(allocator, &imageBuffCreateInfo, &imageBuffAllocInfo, &imageBuff, &imageAlloc, nullptr);

		// TODO: As before, take all the necessary steps to have a descriptor set that references
		// the above created buffers. Make sure the bindings match the shader "primes.comp".
		// Define descriptor pool to provide two needed bindings
		std::vector<vk::DescriptorPoolSize> poolSizes	= { {vk::DescriptorType::eUniformBuffer, 1U},
															{vk::DescriptorType::eStorageBuffer, 2U}};
		vk::DescriptorPoolCreateInfo poolCreateInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1U, poolSizes);
		vk::UniqueDescriptorPool descriptorPool 		= device->createDescriptorPoolUnique(poolCreateInfo);
		// Define bindings for three needed buffers
		vk::DescriptorSetLayoutBinding paramsBinding(0U, vk::DescriptorType::eUniformBuffer, 1U, vk::ShaderStageFlagBits::eCompute);
		vk::DescriptorSetLayoutBinding pointsBinding(1U, vk::DescriptorType::eStorageBuffer, 1U, vk::ShaderStageFlagBits::eCompute);
		vk::DescriptorSetLayoutBinding imageBinding(2U, vk::DescriptorType::eStorageBuffer, 1U, vk::ShaderStageFlagBits::eCompute);
		std::array<vk::DescriptorSetLayoutBinding, 3UL> allBindings	= { paramsBinding, pointsBinding, imageBinding };
		vk::DescriptorSetLayoutCreateInfo descLayoutCreateInfo({}, allBindings);
		vk::UniqueDescriptorSetLayout descriptorSetLayout = device->createDescriptorSetLayoutUnique(descLayoutCreateInfo);
		// Create descriptor set
		vk::DescriptorSetAllocateInfo setAllocateInfo(*descriptorPool, *descriptorSetLayout);
		std::vector<vk::UniqueDescriptorSet> descriptorSets	= device->allocateDescriptorSetsUnique(setAllocateInfo);
		vk::UniqueDescriptorSet& descriptorSet 				= descriptorSets[0];
		// Connect bindings in descriptor set to actual buffers
		vk::DescriptorBufferInfo paramsBuffDescriptorInfo(paramsBuff, 0UL, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo pointsBuffDescriptorInfo(pointsBuff, 0UL, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo imageBuffDescriptorInfo(imageBuff, 0UL, VK_WHOLE_SIZE);
		vk::WriteDescriptorSet paramsDescriptorSetWrite(*descriptorSet, 0U, 0U, 1U, vk::DescriptorType::eUniformBuffer, {}, &paramsBuffDescriptorInfo);
		vk::WriteDescriptorSet pointsDescriptorSetWrite(*descriptorSet, 1U, 0U, 1U, vk::DescriptorType::eStorageBuffer, {}, &pointsBuffDescriptorInfo);
		vk::WriteDescriptorSet imageDescriptorSetWrite(*descriptorSet, 2U, 0U, 1U, vk::DescriptorType::eStorageBuffer, {}, &imageBuffDescriptorInfo);
		device->updateDescriptorSets({paramsDescriptorSetWrite, pointsDescriptorSetWrite, imageDescriptorSetWrite}, {});

		constexpr float fov	= 45.0f;
		glm::vec3 center	= (minimum + maximum) * 0.5f;
		float span 			= glm::length(maximum - minimum);
		auto view 			= glm::lookAt(center + glm::vec3(-0.9f * span, 0, 0.15f * span), center, glm::vec3(0, 0, -1));
		auto projection 	= glm::perspective(fov, ((float)width) / height, 0.1f, 2 * span);
		auto mvp			= projection * view; // No model transformation (i.e. identity model matrix)

		// - Set rendering params 
		// - Write point cloud data to GPU memory
		// - Clear image to 0xFFFFFFFF
		Parameters* paramsMapped; glm::vec4* pointsMapped; uint32_t* imageMapped;
		vmaMapMemory(allocator, paramsAlloc, (void**) &paramsMapped);
		vmaMapMemory(allocator, pointsAlloc, (void**) &pointsMapped);
		vmaMapMemory(allocator, imageAlloc, (void**) &imageMapped);
		paramsMapped->width 	= width;
		paramsMapped->height 	= height;
		paramsMapped->numPoints	= (uint32_t) pointcloud.size();
		paramsMapped->mvp 		= mvp;
		std::memcpy(pointsMapped, pointcloud.data(), pointcloud.size() * sizeof(Point));
		std::memset(imageMapped, 0xFF, sizeof(uint32_t) * width * height);
		vmaUnmapMemory(allocator, paramsAlloc);
		vmaUnmapMemory(allocator, pointsAlloc);
		vmaUnmapMemory(allocator, imageAlloc);
		vmaFlushAllocation(allocator, paramsAlloc, 0U, VK_WHOLE_SIZE);
		vmaFlushAllocation(allocator, pointsAlloc, 0U, VK_WHOLE_SIZE);
		vmaFlushAllocation(allocator, imageAlloc, 0U, VK_WHOLE_SIZE);

		vk::UniqueShaderModule shaderModule;
		vk::UniquePipelineLayout layout;
		vk::UniquePipelineCache cache;
		vk::UniquePipeline pipeline;

		Framework::setupComputePipeline("pointcloud.comp.spv", { *descriptorSetLayout }, *device, shaderModule, layout, cache, pipeline);

		vk::CommandBufferAllocateInfo allocateInfo(*commandPool, vk::CommandBufferLevel::ePrimary, 1);
		auto cmdBuffers = device->allocateCommandBuffersUnique(allocateInfo);

		vk::MemoryBarrier memoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead);
		cmdBuffers[0]->begin(vk::CommandBufferBeginInfo{});
		cmdBuffers[0]->bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
		cmdBuffers[0]->bindDescriptorSets(vk::PipelineBindPoint::eCompute, *layout, 0, *descriptorSet, {});

		// TODO: Pick the number of work groups we need to run. The group size should be 128.
		// Make sure we have enough work groups so there are at least as many threads as there
		// are points (pointcloud.size()).
		cmdBuffers[0]->dispatch(workGroupSize, 1, 1);
		cmdBuffers[0]->pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost, {}, memoryBarrier, {}, {});
		cmdBuffers[0]->end();

		Framework::initDebugging();
		Framework::beginCapture();
		queue.submit(vk::SubmitInfo({}, {}, * cmdBuffers[0]));
		Framework::endCapture();

		device->waitIdle();

		// Save output image to file
		vmaMapMemory(allocator, imageAlloc, (void**) &imageMapped);
		std::vector<uint32_t> image(imageMapped, imageMapped + width * height);
		std::for_each(image.begin(), image.end(), [](uint32_t& rgbd) {
			rgbd = 0xFF000000 | ((rgbd & 0xF) << 4) | ((rgbd & 0xF0) << 8) | ((rgbd & 0xF00) << 12);
		});
		stbi_write_png("output.png", width, height, STBI_rgb_alpha, image.data(), 0);
		vmaUnmapMemory(allocator, imageAlloc);

		// Free allocated resources
		vmaDestroyBuffer(allocator, paramsBuff, paramsAlloc);
		vmaDestroyBuffer(allocator, pointsBuff, pointsAlloc);
		vmaDestroyBuffer(allocator, imageBuff, imageAlloc);
		vmaDestroyAllocator(allocator);

		std::cout	<< "Program finished, please check the image "
					<< (std::filesystem::current_path() / std::filesystem::path("output.png")).string() << std::endl;
	}
	catch (Framework::NotImplemented e)
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}

/*
==================================== Task 7 ====================================
0) Pass the path of an .obj file in the assets directory as a program argument.
1) Create your resources and descriptor sets.
2) Modify the C++ structs so that their contents can be read by the GPU.
3) Compute the number of required work groups to process one point per thread.
4) Complete the shader file pointcloud.comp to perform point cloud rendering.
5) Optional: What happens if you don't use atomics or add the depth to the color?
*/