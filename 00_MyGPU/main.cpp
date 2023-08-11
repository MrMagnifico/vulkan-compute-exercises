#include <vulkan/vulkan.hpp>

#include <format>
#include <iostream>

std::string apiHumanReadable(uint32_t versionNum) {
	// Shift to position of each version part and mask out everything to the left of that
	uint32_t major	= (versionNum >> 22U) & 0x7F;
	uint32_t minor	= (versionNum >> 12U) & 0x3FF;
	uint32_t patch	= versionNum & 0xFFF;

	// Construct final string
	return std::format("{}.{}.{}", major, minor, patch);
}

int main() {
	// Create Vulkan instance
	const vk::ApplicationInfo applicationInfo("Task 0", 1, "None", 1, VK_MAKE_API_VERSION(0, 1, 3, 0), nullptr);
	const auto instance = vk::createInstanceUnique(vk::InstanceCreateInfo({}, &applicationInfo));

	// Query physical devices
	auto physicalDevices = instance->enumeratePhysicalDevices();
	std::cout << "Num physical devices: " << physicalDevices.size() << std::endl;
	for (const auto &device : physicalDevices) {
		const auto properties = device.getProperties();
		std::cout	<< "[DEV] " 			<< properties.deviceName					<< std::endl
					<< "API Version: "		<< apiHumanReadable(properties.apiVersion)	<< std::endl
					<< "Driver Version: "	<< properties.driverVersion 				<< std::endl
					<< "Vendor ID: "		<< properties.vendorID						<< std::endl;
	}

	return 0;
}

/*
==================================== Task 0 ====================================
1) Create a Vulkan instance. Give it an application name "Task 0", a version 
number 1, and an engine name of "None". The API version should be 1.3.
2) Query your physical devices. Retrieve their properties and print out 
- the total number of devices
- their names
- the latest API they support (human-readable! Might be a bit tricky!)
3) Physical devices have several types of property collections available. Out of 
all of them, find three properties that should be relevant to compute jobs and
print them out. 
4) Make sure to clean up everything! Either by explicitly destroying your objects 
(in the right order) or by using the smart pointers of vulkan.hpp.
5) Optional: you can go and explore a bit, checking out the individual features
of your devices and queue families it provides. What kinds does it support? 
Check out an advanced feature using the pNext pointer of VkPhysicalDeviceFeatures2.
*/