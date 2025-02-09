#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cstddef>  // for size_t

// Extension definitions from CHIPBackendOpenCL.hh
#ifndef cl_ext_buffer_device_address
#define cl_ext_buffer_device_address 1
#define CL_DEVICE_PTR_EXT 0xff01
#define CL_MEM_DEVICE_ADDRESS_EXT (1ul << 31)
#define CL_MEM_DEVICE_PTR_EXT 0xff01
typedef cl_ulong cl_mem_device_address_EXT;
typedef cl_int(CL_API_CALL *clSetKernelArgDevicePointerEXT_fn)(
    cl_kernel kernel, cl_uint arg_index, cl_mem_device_address_EXT dev_addr);
#endif

// OpenCL kernel source that adds 1 to each element
const char* kernelSource = R"(
__kernel void increment(__global int* input) {
    int gid = get_global_id(0);
    input[gid] += 1;
}
)";

// Error checking helper function
void checkError(cl_int error, const char* operation) {
    if (error != CL_SUCCESS) {
        std::cerr << "Error during operation " << operation << ": " << error << std::endl;
        exit(1);
    }
}

// Add this helper function at the top level
cl_device_id selectDevice(cl_platform_id& platform) {
    cl_int err;
    
    // Get platforms
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    checkError(err, "getting number of platforms");

    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    checkError(err, "getting platform IDs");

    // For each platform, get and print devices
    for (cl_uint p = 0; p < num_platforms; p++) {
        platform = platforms[p];
        
        // Get number of devices
        cl_uint num_devices;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
        checkError(err, "getting number of devices");

        std::vector<cl_device_id> devices(num_devices);
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr);
        checkError(err, "getting device IDs");

        // Print device info
        std::cout << "Platform " << p << " has " << num_devices << " device(s):" << std::endl;
        for (cl_uint d = 0; d < num_devices; d++) {
            size_t valueSize;
            err = clGetDeviceInfo(devices[d], CL_DEVICE_NAME, 0, nullptr, &valueSize);
            checkError(err, "getting device name size");
            
            std::string deviceName(valueSize, '\0');
            err = clGetDeviceInfo(devices[d], CL_DEVICE_NAME, valueSize, &deviceName[0], nullptr);
            checkError(err, "getting device name");
            
            std::cout << "\t" << d << ": " << deviceName << std::endl;
        }
    }

    // Get user input for device selection
    unsigned int platformIndex, deviceIndex;
    std::cout << "Select platform index: ";
    std::cin >> platformIndex;
    std::cout << "Select device index: ";
    std::cin >> deviceIndex;

    if (platformIndex >= num_platforms) {
        std::cerr << "Invalid platform index" << std::endl;
        exit(1);
    }

    platform = platforms[platformIndex];
    cl_uint num_devices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
    checkError(err, "getting number of devices");

    if (deviceIndex >= num_devices) {
        std::cerr << "Invalid device index" << std::endl;
        exit(1);
    }

    std::vector<cl_device_id> devices(num_devices);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr);
    checkError(err, "getting device IDs");
    return devices[deviceIndex];
}

bool kernelTest(cl_platform_id platform, cl_device_id device) {
    cl_int err;
    // Remove device selection code and start with extension check
    size_t ext_size;
    err = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_size);
    checkError(err, "getting extension size");

    std::vector<char> extensions(ext_size);
    err = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, ext_size, extensions.data(), nullptr);
    checkError(err, "getting extensions");

    bool hasBufferDeviceAddress = 
        std::string(extensions.data()).find("cl_ext_buffer_device_address") != std::string::npos;

    if (!hasBufferDeviceAddress) {
        std::cout << "Warning: Device does not support cl_ext_buffer_device_address extension\n";
        std::cout << "Available extensions:\n" << extensions.data() << std::endl;
        return false;
    }
    std::cout << "Device supports cl_ext_buffer_device_address extension\n";

    // Get the extension function pointer
    clSetKernelArgDevicePointerEXT_fn clSetKernelArgDevicePointerEXT = 
        (clSetKernelArgDevicePointerEXT_fn)
        clGetExtensionFunctionAddressForPlatform(platform, "clSetKernelArgDevicePointerEXT");
    
    if (!clSetKernelArgDevicePointerEXT) {
        std::cout << "Failed to get clSetKernelArgDevicePointerEXT function pointer\n";
        return false;
    }

    // Create context
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    checkError(err, "creating context");

    // Create command queue
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "creating command queue");

    // Create and build the program
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
    checkError(err, "creating program");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        std::cerr << "Build error: " << buffer << std::endl;
        exit(1);
    }

    

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "increment", &err);
    checkError(err, "creating kernel");

    // Create a device buffer with CL_MEM_DEVICE_ADDRESS_EXT flag
    const size_t buffer_size = 1024 * sizeof(int);
    cl_mem device_buffer = clCreateBuffer(context, 
                                        CL_MEM_READ_WRITE | CL_MEM_DEVICE_ADDRESS_EXT,
                                        buffer_size, nullptr, &err);
    checkError(err, "creating device buffer");

    // Get device pointer
    cl_mem_device_address_EXT device_ptr;
    err = clGetMemObjectInfo(device_buffer, CL_MEM_DEVICE_PTR_EXT, 
                            sizeof(device_ptr), &device_ptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to get device pointer (Error: " << err << ")\n";
        return false;
    }
    std::cout << "Successfully obtained device pointer: 0x" << std::hex << device_ptr << std::dec << std::endl;

    // Initialize buffer with test data
    std::vector<int> host_data(1024, 1);  // Initialize with 1s
    err = clEnqueueWriteBuffer(queue, device_buffer, CL_TRUE, 0, buffer_size, 
                              host_data.data(), 0, nullptr, nullptr);
    checkError(err, "writing to buffer");

    // Set kernel argument using the device pointer
    err = clSetKernelArgDevicePointerEXT(kernel, 0, device_ptr);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to set kernel argument with device pointer (Error: " << err << ")\n";
        return false;
    }

    // Execute kernel
    size_t global_size = 1024;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, 
                                nullptr, 0, nullptr, nullptr);
    checkError(err, "enqueueing kernel");

    // Read back results
    std::vector<int> result(1024);
    err = clEnqueueReadBuffer(queue, device_buffer, CL_TRUE, 0, buffer_size, 
                             result.data(), 0, nullptr, nullptr);
    checkError(err, "reading results");

    // Verify results
    bool correct = true;
    for (int i = 0; i < 1024; i++) {
        if (result[i] != 2) {  // Should be 1 + 1
            correct = false;
            break;
        }
    }
    std::cout << "Computation " << (correct ? "successful" : "failed") << std::endl;

    // Clean up
    clReleaseMemObject(device_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return correct;
}

int main() {
    cl_platform_id platform;
    cl_device_id device = selectDevice(platform);

    std::cout << "\nRunning kernel test with device address extension...\n";
    if (!kernelTest(platform, device)) {
        std::cout << "Kernel test failed!\n";
        return 1;
    }
    
    std::cout << "All tests completed successfully\n";
    return 0;
}
