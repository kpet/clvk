cmake_minimum_required(VERSION 3.10)
project(opencl_device_ptr_test)
# Enable debug build
set(CMAKE_BUILD_TYPE Debug)

# Add debug flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")



# Find OpenCL package
find_package(OpenCL REQUIRED)

# Add executable
add_executable(device_ptr_test main.cpp)

# Link OpenCL
target_link_libraries(device_ptr_test OpenCL::OpenCL)
