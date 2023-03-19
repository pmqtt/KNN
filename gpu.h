//
// Created by cefour on 17.03.23.
//

#ifndef KNN_GPU_H
#define KNN_GPU_H
#include <CL/opencl.h>
#include <string>
#include <vector>

void init();

auto write_memory(float *A,std::size_t length) -> cl_mem;

auto create_kernel_from_file(const std::string & filename, const std::string kernelname) -> cl_kernel;

auto create_memory(float *A,std::size_t length) -> cl_mem;

void cleanup(cl_kernel kernel, std::vector<cl_mem> &memory );

auto get_command_queue()->cl_command_queue;

const char *getErrorString(cl_int error);

#endif //KNN_GPU_H
