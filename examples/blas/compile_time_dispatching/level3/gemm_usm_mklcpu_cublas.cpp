/*******************************************************************************
* Copyright 2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: Apache-2.0
*******************************************************************************/

/*
*
*  Content:
*       This example demonstrates use of DPCPP API oneapi::math::blas::gemm
*       using unified shared memory to perform General Matrix-Matrix
*       Multiplication on a INTEL CPU SYCL device and an NVIDIA GPU SYCL device
*
*       C = alpha * op(A) * op(B) + beta * C
*
*       where op() is defined by one of oneapi::math::transpose::{nontrans,trans,conjtrans}
*
*
*       This example demonstrates only single precision (float) data type for
*       gemm matrix data
*
*
*******************************************************************************/

// stl includes
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

// oneMath/SYCL includes
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include "oneapi/math.hpp"

// local includes
#include "example_helper.hpp"

//
// Main example for Gemm consisting of
// initialization of A, B and C matrices as well as
// scalars alpha and beta.  Then the product
//
// C = alpha * op(A) * op(B) + beta * C
//
// is performed and finally the results are post processed.
//
void run_gemm_example(const sycl::device& cpu_dev, const sycl::device& gpu_dev) {
    //
    // Initialize data for Gemm
    //
    // C = alpha * op(A) * op(B)  + beta * C
    //
    oneapi::math::transpose transA = oneapi::math::transpose::trans;
    oneapi::math::transpose transB = oneapi::math::transpose::nontrans;

    // matrix data sizes
    int m = 45;
    int n = 98;
    int k = 67;

    // leading dimensions of data
    int ldA = 103;
    int ldB = 105;
    int ldC = 106;
    int sizea = (transA == oneapi::math::transpose::nontrans) ? ldA * k : ldA * m;
    int sizeb = (transB == oneapi::math::transpose::nontrans) ? ldB * n : ldB * k;
    int sizec = ldC * n;

    // set scalar fp values
    float alpha = set_fp_value(float(2.0), float(-0.5));
    float beta = set_fp_value(float(3.0), float(-1.5));

    // Catch asynchronous exceptions for CPU and GPU
    auto cpu_exception_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const& e) {
                std::cerr << "Caught asynchronous SYCL exception on CPU device during GEMM:"
                          << std::endl;
                std::cerr << "\t" << e.what() << std::endl;
            }
        }
        std::exit(2);
    };
    auto gpu_exception_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const& e) {
                std::cerr << "Caught asynchronous SYCL exception on GPU device during GEMM:"
                          << std::endl;
                std::cerr << "\t" << e.what() << std::endl;
            }
        }
        std::exit(2);
    };

    //
    // Data Preparation on host
    //
    std::vector<float> A(sizea);
    std::vector<float> B(sizeb);
    std::vector<float> C(sizec);
    std::vector<float> result_cpu(sizec);
    std::vector<float> result_gpu(sizec);
    std::fill(A.begin(), A.end(), 0);
    std::fill(B.begin(), B.end(), 0);
    std::fill(C.begin(), C.end(), 0);
    std::fill(result_cpu.begin(), result_cpu.end(), 0);
    std::fill(result_gpu.begin(), result_gpu.end(), 0);

    rand_matrix(A, transA, m, k, ldA);
    rand_matrix(B, transB, k, n, ldB);
    rand_matrix(C, oneapi::math::transpose::nontrans, m, n, ldC);

    //
    // Preparation on CPU
    //
    sycl::queue cpu_queue(cpu_dev, cpu_exception_handler);
    sycl::event cpu_gemm_done;
    sycl::context cpu_cxt = cpu_queue.get_context();

    // allocate on CPU device and copy data from host to SYCL CPU device
    float* cpu_A = sycl::malloc_device<float>(sizea * sizeof(float), cpu_queue);
    float* cpu_B = sycl::malloc_device<float>(sizeb * sizeof(float), cpu_queue);
    float* cpu_C = sycl::malloc_device<float>(sizec * sizeof(float), cpu_queue);
    if (!cpu_A || !cpu_B || !cpu_C) {
        throw std::runtime_error("Failed to allocate USM memory.");
    }
    cpu_queue.memcpy(cpu_A, A.data(), sizea * sizeof(float)).wait();
    cpu_queue.memcpy(cpu_B, B.data(), sizeb * sizeof(float)).wait();
    cpu_queue.memcpy(cpu_C, C.data(), sizec * sizeof(float)).wait();

    //
    // Preparation on GPU
    //
    sycl::queue gpu_queue(gpu_dev, gpu_exception_handler);
    sycl::event gpu_gemm_done;
    sycl::context gpu_cxt = gpu_queue.get_context();

    // allocate on GPU device and copy data from host to SYCL GPU device
    float* gpu_A = sycl::malloc_device<float>(sizea * sizeof(float), gpu_queue);
    float* gpu_B = sycl::malloc_device<float>(sizeb * sizeof(float), gpu_queue);
    float* gpu_C = sycl::malloc_device<float>(sizec * sizeof(float), gpu_queue);
    if (!gpu_A || !gpu_B || !gpu_C) {
        throw std::runtime_error("Failed to allocate USM memory.");
    }
    gpu_queue.memcpy(gpu_A, A.data(), sizea * sizeof(float)).wait();
    gpu_queue.memcpy(gpu_B, B.data(), sizeb * sizeof(float)).wait();
    gpu_queue.memcpy(gpu_C, C.data(), sizec * sizeof(float)).wait();

    //
    // Execute Gemm on CPU and GPU device
    //
    // add oneapi::math::blas::gemm to execution queue
    cpu_gemm_done = oneapi::math::blas::column_major::gemm(
        oneapi::math::backend_selector<oneapi::math::backend::mklcpu>{ cpu_queue }, transA, transB,
        m, n, k, alpha, cpu_A, ldA, cpu_B, ldB, beta, cpu_C, ldC);
    gpu_gemm_done = oneapi::math::blas::column_major::gemm(
        oneapi::math::backend_selector<oneapi::math::backend::cublas>{ gpu_queue }, transA, transB,
        m, n, k, alpha, gpu_A, ldA, gpu_B, ldB, beta, gpu_C, ldC);

    // Wait until calculations are done
    cpu_gemm_done.wait_and_throw();
    gpu_gemm_done.wait_and_throw();

    //
    // Post Processing
    //
    // copy data from CPU back to host
    cpu_queue.memcpy(result_cpu.data(), cpu_C, sizec * sizeof(float)).wait_and_throw();

    // copy data from GPU back to host
    gpu_queue.memcpy(result_gpu.data(), gpu_C, sizec * sizeof(float)).wait_and_throw();

    // print results
    std::cout << "\n\t\tGEMM parameters:" << std::endl;
    std::cout << "\t\t\ttransA = "
              << (transA == oneapi::math::transpose::nontrans
                      ? "nontrans"
                      : (transA == oneapi::math::transpose::trans ? "trans" : "conjtrans"))
              << ", transB = "
              << (transB == oneapi::math::transpose::nontrans
                      ? "nontrans"
                      : (transB == oneapi::math::transpose::trans ? "trans" : "conjtrans"))
              << std::endl;
    std::cout << "\t\t\tm = " << m << ", n = " << n << ", k = " << k << std::endl;
    std::cout << "\t\t\tlda = " << ldA << ", ldB = " << ldB << ", ldC = " << ldC << std::endl;
    std::cout << "\t\t\talpha = " << alpha << ", beta = " << beta << std::endl;

    std::cout << "\n\t\tOutputting 2x2 block of A,B,C matrices:" << std::endl;

    // output the top 2x2 block of A matrix
    print_2x2_matrix_values(A.data(), ldA, "A");

    // output the top 2x2 block of B matrix
    print_2x2_matrix_values(B.data(), ldB, "B");

    // output the top 2x2 block of C matrix from CPU
    print_2x2_matrix_values(result_cpu.data(), ldC, "(CPU) C");

    // output the top 2x2 block of C matrix from GPU
    print_2x2_matrix_values(result_gpu.data(), ldC, "(GPU) C");

    sycl::free(gpu_C, gpu_queue);
    sycl::free(gpu_B, gpu_queue);
    sycl::free(gpu_A, gpu_queue);
    sycl::free(cpu_C, cpu_queue);
    sycl::free(cpu_B, cpu_queue);
    sycl::free(cpu_A, cpu_queue);
}

//
// Description of example setup, apis used and supported floating point type precisions
//
void print_example_banner() {
    std::cout << "" << std::endl;
    std::cout << "########################################################################"
              << std::endl;
    std::cout << "# General Matrix-Matrix Multiplication using Unified Shared Memory Example: "
              << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# C = alpha * A * B + beta * C" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# where A, B and C are general dense matrices and alpha, beta are" << std::endl;
    std::cout << "# floating point type precision scalars." << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using apis:" << std::endl;
    std::cout << "#   gemm" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using single precision (float) data type" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Running on both Intel CPU and Nvidia GPU devices" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "########################################################################"
              << std::endl;
    std::cout << std::endl;
}

//
// Main entry point for example.
//
int main(int argc, char** argv) {
    print_example_banner();

    try {
        sycl::device cpu_dev((sycl::cpu_selector()));
        sycl::device gpu_dev((sycl::gpu_selector()));

        unsigned int vendor_id = gpu_dev.get_info<sycl::info::device::vendor_id>();
        if (vendor_id != NVIDIA_ID) {
            std::cerr << "FAILED: NVIDIA GPU device not found" << std::endl;
            return 1;
        }
        std::cout << "Running BLAS GEMM USM example" << std::endl;
        std::cout << "Running with single precision real data type on:" << std::endl;
        std::cout << "\tCPU device: " << cpu_dev.get_info<sycl::info::device::name>() << std::endl;
        std::cout << "\tGPU device: " << gpu_dev.get_info<sycl::info::device::name>() << std::endl;
        run_gemm_example(cpu_dev, gpu_dev);
        std::cout << "BLAS GEMM USM example ran OK on MKLCPU and CUBLAS" << std::endl;
    }
    catch (sycl::exception const& e) {
        std::cerr << "Caught synchronous SYCL exception during GEMM:" << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
        std::cerr << "\tSYCL error code: " << e.code().value() << std::endl;
        return 1;
    }
    catch (std::exception const& e) {
        std::cerr << "Caught std::exception during GEMM:";
        std::cerr << "\t" << e.what() << std::endl;
        return 1;
    }
    return 0;
}
