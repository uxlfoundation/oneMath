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
*       Multiplication on a SYCL device (HOST, CPU, GPU) that is selected
*       during runtime.
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

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include "oneapi/math.hpp"

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
void run_gemm_example(const sycl::device& dev) {
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

    // Catch asynchronous exceptions
    auto exception_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const& e) {
                std::cerr << "Caught asynchronous SYCL exception during GEMM:" << std::endl;
                std::cerr << "\t" << e.what() << std::endl;
            }
        }
        std::exit(2);
    };

    // create execution queue
    sycl::queue main_queue(dev, exception_handler);
    sycl::event gemm_done;
    sycl::context cxt = main_queue.get_context();

    // allocate matrix on host
    std::vector<float> A(sizea);
    std::vector<float> B(sizeb);
    std::vector<float> C(sizec);
    std::fill(A.begin(), A.end(), 0);
    std::fill(B.begin(), B.end(), 0);
    std::fill(C.begin(), C.end(), 0);

    rand_matrix(A, transA, m, k, ldA);
    rand_matrix(B, transB, k, n, ldB);
    rand_matrix(C, oneapi::math::transpose::nontrans, m, n, ldC);

    // allocate memory on device
    auto dev_A = sycl::malloc_device<float>(sizea * sizeof(float), main_queue);
    auto dev_B = sycl::malloc_device<float>(sizeb * sizeof(float), main_queue);
    auto dev_C = sycl::malloc_device<float>(sizec * sizeof(float), main_queue);
    if (!dev_A || !dev_B || !dev_C) {
        throw std::runtime_error("Failed to allocate USM memory.");
    }

    // copy data from host to device
    main_queue.memcpy(dev_A, A.data(), sizea * sizeof(float)).wait();
    main_queue.memcpy(dev_B, B.data(), sizeb * sizeof(float)).wait();
    main_queue.memcpy(dev_C, C.data(), sizec * sizeof(float)).wait();

    //
    // Execute Gemm
    //
    // add oneapi::math::blas::gemm to execution queue
    gemm_done = oneapi::math::blas::column_major::gemm(main_queue, transA, transB, m, n, k, alpha,
                                                       dev_A, ldA, dev_B, ldB, beta, dev_C, ldC);

    // Wait until calculations are done
    main_queue.wait_and_throw();

    //
    // Post Processing
    //
    // copy data from device back to host
    main_queue.memcpy(C.data(), dev_C, sizec * sizeof(float)).wait_and_throw();

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

    // output the top 2x2 block of C matrix
    print_2x2_matrix_values(C.data(), ldC, "C");

    sycl::free(dev_C, main_queue);
    sycl::free(dev_B, main_queue);
    sycl::free(dev_A, main_queue);
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
    std::cout << "# Device will be selected during runtime." << std::endl;
    std::cout << "# The environment variable ONEAPI_DEVICE_SELECTOR can be used to specify"
              << std::endl;
    std::cout << "# available devices" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "########################################################################"
              << std::endl;
    std::cout << std::endl;
}

//
// Main entry point for example
//
int main(int argc, char** argv) {
    print_example_banner();

    try {
        sycl::device dev = sycl::device();

        if (dev.is_gpu()) {
            std::cout << "Running BLAS GEMM USM example on GPU device." << std::endl;
            std::cout << "Device name is: " << dev.get_info<sycl::info::device::name>()
                      << std::endl;
        }
        else {
            std::cout << "Running BLAS GEMM USM example on CPU device." << std::endl;
            std::cout << "Device name is: " << dev.get_info<sycl::info::device::name>()
                      << std::endl;
        }
        std::cout << "Running with single precision real data type:" << std::endl;

        run_gemm_example(dev);
        std::cout << "BLAS GEMM USM example ran OK." << std::endl;
    }
    catch (sycl::exception const& e) {
        std::cerr << "Caught synchronous SYCL exception during GEMM:" << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
        std::cerr << "\tSYCL error code: " << e.code().value() << std::endl;
        return 1;
    }
    catch (std::exception const& e) {
        std::cerr << "Caught std::exception during GEMM:" << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
        return 1;
    }

    return 0;
}
