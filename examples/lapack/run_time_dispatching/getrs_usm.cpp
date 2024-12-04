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
*       This example demonstrates use of oneapi::math::lapack::getrf and
*       oneapi::math::lapack::getrs to perform LU factorization and compute
*       the solution on a SYCL device (HOST, CPU, GPU) that is selected
*       during runtime.
*
*       This example demonstrates only single precision (float) data type
*       for matrix data
*
*******************************************************************************/

// STL includes
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
// Main example for LU consisting of initialization of
// a general dense A matrix.
// Then the LU factorization
// A = P * L * U
// is performed followed by solving a system of linear
// equations using the computed LU factorization, with
// multiple right-hand sides.
// Finally the results are post processed.
//

void run_getrs_example(const sycl::device& device) {
    // Matrix sizes and leading dimensions
    std::int64_t m = 23;
    std::int64_t n = 23;
    std::int64_t nrhs = 23;
    std::int64_t lda = 32;
    std::int64_t ldb = 32;
    std::int64_t A_size = n * lda;
    std::int64_t B_size = nrhs * ldb;
    std::int64_t ipiv_size = n;
    oneapi::math::transpose trans = oneapi::math::transpose::nontrans;

    // Asynchronous error handler
    auto error_handler = [&](sycl::exception_list exceptions) {
        for (auto const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (oneapi::math::lapack::exception const& e) {
                // Handle LAPACK related exceptions that happened during asynchronous call
                std::cerr << "Caught asynchronous LAPACK exception during GETRF or GETRS:"
                          << std::endl;
                std::cerr << "\t" << e.what() << std::endl;
                std::cerr << "\tinfo: " << e.info() << std::endl;
            }
            catch (sycl::exception const& e) {
                // Handle not LAPACK related exceptions that happened during asynchronous call
                std::cerr << "Caught asynchronous SYCL exception during GETRF or GETRS:"
                          << std::endl;
                std::cerr << "\t" << e.what() << std::endl;
            }
        }
        std::exit(2);
    };

    // Data preparation on host
    std::vector<float> A(A_size);
    std::vector<float> B(B_size);
    std::fill(A.begin(), A.end(), 0);
    std::fill(B.begin(), B.end(), 0);

    rand_matrix(A, trans, m, n, lda);
    rand_matrix(B, trans, n, nrhs, ldb);

    // Data preparation on selected device
    sycl::queue queue(device, error_handler);
    sycl::context context = queue.get_context();
    sycl::event getrf_done;
    sycl::event getrs_done;

    float* dev_A = sycl::malloc_device<float>(A_size * sizeof(float), queue);
    float* dev_B = sycl::malloc_device<float>(B_size * sizeof(float), queue);
    std::int64_t* dev_ipiv =
        sycl::malloc_device<std::int64_t>(ipiv_size * sizeof(std::int64_t), queue);

    std::int64_t getrf_scratchpad_size =
        oneapi::math::lapack::getrf_scratchpad_size<float>(queue, m, n, lda);
    std::int64_t getrs_scratchpad_size =
        oneapi::math::lapack::getrs_scratchpad_size<float>(queue, trans, n, nrhs, lda, ldb);
    float* getrf_scratchpad =
        sycl::malloc_shared<float>(getrf_scratchpad_size * sizeof(float), device, context);
    float* getrs_scratchpad =
        sycl::malloc_shared<float>(getrs_scratchpad_size * sizeof(float), device, context);
    if (!dev_A || !dev_B || !dev_ipiv) {
        throw std::runtime_error("Failed to allocate USM memory.");
    }
    // Skip checking getrf scratchpad memory allocation on rocsolver because with rocsolver
    // backend getrf does not use scrachpad memory
    if (device.is_cpu() || device.get_info<sycl::info::device::vendor_id>() != AMD_ID) {
        if (!getrf_scratchpad) {
            throw std::runtime_error("Failed to allocate USM memory.");
        }
    }
    // Skip checking getrs scratchpad memory allocation on cusolver/rocsolver because with
    // cusolver/rocsolver backend getrs does not use scrachpad memory
    if (device.is_cpu() || (device.get_info<sycl::info::device::vendor_id>() != NVIDIA_ID &&
                            device.get_info<sycl::info::device::vendor_id>() != AMD_ID)) {
        if (!getrs_scratchpad) {
            throw std::runtime_error("Failed to allocate USM memory.");
        }
    }

    // copy data from host to device
    queue.memcpy(dev_A, A.data(), A_size * sizeof(float)).wait();
    queue.memcpy(dev_B, B.data(), B_size * sizeof(float)).wait();

    // Execute on device
    getrf_done = oneapi::math::lapack::getrf(queue, m, n, dev_A, lda, dev_ipiv, getrf_scratchpad,
                                             getrf_scratchpad_size);
    getrs_done =
        oneapi::math::lapack::getrs(queue, trans, n, nrhs, dev_A, lda, dev_ipiv, dev_B, ldb,
                                    getrs_scratchpad, getrs_scratchpad_size, { getrf_done });

    // Wait until calculations are done
    queue.wait_and_throw();

    // Copy data from device back to host
    queue.memcpy(B.data(), dev_B, B_size * sizeof(float)).wait_and_throw();

    // Print results
    std::cout << "\n\t\tGETRF and GETRS parameters:" << std::endl;
    std::cout << "\t\t\ttrans = "
              << (trans == oneapi::math::transpose::nontrans
                      ? "nontrans"
                      : (trans == oneapi::math::transpose::trans ? "trans" : "conjtrans"))
              << std::endl;
    std::cout << "\t\t\tm = " << m << ", n = " << n << ", nrhs = " << nrhs << std::endl;
    std::cout << "\t\t\tlda = " << lda << ", ldb = " << ldb << std::endl;

    std::cout << "\n\t\tOutputting 2x2 block of A and X matrices:" << std::endl;
    // output the top 2x2 block of A matrix
    print_2x2_matrix_values(A.data(), lda, "A");

    // output the top 2x2 block of X matrix
    print_2x2_matrix_values(B.data(), ldb, "X");

    sycl::free(getrs_scratchpad, queue);
    sycl::free(getrf_scratchpad, queue);
    sycl::free(dev_ipiv, queue);
    sycl::free(dev_B, queue);
    sycl::free(dev_A, queue);
}

//
// Description of example setup, apis used and supported floating point type precisions
//

void print_example_banner() {
    std::cout << "" << std::endl;
    std::cout << "########################################################################"
              << std::endl;
    std::cout << "# LU Factorization and Solve Example: " << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Computes LU Factorization A = P * L * U" << std::endl;
    std::cout << "# and uses it to solve for X in a system of linear equations:" << std::endl;
    std::cout << "#   AX = B" << std::endl;
    std::cout << "# where A is a general dense matrix and B is a matrix whose columns" << std::endl;
    std::cout << "# are the right-hand sides for the systems of equations." << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using apis:" << std::endl;
    std::cout << "#   getrf and getrs" << std::endl;
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
// Main entry point for example.
//
int main(int argc, char** argv) {
    print_example_banner();

    try {
        sycl::device dev = sycl::device();
        if (dev.is_gpu()) {
            std::cout << "Running LAPACK getrs example on GPU device." << std::endl;
            std::cout << "Device name is: " << dev.get_info<sycl::info::device::name>()
                      << std::endl;
        }
        else {
            std::cout << "Running LAPACK getrs example on CPU device." << std::endl;
            std::cout << "Device name is: " << dev.get_info<sycl::info::device::name>()
                      << std::endl;
        }

        std::cout << "Running with single precision real data type:" << std::endl;
        run_getrs_example(dev);
        std::cout << "LAPACK GETRS USM example ran OK" << std::endl;
    }
    catch (oneapi::math::lapack::exception const& e) {
        // Handle LAPACK related exceptions that happened during synchronous call
        std::cerr << "Caught synchronous LAPACK exception:" << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
        std::cerr << "\tinfo: " << e.info() << std::endl;
        return 1;
    }
    catch (sycl::exception const& e) {
        // Handle not LAPACK related exceptions that happened during synchronous call
        std::cerr << "Caught synchronous SYCL exception:" << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
        std::cerr << "\tSYCL error code: " << e.code().value() << std::endl;
        return 1;
    }
    catch (std::exception const& e) {
        // Handle not SYCL related exceptions that happened during synchronous call
        std::cerr << "Caught synchronous std::exception:" << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
        return 1;
    }

    return 0;
}
