/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef _RNG_TEST_SKIP_AHEAD_TEST_HPP__
#define _RNG_TEST_SKIP_AHEAD_TEST_HPP__

#include <cstdint>
#include <iostream>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math.hpp"

#include "rng_test_common.hpp"

template <typename Engine>
class skip_ahead_test {
public:
    template <typename Queue>
    void operator()(Queue queue) {
        // Prepare arrays for random numbers
        std::vector<std::uint32_t> r1(N_GEN_SERVICE);
        std::vector<std::uint32_t> r2(N_GEN_SERVICE);

        try {
            // Initialize rng objects
            Engine engine(queue);
            std::vector<Engine*> engines;

            oneapi::math::rng::bits<std::uint32_t> distr;

            // Perform skip
            for (int i = 0; i < N_ENGINES; i++) {
                engines.push_back(new Engine(queue));
                oneapi::math::rng::skip_ahead(*(engines[i]), i * N_PORTION);
            }

            sycl::buffer<std::uint32_t, 1> r_buffer(r1.data(), r1.size());
            std::vector<sycl::buffer<std::uint32_t, 1>> r_buffers;
            for (int i = 0; i < N_ENGINES; i++) {
                r_buffers.push_back(
                    sycl::buffer<std::uint32_t, 1>(r2.data() + i * N_PORTION, N_PORTION));
            }

            oneapi::math::rng::generate(distr, engine, N_GEN_SERVICE, r_buffer);
            for (int i = 0; i < N_ENGINES; i++) {
                oneapi::math::rng::generate(distr, *(engines[i]), N_PORTION, r_buffers[i]);
            }
            QUEUE_WAIT(queue);

            // Clear memory
            for (int i = 0; i < N_ENGINES; i++) {
                delete engines[i];
            }
        }
        catch (const oneapi::math::unimplemented& e) {
            status = test_skipped;
            return;
        }
        catch (sycl::exception const& e) {
            std::cout << "SYCL exception during generation" << std::endl << e.what() << std::endl;
            print_error_code(e);
            status = test_failed;
            return;
        }

        // Validation
        status = check_equal_vector(r1, r2);
    }

    int status = test_passed;
};

template <typename Engine>
class skip_ahead_ex_test {
public:
    template <typename Queue>
    void operator()(Queue queue) {
        // Prepare arrays for random numbers
        std::vector<std::uint32_t> r1(N_GEN);
        std::vector<std::uint32_t> r2(N_GEN);

        try {
            // Initialize rng objects
            Engine engine1(queue);
            Engine engine2(queue);

            oneapi::math::rng::bits<std::uint32_t> distr;

            // Perform skip
            for (int j = 0; j < SKIP_TIMES; j++) {
                oneapi::math::rng::skip_ahead(engine1, N_SKIP);
            }
            oneapi::math::rng::skip_ahead(engine2, NUM_TO_SKIP);

            sycl::buffer<std::uint32_t, 1> r1_buffer(r1.data(), r1.size());
            sycl::buffer<std::uint32_t, 1> r2_buffer(r2.data(), r2.size());

            oneapi::math::rng::generate(distr, engine1, N_GEN, r1_buffer);
            oneapi::math::rng::generate(distr, engine2, N_GEN, r2_buffer);
            QUEUE_WAIT(queue);
        }
        catch (const oneapi::math::unimplemented& e) {
            status = test_skipped;
            return;
        }
        catch (sycl::exception const& e) {
            std::cout << "SYCL exception during generation" << std::endl << e.what() << std::endl;
            print_error_code(e);
            status = test_failed;
            return;
        }

        // validation
        status = check_equal_vector(r1, r2);
    }

    int status = test_passed;
};

#endif // _RNG_TEST_SKIP_AHEAD_TEST_HPP__
