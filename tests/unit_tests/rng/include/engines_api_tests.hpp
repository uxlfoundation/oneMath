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

#ifndef _RNG_ENGINES_API_TESTS_HPP__
#define _RNG_ENGINES_API_TESTS_HPP__

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
class engines_constructors_test {
public:
    template <typename Queue, typename... Args>
    void operator()(Queue queue, Args... args) {
        // Prepare arrays for random numbers
        std::vector<std::uint32_t> r1(N_GEN);
        std::vector<std::uint32_t> r2(N_GEN);
        std::vector<std::uint32_t> r3(N_GEN);
        std::vector<std::uint32_t> r4(N_GEN);

        try {
            // Initialize rng objects
            Engine engine1(queue, SEED);
            Engine engine2(queue, args...);
            Engine engine3(engine1);
            Engine engine4 = std::move(Engine(queue, SEED));

            oneapi::math::rng::bits<std::uint32_t> distr;

            sycl::buffer<std::uint32_t, 1> r1_buffer(r1.data(), r1.size());
            sycl::buffer<std::uint32_t, 1> r2_buffer(r2.data(), r2.size());
            sycl::buffer<std::uint32_t, 1> r3_buffer(r3.data(), r3.size());
            sycl::buffer<std::uint32_t, 1> r4_buffer(r4.data(), r4.size());

            oneapi::math::rng::generate(distr, engine1, N_GEN, r1_buffer);
            oneapi::math::rng::generate(distr, engine2, N_GEN, r2_buffer);
            oneapi::math::rng::generate(distr, engine3, N_GEN, r3_buffer);
            oneapi::math::rng::generate(distr, engine4, N_GEN, r4_buffer);
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
        status = (check_equal_vector(r1, r2) && check_equal_vector(r1, r3) &&
                  check_equal_vector(r1, r4));
    }

    int status = test_passed;
};

template <typename Engine>
class engines_copy_test {
public:
    template <typename Queue>
    void operator()(Queue queue) {
        // Prepare arrays for random numbers
        std::vector<std::uint32_t> r1(N_GEN);
        std::vector<std::uint32_t> r2(N_GEN);
        std::vector<std::uint32_t> r3(N_GEN);

        try {
            // Initialize rng objects
            Engine engine1(queue, SEED);
            Engine engine2(engine1);

            oneapi::math::rng::bits<std::uint32_t> distr;
            {
                sycl::buffer<std::uint32_t, 1> r1_buffer(r1.data(), r1.size());
                sycl::buffer<std::uint32_t, 1> r2_buffer(r2.data(), r2.size());

                oneapi::math::rng::generate(distr, engine1, N_GEN, r1_buffer);
                oneapi::math::rng::generate(distr, engine2, N_GEN, r2_buffer);
            }

            Engine engine3 = engine1;
            Engine engine4 = std::move(engine2);
            {
                sycl::buffer<std::uint32_t, 1> r1_buffer(r1.data(), r1.size());
                sycl::buffer<std::uint32_t, 1> r2_buffer(r2.data(), r2.size());
                sycl::buffer<std::uint32_t, 1> r3_buffer(r3.data(), r3.size());

                oneapi::math::rng::generate(distr, engine1, N_GEN, r1_buffer);
                oneapi::math::rng::generate(distr, engine3, N_GEN, r2_buffer);
                oneapi::math::rng::generate(distr, engine4, N_GEN, r3_buffer);
            }
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

        // Validation
        status = (check_equal_vector(r1, r2) && check_equal_vector(r1, r3));
    }

    int status = test_passed;
};

#endif // _RNG_ENGINES_API_TESTS_HPP__
