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

#ifndef _RNG_TEST_STATISTICS_CHECK_TEST_HPP__
#define _RNG_TEST_STATISTICS_CHECK_TEST_HPP__

#include <cstdint>
#include <iostream>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math.hpp"

#include "statistics_check.hpp"

#define UNIFORM_ARGS_FLOAT  -1.0f, 5.0f
#define UNIFORM_ARGS_DOUBLE -1.0, 5.0
#define UNIFORM_ARGS_INT    -1, 5

#define GAUSSIAN_ARGS_FLOAT  -1.0f, 5.0f
#define GAUSSIAN_ARGS_DOUBLE -1.0, 5.0

#define LOGNORMAL_ARGS_FLOAT  -1.0f, 5.0f, 1.0f, 2.0f
#define LOGNORMAL_ARGS_DOUBLE -1.0, 5.0, 1.0, 2.0

#define BERNOULLI_ARGS 0.5f

#define POISSON_ARGS 0.5

template <typename Distr, typename Engine>
class statistics_test {
public:
    template <typename Queue, typename... Args>
    void operator()(Queue queue, std::int64_t n_gen, Args... args) {
        using Type = typename Distr::result_type;

        std::vector<Type> r(n_gen);

        try {
            sycl::buffer<Type, 1> r_buffer(r.data(), r.size());

            Engine engine(queue, SEED);
            Distr distr(args...);
            oneapi::math::rng::generate(distr, engine, n_gen, r_buffer);
            QUEUE_WAIT(queue);
        }
        catch (sycl::exception const& e) {
            std::cout << "Caught synchronous SYCL exception during generation:\n"
                      << e.what() << std::endl;
            print_error_code(e);
        }
        catch (const oneapi::math::unimplemented& e) {
            status = test_skipped;
            return;
        }
        catch (const std::runtime_error& error) {
            std::cout << "Error raised during execution:\n" << error.what() << std::endl;
        }

        status = statistics<Distr>{}.check(r, Distr{ args... });
    }

    int status = test_passed;
};

template <typename Distr, typename Engine>
class statistics_usm_test {
public:
    template <typename Queue, typename... Args>
    void operator()(Queue queue, std::int64_t n_gen, Args... args) {
        using Type = typename Distr::result_type;

#ifdef CALL_RT_API
        auto ua = sycl::usm_allocator<Type, sycl::usm::alloc::shared, 64>(queue);
#else
        auto ua = sycl::usm_allocator<Type, sycl::usm::alloc::shared, 64>(queue.get_queue());
#endif
        std::vector<Type, decltype(ua)> r(n_gen, ua);

        try {
            Engine engine(queue, SEED);
            Distr distr(args...);
            auto event = oneapi::math::rng::generate(distr, engine, n_gen, r.data());
            event.wait_and_throw();
        }
        catch (sycl::exception const& e) {
            std::cout << "Caught synchronous SYCL exception during generation:\n"
                      << e.what() << std::endl;
            print_error_code(e);
        }
        catch (const oneapi::math::unimplemented& e) {
            status = test_skipped;
            return;
        }
        catch (const std::runtime_error& error) {
            std::cout << "Error raised during execution:\n" << error.what() << std::endl;
        }

        status = statistics<Distr>{}.check(r, Distr{ args... });
    }

    int status = test_passed;
};

#endif // _RNG_TEST_STATISTICS_CHECK_TEST_HPP__
