/*******************************************************************************
* Copyright 2023 Intel Corporation
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
*       oneapi::math::rng::device:: engines skip_ahead and skip_ahead_ex tests
*       (SYCL interface)
*
*******************************************************************************/

#ifndef _RNG_DEVICE_SKIP_AHEAD_TEST_HPP__
#define _RNG_DEVICE_SKIP_AHEAD_TEST_HPP__

#include <cstdint>
#include <iostream>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math/rng/device.hpp"

#include "rng_device_test_common.hpp"

template <typename Engine>
class skip_ahead_test {
public:
    template <typename Queue>
    void operator()(Queue queue) {
        using UIntType = std::conditional_t<is_mcg59<Engine>::value, std::uint64_t, std::uint32_t>;

        std::vector<UIntType> r(N_GEN);
        std::vector<UIntType> r_ref(N_GEN);

        try {
            sycl::range<1> range(N_GEN / Engine::vec_size);

            sycl::buffer<UIntType> buf(r);
            auto event = queue.submit([&](sycl::handler& cgh) {
                sycl::accessor acc(buf, cgh, sycl::write_only);
                cgh.parallel_for(range, [=](sycl::item<1> item) {
                    size_t id = item.get_id(0);
                    Engine engine(SEED);
                    oneapi::math::rng::device::skip_ahead(engine, id * Engine::vec_size);
                    oneapi::math::rng::device::bits<UIntType> distr;
                    auto res = oneapi::math::rng::device::generate(distr, engine);
                    if constexpr (Engine::vec_size == 1) {
                        acc[id] = res;
                    }
                    else {
                        res.store(id, get_multi_ptr(acc));
                    }
                });
            });
            event.wait_and_throw();
        }
        catch (const oneapi::math::unimplemented& e) {
            status = test_skipped;
            return;
        }
        catch (sycl::exception const& e) {
            std::cout << "SYCL exception during generation" << std::endl
                      << e.what() << std::endl
                      << "Error code: " << get_error_code(e) << std::endl;
            status = test_failed;
            return;
        }

        // validation
        Engine engine(SEED);
        oneapi::math::rng::device::bits<UIntType> distr;
        for (int i = 0; i < N_GEN; i += Engine::vec_size) {
            auto res = oneapi::math::rng::device::generate(distr, engine);
            if constexpr (Engine::vec_size == 1) {
                r_ref[i] = res;
            }
            else {
                for (int j = 0; j < Engine::vec_size; ++j) {
                    r_ref[i + j] = res[j];
                }
            }
        }

        status = check_equal_vector_device(r, r_ref);
    }

    int status = test_passed;
};

template <class Engine>
class skip_ahead_ex_test {
public:
    template <typename Queue>
    void operator()(Queue queue) {
        std::vector<std::uint32_t> r(N_GEN);
        std::vector<std::uint32_t> r_ref(N_GEN);

        try {
            sycl::range<1> range(N_GEN / Engine::vec_size);

            sycl::buffer<std::uint32_t, 1> buf(r);
            std::uint64_t skip_num = (std::uint64_t)pow(2, 12);
            auto event = queue.submit([&](sycl::handler& cgh) {
                sycl::accessor acc(buf, cgh, sycl::write_only);
                cgh.parallel_for(range, [=](sycl::item<1> item) {
                    size_t id = item.get_id(0);
                    Engine engine(SEED);
                    oneapi::math::rng::device::skip_ahead(engine,
                                                          { id * Engine::vec_size, skip_num });
                    oneapi::math::rng::device::bits<> distr;
                    auto res = oneapi::math::rng::device::generate(distr, engine);
                    if constexpr (Engine::vec_size == 1) {
                        acc[id] = res;
                    }
                    else {
                        res.store(id, get_multi_ptr(acc));
                    }
                });
            });
            event.wait_and_throw();
        }
        catch (const oneapi::math::unimplemented& e) {
            status = test_skipped;
            return;
        }
        catch (sycl::exception const& e) {
            std::cout << "SYCL exception during generation" << std::endl
                      << e.what() << std::endl
                      << "Error code: " << get_error_code(e) << std::endl;
            status = test_failed;
            return;
        }

        // validation
        Engine engine(SEED);
        for (int j = 0; j < SKIP_TIMES; j++) {
            oneapi::math::rng::device::skip_ahead(engine, N_SKIP);
        }
        oneapi::math::rng::device::bits<> distr;
        for (int i = 0; i < N_GEN; i += Engine::vec_size) {
            auto res = oneapi::math::rng::device::generate(distr, engine);
            if constexpr (Engine::vec_size == 1) {
                r_ref[i] = res;
            }
            else {
                for (int j = 0; j < Engine::vec_size; ++j) {
                    r_ref[i + j] = res[j];
                }
            }
        }

        status = check_equal_vector_device(r, r_ref);
    }

    int status = test_passed;
};

#endif // _RNG_DEVICE_SKIP_AHEAD_TEST_HPP__
