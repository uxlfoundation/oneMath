/*******************************************************************************
* Copyright 2025 Intel Corporation
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
*       oneapi::math::rng::device::count_engine_adaptor:: engines tests
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
class count_adaptor_test {
public:
    template <typename Queue>
    void operator()(Queue queue) {
        using UIntType = std::conditional_t<is_mcg59<Engine>::value, std::uint64_t, std::uint32_t>;

        using allocator_t = sycl::usm_allocator<UIntType, sycl::usm::alloc::shared>;
        allocator_t allocator(queue);

        std::vector<UIntType> r_ref(N_GEN);
        std::vector<UIntType, allocator_t> r_vec(N_GEN, allocator);
        std::vector<UIntType, allocator_t> r_count_vec(1, allocator);
        UIntType* r = r_vec.data();
        UIntType* r_count = r_count_vec.data();
        const int count_idx = std::min(N_GEN / Engine::vec_size - 1, 42);

        try {
            sycl::range<1> range(N_GEN / Engine::vec_size);

            auto event = queue.parallel_for(range, [=](sycl::item<1> item) {
                size_t id = item.get_id(0);
                oneapi::math::rng::device::count_engine_adaptor<Engine> engine(
                    SEED, 2 * id * Engine::vec_size);
                oneapi::math::rng::device::bits<UIntType> distr;
                auto res = oneapi::math::rng::device::generate(distr, engine);

                Engine base_engine = engine.base();
                res = oneapi::math::rng::device::generate(distr, base_engine);
                if constexpr (Engine::vec_size == 1) {
                    r[id] = res;
                }
                else {
                    for (int j = 0; j < Engine::vec_size; ++j) {
                        r[id * Engine::vec_size + j] = res[j];
                    }
                }

                if (id == count_idx)
                    r_count[0] = engine.get_count();
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
        if (r_count[0] != Engine::vec_size) {
            std::cout << "Error: count = " << r_count[0] << ", but should be " << Engine::vec_size
                      << std::endl;
            status = test_failed;
            return;
        }

        Engine engine(SEED);
        oneapi::math::rng::device::bits<UIntType> distr;
        for (int i = 0; i < N_GEN; i += Engine::vec_size) {
            auto res = oneapi::math::rng::device::generate(distr, engine);
            res = oneapi::math::rng::device::generate(distr, engine);
            if constexpr (Engine::vec_size == 1) {
                r_ref[i] = res;
            }
            else {
                for (int j = 0; j < Engine::vec_size; ++j) {
                    r_ref[i + j] = res[j];
                }
            }
        }

        status = check_equal_vector_device(r, r_ref.data(), N_GEN);
    }

    int status = test_passed;
};

#endif // _RNG_DEVICE_SKIP_AHEAD_TEST_HPP__
