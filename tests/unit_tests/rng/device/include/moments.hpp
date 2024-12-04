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
*       oneapi::math::rng::device:: distributions moments test (SYCL interface)
*
*******************************************************************************/

#ifndef _RNG_DEVICE_DISTR_MOMENTS_TEST_HPP_
#define _RNG_DEVICE_DISTR_MOMENTS_TEST_HPP_

#include <iostream>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math/rng/device.hpp"

#include "rng_device_test_common.hpp"

template <class Engine, class Distribution>
class moments_test {
public:
    template <typename Queue>
    void operator()(Queue queue) {
        // Note: the following methods of discrete distributions require double precision support
        if ((std::is_same_v<
                 Distribution,
                 oneapi::math::rng::device::uniform<
                     std::uint32_t, oneapi::math::rng::device::uniform_method::accurate>> ||
             std::is_same_v<
                 Distribution,
                 oneapi::math::rng::device::uniform<
                     std::int32_t, oneapi::math::rng::device::uniform_method::accurate>> ||
             std::is_same_v<
                 Distribution,
                 oneapi::math::rng::device::poisson<
                     std::uint32_t, oneapi::math::rng::device::poisson_method::devroye>> ||
             std::is_same_v<
                 Distribution,
                 oneapi::math::rng::device::poisson<
                     std::int32_t, oneapi::math::rng::device::poisson_method::devroye>>) &&
            !queue.get_device().has(sycl::aspect::fp64)) {
            status = test_skipped;
            return;
        }
        using Type = typename Distribution::result_type;
        // prepare array for random numbers
        std::vector<Type> r(N_GEN);

        try {
            sycl::range<1> range(N_GEN / Engine::vec_size);

            sycl::buffer<Type> buf(r);
            auto event = queue.submit([&](sycl::handler& cgh) {
                sycl::accessor acc(buf, cgh, sycl::write_only);
                cgh.parallel_for(range, [=](sycl::item<1> item) {
                    size_t id = item.get_id(0);
                    auto multiplier = Engine::vec_size;
                    if constexpr (std::is_same_v<Distribution,
                                                 oneapi::math::rng::device::uniform_bits<uint64_t>>)
                        multiplier *= 2;
                    Engine engine(SEED, id * multiplier);
                    Distribution distr;
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

        // validation (statistics check is turned out for mcg59)
        if constexpr (!std::is_same<Engine,
                                    oneapi::math::rng::device::mcg59<Engine::vec_size>>::value) {
            statistics_device<Distribution> stat;
            status = stat.check(r, Distribution{});
        }
        return;
    }

    int status = test_passed;
};

#endif // _RNG_DEVICE_DISTR_MOMENTS_TEST_HPP_
