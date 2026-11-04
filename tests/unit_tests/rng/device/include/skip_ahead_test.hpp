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
#include <type_traits>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math/rng/device.hpp"

#include "rng_device_test_common.hpp"

// engines supported by uniform_bits
template <typename Engine>
struct has_uniform_bits : std::false_type {};

template <std::int32_t VecSize>
struct has_uniform_bits<oneapi::math::rng::device::philox4x32x10<VecSize>> : std::true_type {};

template <std::int32_t VecSize>
struct has_uniform_bits<oneapi::math::rng::device::mcg59<VecSize>> : std::true_type {};

template <typename Engine>
class skip_ahead_test {
public:
    template <typename Queue>
    void operator()(Queue queue) {
        using UIntType = std::conditional_t<is_mcg59<Engine>::value, std::uint64_t, std::uint32_t>;

        // reference: stream from offset 0, one extra number for the engines started at offset 1
        std::vector<UIntType> r_ref(N_GEN + Engine::vec_size);
        fill_reference(oneapi::math::rng::device::bits<UIntType>{}, r_ref);

        // work-item id must produce r_ref[start + id * vec_size ...]:
        //   Engine(SEED)    + skip_ahead(id * vec_size), start = 0
        //   Engine(SEED, id * vec_size),                 start = 0
        //   Engine(SEED, 1) + skip_ahead(id * vec_size), start = 1 (odd offset, then
        //       a skip with a pending 32-bit half for 64-bit engines like pcg64_dxsm)
        const char* names[] = { "Engine(SEED) + skip_ahead", "Engine(SEED, offset)",
                                "Engine(SEED, 1) + skip_ahead" };
        for (int mode = 0; mode < 3; mode++) {
            std::vector<UIntType> r(N_GEN);
            try {
                sycl::range<1> range(N_GEN / Engine::vec_size);

                sycl::buffer<UIntType> buf(r);
                auto event = queue.submit([&](sycl::handler& cgh) {
                    sycl::accessor acc(buf, cgh, sycl::write_only);
                    cgh.parallel_for(range, [=](sycl::item<1> item) {
                        size_t id = item.get_id(0);
                        std::uint64_t offset = id * Engine::vec_size;
                        std::uint64_t ctor_offset = (mode == 1) ? offset : (mode == 2) ? 1 : 0;
                        Engine engine(SEED, ctor_offset);
                        if (mode != 1) {
                            oneapi::math::rng::device::skip_ahead(engine, offset);
                        }
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
            if (!check_equal_vector_device(r.data(), r_ref.data() + (mode == 2 ? 1 : 0), N_GEN)) {
                std::cout << names[mode] << " failed" << std::endl;
                status = test_failed;
                return;
            }
        }

        // uniform_bits<std::uint64_t>: a 64-bit number takes two consecutive
        // uniform_bits<std::uint32_t> positions; Engine(SEED, 1) + skip_ahead(2 * id * vec_size)
        // starts in the middle of a pair
        if constexpr (has_uniform_bits<Engine>::value) {
            std::vector<std::uint32_t> r_ref32(2 * N_GEN + Engine::vec_size);
            fill_reference(oneapi::math::rng::device::uniform_bits<std::uint32_t>{}, r_ref32);
            std::vector<std::uint64_t> r64_ref(N_GEN);
            for (int i = 0; i < N_GEN; i++) {
                r64_ref[i] =
                    r_ref32[2 * i + 1] | (static_cast<std::uint64_t>(r_ref32[2 * i + 2]) << 32);
            }

            std::vector<std::uint64_t> r64(N_GEN);
            try {
                sycl::range<1> range(N_GEN / Engine::vec_size);

                sycl::buffer<std::uint64_t> buf(r64);
                auto event = queue.submit([&](sycl::handler& cgh) {
                    sycl::accessor acc(buf, cgh, sycl::write_only);
                    cgh.parallel_for(range, [=](sycl::item<1> item) {
                        size_t id = item.get_id(0);
                        Engine engine(SEED, 1);
                        oneapi::math::rng::device::skip_ahead(engine, 2 * id * Engine::vec_size);
                        oneapi::math::rng::device::uniform_bits<std::uint64_t> distr;
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

            if (!check_equal_vector_device(r64, r64_ref)) {
                std::cout << "uniform_bits<std::uint64_t> after Engine(SEED, 1) + skip_ahead failed"
                          << std::endl;
                status = test_failed;
                return;
            }
        }
    }

    int status = test_passed;

private:
    // stream from offset 0; r.size() must be a multiple of vec_size
    template <typename Distr, typename UIntType>
    static void fill_reference(Distr distr, std::vector<UIntType>& r) {
        Engine engine(SEED);
        for (size_t i = 0; i < r.size(); i += Engine::vec_size) {
            auto res = oneapi::math::rng::device::generate(distr, engine);
            if constexpr (Engine::vec_size == 1) {
                r[i] = res;
            }
            else {
                for (int j = 0; j < Engine::vec_size; ++j) {
                    r[i + j] = res[j];
                }
            }
        }
    }
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
