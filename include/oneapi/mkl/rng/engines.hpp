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

#ifndef _ONEMKL_RNG_ENGINES_HPP_
#define _ONEMKL_RNG_ENGINES_HPP_

#include <cstdint>
#include <limits>
#include <memory>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/detail/backend_selector.hpp"

#include "oneapi/mkl/rng/detail/engine_impl.hpp"
#include "oneapi/mkl/rng/detail/rng_loader.hpp"

#ifdef ONEMKL_ENABLE_MKLCPU_BACKEND
#include "oneapi/mkl/rng/detail/mklcpu/onemkl_rng_mklcpu.hpp"
#endif
#ifdef ONEMKL_ENABLE_MKLGPU_BACKEND
#include "oneapi/mkl/rng/detail/mklgpu/onemkl_rng_mklgpu.hpp"
#endif
#ifdef ONEMKL_ENABLE_CURAND_BACKEND
#include "oneapi/mkl/rng/detail/curand/onemkl_rng_curand.hpp"
#endif
#ifdef ONEMKL_ENABLE_ROCRAND_BACKEND
#include "oneapi/mkl/rng/detail/rocrand/onemkl_rng_rocrand.hpp"
#endif

namespace oneapi {
namespace mkl {
namespace rng {

// Class oneapi::mkl::rng::philox4x32x10
//
// Represents Philox4x32-10 counter-based pseudorandom number generator
//
// Supported parallelization methods:
//      skip_ahead
class philox4x32x10 {
public:
    static constexpr std::uint64_t default_seed = 0;

    philox4x32x10(sycl::queue queue, std::uint64_t seed = default_seed)
            : pimpl_(detail::create_philox4x32x10(get_device_id(queue), queue, seed)) {}

    philox4x32x10(sycl::queue queue, std::initializer_list<std::uint64_t> seed)
            : pimpl_(detail::create_philox4x32x10(get_device_id(queue), queue, seed)) {}

#ifdef ONEMKL_ENABLE_MKLCPU_BACKEND
    philox4x32x10(backend_selector<backend::mklcpu> selector, std::uint64_t seed = default_seed)
            : pimpl_(mklcpu::create_philox4x32x10(selector.get_queue(), seed)) {}

    philox4x32x10(backend_selector<backend::mklcpu> selector,
                  std::initializer_list<std::uint64_t> seed)
            : pimpl_(mklcpu::create_philox4x32x10(selector.get_queue(), seed)) {}
#endif

#ifdef ONEMKL_ENABLE_MKLGPU_BACKEND
    philox4x32x10(backend_selector<backend::mklgpu> selector, std::uint64_t seed = default_seed)
            : pimpl_(mklgpu::create_philox4x32x10(selector.get_queue(), seed)) {}

    philox4x32x10(backend_selector<backend::mklgpu> selector,
                  std::initializer_list<std::uint64_t> seed)
            : pimpl_(mklgpu::create_philox4x32x10(selector.get_queue(), seed)) {}
#endif

#ifdef ONEMKL_ENABLE_CURAND_BACKEND
    philox4x32x10(backend_selector<backend::curand> selector, std::uint64_t seed = default_seed)
            : pimpl_(curand::create_philox4x32x10(selector.get_queue(), seed)) {}

    philox4x32x10(backend_selector<backend::curand> selector,
                  std::initializer_list<std::uint64_t> seed)
            : pimpl_(curand::create_philox4x32x10(selector.get_queue(), seed)) {}
#endif
#ifdef ONEMKL_ENABLE_ROCRAND_BACKEND
    philox4x32x10(backend_selector<backend::rocrand> selector, std::uint64_t seed = default_seed)
            : pimpl_(rocrand::create_philox4x32x10(selector.get_queue(), seed)) {}

    philox4x32x10(backend_selector<backend::rocrand> selector,
                  std::initializer_list<std::uint64_t> seed)
            : pimpl_(rocrand::create_philox4x32x10(selector.get_queue(), seed)) {}
#endif

    philox4x32x10(const philox4x32x10& other) {
        pimpl_.reset(other.pimpl_.get()->copy_state());
    }

    philox4x32x10(philox4x32x10&& other) {
        pimpl_ = std::move(other.pimpl_);
    }

    philox4x32x10& operator=(const philox4x32x10& other) {
        if (this == &other)
            return *this;
        pimpl_.reset(other.pimpl_.get()->copy_state());
        return *this;
    }

    philox4x32x10& operator=(philox4x32x10&& other) {
        if (this == &other)
            return *this;
        pimpl_ = std::move(other.pimpl_);
        return *this;
    }

private:
    std::unique_ptr<detail::engine_impl> pimpl_;

    template <typename Engine>
    friend void skip_ahead(Engine& engine, std::uint64_t num_to_skip);

    template <typename Engine>
    friend void skip_ahead(Engine& engine, std::initializer_list<std::uint64_t> num_to_skip);

    template <typename Distr, typename Engine>
    friend void generate(const Distr& distr, Engine& engine, std::int64_t n,
                         sycl::buffer<typename Distr::result_type, 1>& r);

    template <typename Distr, typename Engine>
    friend sycl::event generate(const Distr& distr, Engine& engine, std::int64_t n,
                                typename Distr::result_type* r,
                                const std::vector<sycl::event>& dependencies);
};

// Class oneapi::mkl::rng::mrg32k3a
//
// Represents the combined recurcive pseudorandom number generator
//
// Supported parallelization methods:
//      skip_ahead
class mrg32k3a {
public:
    static constexpr std::uint32_t default_seed = 1;

    mrg32k3a(sycl::queue queue, std::uint32_t seed = default_seed)
            : pimpl_(detail::create_mrg32k3a(get_device_id(queue), queue, seed)) {}

    mrg32k3a(sycl::queue queue, std::initializer_list<std::uint32_t> seed)
            : pimpl_(detail::create_mrg32k3a(get_device_id(queue), queue, seed)) {}

#ifdef ONEMKL_ENABLE_MKLCPU_BACKEND
    mrg32k3a(backend_selector<backend::mklcpu> selector, std::uint32_t seed = default_seed)
            : pimpl_(mklcpu::create_mrg32k3a(selector.get_queue(), seed)) {}

    mrg32k3a(backend_selector<backend::mklcpu> selector, std::initializer_list<std::uint32_t> seed)
            : pimpl_(mklcpu::create_mrg32k3a(selector.get_queue(), seed)) {}
#endif

#ifdef ONEMKL_ENABLE_MKLGPU_BACKEND
    mrg32k3a(backend_selector<backend::mklgpu> selector, std::uint32_t seed = default_seed)
            : pimpl_(mklgpu::create_mrg32k3a(selector.get_queue(), seed)) {}

    mrg32k3a(backend_selector<backend::mklgpu> selector, std::initializer_list<std::uint32_t> seed)
            : pimpl_(mklgpu::create_mrg32k3a(selector.get_queue(), seed)) {}
#endif

#ifdef ONEMKL_ENABLE_CURAND_BACKEND
    mrg32k3a(backend_selector<backend::curand> selector, std::uint32_t seed = default_seed)
            : pimpl_(curand::create_mrg32k3a(selector.get_queue(), seed)) {}

    mrg32k3a(backend_selector<backend::curand> selector, std::initializer_list<std::uint32_t> seed)
            : pimpl_(curand::create_mrg32k3a(selector.get_queue(), seed)) {}
#endif

#ifdef ONEMKL_ENABLE_ROCRAND_BACKEND
    mrg32k3a(backend_selector<backend::rocrand> selector, std::uint32_t seed = default_seed)
            : pimpl_(rocrand::create_mrg32k3a(selector.get_queue(), seed)) {}

    mrg32k3a(backend_selector<backend::rocrand> selector, std::initializer_list<std::uint32_t> seed)
            : pimpl_(rocrand::create_mrg32k3a(selector.get_queue(), seed)) {}
#endif

    mrg32k3a(const mrg32k3a& other) {
        pimpl_.reset(other.pimpl_.get()->copy_state());
    }

    mrg32k3a(mrg32k3a&& other) {
        pimpl_ = std::move(other.pimpl_);
    }

    mrg32k3a& operator=(const mrg32k3a& other) {
        if (this == &other)
            return *this;
        pimpl_.reset(other.pimpl_.get()->copy_state());
        return *this;
    }

    mrg32k3a& operator=(mrg32k3a&& other) {
        if (this == &other)
            return *this;
        pimpl_ = std::move(other.pimpl_);
        return *this;
    }

private:
    std::unique_ptr<detail::engine_impl> pimpl_;

    template <typename Engine>
    friend void skip_ahead(Engine& engine, std::uint64_t num_to_skip);

    template <typename Engine>
    friend void skip_ahead(Engine& engine, std::initializer_list<std::uint64_t> num_to_skip);

    template <typename Distr, typename Engine>
    friend void generate(const Distr& distr, Engine& engine, std::int64_t n,
                         sycl::buffer<typename Distr::result_type, 1>& r);

    template <typename Distr, typename Engine>
    friend sycl::event generate(const Distr& distr, Engine& engine, std::int64_t n,
                                typename Distr::result_type* r,
                                const std::vector<sycl::event>& dependencies);
};

// Default engine to be used for common cases
using default_engine = philox4x32x10;

} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif //_ONEMKL_RNG_ENGINES_HPP_
