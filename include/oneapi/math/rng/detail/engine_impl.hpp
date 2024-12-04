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

#ifndef _ONEMATH_RNG_ENGINE_IMPL_HPP_
#define _ONEMATH_RNG_ENGINE_IMPL_HPP_

#include <cstdint>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math/detail/export.hpp"
#include "oneapi/math/detail/get_device_id.hpp"

#include "oneapi/math/rng/distributions.hpp"
#include "oneapi/math/types.hpp"

namespace oneapi {
namespace math {
namespace rng {
namespace detail {

class engine_impl {
public:
    engine_impl(sycl::queue queue) : queue_(queue) {}

    engine_impl(const engine_impl& other) : queue_(other.queue_) {}

    // Buffers API
    virtual void generate(const uniform<float, uniform_method::standard>& distr, std::int64_t n,
                          sycl::buffer<float, 1>& r) = 0;

    virtual void generate(const uniform<double, uniform_method::standard>& distr, std::int64_t n,
                          sycl::buffer<double, 1>& r) = 0;

    virtual void generate(const uniform<std::int32_t, uniform_method::standard>& distr,
                          std::int64_t n, sycl::buffer<std::int32_t, 1>& r) = 0;

    virtual void generate(const uniform<float, uniform_method::accurate>& distr, std::int64_t n,
                          sycl::buffer<float, 1>& r) = 0;

    virtual void generate(const uniform<double, uniform_method::accurate>& distr, std::int64_t n,
                          sycl::buffer<double, 1>& r) = 0;

    virtual void generate(const gaussian<float, gaussian_method::box_muller2>& distr,
                          std::int64_t n, sycl::buffer<float, 1>& r) = 0;

    virtual void generate(const gaussian<double, gaussian_method::box_muller2>& distr,
                          std::int64_t n, sycl::buffer<double, 1>& r) = 0;

    virtual void generate(const gaussian<float, gaussian_method::icdf>& distr, std::int64_t n,
                          sycl::buffer<float, 1>& r) = 0;

    virtual void generate(const gaussian<double, gaussian_method::icdf>& distr, std::int64_t n,
                          sycl::buffer<double, 1>& r) = 0;

    virtual void generate(const lognormal<float, lognormal_method::box_muller2>& distr,
                          std::int64_t n, sycl::buffer<float, 1>& r) = 0;

    virtual void generate(const lognormal<double, lognormal_method::box_muller2>& distr,
                          std::int64_t n, sycl::buffer<double, 1>& r) = 0;

    virtual void generate(const lognormal<float, lognormal_method::icdf>& distr, std::int64_t n,
                          sycl::buffer<float, 1>& r) = 0;

    virtual void generate(const lognormal<double, lognormal_method::icdf>& distr, std::int64_t n,
                          sycl::buffer<double, 1>& r) = 0;

    virtual void generate(const bernoulli<std::int32_t, bernoulli_method::icdf>& distr,
                          std::int64_t n, sycl::buffer<std::int32_t, 1>& r) = 0;

    virtual void generate(const bernoulli<std::uint32_t, bernoulli_method::icdf>& distr,
                          std::int64_t n, sycl::buffer<std::uint32_t, 1>& r) = 0;

    virtual void generate(const poisson<std::int32_t, poisson_method::gaussian_icdf_based>& distr,
                          std::int64_t n, sycl::buffer<std::int32_t, 1>& r) = 0;

    virtual void generate(const poisson<std::uint32_t, poisson_method::gaussian_icdf_based>& distr,
                          std::int64_t n, sycl::buffer<std::uint32_t, 1>& r) = 0;

    virtual void generate(const bits<std::uint32_t>& distr, std::int64_t n,
                          sycl::buffer<std::uint32_t, 1>& r) = 0;

    // USM APIs
    virtual sycl::event generate(const uniform<float, uniform_method::standard>& distr,
                                 std::int64_t n, float* r,
                                 const std::vector<sycl::event>& dependencies) = 0;

    virtual sycl::event generate(const uniform<double, uniform_method::standard>& distr,
                                 std::int64_t n, double* r,
                                 const std::vector<sycl::event>& dependencies) = 0;

    virtual sycl::event generate(const uniform<std::int32_t, uniform_method::standard>& distr,
                                 std::int64_t n, std::int32_t* r,
                                 const std::vector<sycl::event>& dependencies) = 0;

    virtual sycl::event generate(const uniform<float, uniform_method::accurate>& distr,
                                 std::int64_t n, float* r,
                                 const std::vector<sycl::event>& dependencies) = 0;

    virtual sycl::event generate(const uniform<double, uniform_method::accurate>& distr,
                                 std::int64_t n, double* r,
                                 const std::vector<sycl::event>& dependencies) = 0;

    virtual sycl::event generate(const gaussian<float, gaussian_method::box_muller2>& distr,
                                 std::int64_t n, float* r,
                                 const std::vector<sycl::event>& dependencies) = 0;

    virtual sycl::event generate(const gaussian<double, gaussian_method::box_muller2>& distr,
                                 std::int64_t n, double* r,
                                 const std::vector<sycl::event>& dependencies) = 0;

    virtual sycl::event generate(const gaussian<float, gaussian_method::icdf>& distr,
                                 std::int64_t n, float* r,
                                 const std::vector<sycl::event>& dependencies) = 0;

    virtual sycl::event generate(const gaussian<double, gaussian_method::icdf>& distr,
                                 std::int64_t n, double* r,
                                 const std::vector<sycl::event>& dependencies) = 0;

    virtual sycl::event generate(const lognormal<float, lognormal_method::box_muller2>& distr,
                                 std::int64_t n, float* r,
                                 const std::vector<sycl::event>& dependencies) = 0;

    virtual sycl::event generate(const lognormal<double, lognormal_method::box_muller2>& distr,
                                 std::int64_t n, double* r,
                                 const std::vector<sycl::event>& dependencies) = 0;

    virtual sycl::event generate(const lognormal<float, lognormal_method::icdf>& distr,
                                 std::int64_t n, float* r,
                                 const std::vector<sycl::event>& dependencies) = 0;

    virtual sycl::event generate(const lognormal<double, lognormal_method::icdf>& distr,
                                 std::int64_t n, double* r,
                                 const std::vector<sycl::event>& dependencies) = 0;

    virtual sycl::event generate(const bernoulli<std::int32_t, bernoulli_method::icdf>& distr,
                                 std::int64_t n, std::int32_t* r,
                                 const std::vector<sycl::event>& dependencies) = 0;

    virtual sycl::event generate(const bernoulli<std::uint32_t, bernoulli_method::icdf>& distr,
                                 std::int64_t n, std::uint32_t* r,
                                 const std::vector<sycl::event>& dependencies) = 0;

    virtual sycl::event generate(
        const poisson<std::int32_t, poisson_method::gaussian_icdf_based>& distr, std::int64_t n,
        std::int32_t* r, const std::vector<sycl::event>& dependencies) = 0;

    virtual sycl::event generate(
        const poisson<std::uint32_t, poisson_method::gaussian_icdf_based>& distr, std::int64_t n,
        std::uint32_t* r, const std::vector<sycl::event>& dependencies) = 0;

    virtual sycl::event generate(const bits<std::uint32_t>& distr, std::int64_t n, std::uint32_t* r,
                                 const std::vector<sycl::event>& dependencies) = 0;

    virtual engine_impl* copy_state() = 0;

    virtual void skip_ahead(std::uint64_t num_to_skip) = 0;

    virtual void skip_ahead(std::initializer_list<std::uint64_t> num_to_skip) = 0;

    virtual void leapfrog(std::uint64_t idx, std::uint64_t stride) = 0;

    virtual ~engine_impl() {}

    sycl::queue& get_queue() {
        return queue_;
    }

protected:
    sycl::queue queue_;
};

} // namespace detail
} // namespace rng
} // namespace math
} // namespace oneapi

#endif //_ONEMATH_RNG_ENGINE_IMPL_HPP_
