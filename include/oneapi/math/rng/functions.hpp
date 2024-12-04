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

#ifndef _ONEMATH_RNG_FUNCTIONS_HPP_
#define _ONEMATH_RNG_FUNCTIONS_HPP_

#include <cstdint>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math/exceptions.hpp"
#include "oneapi/math/rng/predicates.hpp"

namespace oneapi {
namespace math {
namespace rng {

// Function oneapi::math::rng::generate().Buffer API
// Provides random numbers from a given engine with a given statistics
//
// Input parameters:
//      const Distr& distr              - distribution object
//      Engine& engine                   - engine object
//      std::int64_t n                   - number of random values to be generated
//
// Output parameters:
//      sycl::buffer<typename Distr::result_type, 1>& r - sycl::buffer to the output vector
template <typename Distr, typename Engine>
static inline void generate(const Distr& distr, Engine& engine, std::int64_t n,
                            sycl::buffer<typename Distr::result_type, 1>& r) {
    generate_precondition(distr, engine, n, r);
    engine.pimpl_->generate(distr, n, r);
}

// Function oneapi::math::rng::generate(). USM API
// Provides random numbers from a given engine with a given statistics
//
// Input parameters:
//      const Distr& distr               - distribution object
//      Engine& engine                   - engine object
//      std::int64_t n                   - number of random values to be generated
//      const std::vector<sycl::event>& dependencies - list of events to wait for
//                  before starting computation, if any. If omitted, defaults to no dependencies
//
// Output parameters:
//      typename Distr::result_type* - pointer to the output vector
//
// Returns:
//      sycl::event - event for the submitted to the engine's queue task
template <typename Distr, typename Engine>
static inline sycl::event generate(const Distr& distr, Engine& engine, std::int64_t n,
                                   typename Distr::result_type* r,
                                   const std::vector<sycl::event>& dependencies = {}) {
    generate_precondition(distr, engine, n, r, dependencies);
    return engine.pimpl_->generate(distr, n, r, dependencies);
}

//  SERVICE FUNCTIONS

// Function oneapi::math::rng::skip_ahead(). Common interface
//
// Proceeds state of engine using the skip-ahead method
//
// Input parameters:
//      Engine& engine             - engine object
//      const std::int64_t num_to_skip - number of skipped elements
template <typename Engine>
static inline void skip_ahead(Engine& engine, std::uint64_t num_to_skip) {
    engine.pimpl_->skip_ahead(num_to_skip);
}

// Function oneapi::math::rng::skip_ahead(). Interface with partitioned number of skipped elements
//
// Proceeds state of engine using the skip-ahead method
//
// Input parameters:
//      Engine& engine                               - engine object
//      std::initializer_list<std::uint64_t> num_to_skip - number of skipped elements
template <typename Engine>
static inline void skip_ahead(Engine& engine, std::initializer_list<std::uint64_t> num_to_skip) {
    engine.pimpl_->skip_ahead(num_to_skip);
}

// Function oneapi::math::rng::leapfrog()
//
// Proceeds state of engine using the leapfrog method
//
// Input parameters:
//      Engine& engine  - engine object
//      std::uint64_t idx    - index of the computational node
//      std::uint64_t stride - largest number of computational nodes, or stride
template <typename Engine>
static inline void leapfrog(Engine& engine, std::uint64_t idx, std::uint64_t stride) {
    engine.pimpl_->leapfrog(idx, stride);
}

} // namespace rng
} // namespace math
} // namespace oneapi

#endif //_ONEMATH_RNG_FUNCTIONS_HPP_
