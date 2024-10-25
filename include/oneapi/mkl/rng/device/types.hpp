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

#ifndef _MKL_RNG_DEVICE_TYPES_HPP_
#define _MKL_RNG_DEVICE_TYPES_HPP_

namespace oneapi::mkl::rng::device {

// METHODS FOR DISTRIBUTIONS

namespace uniform_method {
struct standard {};
struct accurate {};
using by_default = standard;
} // namespace uniform_method

namespace gaussian_method {
struct box_muller2 {};
struct icdf {};
using by_default = box_muller2;
} // namespace gaussian_method

namespace lognormal_method {
struct box_muller2 {};
using by_default = box_muller2;
} // namespace lognormal_method

namespace exponential_method {
struct icdf {};
struct icdf_accurate {};
using by_default = icdf;
} // namespace exponential_method

namespace poisson_method {
struct devroye {};
using by_default = devroye;
} // namespace poisson_method

namespace bernoulli_method {
struct icdf {};
using by_default = icdf;
} // namespace bernoulli_method

namespace beta_method {
struct cja {};
struct cja_accurate {};
using by_default = cja;
} // namespace beta_method

namespace gamma_method {
struct marsaglia {};
struct marsaglia_accurate {};
using by_default = marsaglia;
} // namespace gamma_method

} // namespace oneapi::mkl::rng::device

#endif // _MKL_RNG_DEVICE_TYPES_HPP_
