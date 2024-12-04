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

#ifndef _RNG_TEST_STATISTICS_CHECK_HPP__
#define _RNG_TEST_STATISTICS_CHECK_HPP__

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math.hpp"

#include "rng_test_common.hpp"

template <typename Type, typename AllocType>
bool compare_moments(std::vector<Type, AllocType>& r, double tM, double tD, double tQ) {
    double tD2;
    double sM, sD;
    double sum, sum2;
    double n, s;
    double DeltaM, DeltaD;

    // Sample moments
    sum = 0.0;
    sum2 = 0.0;
    for (int i = 0; i < r.size(); i++) {
        sum += (double)r[i];
        sum2 += (double)r[i] * (double)r[i];
    }
    sM = sum / ((double)r.size());
    sD = sum2 / (double)r.size() - (sM * sM);

    // Comparison of theoretical and sample moments
    n = (double)r.size();
    tD2 = tD * tD;
    s = ((tQ - tD2) / n) - (2 * (tQ - 2 * tD2) / (n * n)) + ((tQ - 3 * tD2) / (n * n * n));

    DeltaM = (tM - sM) / sqrt(tD / n);
    DeltaD = (tD - sD) / sqrt(s);
    if (fabs(DeltaM) > 3.0 || fabs(DeltaD) > 10.0) {
        std::cout << "Error: sample moments (mean=" << sM << ", variance=" << sD
                  << ") disagree with theory (mean=" << tM << ", variance=" << tD << ")"
                  << " N_GEN = " << r.size() << std::endl;
        return false;
    }
    return true;
}

template <typename Distribution>
struct statistics {};

template <typename Type, typename Method>
struct statistics<oneapi::math::rng::uniform<Type, Method>> {
    template <typename AllocType>
    bool check(std::vector<Type, AllocType>& r,
               const oneapi::math::rng::uniform<Type, Method>& distr) {
        double tM, tD, tQ;
        Type a = distr.a();
        Type b = distr.b();

        // Theoretical moments
        tM = (b + a) / 2.0;
        tD = ((b - a) * (b - a)) / 12.0;
        tQ = ((b - a) * (b - a) * (b - a) * (b - a)) / 80.0;

        return compare_moments(r, tM, tD, tQ);
    }
};

template <typename Method>
struct statistics<oneapi::math::rng::uniform<std::int32_t, Method>> {
    template <typename AllocType>
    bool check(std::vector<int32_t, AllocType>& r,
               const oneapi::math::rng::uniform<int32_t, Method>& distr) {
        double tM, tD, tQ;
        int32_t a = distr.a();
        int32_t b = distr.b();

        // Theoretical moments
        tM = (a + b - 1.0) / 2.0;
        tD = ((b - a) * (b - a) - 1.0) / 12.0;
        tQ = (((b - a) * (b - a)) * ((1.0 / 80.0) * (b - a) * (b - a) - (1.0 / 24.0))) +
             (7.0 / 240.0);

        return compare_moments(r, tM, tD, tQ);
    }
};

template <typename Type, typename Method>
struct statistics<oneapi::math::rng::gaussian<Type, Method>> {
    template <typename AllocType>
    bool check(std::vector<Type, AllocType>& r,
               const oneapi::math::rng::gaussian<Type, Method>& distr) {
        double tM, tD, tQ;
        Type a = distr.mean();
        Type sigma = distr.stddev();

        // Theoretical moments
        tM = a;
        tD = sigma * sigma;
        tQ = 720.0 * sigma * sigma * sigma * sigma;

        return compare_moments(r, tM, tD, tQ);
    }
};

template <typename Type, typename Method>
struct statistics<oneapi::math::rng::lognormal<Type, Method>> {
    template <typename AllocType>
    bool check(std::vector<Type, AllocType>& r,
               const oneapi::math::rng::lognormal<Type, Method>& distr) {
        double tM, tD, tQ;
        Type a = distr.m();
        Type b = distr.displ();
        Type sigma = distr.s();
        Type beta = distr.scale();

        // Theoretical moments
        tM = b + beta * exp(a + sigma * sigma * 0.5);
        tD = beta * beta * exp(2.0 * a + sigma * sigma) * (exp(sigma * sigma) - 1.0);
        tQ = beta * beta * beta * beta * exp(4.0 * a + 2.0 * sigma * sigma) *
             (exp(6.0 * sigma * sigma) - 4.0 * exp(3.0 * sigma * sigma) + 6.0 * exp(sigma * sigma) -
              3.0);

        return compare_moments(r, tM, tD, tQ);
    }
};

template <typename Type, typename Method>
struct statistics<oneapi::math::rng::bernoulli<Type, Method>> {
    template <typename AllocType>
    bool check(std::vector<Type, AllocType>& r,
               const oneapi::math::rng::bernoulli<Type, Method>& distr) {
        double tM, tD, tQ;
        double p = distr.p();

        tM = p;
        tD = p * (1.0 - p);
        tQ = p * (1.0 - 4.0 * p + 6.0 * p * p - 3.0 * p * p * p);

        return compare_moments(r, tM, tD, tQ);
    }
};

template <typename Type, typename Method>
struct statistics<oneapi::math::rng::poisson<Type, Method>> {
    template <typename AllocType>
    bool check(std::vector<Type, AllocType>& r,
               const oneapi::math::rng::poisson<Type, Method>& distr) {
        double tM, tD, tQ;
        double lambda = distr.lambda();

        tM = lambda;
        tD = lambda;
        tQ = 4 * lambda * lambda + lambda;

        return compare_moments(r, tM, tD, tQ);
    }
};

#endif // _RNG_TEST_STATISTICS_CHECK_HPP__
