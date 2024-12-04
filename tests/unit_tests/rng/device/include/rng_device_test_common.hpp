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

#ifndef _RNG_DEVICE_TEST_COMMON_HPP__
#define _RNG_DEVICE_TEST_COMMON_HPP__

#include <iostream>
#include <limits>

#include "test_helper.hpp"

#define SEED  777
#define N_GEN 960

// Defines for skip_ahead and leapfrog tests
#define N_ENGINES     5
#define N_PORTION     100
#define N_GEN_SERVICE (N_ENGINES * N_PORTION)

// defines for skip_ahead_ex tests
#define N_SKIP      ((std::uint64_t)pow(2, 62))
#define SKIP_TIMES  ((std::int32_t)pow(2, 14))
#define NUM_TO_SKIP { 0, (std::uint64_t)pow(2, 12) }

// Correctness checking.
static inline bool check_equal_device(float x, float x_ref) {
    float bound = std::numeric_limits<float>::epsilon();
    float aerr = std::abs(x - x_ref);
    return (aerr <= bound);
}

static inline bool check_equal_device(double x, double x_ref) {
    double bound = std::numeric_limits<double>::epsilon();
    double aerr = std::abs(x - x_ref);
    return (aerr <= bound);
}

static inline bool check_equal_device(std::uint32_t x, std::uint32_t x_ref) {
    return x == x_ref;
}

static inline bool check_equal_device(std::uint64_t x, std::uint64_t x_ref) {
    return x == x_ref;
}

template <typename Fp, typename AllocType>
static inline bool check_equal_vector_device(std::vector<Fp, AllocType>& r1,
                                             std::vector<Fp, AllocType>& r2) {
    bool good = true;
    for (int i = 0; i < r1.size(); i++) {
        if (!check_equal_device(r1[i], r2[i])) {
            good = false;
            break;
        }
    }
    return good;
}

template <typename Test>
class rng_device_test {
public:
    // method to call any tests, switch between rt and ct
    template <typename... Args>
    int operator()(sycl::device* dev, Args... args) {
        auto exception_handler = [](sycl::exception_list exceptions) {
            for (std::exception_ptr const& e : exceptions) {
                try {
                    std::rethrow_exception(e);
                }
                catch (sycl::exception const& e) {
                    std::cout << "Caught asynchronous SYCL exception during ASUM:\n"
                              << e.what() << std::endl;
                    print_error_code(e);
                }
            }
        };

        sycl::queue queue(*dev, exception_handler);

        test_(queue, args...);

        return test_.status;
    }

protected:
    Test test_;
};

template <typename T, typename = void>
struct has_member_code_meta : std::false_type {};

template <typename T>
struct has_member_code_meta<T, std::void_t<decltype(std::declval<T>().get_multi_ptr())>>
        : std::true_type {};

template <typename T, typename std::enable_if<has_member_code_meta<T>::value>::type* = nullptr>
auto get_multi_ptr(T acc) {
#ifndef __HIPSYCL__
    return acc.get_multi_ptr();
#else
    return acc.get_pointer();
#endif
};

template <typename T, typename std::enable_if<!has_member_code_meta<T>::value>::type* = nullptr>
auto get_multi_ptr(T acc) {
#ifndef __HIPSYCL__
    return acc.template get_multi_ptr<sycl::access::decorated::yes>();
#else
    return acc.get_pointer();
#endif
};

template <typename T>
auto get_error_code(T x) {
    return x.code().value();
};

template <typename Fp, typename AllocType>
bool compare_moments(const std::vector<Fp, AllocType>& r, double tM, double tD, double tQ) {
    double tD2;
    double sM, sD;
    double sum, sum2;
    double n, s;
    double DeltaM, DeltaD;

    // sample moments
    sum = 0.0;
    sum2 = 0.0;
    for (int i = 0; i < N_GEN; i++) {
        sum += (double)r[i];
        sum2 += (double)r[i] * (double)r[i];
    }
    sM = sum / ((double)N_GEN);
    sD = sum2 / (double)N_GEN - (sM * sM);

    // Comparison of theoretical and sample moments
    n = (double)N_GEN;
    tD2 = tD * tD;
    s = ((tQ - tD2) / n) - (2 * (tQ - 2 * tD2) / (n * n)) + ((tQ - 3 * tD2) / (n * n * n));

    DeltaM = (tM - sM) / std::sqrt(tD / n);
    DeltaD = (tD - sD) / std::sqrt(s);
    if (fabs(DeltaM) > 3.0 || fabs(DeltaD) > 10.0) {
        std::cout << "Error: sample moments (mean=" << sM << ", variance=" << sD
                  << ") disagree with theory (mean=" << tM << ", variance=" << tD << ")"
                  << " N_GEN = " << N_GEN << std::endl;
        return false;
    }
    return true;
}

template <typename Distribution>
struct statistics_device {};

template <typename Fp, typename Method>
struct statistics_device<oneapi::math::rng::device::uniform<Fp, Method>> {
    template <typename AllocType>
    bool check(const std::vector<Fp, AllocType>& r,
               const oneapi::math::rng::device::uniform<Fp, Method>& distr) {
        double tM, tD, tQ;
        Fp a = distr.a();
        Fp b = distr.b();

        // Theoretical moments
        tM = (b + a) / 2.0;
        tD = ((b - a) * (b - a)) / 12.0;
        tQ = ((b - a) * (b - a) * (b - a) * (b - a)) / 80.0;

        return compare_moments(r, tM, tD, tQ);
    }
};

template <typename Method>
struct statistics_device<oneapi::math::rng::device::uniform<std::int32_t, Method>> {
    template <typename AllocType>
    bool check(const std::vector<std::int32_t, AllocType>& r,
               const oneapi::math::rng::device::uniform<std::int32_t, Method>& distr) {
        double tM, tD, tQ;
        double a = distr.a();
        double b = distr.b();

        // Theoretical moments
        tM = (a + b - 1.0) / 2.0;
        tD = ((b - a) * (b - a) - 1.0) / 12.0;
        tQ = (((b - a) * (b - a)) * ((1.0 / 80.0) * (b - a) * (b - a) - (1.0 / 24.0))) +
             (7.0 / 240.0);

        return compare_moments(r, tM, tD, tQ);
    }
};

template <typename Method>
struct statistics_device<oneapi::math::rng::device::uniform<std::uint32_t, Method>> {
    template <typename AllocType>
    bool check(const std::vector<std::uint32_t, AllocType>& r,
               const oneapi::math::rng::device::uniform<std::uint32_t, Method>& distr) {
        double tM, tD, tQ;
        double a = distr.a();
        double b = distr.b();

        // Theoretical moments
        tM = (a + b - 1.0) / 2.0;
        tD = ((b - a) * (b - a) - 1.0) / 12.0;
        tQ = (((b - a) * (b - a)) * ((1.0 / 80.0) * (b - a) * (b - a) - (1.0 / 24.0))) +
             (7.0 / 240.0);

        return compare_moments(r, tM, tD, tQ);
    }
};

template <typename Method>
struct statistics_device<oneapi::math::rng::device::uniform<std::int64_t, Method>> {
    template <typename AllocType>
    bool check(const std::vector<std::int64_t, AllocType>& r,
               const oneapi::math::rng::device::uniform<std::int64_t, Method>& distr) {
        double tM, tD, tQ;
        double a = distr.a();
        double b = distr.b();

        // Theoretical moments
        tM = (a + b - 1.0) / 2.0;
        tD = ((b - a) * (b - a) - 1.0) / 12.0;
        tQ = (((b - a) * (b - a)) * ((1.0 / 80.0) * (b - a) * (b - a) - (1.0 / 24.0))) +
             (7.0 / 240.0);

        return compare_moments(r, tM, tD, tQ);
    }
};

template <typename Method>
struct statistics_device<oneapi::math::rng::device::uniform<std::uint64_t, Method>> {
    template <typename AllocType>
    bool check(const std::vector<std::uint64_t, AllocType>& r,
               const oneapi::math::rng::device::uniform<std::uint64_t, Method>& distr) {
        double tM, tD, tQ;
        double a = distr.a();
        double b = distr.b();

        // Theoretical moments
        tM = (a + b - 1.0) / 2.0;
        tD = ((b - a) * (b - a) - 1.0) / 12.0;
        tQ = (((b - a) * (b - a)) * ((1.0 / 80.0) * (b - a) * (b - a) - (1.0 / 24.0))) +
             (7.0 / 240.0);

        return compare_moments(r, tM, tD, tQ);
    }
};

template <typename Fp, typename Method>
struct statistics_device<oneapi::math::rng::device::gaussian<Fp, Method>> {
    template <typename AllocType>
    bool check(const std::vector<Fp, AllocType>& r,
               const oneapi::math::rng::device::gaussian<Fp, Method>& distr) {
        double tM, tD, tQ;
        Fp a = distr.mean();
        Fp sigma = distr.stddev();

        // Theoretical moments
        tM = a;
        tD = sigma * sigma;
        tQ = 720.0 * sigma * sigma * sigma * sigma;

        return compare_moments(r, tM, tD, tQ);
    }
};

template <typename Fp, typename Method>
struct statistics_device<oneapi::math::rng::device::lognormal<Fp, Method>> {
    template <typename AllocType>
    bool check(const std::vector<Fp, AllocType>& r,
               const oneapi::math::rng::device::lognormal<Fp, Method>& distr) {
        double tM, tD, tQ;
        Fp a = distr.m();
        Fp b = distr.displ();
        Fp sigma = distr.s();
        Fp beta = distr.scale();

        // Theoretical moments
        tM = b + beta * std::exp(a + sigma * sigma * 0.5);
        tD = beta * beta * std::exp(2.0 * a + sigma * sigma) * (std::exp(sigma * sigma) - 1.0);
        tQ = beta * beta * beta * beta * std::exp(4.0 * a + 2.0 * sigma * sigma) *
             (std::exp(6.0 * sigma * sigma) - 4.0 * std::exp(3.0 * sigma * sigma) +
              6.0 * std::exp(sigma * sigma) - 3.0);

        return compare_moments(r, tM, tD, tQ);
    }
};

template <typename Fp, typename Method>
struct statistics_device<oneapi::math::rng::device::exponential<Fp, Method>> {
    template <typename AllocType>
    bool check(const std::vector<Fp, AllocType>& r,
               const oneapi::math::rng::device::exponential<Fp, Method>& distr) {
        double tM, tD, tQ;
        Fp a = distr.a();
        Fp beta = distr.beta();

        tM = a + beta;
        tD = beta * beta;
        tQ = 9.0 * beta * beta * beta * beta;

        return compare_moments(r, tM, tD, tQ);
    }
};

template <typename Fp, typename Method>
struct statistics_device<oneapi::math::rng::device::poisson<Fp, Method>> {
    template <typename AllocType>
    bool check(const std::vector<Fp, AllocType>& r,
               const oneapi::math::rng::device::poisson<Fp, Method>& distr) {
        double tM, tD, tQ;
        double lambda = distr.lambda();

        tM = lambda;
        tD = lambda;
        tQ = 4 * lambda * lambda + lambda;

        return compare_moments(r, tM, tD, tQ);
    }
};

template <typename Fp, typename Method>
struct statistics_device<oneapi::math::rng::device::bernoulli<Fp, Method>> {
    template <typename AllocType>
    bool check(const std::vector<Fp, AllocType>& r,
               const oneapi::math::rng::device::bernoulli<Fp, Method>& distr) {
        double tM, tD, tQ;
        double p = static_cast<double>(distr.p());

        tM = p;
        tD = p * (1.0 - p);
        tQ = p * (1.0 - 4.0 * p + 6.0 * p * p - 3.0 * p * p * p);

        return compare_moments(r, tM, tD, tQ);
    }
};

template <typename Fp, typename Method>
struct statistics_device<oneapi::math::rng::device::beta<Fp, Method>> {
    template <typename AllocType>
    bool check(const std::vector<Fp, AllocType>& r,
               const oneapi::math::rng::device::beta<Fp, Method>& distr) {
        double tM, tD, tQ;
        double b, c, d, e, e2, b2, sum_pq;
        Fp p = distr.p();
        Fp q = distr.q();
        Fp a = distr.a();
        Fp beta = distr.b();

        b2 = beta * beta;
        sum_pq = p + q;
        b = (p + 1.0) / (sum_pq + 1.0);
        c = (p + 2.0) / (sum_pq + 2.0);
        d = (p + 3.0) / (sum_pq + 3.0);
        e = p / sum_pq;
        e2 = e * e;

        tM = a + e * beta;
        tD = b2 * p * q / (sum_pq * sum_pq * (sum_pq + 1.0));
        tQ = b2 * b2 * (e * b * c * d - 4.0 * e2 * b * c + 6.0 * e2 * e * b - 3.0 * e2 * e2);

        return compare_moments(r, tM, tD, tQ);
    }
};

template <typename Fp, typename Method>
struct statistics_device<oneapi::math::rng::device::gamma<Fp, Method>> {
    template <typename AllocType>
    bool check(const std::vector<Fp, AllocType>& r,
               const oneapi::math::rng::device::gamma<Fp, Method>& distr) {
        double tM, tD, tQ;
        Fp a = distr.a();
        Fp alpha = distr.alpha();
        Fp beta = distr.beta();

        tM = a + beta * alpha;
        tD = beta * beta * alpha;
        tQ = beta * beta * beta * beta * 3 * alpha * (alpha + 2);

        return compare_moments(r, tM, tD, tQ);
    }
};

template <typename Fp>
struct statistics_device<oneapi::math::rng::device::bits<Fp>> {
    template <typename AllocType>
    bool check(const std::vector<Fp, AllocType>& r,
               const oneapi::math::rng::device::bits<Fp>& distr) {
        return true;
    }
};

template <typename Fp>
struct statistics_device<oneapi::math::rng::device::uniform_bits<Fp>> {
    template <typename AllocType>
    bool check(const std::vector<Fp, AllocType>& r,
               const oneapi::math::rng::device::uniform_bits<Fp>& distr) {
        return true;
    }
};

template <typename Engine>
struct is_mcg59 : std::false_type {};

template <std::int32_t VecSize>
struct is_mcg59<oneapi::math::rng::device::mcg59<VecSize>> : std::true_type {};

#endif // _RNG_DEVICE_TEST_COMMON_HPP__
