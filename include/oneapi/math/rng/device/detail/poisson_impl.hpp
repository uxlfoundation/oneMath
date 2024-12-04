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

#ifndef ONEMATH_RNG_DEVICE_POISSON_IMPL_HPP_
#define ONEMATH_RNG_DEVICE_POISSON_IMPL_HPP_

#include <limits>

namespace oneapi::math::rng::device::detail {

// Implementation of Poisson distribution uses 3 methods depending on lambda parameter:
//    - table-lookup method [1] for small lambdas (lambda < 60)
//    - Devroye's method [2] for medium lambdas (60 <= lambda < 1000)
//    - Gaussian approximation [1] for huge lambdas (lambda >= 1000)
//
// References:
// [1] Michael B. Giles // Algorithm 955: approximation of the inverse Poisson cumulative
// distribution function
// [2] Devroye, L. Non-Uniform Random Variates Generation. Springer-Verlag,
// New York, 1986, Ch. X, Sects. 3.3 & 3.4 + Errata

#define RNG_POISSON_LAMBDA_HUGE_BOUND 1000.0
#define RNG_POISSON_LAMBDA_LOW_BOUND  60.0
#define RNG_POISSON_N_PRECOMPUTED_CDF 32

struct poisson_parameters {
    void set_lambda(double lambda) {
        if (lambda >= RNG_POISSON_LAMBDA_HUGE_BOUND) {
            sqrt_lambda_ = sycl::sqrt(lambda);
        }
        else if (lambda >= RNG_POISSON_LAMBDA_LOW_BOUND) {
            floored_lambda_ = sycl::floor(lambda);
            log_lambda_ = sycl::log(lambda);
            lgamma_floored_lambda_ = sycl::lgamma(floored_lambda_ + 1.0);
            sqrt_floored_lambda_ = sycl::sqrt(floored_lambda_);
            dx_ = sycl::sqrt(2.0 * floored_lambda_ * sycl::log(32.0 * floored_lambda_ / pi_4_));
            delta_ = sycl::round((sycl::max)(6.0, (sycl::min)(floored_lambda_, dx_)));
            dpdfl_ = delta_ + 2.0 * floored_lambda_;
            sqrt_half_dpdfl_ = sycl::sqrt(dpdfl_ / 2.0);
            inv_dpdfl_ = 1.0 / dpdfl_;
            c2_add_coeff_ = sycl::sqrt(pi_4_ * dpdfl_) * sycl::exp(inv_dpdfl_);
            c_add_coeff_ =
                2.0 * dpdfl_ * sycl::exp(-delta_ * inv_dpdfl_ * (1.0 + delta_ / 2.0)) / delta_;
            c1_ = sqrt_floored_lambda_ * spi_2_;
            c2_ = c2_add_coeff_ + c1_;
            c3_ = c2_ + 1.0;
            c4_ = c2_ + 2.0;
            c5_ = c4_ + exp_one_by_78;
            c_ = c5_ + c_add_coeff_;
        }
        else {
            prob[0] = sycl::exp(-lambda);
            double tmp = prob[0];
            for (int i = 1; i < RNG_POISSON_N_PRECOMPUTED_CDF; ++i) {
                tmp *= lambda / (double)i;
                prob[i] = prob[i - 1] + tmp;
            }
        }
    }

    poisson_parameters& operator=(const poisson_parameters& other) {
        if (this == &other) {
            return *this;
        }
        for (int i = 0; i < RNG_POISSON_N_PRECOMPUTED_CDF; i++) {
            prob[i] = other.prob[i];
        }
        floored_lambda_ = other.floored_lambda_;
        log_lambda_ = other.log_lambda_;
        lgamma_floored_lambda_ = other.lgamma_floored_lambda_;
        sqrt_lambda_ = other.sqrt_lambda_;
        sqrt_floored_lambda_ = other.sqrt_floored_lambda_;
        dx_ = other.dx_;
        delta_ = other.delta_;
        dpdfl_ = other.dpdfl_;
        sqrt_half_dpdfl_ = other.sqrt_half_dpdfl_;
        inv_dpdfl_ = other.inv_dpdfl_;
        c2_add_coeff_ = other.c2_add_coeff_;
        c_add_coeff_ = other.c_add_coeff_;
        c1_ = other.c1_;
        c2_ = other.c2_;
        c3_ = other.c3_;
        c4_ = other.c4_;
        c5_ = other.c5_;
        c_ = other.c_;
        return *this;
    }
    double prob[RNG_POISSON_N_PRECOMPUTED_CDF];
    double floored_lambda_ = 0.0;
    double log_lambda_ = 0.0;
    double lgamma_floored_lambda_ = 0.0;
    double sqrt_lambda_ = 0.0;
    double sqrt_floored_lambda_ = 0.0;
    double dx_ = 0.0;
    double delta_ = 0.0;
    double dpdfl_ = 0.0;
    double sqrt_half_dpdfl_ = 0.0;
    double inv_dpdfl_ = 0.0;
    double c2_add_coeff_ = 0.0;
    double c_add_coeff_ = 0.0;
    double c1_ = 0.0;
    double c2_ = 0.0;
    double c3_ = 0.0;
    double c4_ = 0.0;
    double c5_ = 0.0;
    double c_ = 0.0;
    const double exp_one_by_78 = 1.0129030479320018583185514777512983L;
    const double pi_4_ = 0.7853981633974483096156608458198757L;
    const double spi_2_ = 1.2533141373155002512078826424055226L;
};

template <typename IntType>
class distribution_base<oneapi::math::rng::device::poisson<IntType, poisson_method::devroye>> {
public:
    struct param_type {
        param_type(double lambda) : lambda_(lambda) {}
        double lambda_;
    };

    distribution_base(double lambda) : lambda_(lambda) {
#ifndef __SYCL_DEVICE_ONLY__
        if (lambda_ <= 0.0) {
            throw oneapi::math::invalid_argument("rng", "poisson", "lambda <= 0");
        }
#endif
        params_.set_lambda(lambda_);
    }

    double lambda() const {
        return lambda_;
    }

    param_type param() const {
        return param_type(lambda_);
    }

    void param(const param_type& pt) {
#ifndef __SYCL_DEVICE_ONLY__
        if (pt.lambda_ <= 0.0) {
            throw oneapi::math::invalid_argument("rng", "poisson", "lambda <= 0");
        }
#endif
        lambda_ = pt.lambda_;
        params_.set_lambda(lambda_);
    }

protected:
    IntType get_one_num_small_lambdas(double uniform_var) {
        IntType res = 0;
        if (uniform_var < params_.prob[0]) {
            return res;
        }
        else {
            for (res = 1; res < RNG_POISSON_N_PRECOMPUTED_CDF; ++res) {
                if (uniform_var < params_.prob[res]) {
                    return res;
                }
            }
            // in case uniform_var is still bigger than CDF[31] compute additional CDF coefficients
            double prob_less_than_k = params_.prob[--res];
            double prob_that_k = prob_less_than_k - params_.prob[res - 1];
            do {
                prob_that_k *= lambda_ / (double)(res++ + 1);
                prob_less_than_k += prob_that_k;
            } while (uniform_var >= prob_less_than_k);

            return res;
        }
    }
    template <typename EngineType>
    IntType get_one_num_med_lambdas(EngineType& engine) {
        const double rounding_coeff = (1.0 - std::numeric_limits<double>::epsilon()) / 2.0;
        const double max_inttype_val = (std::numeric_limits<IntType>::max)() + rounding_coeff;
        double res_;
        bool rejection_flag = true;
        do {
            const double uniform_var = params_.c_ * engine.generate_single(0.0, 1.0);
            const double exponential_var = exponential_.generate_single(engine);
            double w = 0.0;
            if (uniform_var <= params_.c1_) {
                const double gaussian_var = gaussian_.generate_single(engine);
                const double y = -sycl::fabs(gaussian_var) * params_.sqrt_floored_lambda_ - 1.0;
                res_ = sycl::floor(y);
                w = -gaussian_var * gaussian_var / 2.0;
                if (res_ < -params_.floored_lambda_)
                    continue;
            }
            else if (uniform_var <= params_.c2_) {
                const double gaussian_var = gaussian_.generate_single(engine);
                const double y = 1.0 + sycl::fabs(gaussian_var) * params_.sqrt_half_dpdfl_;
                res_ = sycl::ceil(y);
                w = y * (2.0 - y) * params_.inv_dpdfl_;
                if (res_ > params_.delta_)
                    continue;
            }
            else if (uniform_var <= params_.c3_)
                res_ = -1.0;
            else if (uniform_var <= params_.c4_)
                res_ = 0.0;
            else if (uniform_var <= params_.c5_)
                res_ = 1.0;
            else {
                const double exponential_var_1 = exponential_.generate_single(engine);
                const double y =
                    params_.delta_ + exponential_var_1 * 2.0 * params_.dpdfl_ / params_.delta_;
                res_ = sycl::ceil(y);
                w = -params_.delta_ * params_.inv_dpdfl_ * (1.0 + y / 2.0);
            }

            rejection_flag = ((w - exponential_var - res_ * params_.log_lambda_) >
                              (params_.lgamma_floored_lambda_ -
                               sycl::lgamma(res_ + params_.floored_lambda_ + 1.0)));

            rejection_flag |= (res_ + params_.floored_lambda_) >= max_inttype_val;

        } while (rejection_flag);

        return ((IntType)(res_ + params_.floored_lambda_ + rounding_coeff));
    }

    template <typename EngineType>
    auto generate(EngineType& engine) ->
        typename std::conditional<EngineType::vec_size == 1, IntType,
                                  sycl::vec<IntType, EngineType::vec_size>>::type {
        using OutType = typename std::conditional<EngineType::vec_size == 1, IntType,
                                                  sycl::vec<IntType, EngineType::vec_size>>::type;
        OutType res;
        if constexpr (EngineType::vec_size == 1) {
            res = 0;
            if (lambda_ < RNG_POISSON_LAMBDA_LOW_BOUND) {
                double uniform_var = engine.generate(0.0, 1.0);
                return get_one_num_small_lambdas(uniform_var);
            }
            else if (lambda_ < RNG_POISSON_LAMBDA_HUGE_BOUND) {
                const double rounding_coeff = (1.0 - std::numeric_limits<double>::epsilon()) / 2.0;
                const double max_inttype_val =
                    (std::numeric_limits<IntType>::max)() + rounding_coeff;
                double res_;
                bool rejection_flag = true;
                do {
                    const double uniform_var = params_.c_ * engine.generate(0.0, 1.0);
                    const double exponential_var = exponential_.generate(engine);
                    double w = 0.0;
                    if (uniform_var <= params_.c1_) {
                        const double gaussian_var = gaussian_.generate(engine);
                        const double y =
                            -sycl::fabs(gaussian_var) * params_.sqrt_floored_lambda_ - 1.0;
                        res_ = sycl::floor(y);
                        w = -gaussian_var * gaussian_var / 2.0;
                        if (res_ < -params_.floored_lambda_)
                            continue;
                    }
                    else if (uniform_var <= params_.c2_) {
                        const double gaussian_var = gaussian_.generate(engine);
                        const double y = 1.0 + sycl::fabs(gaussian_var) * params_.sqrt_half_dpdfl_;
                        res_ = sycl::ceil(y);
                        w = y * (2.0 - y) * params_.inv_dpdfl_;
                        if (res_ > params_.delta_)
                            continue;
                    }
                    else if (uniform_var <= params_.c3_)
                        res_ = -1.0;
                    else if (uniform_var <= params_.c4_)
                        res_ = 0.0;
                    else if (uniform_var <= params_.c5_)
                        res_ = 1.0;
                    else {
                        const double exponential_var_1 = exponential_.generate(engine);
                        const double y = params_.delta_ +
                                         exponential_var_1 * 2.0 * params_.dpdfl_ / params_.delta_;
                        res_ = sycl::ceil(y);
                        w = -params_.delta_ * params_.inv_dpdfl_ * (1.0 + y / 2.0);
                    }

                    rejection_flag = ((w - exponential_var - res_ * params_.log_lambda_) >
                                      (params_.lgamma_floored_lambda_ -
                                       sycl::lgamma(res_ + params_.floored_lambda_ + 1.0)));

                    rejection_flag |= (res_ + params_.floored_lambda_) >= max_inttype_val;

                } while (rejection_flag);

                return ((IntType)(res_ + params_.floored_lambda_ + rounding_coeff));
            }
            else {
                res = static_cast<IntType>(lambda_ +
                                           params_.sqrt_lambda_ * gaussian_.generate(engine));
            }
        }
        else {
            if (lambda_ < RNG_POISSON_LAMBDA_LOW_BOUND) {
                auto uniform_var = engine.generate(0.0, 1.0);
                for (int i = 0; i < EngineType::vec_size; ++i) {
                    res[i] = get_one_num_small_lambdas(uniform_var[i]);
                }
                return res;
            }
            else if (lambda_ < RNG_POISSON_LAMBDA_HUGE_BOUND) {
                for (int i = 0; i < EngineType::vec_size; ++i) {
                    res[i] = get_one_num_med_lambdas(engine);
                }
                return res;
            }
            else {
                sycl::vec<double, EngineType::vec_size> res_fp =
                    lambda_ + params_.sqrt_lambda_ * gaussian_.generate(engine);
                res_fp = sycl::floor(res_fp);
                res = res_fp.template convert<IntType>();
            }
        }
        return res;
    }

    template <typename EngineType>
    IntType generate_single(EngineType& engine) {
        IntType res = 0;
        if (lambda_ < RNG_POISSON_LAMBDA_LOW_BOUND) {
            double uniform_var = engine.generate_single(0.0, 1.0);
            return get_one_num_small_lambdas(uniform_var);
        }
        else if (lambda_ < RNG_POISSON_LAMBDA_HUGE_BOUND) {
            return get_one_num_med_lambdas(engine);
        }
        else {
            res = static_cast<IntType>(lambda_ +
                                       params_.sqrt_lambda_ * gaussian_.generate_single(engine));
        }
        return res;
    }

    distribution_base<oneapi::math::rng::device::gaussian<double>> gaussian_ = { 0.0, 1.0 };
    distribution_base<oneapi::math::rng::device::exponential<double>> exponential_ = { 0.0, 1.0 };
    poisson_parameters params_;
    double lambda_;
};

} // namespace oneapi::math::rng::device::detail

#endif // ONEMATH_RNG_DEVICE_POISSON_IMPL_HPP_
