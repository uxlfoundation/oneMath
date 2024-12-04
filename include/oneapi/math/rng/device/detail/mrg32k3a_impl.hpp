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

// References:
// [1] Bradley, Thomas & du Toit, Jacques & Giles, Mike & Tong, Robert & Woodhams, Paul.
// (2011). Parallelisation Techniques for Random Number Generators.
// GPU Computing Gems Emerald Edition. 10.1016/B978-0-12-384988-5.00016-4

#ifndef ONEMATH_RNG_DEVICE_MRG32K3A_IMPL_HPP_
#define ONEMATH_RNG_DEVICE_MRG32K3A_IMPL_HPP_

#include "oneapi/math/rng/device/detail/mrg32k3a_skip_ahead_matrix.hpp"

namespace oneapi::math::rng::device {

template <std::int32_t VecSize = 1>
class mrg32k3a;

namespace detail {

template <std::int32_t VecSize>
struct engine_state<oneapi::math::rng::device::mrg32k3a<VecSize>> {
    std::uint32_t s[6];
};

namespace mrg32k3a_impl {

struct mrg32k3a_params {
    static constexpr std::uint32_t m1 = 4294967087;
    static constexpr std::uint32_t m2 = 4294944443;
    static constexpr std::uint32_t a12 = 1403580;
    static constexpr std::uint32_t a13 = 4294156359;
    static constexpr std::uint32_t a21 = 527612;
    static constexpr std::uint32_t a23 = 4293573854;
    static constexpr std::uint32_t a13n = 810728;
    static constexpr std::uint32_t a23n = 1370589;
};

template <std::uint32_t M>
struct two_pow_32_minus_m {};

template <>
struct two_pow_32_minus_m<mrg32k3a_params::m1> {
    static constexpr std::int64_t val = 209;
};

template <>
struct two_pow_32_minus_m<mrg32k3a_params::m2> {
    static constexpr std::int64_t val = 22853;
};

template <std::int64_t M, typename T>
static inline void bit_shift_and_mask(T& in) {
    T mask;
    if constexpr (std::is_same_v<T, std::uint64_t>) {
        mask = 0x00000000ffffffffu;
    }
    else {
        mask = 0x00000000ffffffff;
    }
    in = ((in >> 32) * two_pow_32_minus_m<M>::val + (in & mask));
}

template <std::uint32_t M>
static inline void matr3x3_vec_mul_mod(std::uint32_t a[3][3], std::uint32_t x[3],
                                       std::uint32_t y[3]) {
    std::uint64_t temp[3] = { 0ull, 0ull, 0ull };
    for (int i = 0; i < 3; ++i) {
        for (int k = 0; k < 3; ++k) {
            std::uint64_t tmp =
                static_cast<std::uint64_t>(a[i][k]) * static_cast<std::uint64_t>(x[k]);
            bit_shift_and_mask<M>(tmp);
            bit_shift_and_mask<M>(tmp);
            if (tmp >= M) {
                tmp -= M;
            }
            temp[i] += tmp;
        }
        bit_shift_and_mask<M>(temp[i]);
        if (temp[i] >= M) {
            temp[i] -= M;
        }
    }

    for (int k = 0; k < 3; k++) {
        y[k] = static_cast<std::uint32_t>(temp[k]);
    }

    return;
}

template <std::uint32_t M>
static inline void matr3x3_mul_mod(std::uint32_t B[3][3],
                                   const std::uint32_t _skip_ahead_matrix[3][3]) {
    std::uint64_t temp[3][3] = { { 0ull, 0ull, 0ull }, { 0ull, 0ull, 0ull }, { 0ull, 0ull, 0ull } };

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                std::uint64_t tmp = static_cast<std::uint64_t>(B[i][k]) *
                                    static_cast<std::uint64_t>(_skip_ahead_matrix[k][j]);
                bit_shift_and_mask<M>(tmp);
                if constexpr (mrg32k3a_params::m2 == M) {
                    bit_shift_and_mask<M>(tmp);
                }
                if (tmp >= M) {
                    tmp -= M;
                }
                temp[i][j] += tmp;
            }
            bit_shift_and_mask<M>(temp[i][j]);
            if (temp[i][j] >= M) {
                temp[i][j] -= M;
            }
        }
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            B[i][j] = static_cast<std::uint32_t>(temp[i][j]);
        }
    }
}

template <std::uint32_t M>
static inline void vec3_pow_mod(
    std::uint32_t x[3], std::uint64_t n, const std::uint64_t* skip_params,
    const std::uint32_t _skip_ahead_matrix[quantity_of_3x3_matrices][3][3]) {
    std::uint32_t B[3][3] = { { 1u, 0u, 0u }, { 0u, 1u, 0u }, { 0u, 0u, 1u } };

    std::uint32_t off;
    std::uint32_t mod;
    std::uint64_t skip_param;
    std::uint32_t bit_count = 0; // can be 0, 1, 2
    std::uint32_t bit_count_tmp;

    for (std::uint32_t j = 0; j < n; j++) {
        skip_param = skip_params[j];
        off = 0;
        bit_count_tmp = bit_count;
        while (skip_param) {
            // we have to multiply skip_param[1] by 2 and skip_params[2] by 4 only for the 1st iteration
            // of the loop to get the required power of a power-of-eight matrice from a power of two
            mod = (skip_param << static_cast<std::uint64_t>(bit_count_tmp)) &
                  7ull; // == (skip_param * _mult) % 8, _mult={1,2,4}
            if (mod) {
                // 7 - number of 3x3 matrices of some power of 8: 1*8^x, 2*8^x, ..., 7*8^x
                // 7 * 21 - number of 3x3 matrices for each skip parameter
                matr3x3_mul_mod<M>(B, _skip_ahead_matrix[7 * 21 * j + off * 7 + (mod - 1)]);
            }
            skip_param =
                skip_param /
                (8ull >> static_cast<std::uint64_t>(bit_count_tmp)); // == skip_param / (8 / _mult)
            ++off;
            bit_count_tmp = 0;
        }
        ++bit_count;
    }
    matr3x3_vec_mul_mod<M>(B, x, x);
}

template <std::int32_t VecSize>
static inline void skip_ahead(engine_state<oneapi::math::rng::device::mrg32k3a<VecSize>>& state,
                              std::uint64_t n, const std::uint64_t* num_to_skip_ptr) {
    if (n > 3) {
        n = 3;
#ifndef __SYCL_DEVICE_ONLY__
        throw oneapi::math::invalid_argument("rng", "mrg32k3a",
                                             "period is 2 ^ 191, skip on more than 2^192");
#endif
    }
    vec3_pow_mod<mrg32k3a_params::m1>(state.s, n, num_to_skip_ptr, skip_ahead_matrix[0]);
    vec3_pow_mod<mrg32k3a_params::m2>(state.s + 3, n, num_to_skip_ptr, skip_ahead_matrix[1]);
}

template <std::int32_t VecSize>
static inline void validate_seed(
    engine_state<oneapi::math::rng::device::mrg32k3a<VecSize>>& state) {
    int i;
    for (i = 0; i < 3; i++) {
        if (state.s[i] >= mrg32k3a_params::m1) {
            state.s[i] -= mrg32k3a_params::m1;
        }
    }
    for (; i < 6; i++) {
        if (state.s[i] >= mrg32k3a_params::m2) {
            state.s[i] -= mrg32k3a_params::m2;
        }
    }

    if ((state.s[0]) == 0 && (state.s[1]) == 0 && (state.s[2]) == 0) {
        state.s[0] = 1;
    }
    if ((state.s[3]) == 0 && (state.s[4]) == 0 && (state.s[5]) == 0) {
        state.s[3] = 1;
    }
}

template <std::int32_t VecSize>
static inline void init(engine_state<oneapi::math::rng::device::mrg32k3a<VecSize>>& state,
                        std::uint64_t n, const std::uint32_t* seed_ptr, std::uint64_t n_offset,
                        const std::uint64_t* offset_ptr) {
    std::uint64_t i;
    if (n > 6) {
        n = 6;
    }
    for (i = 0; i < n; i++) {
        state.s[i] = seed_ptr[i];
    }
    for (; i < 6; i++) {
        state.s[i] = 1;
    }
    validate_seed(state);
    mrg32k3a_impl::skip_ahead(state, n_offset, offset_ptr);
}

template <std::int32_t VecSize>
static inline sycl::vec<std::uint32_t, VecSize> generate(
    engine_state<oneapi::math::rng::device::mrg32k3a<VecSize>>& state) {
    const std::int32_t num_elements = VecSize;
    sycl::vec<std::uint32_t, VecSize> res;
    std::int64_t x, y;
    std::int32_t i = 0;
    for (i = 0; i < num_elements; i++) {
        x = mrg32k3a_params::a12 * static_cast<std::int64_t>(state.s[1]) -
            mrg32k3a_params::a13n * static_cast<std::int64_t>(state.s[0]);
        // perform modulus
        bit_shift_and_mask<mrg32k3a_params::m1>(x);
        if (x >= mrg32k3a_params::m1)
            x -= mrg32k3a_params::m1;
        x += ((x & 0x8000000000000000) >> 63) * mrg32k3a_params::m1;
        y = mrg32k3a_params::a21 * static_cast<std::int64_t>(state.s[5]) -
            mrg32k3a_params::a23n * static_cast<std::int64_t>(state.s[3]);
        // perform modulus
        bit_shift_and_mask<mrg32k3a_params::m2>(y);
        bit_shift_and_mask<mrg32k3a_params::m2>(y);
        if (y >= mrg32k3a_params::m2)
            y -= mrg32k3a_params::m2;
        y += ((y & 0x8000000000000000) >> 63) * mrg32k3a_params::m2;
        state.s[0] = state.s[1];
        state.s[1] = state.s[2];
        state.s[2] = x;
        state.s[3] = state.s[4];
        state.s[4] = state.s[5];
        state.s[5] = y;
        if (x <= y) {
            res[i] = x + (mrg32k3a_params::m1 - y);
        }
        else {
            res[i] = x - y;
        }
    }
    return res;
}

template <std::int32_t VecSize>
static inline std::uint32_t generate_single(
    engine_state<oneapi::math::rng::device::mrg32k3a<VecSize>>& state) {
    std::uint32_t res;
    std::int64_t x, y;
    x = mrg32k3a_params::a12 * static_cast<std::int64_t>(state.s[1]) -
        mrg32k3a_params::a13n * static_cast<std::int64_t>(state.s[0]);
    // perform modulus
    bit_shift_and_mask<mrg32k3a_params::m1>(x);
    if (x >= mrg32k3a_params::m1)
        x -= mrg32k3a_params::m1;
    x += ((x & 0x8000000000000000) >> 63) * mrg32k3a_params::m1;
    y = mrg32k3a_params::a21 * static_cast<std::int64_t>(state.s[5]) -
        mrg32k3a_params::a23n * static_cast<std::int64_t>(state.s[3]);
    // perform modulus
    bit_shift_and_mask<mrg32k3a_params::m2>(y);
    bit_shift_and_mask<mrg32k3a_params::m2>(y);
    if (y >= mrg32k3a_params::m2)
        y -= mrg32k3a_params::m2;
    y += ((y & 0x8000000000000000) >> 63) * mrg32k3a_params::m2;
    state.s[0] = state.s[1];
    state.s[1] = state.s[2];
    state.s[2] = x;
    state.s[3] = state.s[4];
    state.s[4] = state.s[5];
    state.s[5] = y;
    if (x <= y) {
        res = x + (mrg32k3a_params::m1 - y);
    }
    else {
        res = x - y;
    }

    return res;
}

} // namespace mrg32k3a_impl

template <std::int32_t VecSize>
class engine_base<oneapi::math::rng::device::mrg32k3a<VecSize>> {
protected:
    engine_base(std::uint32_t seed, std::uint64_t offset = 0) {
        mrg32k3a_impl::init(this->state_, 1, &seed, 1, &offset);
    }

    engine_base(std::uint64_t n, const std::uint32_t* seed, std::uint64_t offset = 0) {
        mrg32k3a_impl::init(this->state_, n, seed, 1, &offset);
    }

    engine_base(std::uint32_t seed, std::uint64_t n_offset, const std::uint64_t* offset_ptr) {
        mrg32k3a_impl::init(this->state_, 1, &seed, n_offset, offset_ptr);
    }

    engine_base(std::uint64_t n, const std::uint32_t* seed, std::uint64_t n_offset,
                const std::uint64_t* offset_ptr) {
        mrg32k3a_impl::init(this->state_, n, seed, n_offset, offset_ptr);
    }

    template <typename RealType>
    auto generate(RealType a, RealType b) ->
        typename std::conditional<VecSize == 1, RealType, sycl::vec<RealType, VecSize>>::type {
        sycl::vec<RealType, VecSize> res;
        sycl::vec<std::uint32_t, VecSize> res_uint;
        RealType c;

        c = (b - a) / (static_cast<RealType>(mrg32k3a_impl::mrg32k3a_params::m1));

        res_uint = mrg32k3a_impl::generate(this->state_);

        for (int i = 0; i < VecSize; i++) {
            res[i] = (RealType)(res_uint[i]) * c + a;
        }
        return res;
    }

    auto generate() -> typename std::conditional<VecSize == 1, std::uint32_t,
                                                 sycl::vec<std::uint32_t, VecSize>>::type {
        return mrg32k3a_impl::generate(this->state_);
    }

    template <typename RealType>
    RealType generate_single(RealType a, RealType b) {
        RealType res;
        std::uint32_t res_uint;
        RealType c;

        c = (b - a) / (static_cast<RealType>(mrg32k3a_impl::mrg32k3a_params::m1));

        res_uint = mrg32k3a_impl::generate_single(this->state_);

        res = (RealType)(res_uint)*c + a;

        return res;
    }

    std::uint32_t generate_single() {
        return mrg32k3a_impl::generate_single(this->state_);
    }

    void skip_ahead(std::uint64_t num_to_skip) {
        detail::mrg32k3a_impl::skip_ahead(this->state_, 1, &num_to_skip);
    }

    void skip_ahead(std::initializer_list<std::uint64_t> num_to_skip) {
        detail::mrg32k3a_impl::skip_ahead(this->state_, num_to_skip.size(), num_to_skip.begin());
    }

    engine_state<oneapi::math::rng::device::mrg32k3a<VecSize>> state_;
};

} // namespace detail
} // namespace oneapi::math::rng::device

#endif // ONEMATH_RNG_DEVICE_MRG32K3A_IMPL_HPP_
