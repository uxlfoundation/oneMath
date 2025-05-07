/*******************************************************************************
* Copyright (C) 2025 Intel Corporation
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

#ifndef ONEMATH_RNG_DEVICE_PCG64_DXSM_IMPL_HPP_
#define ONEMATH_RNG_DEVICE_PCG64_DXSM_IMPL_HPP_

namespace oneapi::math::rng::device {

template <std::int32_t VecSize = 1>
class pcg64_dxsm;

namespace detail {

struct pcg128_t {
    std::uint64_t high;
    std::uint64_t low;
};

struct pcg64_dxsm_param {
    static constexpr std::uint64_t cheap_multiplier = 0xda942042e4dd58b5ULL;
    static constexpr pcg128_t increment = { 0x5851f42d4c957f2dULL, 0x14057b7ef767814fULL };
};

template <std::int32_t VecSize>
struct engine_state<oneapi::math::rng::device::pcg64_dxsm<VecSize>> {
    pcg128_t s;
    std::uint32_t result;
    bool has_32;
};

namespace pcg64_dxsm_impl {

static inline pcg128_t mul_64x64(std::uint64_t x, std::uint64_t y) {
    const std::uint64_t x_lo = x & 0xFFFFFFFFULL;
    const std::uint64_t x_hi = x >> 32;
    const std::uint64_t y_lo = y & 0xFFFFFFFFULL;
    const std::uint64_t y_hi = y >> 32;

    const std::uint64_t xy_md = x_hi * y_lo;
    const std::uint64_t yx_md = y_hi * x_lo;

    return { (x_hi * y_hi) + (xy_md >> 32) + (yx_md >> 32) +
                 (((xy_md & 0xFFFFFFFFULL) + (yx_md & 0xFFFFFFFFULL) + ((x_lo * y_lo) >> 32)) >>
                  32),
             x * y };
}

static inline pcg128_t mul_lo_128x128(pcg128_t x, pcg128_t y) {
    pcg128_t result = mul_64x64(x.low, y.low);
    result.high += (x.high * y.low) + (x.low * y.high);
    return result;
}

static inline pcg128_t mul_lo_128x64(pcg128_t x, std::uint64_t y) {
    std::uint64_t x_high = x.high;
    x = mul_64x64(x.low, y);
    x.high += x_high * y;
    return x;
}

static inline pcg128_t add_128x128(pcg128_t x, pcg128_t y) {
    x.low += y.low;
    x.high += (y.high + (x.low < y.low));
    return x;
}

static inline std::uint64_t prepare_output(const pcg128_t local_state) {
    std::uint64_t tmp = local_state.high;

    tmp ^= tmp >> 32;
    tmp *= pcg64_dxsm_param::cheap_multiplier;
    tmp ^= tmp >> 48;
    tmp *= (local_state.low | 1);

    return tmp;
}

static inline void update_state(pcg128_t& local_state) {
    local_state = add_128x128(pcg64_dxsm_param::increment,
                              mul_lo_128x64(local_state, pcg64_dxsm_param::cheap_multiplier));
}

template <std::int32_t VecSize>
static inline void bump32(engine_state<oneapi::math::rng::device::pcg64_dxsm<VecSize>>& state) {
    if (state.has_32) {
        state.has_32 = false;
    }
    else {
        std::uint64_t tmp = prepare_output(state.s);
        state.result = static_cast<std::uint32_t>(tmp >> 32);

        update_state(state.s);
        state.has_32 = true;
    }
}

template <std::int32_t VecSize>
static inline void skip_ahead(engine_state<oneapi::math::rng::device::pcg64_dxsm<VecSize>>& state,
                              pcg128_t num_to_skip) {
    pcg128_t acc_mul = { 0ULL, 1ULL };
    pcg128_t acc_inc = { 0ULL, 0ULL };
    pcg128_t tmp_mul = { 0ULL, pcg64_dxsm_param::cheap_multiplier };
    pcg128_t tmp_inc = pcg64_dxsm_param::increment;
    bool is_skip_odd = num_to_skip.low & 1;
    num_to_skip.low >>= 1;
    num_to_skip.low |= num_to_skip.high << 63;
    num_to_skip.high >>= 1;

    while (num_to_skip.low || num_to_skip.high) {
        if (num_to_skip.low & 1) {
            acc_mul = mul_lo_128x128(acc_mul, tmp_mul);
            acc_inc = add_128x128(mul_lo_128x128(acc_inc, tmp_mul), tmp_inc);
        }

        tmp_inc = mul_lo_128x128(add_128x128(tmp_mul, { 0, 1 }), tmp_inc);
        tmp_mul = mul_lo_128x128(tmp_mul, tmp_mul);

        num_to_skip.low >>= 1;
        num_to_skip.low |= num_to_skip.high << 63;
        num_to_skip.high >>= 1;
    }

    state.s = add_128x128(mul_lo_128x128(state.s, acc_mul), acc_inc);

    if (is_skip_odd) {
        bump32(state);
    }
}

template <std::int32_t VecSize>
static inline void skip_ahead(engine_state<oneapi::math::rng::device::pcg64_dxsm<VecSize>>& state,
                              std::uint64_t num_to_skip) {
    pcg128_t acc_mul = { 0ULL, 1ULL };
    pcg128_t acc_inc = { 0ULL, 0ULL };
    pcg128_t tmp_mul = { 0ULL, pcg64_dxsm_param::cheap_multiplier };
    pcg128_t tmp_inc = pcg64_dxsm_param::increment;
    bool is_skip_odd = num_to_skip & 1;
    num_to_skip >>= 1;

    while (num_to_skip) {
        if (num_to_skip & 1) {
            acc_mul = mul_lo_128x128(acc_mul, tmp_mul);
            acc_inc = add_128x128(mul_lo_128x128(acc_inc, tmp_mul), tmp_inc);
        }

        tmp_inc = mul_lo_128x128(add_128x128(tmp_mul, { 0, 1 }), tmp_inc);
        tmp_mul = mul_lo_128x128(tmp_mul, tmp_mul);

        num_to_skip >>= 1;
    }
    state.s = add_128x128(mul_lo_128x128(state.s, acc_mul), acc_inc);

    if (is_skip_odd) {
        bump32(state);
    }
}

template <std::int32_t VecSize>
static inline void init(engine_state<oneapi::math::rng::device::pcg64_dxsm<VecSize>>& state,
                        std::uint64_t n, const std::uint64_t* seed_ptr, std::uint64_t n_offset,
                        const std::uint64_t* offset_ptr) {
    state.s.low = seed_ptr[0];
    state.s.high = (n > 1 ? seed_ptr[1] : 0ULL);

    state.s = add_128x128(state.s, pcg64_dxsm_param::increment);
    state.s = add_128x128(pcg64_dxsm_param::increment,
                          mul_lo_128x64(state.s, pcg64_dxsm_param::cheap_multiplier));

    if (n_offset > 1)
        pcg64_dxsm_impl::skip_ahead(state, { offset_ptr[1], offset_ptr[0] });
    else
        pcg64_dxsm_impl::skip_ahead(state, offset_ptr[0]);

    state.has_32 = false;
}

template <std::int32_t VecSize>
static inline sycl::vec<std::uint64_t, VecSize> generate64(
    engine_state<oneapi::math::rng::device::pcg64_dxsm<VecSize>>& state) {
    sycl::vec<std::uint64_t, VecSize> res;

    for (int i = 0; i < VecSize; i++) {
        res[i] = prepare_output(state.s);
        update_state(state.s);
    }

    return res;
}

template <std::int32_t VecSize>
static inline sycl::vec<std::uint32_t, VecSize> generate32_even(
    engine_state<oneapi::math::rng::device::pcg64_dxsm<VecSize>>& state) {
    sycl::vec<std::uint32_t, VecSize> res;
    std::uint64_t tmp;

    for (int i = 0; i < (VecSize / 2); i++) {
        tmp = prepare_output(state.s);
        res[2 * i] = static_cast<std::uint32_t>(tmp);
        res[(2 * i) + 1] = static_cast<std::uint32_t>(tmp >> 32);

        update_state(state.s);
    }

    return res;
}

template <std::int32_t VecSize>
static inline sycl::vec<std::uint32_t, VecSize> generate32_odd(
    engine_state<oneapi::math::rng::device::pcg64_dxsm<VecSize>>& state) {
    sycl::vec<std::uint32_t, VecSize> res;

    if constexpr (VecSize == 1) {
        if (state.has_32) {
            res[0] = state.result;
            state.has_32 = false;
        }
        else {
            std::uint64_t tmp = prepare_output(state.s);
            res[0] = static_cast<std::uint32_t>(tmp);
            state.result = static_cast<std::uint32_t>(tmp >> 32);

            update_state(state.s);
            state.has_32 = true;
        }
    }
    else {
        std::uint64_t tmp;
        int shift = (state.has_32 ? 1 : 0);

        tmp = prepare_output(state.s);
        res[shift] = static_cast<std::uint32_t>(tmp);
        res[shift + 1] = static_cast<std::uint32_t>(tmp >> 32);

        update_state(state.s);

        if (state.has_32) {
            res[0] = state.result;
            state.has_32 = false;
        }
        else {
            tmp = prepare_output(state.s);

            res[2] = static_cast<std::uint32_t>(tmp);
            state.result = static_cast<std::uint32_t>(tmp >> 32);

            update_state(state.s);
            state.has_32 = true;
        }
    }

    return res;
}

template <std::int32_t VecSize>
static inline sycl::vec<std::uint32_t, VecSize> generate32(
    engine_state<oneapi::math::rng::device::pcg64_dxsm<VecSize>>& state) {
    if constexpr (VecSize % 2 == 1)
        return generate32_odd<VecSize>(state);
    else
        return generate32_even<VecSize>(state);
}

template <std::int32_t VecSize>
static inline std::uint64_t generate_single64(
    engine_state<oneapi::math::rng::device::pcg64_dxsm<VecSize>>& state) {
    std::uint64_t res = prepare_output(state.s);
    update_state(state.s);
    return res;
}

template <std::int32_t VecSize>
static inline std::uint64_t generate_single32(
    engine_state<oneapi::math::rng::device::pcg64_dxsm<VecSize>>& state) {
    std::uint32_t res;

    if (state.has_32) {
        res = state.result;
        state.has_32 = false;
    }
    else {
        std::uint64_t tmp = prepare_output(state.s);

        res = static_cast<std::uint32_t>(tmp);
        state.result = static_cast<std::uint32_t>(tmp >> 32);

        update_state(state.s);
        state.has_32 = true;
    }

    return res;
}

} // namespace pcg64_dxsm_impl

template <std::int32_t VecSize>
class engine_base<oneapi::math::rng::device::pcg64_dxsm<VecSize>> {
protected:
    engine_base(std::uint64_t seed, std::uint64_t offset = 0) {
        pcg64_dxsm_impl::init(this->state_, 1, &seed, 1, &offset);
    }

    engine_base(std::uint64_t n, const std::uint64_t* seed, std::uint64_t offset = 0) {
        pcg64_dxsm_impl::init(this->state_, n, seed, 1, &offset);
    }

    engine_base(std::uint64_t seed, std::uint64_t n_offset, const std::uint64_t* offset_ptr) {
        pcg64_dxsm_impl::init(this->state_, 1, &seed, n_offset, offset_ptr);
    }

    engine_base(std::uint64_t n, const std::uint64_t* seed, std::uint64_t n_offset,
                const std::uint64_t* offset_ptr) {
        pcg64_dxsm_impl::init(this->state_, n, seed, n_offset, offset_ptr);
    }

    template <typename RealType>
    inline auto generate(RealType a, RealType b) ->
        typename std::conditional<VecSize == 1, RealType, sycl::vec<RealType, VecSize>>::type {
        sycl::vec<RealType, VecSize> res;
        sycl::vec<std::uint32_t, VecSize> res_uint;
        RealType a1 = (b + a) / static_cast<RealType>(2.0);
        RealType c1 =
            (b - a) / (static_cast<RealType>((std::numeric_limits<std::uint32_t>::max)()) + 1);

        res_uint = pcg64_dxsm_impl::generate32(this->state_);
        for (int i = 0; i < VecSize; i++) {
            res[i] = static_cast<RealType>(static_cast<std::int32_t>(res_uint[i])) * c1 + a1;
        }
        return res;
    }

    inline auto generate() -> typename std::conditional<VecSize == 1, std::uint32_t,
                                                        sycl::vec<std::uint32_t, VecSize>>::type {
        return pcg64_dxsm_impl::generate32(this->state_);
    }

    template <typename UIntType>
    inline auto generate_uniform_bits() ->
        typename std::conditional<VecSize == 1, UIntType, sycl::vec<UIntType, VecSize>>::type {
        if constexpr (std::is_same<UIntType, std::uint32_t>::value) {
            return pcg64_dxsm_impl::generate32(this->state_);
        }
        else {
            return pcg64_dxsm_impl::generate64(this->state_);
        }
    }

    template <typename RealType>
    inline RealType generate_single(RealType a, RealType b) {
        std::uint32_t res_uint;
        RealType res;
        RealType a1 = (b + a) / static_cast<RealType>(2.0);
        RealType c1 =
            (b - a) / (static_cast<RealType>((std::numeric_limits<std::uint32_t>::max)()) + 1);

        res_uint = pcg64_dxsm_impl::generate_single32(this->state_);

        res = static_cast<RealType>(static_cast<std::int32_t>(res_uint)) * c1 + a1;

        return res;
    }

    inline std::uint32_t generate_single() {
        return pcg64_dxsm_impl::generate_single32(this->state_);
    }

    template <typename UIntType>
    inline auto generate_single_uniform_bits() {
        if constexpr (std::is_same<UIntType, std::uint32_t>::value) {
            return pcg64_dxsm_impl::generate_single32(this->state_);
        }
        else {
            return pcg64_dxsm_impl::generate_single64(this->state_);
        }
    }

    void skip_ahead(std::uint64_t num_to_skip) {
        detail::pcg64_dxsm_impl::skip_ahead(this->state_, num_to_skip);
    }

    void skip_ahead(std::initializer_list<std::uint64_t> num_to_skip) {
        if (num_to_skip.size() > 1) {
            detail::pcg64_dxsm_impl::skip_ahead(this->state_,
                                                { num_to_skip.begin()[1], num_to_skip.begin()[0] });
        }
        else {
            detail::pcg64_dxsm_impl::skip_ahead(this->state_, num_to_skip.begin()[0]);
        }
    }

    engine_state<oneapi::math::rng::device::pcg64_dxsm<VecSize>> state_;
    friend class oneapi::math::rng::device::count_engine_adaptor<
        oneapi::math::rng::device::pcg64_dxsm<VecSize>>;
};

} // namespace detail
} // namespace oneapi::math::rng::device

#endif // ONEMATH_RNG_DEVICE_PCG64_DXSM_IMPL_HPP_
