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

#ifndef ONEMATH_RNG_DEVICE_PHILOX4X32X10_IMPL_HPP_
#define ONEMATH_RNG_DEVICE_PHILOX4X32X10_IMPL_HPP_

#include <utility> // std::pair

namespace oneapi::math::rng::device {

template <std::int32_t VecSize = 1>
class philox4x32x10;

namespace detail {

template <std::int32_t VecSize>
struct engine_state<oneapi::math::rng::device::philox4x32x10<VecSize>> {
    std::uint32_t key[2];
    std::uint32_t counter[4];
    std::uint32_t part;
    std::uint32_t result[4];
};

namespace philox4x32x10_impl {

static inline void add128(std::uint32_t* a, std::uint64_t b) {
    std::uint64_t tmp = ((static_cast<std::uint64_t>(a[1]) << 32) | a[0]);

    tmp += b;

    a[0] = static_cast<std::uint32_t>(tmp);
    a[1] = static_cast<std::uint32_t>(tmp >> 32);

    if (tmp < b) {
        tmp = ((static_cast<std::uint64_t>(a[3]) << 32) | a[2]) + 1;

        a[2] = static_cast<std::uint32_t>(tmp);
        a[3] = static_cast<std::uint32_t>(tmp >> 32);
    }
    return;
}

static inline void add128_1(std::uint32_t* a) {
    if (++a[0]) {
        return;
    }
    if (++a[1]) {
        return;
    }
    if (++a[2]) {
        return;
    }
    ++a[3];
}

static inline std::pair<std::uint32_t, std::uint32_t> mul_hilo_32(std::uint32_t a,
                                                                  std::uint32_t b) {
    std::uint64_t res_64 = static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b);
    return std::make_pair(static_cast<std::uint32_t>(res_64),
                          static_cast<std::uint32_t>(res_64 >> 32));
}

static inline void round(std::uint32_t* cnt, std::uint32_t* k) {
    auto [L0, H0] = mul_hilo_32(0xD2511F53, cnt[0]);
    auto [L1, H1] = mul_hilo_32(0xCD9E8D57, cnt[2]);

    cnt[0] = H1 ^ cnt[1] ^ k[0];
    cnt[1] = L1;
    cnt[2] = H0 ^ cnt[3] ^ k[1];
    cnt[3] = L0;
}

static inline void round_10(std::uint32_t* cnt, std::uint32_t* k) {
    round(cnt, k); // 1
    // increasing keys with philox4x32x10 constants
    k[0] += 0x9E3779B9;
    k[1] += 0xBB67AE85;
    round(cnt, k); // 2
    k[0] += 0x9E3779B9;
    k[1] += 0xBB67AE85;
    round(cnt, k); // 3
    k[0] += 0x9E3779B9;
    k[1] += 0xBB67AE85;
    round(cnt, k); // 4
    k[0] += 0x9E3779B9;
    k[1] += 0xBB67AE85;
    round(cnt, k); // 5
    k[0] += 0x9E3779B9;
    k[1] += 0xBB67AE85;
    round(cnt, k); // 6
    k[0] += 0x9E3779B9;
    k[1] += 0xBB67AE85;
    round(cnt, k); // 7
    k[0] += 0x9E3779B9;
    k[1] += 0xBB67AE85;
    round(cnt, k); // 8
    k[0] += 0x9E3779B9;
    k[1] += 0xBB67AE85;
    round(cnt, k); // 9
    k[0] += 0x9E3779B9;
    k[1] += 0xBB67AE85;
    round(cnt, k); // 10
}

template <std::int32_t VecSize>
static inline void skip_ahead(
    engine_state<oneapi::math::rng::device::philox4x32x10<VecSize>>& state,
    std::uint64_t num_to_skip) {
    std::uint64_t num_to_skip_tmp = num_to_skip;
    std::uint64_t c_inc;
    std::uint32_t counter[4];
    std::uint32_t key[2];
    std::uint64_t tail;
    if (num_to_skip_tmp <= state.part) {
        state.part -= num_to_skip_tmp;
    }
    else {
        tail = num_to_skip % 4;
        if ((tail == 0) && (state.part == 0)) {
            add128(state.counter, num_to_skip / 4);
        }
        else {
            num_to_skip_tmp = num_to_skip_tmp - state.part;
            state.part = 0;
            c_inc = (num_to_skip_tmp - 1) / 4;
            state.part = (4 - num_to_skip_tmp % 4) % 4;
            add128(state.counter, c_inc);
            counter[0] = state.counter[0];
            counter[1] = state.counter[1];
            counter[2] = state.counter[2];
            counter[3] = state.counter[3];
            key[0] = state.key[0];
            key[1] = state.key[1];
            round_10(counter, key);
            state.result[0] = counter[0];
            state.result[1] = counter[1];
            state.result[2] = counter[2];
            state.result[3] = counter[3];
            add128_1(state.counter);
        }
    }
}

template <std::int32_t VecSize>
static inline void skip_ahead(
    engine_state<oneapi::math::rng::device::philox4x32x10<VecSize>>& state, std::uint64_t n,
    const std::uint64_t* num_to_skip_ptr) {
    constexpr std::uint64_t uint_max = 0xFFFFFFFFFFFFFFFF;
    std::uint64_t post_buffer, pre_buffer;
    std::int32_t num_elements = 0;
    std::int32_t remained_counter;
    std::uint64_t tmp_skip_array[3] = { 0, 0, 0 };

    for (std::uint64_t i = 0; (i < 3) && (i < n); i++) {
        tmp_skip_array[i] = num_to_skip_ptr[i];
        if (tmp_skip_array[i]) {
            num_elements = i + 1;
        }
    }

    if (num_elements == 0) {
        return;
    }
    if ((num_elements == 1) && (tmp_skip_array[0] <= state.part)) {
        state.part -= static_cast<std::uint32_t>(tmp_skip_array[0]);
        return;
    }
    std::uint32_t counter[4];
    std::uint32_t key[2];

    if ((tmp_skip_array[0] - state.part) <= tmp_skip_array[0]) {
        tmp_skip_array[0] = tmp_skip_array[0] - state.part;
    }
    else if ((num_elements == 2) || (tmp_skip_array[1] - 1 < tmp_skip_array[1])) {
        tmp_skip_array[1] = tmp_skip_array[1] - 1;
        tmp_skip_array[0] = uint_max - state.part + tmp_skip_array[0];
    }
    else {
        tmp_skip_array[2] = tmp_skip_array[2] - 1;
        tmp_skip_array[1] = uint_max - 1;
        tmp_skip_array[0] = uint_max - state.part + tmp_skip_array[0];
    }

    state.part = 0;

    post_buffer = 0;

    remained_counter = static_cast<std::uint32_t>(tmp_skip_array[0] % 4);

    for (int i = num_elements - 1; i >= 0; i--) {
        pre_buffer = (tmp_skip_array[i] << 62);
        tmp_skip_array[i] >>= 2;
        tmp_skip_array[i] |= post_buffer;
        post_buffer = pre_buffer;
    }

    state.part = 4 - remained_counter;

    std::uint64_t counter64[] = { state.counter[1], state.counter[3] };
    counter64[0] = ((counter64[0] << 32ull) | state.counter[0]);
    counter64[1] = ((counter64[1] << 32ull) | state.counter[2]);

    counter64[0] += tmp_skip_array[0];

    if (counter64[0] < tmp_skip_array[0]) {
        counter64[1]++;
    }

    counter64[1] += tmp_skip_array[1];

    counter[0] = static_cast<std::uint32_t>(counter64[0]);
    counter[1] = static_cast<std::uint32_t>(counter64[0] >> 32);
    counter[2] = static_cast<std::uint32_t>(counter64[1]);
    counter[3] = static_cast<std::uint32_t>(counter64[1] >> 32);

    key[0] = state.key[0];
    key[1] = state.key[1];

    round_10(counter, key);

    state.result[0] = counter[0];
    state.result[1] = counter[1];
    state.result[2] = counter[2];
    state.result[3] = counter[3];

    counter64[0]++;

    if (counter64[0] < 1) {
        counter64[1]++;
    }

    state.counter[0] = static_cast<std::uint32_t>(counter64[0]);
    state.counter[1] = static_cast<std::uint32_t>(counter64[0] >> 32);
    state.counter[2] = static_cast<std::uint32_t>(counter64[1]);
    state.counter[3] = static_cast<std::uint32_t>(counter64[1] >> 32);
}

template <std::int32_t VecSize>
static inline void init(engine_state<oneapi::math::rng::device::philox4x32x10<VecSize>>& state,
                        std::uint64_t n, const std::uint64_t* seed_ptr, std::uint64_t offset) {
    state.key[0] = static_cast<std::uint32_t>(seed_ptr[0]);
    state.key[1] = static_cast<std::uint32_t>(seed_ptr[0] >> 32);

    state.counter[0] = (n >= 2 ? static_cast<std::uint32_t>(seed_ptr[1]) : 0);
    state.counter[1] = (n >= 2 ? static_cast<std::uint32_t>(seed_ptr[1] >> 32) : 0);

    state.counter[2] = (n >= 3 ? static_cast<std::uint32_t>(seed_ptr[2]) : 0);
    state.counter[3] = (n >= 3 ? static_cast<std::uint32_t>(seed_ptr[2] >> 32) : 0);

    state.part = 0;
    state.result[0] = 0;
    state.result[1] = 0;
    state.result[2] = 0;
    state.result[3] = 0;
    skip_ahead(state, offset);
}

template <std::int32_t VecSize>
static inline void init(engine_state<oneapi::math::rng::device::philox4x32x10<VecSize>>& state,
                        std::uint64_t n, const std::uint64_t* seed_ptr, std::uint64_t n_offset,
                        const std::uint64_t* offset_ptr) {
    state.key[0] = static_cast<std::uint32_t>(seed_ptr[0]);
    state.key[1] = static_cast<std::uint32_t>(seed_ptr[0] >> 32);

    state.counter[0] = (n >= 2 ? static_cast<std::uint32_t>(seed_ptr[1]) : 0);
    state.counter[1] = (n >= 2 ? static_cast<std::uint32_t>(seed_ptr[1] >> 32) : 0);

    state.counter[2] = (n >= 3 ? static_cast<std::uint32_t>(seed_ptr[2]) : 0);
    state.counter[3] = (n >= 3 ? static_cast<std::uint32_t>(seed_ptr[2] >> 32) : 0);

    state.part = 0;
    state.result[0] = 0;
    state.result[1] = 0;
    state.result[2] = 0;
    state.result[3] = 0;
    skip_ahead(state, n_offset, offset_ptr);
}

// for VecSize > 4
template <std::int32_t VecSize>
__attribute__((always_inline)) static inline sycl::vec<std::uint32_t, VecSize> generate_full(
    engine_state<oneapi::math::rng::device::philox4x32x10<VecSize>>& state) {
    const std::int32_t num_elements = VecSize;
    sycl::vec<std::uint32_t, VecSize> res;

    std::uint32_t counter[4];

    int i = 0;
    int part = (int)state.part;
    while (part && (i < num_elements)) {
        res[i++] = state.result[3 - (--part)];
    }
    if (i == num_elements) {
        skip_ahead(state, num_elements);
        return res;
    }

    counter[0] = state.counter[0];
    counter[1] = state.counter[1];
    counter[2] = state.counter[2];
    counter[3] = state.counter[3];

    std::uint32_t cntTmp[4];
    std::uint32_t keyTmp[2];
    for (; i < num_elements; i += 4) {
        cntTmp[0] = counter[0];
        cntTmp[1] = counter[1];
        cntTmp[2] = counter[2];
        cntTmp[3] = counter[3];

        keyTmp[0] = state.key[0];
        keyTmp[1] = state.key[1];

        round_10(cntTmp, keyTmp);

        if (i + 4 <= num_elements) {
            for (int j = 0; j < 4; j++) {
                res[i + j] = cntTmp[j];
            }
            add128_1(counter);
        }
        else {
            // here if last iteration
            for (int j = 0; i < num_elements; i++, j++) {
                res[i] = cntTmp[j];
            }
        }
    }
    skip_ahead(state, num_elements);
    return res;
}

// for VecSize <= 4
template <std::int32_t VecSize>
__attribute__((always_inline)) static inline sycl::vec<std::uint32_t, VecSize> generate_small(
    engine_state<oneapi::math::rng::device::philox4x32x10<VecSize>>& state) {
    const std::int32_t num_elements = VecSize;
    sycl::vec<std::uint32_t, VecSize> res;

    std::uint32_t counter[4];
    std::uint32_t key[2];

    int i = 0;
    int part = (int)state.part;
    while (part && (i < num_elements)) {
        res[i++] = state.result[3 - (--part)];
    }
    if (i == num_elements) {
        skip_ahead(state, num_elements);
        return res;
    }

    counter[0] = state.counter[0];
    counter[1] = state.counter[1];
    counter[2] = state.counter[2];
    counter[3] = state.counter[3];
    key[0] = state.key[0];
    key[1] = state.key[1];

    round_10(counter, key);

    for (int j = 0; i < num_elements; i++, j++) {
        res[i] = counter[j];
    }

    skip_ahead(state, num_elements);
    return res;
}

template <int VecSize>
__attribute__((always_inline)) static inline std::uint32_t generate_single(
    engine_state<oneapi::math::rng::device::philox4x32x10<VecSize>>& state) {
    std::uint32_t res;

    std::uint32_t counter[4];
    std::uint32_t key[2];

    std::int32_t part = static_cast<std::int32_t>(state.part);
    if (part != 0) {
        res = state.result[3 - (--part)];
        skip_ahead(state, 1);
        return res;
    }
    counter[0] = state.counter[0];
    counter[1] = state.counter[1];
    counter[2] = state.counter[2];
    counter[3] = state.counter[3];
    key[0] = state.key[0];
    key[1] = state.key[1];

    round_10(counter, key);

    res = counter[0];

    skip_ahead(state, 1);
    return res;
}

} // namespace philox4x32x10_impl

template <std::int32_t VecSize>
class engine_base<oneapi::math::rng::device::philox4x32x10<VecSize>> {
protected:
    engine_base(std::uint64_t seed, std::uint64_t offset = 0) {
        philox4x32x10_impl::init(this->state_, 1, &seed, offset);
    }

    engine_base(std::uint64_t n, const std::uint64_t* seed, std::uint64_t offset = 0) {
        philox4x32x10_impl::init(this->state_, n, seed, offset);
    }

    engine_base(std::uint64_t seed, std::uint64_t n_offset, const std::uint64_t* offset_ptr) {
        philox4x32x10_impl::init(this->state_, 1, &seed, n_offset, offset_ptr);
    }

    engine_base(std::uint64_t n, const std::uint64_t* seed, std::uint64_t n_offset,
                const std::uint64_t* offset_ptr) {
        philox4x32x10_impl::init(this->state_, n, seed, n_offset, offset_ptr);
    }

    template <typename RealType>
    __attribute__((always_inline)) inline auto generate(RealType a, RealType b) ->
        typename std::conditional<VecSize == 1, RealType, sycl::vec<RealType, VecSize>>::type {
        sycl::vec<RealType, VecSize> res;
        sycl::vec<std::uint32_t, VecSize> res_uint;
        RealType a1;
        RealType c1;

        c1 = (b - a) / (static_cast<RealType>((std::numeric_limits<std::uint32_t>::max)()) + 1);
        a1 = (b + a) / static_cast<RealType>(2.0);

        if constexpr (VecSize > 4) {
            res_uint = philox4x32x10_impl::generate_full(this->state_);
        }
        else {
            res_uint = philox4x32x10_impl::generate_small(this->state_);
        }
        for (int i = 0; i < VecSize; i++) {
            res[i] = static_cast<RealType>(static_cast<std::int32_t>(res_uint[i])) * c1 + a1;
        }
        return res;
    }

    __attribute__((always_inline)) inline auto generate() ->
        typename std::conditional<VecSize == 1, std::uint32_t,
                                  sycl::vec<std::uint32_t, VecSize>>::type {
        if constexpr (VecSize > 4) {
            return philox4x32x10_impl::generate_full(this->state_);
        }
        return philox4x32x10_impl::generate_small(this->state_);
    }

    template <typename UIntType>
    __attribute__((always_inline)) inline auto generate_uniform_bits() ->
        typename std::conditional<VecSize == 1, UIntType, sycl::vec<UIntType, VecSize>>::type {
        if constexpr (std::is_same<UIntType, std::uint32_t>::value) {
            return generate();
        }
        else {
            auto uni_res1 = generate();
            auto uni_res2 = generate();

            if constexpr (VecSize == 1) {
                return (static_cast<std::uint64_t>(uni_res2) << 32) + uni_res1;
            }
            else {
                sycl::vec<std::uint64_t, VecSize> vec_out;

                if constexpr (VecSize != 3) {
                    for (int i = 0; i < VecSize / 2; i++) {
                        vec_out[i] = (static_cast<std::uint64_t>(uni_res1[2 * i + 1]) << 32) +
                                     uni_res1[2 * i];
                        vec_out[i + VecSize / 2] =
                            (static_cast<std::uint64_t>(uni_res2[2 * i + 1]) << 32) +
                            uni_res2[2 * i];
                    }
                }
                else {
                    vec_out[0] = (static_cast<std::uint64_t>(uni_res1[1]) << 32) + uni_res1[0];
                    vec_out[1] = (static_cast<std::uint64_t>(uni_res2[0]) << 32) + uni_res1[2];
                    vec_out[2] = (static_cast<std::uint64_t>(uni_res2[2]) << 32) + uni_res2[1];
                }

                return vec_out;
            }
        }
    }

    template <typename RealType>
    RealType generate_single(RealType a, RealType b) {
        RealType res;
        std::uint32_t res_uint;
        RealType a1;
        RealType c1;

        c1 = (b - a) / (static_cast<RealType>((std::numeric_limits<std::uint32_t>::max)()) + 1);
        a1 = (b + a) / static_cast<RealType>(2.0);

        res_uint = philox4x32x10_impl::generate_single(this->state_);

        res = static_cast<RealType>(static_cast<std::int32_t>(res_uint)) * c1 + a1;

        return res;
    }

    __attribute__((always_inline)) inline std::uint32_t generate_single() {
        return philox4x32x10_impl::generate_single(this->state_);
    }

    template <typename UIntType>
    __attribute__((always_inline)) inline auto generate_single_uniform_bits() {
        if constexpr (std::is_same<UIntType, std::uint32_t>::value) {
            return philox4x32x10_impl::generate_single(this->state_);
        }
        else {
            auto uni_res1 = philox4x32x10_impl::generate_single(this->state_);
            auto uni_res2 = philox4x32x10_impl::generate_single(this->state_);

            return (static_cast<std::uint64_t>(uni_res2) << 32) + uni_res1;
        }
    }

    void skip_ahead(std::uint64_t num_to_skip) {
        detail::philox4x32x10_impl::skip_ahead(this->state_, num_to_skip);
    }

    void skip_ahead(std::initializer_list<std::uint64_t> num_to_skip) {
        detail::philox4x32x10_impl::skip_ahead(this->state_, num_to_skip.size(),
                                               num_to_skip.begin());
    }

    engine_state<oneapi::math::rng::device::philox4x32x10<VecSize>> state_;
};

} // namespace detail
} // namespace oneapi::math::rng::device

#endif // ONEMATH_RNG_DEVICE_PHILOX4X32X10_IMPL_HPP_
