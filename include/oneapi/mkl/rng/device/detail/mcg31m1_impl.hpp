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

#ifndef _MKL_RNG_DEVICE_MCG31M1_IMPL_HPP_
#define _MKL_RNG_DEVICE_MCG31M1_IMPL_HPP_

namespace oneapi::mkl::rng::device {

template <std::int32_t VecSize = 1>
class mcg31m1;

namespace detail {

template <std::uint64_t VecSize>
constexpr sycl::vec<std::uint64_t, VecSize> select_vector_a_mcg31m1() {
    if constexpr (VecSize == 1)
        return sycl::vec<std::uint64_t, 1>(UINT64_C(1));
    else if constexpr (VecSize == 2)
        return sycl::vec<std::uint64_t, 2>({ UINT64_C(1), UINT64_C(1132489760) });
    else if constexpr (VecSize == 3)
        return sycl::vec<std::uint64_t, 3>(
            { UINT64_C(1), UINT64_C(1132489760), UINT64_C(826537482) });
    else if constexpr (VecSize == 4)
        return sycl::vec<std::uint64_t, 4>(
            { UINT64_C(1), UINT64_C(1132489760), UINT64_C(826537482), UINT64_C(289798557) });
    else if constexpr (VecSize == 8)
        return sycl::vec<std::uint64_t, 8>({ UINT64_C(1), UINT64_C(1132489760), UINT64_C(826537482),
                                             UINT64_C(289798557), UINT64_C(480863449),
                                             UINT64_C(1381340036), UINT64_C(1582925527),
                                             UINT64_C(1918178478) });
    else
        return sycl::vec<std::uint64_t, 16>(
            { UINT64_C(1), UINT64_C(1132489760), UINT64_C(826537482), UINT64_C(289798557),
              UINT64_C(480863449), UINT64_C(1381340036), UINT64_C(1582925527), UINT64_C(1918178478),
              UINT64_C(1286028348), UINT64_C(482167044), UINT64_C(262060616), UINT64_C(1856662125),
              UINT64_C(839877947), UINT64_C(1997268203), UINT64_C(458714024),
              UINT64_C(650347998) });
}

// hipSYCL (AdaptiveCpp) doesn't support constexpr sycl::vec constructor
// that's why in case of hipSYCL backend sycl::vec is created as a local variable
#ifndef __HIPSYCL__
template <std::uint64_t VecSize>
struct mcg31m1_vector_a {
    static constexpr sycl::vec<std::uint64_t, VecSize> vector_a =
        select_vector_a_mcg31m1<VecSize>(); // powers of a
};
#endif

struct mcg31m1_param {
    static constexpr std::uint32_t a = 1132489760;
    static constexpr std::uint64_t m_64 = 0x000000007FFFFFFF; // 2^31 - 1
    static constexpr double m_fl = 2147483647.0; // 2^31 - 1
    static constexpr std::uint64_t bits = 31;
};

template <std::int32_t VecSize>
struct engine_state<oneapi::mkl::rng::device::mcg31m1<VecSize>> {
    std::uint32_t s;
};

namespace mcg31m1_impl {

// Improved modulus x % (2^31 - 1) operation (possible to do for divisor (2^N
// -1), but MCG31M1 needs only 2^31 - 1) if we want to do x % (2^N -1) we can
// find out that: x = A + B * 2^N, where A = x % 2^N = x & 00..01..11 (binary)
// where quantity of 1 is N, B = x / 2^N = x >> N also x = A + B * (2^N - 1 + 1)
// = (A + B) + B * (2^N - 1), but (A + B) may be greater than (2^N - 1), that's
// why we put x1 = A + B = A' + B' * 2^N = ... until new (A + B) < (2^N - 1) for
// MCG31m1 N = 31
template <typename T>
static inline T custom_mod(std::uint64_t x) {
    std::uint64_t b = x >> mcg31m1_param::bits;
    std::uint64_t a = x & mcg31m1_param::m_64;
    x = a + b;
    b = x >> mcg31m1_param::bits;
    a = x & mcg31m1_param::m_64;
    return static_cast<T>(a + b);
}

template <std::int32_t VecSize>
static inline sycl::vec<std::uint32_t, VecSize> custom_mod(
    const sycl::vec<std::uint64_t, VecSize>& x) {
    sycl::vec<std::uint64_t, VecSize> b = x >> mcg31m1_param::bits;
    sycl::vec<std::uint64_t, VecSize> a = x & mcg31m1_param::m_64;
    sycl::vec<std::uint64_t, VecSize> res = a + b;
    b = res >> mcg31m1_param::bits;
    a = res & mcg31m1_param::m_64;
    res = a + b;
    return res.template convert<std::uint32_t>();
}

static inline std::uint64_t power(std::uint64_t a, std::uint64_t n) {
    std::uint64_t a2;
    // initialize result by 1 for recurrence
    std::uint32_t result = 1;

    if (n == 0) {
        // return (a^0)%m = 1
        return std::uint64_t{ 1 };
    }

    // Recurrence loop
    do {
        // For each odd n
        if (n & 1) {
            a2 = static_cast<std::uint64_t>(result) * a;
            result = custom_mod<std::uint32_t>(a2);
        }
        // n /= 2
        n >>= 1;

        a2 = a * a;
        a = custom_mod<std::uint64_t>(a2);
    } while (n);

    return static_cast<std::uint64_t>(result);
}

template <std::int32_t VecSize>
static inline void skip_ahead(engine_state<oneapi::mkl::rng::device::mcg31m1<VecSize>>& state,
                              std::uint64_t num_to_skip) {
    std::uint64_t loc_A = power(static_cast<std::uint64_t>(mcg31m1_param::a), num_to_skip);
    state.s = custom_mod<std::uint32_t>(loc_A * static_cast<std::uint64_t>(state.s));
}

template <std::int32_t VecSize>
static inline void init(engine_state<oneapi::mkl::rng::device::mcg31m1<VecSize>>& state,
                        std::uint32_t seed, std::uint64_t offset) {
    state.s = custom_mod<std::uint32_t>(seed);
    if (state.s == 0)
        state.s = 1;
    skip_ahead(state, offset);
}

template <std::int32_t VecSize>
static inline sycl::vec<std::uint32_t, VecSize> generate(
    engine_state<oneapi::mkl::rng::device::mcg31m1<VecSize>>& state) {
    sycl::vec<std::uint64_t, VecSize> x(state.s);
    sycl::vec<std::uint32_t, VecSize> res;
#ifndef __HIPSYCL__
    res = custom_mod(mcg31m1_vector_a<VecSize>::vector_a * x);
#else
    // a workaround for hipSYCL (AdaptiveCpp)
    res = custom_mod(select_vector_a_mcg31m1<VecSize>() * x);
#endif
    state.s =
        custom_mod<std::uint32_t>(mcg31m1_param::a * static_cast<std::uint64_t>(res[VecSize - 1]));
    return res;
}

template <std::int32_t VecSize>
static inline std::uint32_t generate_single(
    engine_state<oneapi::mkl::rng::device::mcg31m1<VecSize>>& state) {
    std::uint32_t x = state.s;
    state.s = custom_mod<std::uint32_t>(mcg31m1_param::a * static_cast<std::uint64_t>(state.s));
    return x;
}

} // namespace mcg31m1_impl

template <std::int32_t VecSize>
class engine_base<oneapi::mkl::rng::device::mcg31m1<VecSize>> {
protected:
    engine_base(std::uint32_t seed, std::uint64_t offset = 0) {
        mcg31m1_impl::init(this->state_, seed, offset);
    }

    template <typename RealType>
    auto generate(RealType a, RealType b) ->
        typename std::conditional<VecSize == 1, RealType, sycl::vec<RealType, VecSize>>::type {
        sycl::vec<RealType, VecSize> res;
        sycl::vec<std::uint32_t, VecSize> res_uint;

        RealType c = (b - a) / static_cast<RealType>(mcg31m1_param::m_fl);

        res_uint = mcg31m1_impl::generate(this->state_);

        res = res_uint.template convert<RealType>() * c + a;

        return res;
    }

    auto generate() -> typename std::conditional<VecSize == 1, std::uint32_t,
                                                 sycl::vec<std::uint32_t, VecSize>>::type {
        return mcg31m1_impl::generate(this->state_);
    }

    template <typename RealType>
    RealType generate_single(RealType a, RealType b) {
        RealType res;
        std::uint32_t res_uint;

        RealType c = (b - a) / static_cast<RealType>(mcg31m1_param::m_fl);

        res_uint = mcg31m1_impl::generate_single(this->state_);

        res = static_cast<RealType>(res_uint) * c + a;
        return res;
    }

    std::uint32_t generate_single() {
        return mcg31m1_impl::generate_single(this->state_);
    }

    void skip_ahead(std::uint64_t num_to_skip) {
        detail::mcg31m1_impl::skip_ahead(this->state_, num_to_skip);
    }

    engine_state<oneapi::mkl::rng::device::mcg31m1<VecSize>> state_;
};

} // namespace detail

} // namespace oneapi::mkl::rng::device

#endif // _MKL_RNG_DEVICE_MCG31M1_IMPL_HPP_
