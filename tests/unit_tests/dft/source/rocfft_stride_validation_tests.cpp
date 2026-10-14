/*******************************************************************************
* Copyright contributors to the oneMath project
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

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "test_helper.hpp"
#include "test_common.hpp"
#include "dft/backends/rocfft/stride_validation.hpp"
#include <gtest/gtest.h>

extern std::vector<sycl::device*> devices;

namespace {

using oneapi::math::dft::rocfft::detail::get_stride_validity;
using oneapi::math::dft::rocfft::detail::stride_validity;
using sizes_t = std::vector<std::size_t>;

// oneMath strides are {offset, outermost, ..., innermost}; rocFFT wants innermost first
// without the offset.
sizes_t to_rocfft_strides(sizes_t strides) {
    std::reverse(strides.begin(), strides.end());
    strides.pop_back();
    return strides;
}

sizes_t to_rocfft_lengths(sizes_t lengths) {
    std::reverse(lengths.begin(), lengths.end());
    return lengths;
}

stride_validity validate_complex(const sizes_t& lengths, const sizes_t& strides_a,
                                 const sizes_t& strides_b, bool fb_strides) {
    const sizes_t rocfft_lengths = to_rocfft_lengths(lengths);
    return get_stride_validity(lengths.size(), to_rocfft_strides(strides_a),
                               to_rocfft_strides(strides_b), rocfft_lengths, rocfft_lengths,
                               fb_strides);
}

stride_validity validate_real(const sizes_t& lengths, const sizes_t& strides_a,
                              const sizes_t& strides_b, bool fb_strides) {
    const sizes_t fwd_lengths = to_rocfft_lengths(lengths);
    sizes_t bwd_lengths = fwd_lengths;
    bwd_lengths[0] = bwd_lengths[0] / 2 + 1;
    return get_stride_validity(lengths.size(), to_rocfft_strides(strides_a),
                               to_rocfft_strides(strides_b), fwd_lengths, bwd_lengths, fb_strides);
}

// ---------------------------------------------------------------------------
// Hardware-independent tests of the rocFFT backend's stride validation (#760).
// ---------------------------------------------------------------------------

TEST(RocfftStrideValidation, AcceptsDefaultRealOutOfPlaceStrides) {
    const auto v = validate_real({ 4, 8 }, { 0, 8, 1 }, { 0, 5, 1 }, true);
    EXPECT_TRUE(v.forward);
    EXPECT_TRUE(v.backward);
}

// Real rows of 8 elements placed 5 apart overlap. Validating the forward strides against the
// backward-domain lengths (5 complex elements per row) used to accept this.
TEST(RocfftStrideValidation, RejectsRealFwdStridesThatOnlyFitBackwardDomain) {
    const auto v = validate_real({ 4, 8 }, { 0, 5, 1 }, { 0, 8, 1 }, true);
    EXPECT_FALSE(v.forward);
    EXPECT_FALSE(v.backward);
}

// With INPUT/OUTPUT_STRIDES the domains swap between directions, so real-to-complex
// strides are only valid for the forward direction.
TEST(RocfftStrideValidation, RealIoStridesValidOnlyForForward) {
    const auto v = validate_real({ 4, 8 }, { 0, 8, 1 }, { 0, 5, 1 }, false);
    EXPECT_TRUE(v.forward);
    EXPECT_FALSE(v.backward);
}

TEST(RocfftStrideValidation, AcceptsNonUnitInnermostStride) {
    auto v = validate_complex({ 4, 4 }, { 0, 8, 2 }, { 0, 8, 2 }, true);
    EXPECT_TRUE(v.forward);
    EXPECT_TRUE(v.backward);
    v = validate_complex({ 2, 4, 4 }, { 0, 32, 8, 2 }, { 0, 32, 8, 2 }, true);
    EXPECT_TRUE(v.forward);
    EXPECT_TRUE(v.backward);
}

// An innermost stride of 2 makes each row of 4 span 8 elements, so rows 4 apart overlap.
// Comparing the length alone against the next stride (4 <= 4) used to accept this.
TEST(RocfftStrideValidation, RejectsOverlapFromNonUnitInnermostStride) {
    auto v = validate_complex({ 4, 4 }, { 0, 4, 2 }, { 0, 4, 2 }, true);
    EXPECT_FALSE(v.forward);
    EXPECT_FALSE(v.backward);
    v = validate_complex({ 2, 4, 4 }, { 0, 32, 4, 2 }, { 0, 32, 4, 2 }, true);
    EXPECT_FALSE(v.forward);
    EXPECT_FALSE(v.backward);
}

#ifdef ONEMATH_ENABLE_ROCFFT_BACKEND

// ---------------------------------------------------------------------------
// rocFFT backend commit tests. These assert the commit result directly instead of treating
// unimplemented as a skipped test.
// ---------------------------------------------------------------------------

std::vector<sycl::device*> get_amd_gpus() {
    std::vector<sycl::device*> amd_gpus;
    for (auto* dev : devices) {
        if (dev->is_gpu() &&
            static_cast<unsigned int>(dev->get_info<sycl::info::device::vendor_id>()) == AMD_ID) {
            amd_gpus.push_back(dev);
        }
    }
    return amd_gpus;
}

template <typename descriptor_t>
void commit_on_rocfft(descriptor_t& descriptor, sycl::queue& queue) {
#ifdef CALL_RT_API
    descriptor.commit(queue);
#else
    descriptor.commit(oneapi::math::backend_selector<oneapi::math::backend::rocfft>{ queue });
#endif
}

template <typename descriptor_t>
void expect_invalid_strides(descriptor_t& descriptor, sycl::queue& queue) {
    try {
        commit_on_rocfft(descriptor, queue);
        ADD_FAILURE() << "commit accepted invalid strides";
    }
    catch (const oneapi::math::unimplemented& e) {
        ADD_FAILURE() << "expected an invalid strides error, got unimplemented: " << e.what();
    }
    catch (const oneapi::math::exception& e) {
        EXPECT_NE(std::string(e.what()).find("Invalid strides"), std::string::npos) << e.what();
    }
}

class RocfftCommit : public ::testing::Test {
protected:
    void SetUp() override {
        amd_gpus = get_amd_gpus();
        if (amd_gpus.empty()) {
            GTEST_SKIP() << "No AMD GPU available";
        }
    }
    std::vector<sycl::device*> amd_gpus;
};

TEST_F(RocfftCommit, RejectsRealFwdStridesThatOnlyFitBackwardDomain) {
    for (auto* dev : amd_gpus) {
        sycl::queue queue(*dev);
        oneapi::math::dft::descriptor<oneapi::math::dft::precision::SINGLE,
                                      oneapi::math::dft::domain::REAL>
            descriptor({ 4, 8 });
        std::vector<std::int64_t> fwd_strides{ 0, 5, 1 };
        std::vector<std::int64_t> bwd_strides{ 0, 8, 1 };
        descriptor.set_value(oneapi::math::dft::config_param::PLACEMENT,
                             oneapi::math::dft::config_value::NOT_INPLACE);
        descriptor.set_value(oneapi::math::dft::config_param::FWD_STRIDES, fwd_strides.data());
        descriptor.set_value(oneapi::math::dft::config_param::BWD_STRIDES, bwd_strides.data());
        expect_invalid_strides(descriptor, queue);
    }
}

TEST_F(RocfftCommit, RejectsOverlapFromNonUnitInnermostStride) {
    for (auto* dev : amd_gpus) {
        sycl::queue queue(*dev);
        oneapi::math::dft::descriptor<oneapi::math::dft::precision::SINGLE,
                                      oneapi::math::dft::domain::COMPLEX>
            descriptor({ 4, 4 });
        std::vector<std::int64_t> strides{ 0, 4, 2 };
        descriptor.set_value(oneapi::math::dft::config_param::FWD_STRIDES, strides.data());
        descriptor.set_value(oneapi::math::dft::config_param::BWD_STRIDES, strides.data());
        expect_invalid_strides(descriptor, queue);
    }
}

TEST_F(RocfftCommit, AcceptsNonUnitInnermostStride) {
    for (auto* dev : amd_gpus) {
        sycl::queue queue(*dev);
        oneapi::math::dft::descriptor<oneapi::math::dft::precision::SINGLE,
                                      oneapi::math::dft::domain::COMPLEX>
            descriptor({ 4, 4 });
        std::vector<std::int64_t> strides{ 0, 8, 2 };
        descriptor.set_value(oneapi::math::dft::config_param::FWD_STRIDES, strides.data());
        descriptor.set_value(oneapi::math::dft::config_param::BWD_STRIDES, strides.data());
        EXPECT_NO_THROW(commit_on_rocfft(descriptor, queue));
    }
}

TEST_F(RocfftCommit, RejectsRank4AsUnimplemented) {
    for (auto* dev : amd_gpus) {
        sycl::queue queue(*dev);
        oneapi::math::dft::descriptor<oneapi::math::dft::precision::SINGLE,
                                      oneapi::math::dft::domain::COMPLEX>
            descriptor({ 2, 2, 2, 2 });
        EXPECT_THROW(commit_on_rocfft(descriptor, queue), oneapi::math::unimplemented);
    }
}

#endif // ONEMATH_ENABLE_ROCFFT_BACKEND

} // anonymous namespace
