/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  For your convenience, a copy of the License has been included in this
*  repository.
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
**************************************************************************/

#ifndef ONEMATH_COMPUTE_OUT_OF_PLACE_HPP
#define ONEMATH_COMPUTE_OUT_OF_PLACE_HPP

#include "compute_tester.hpp"
#include <numeric>

template <oneapi::math::dft::precision precision, oneapi::math::dft::domain domain>
int DFT_Test<precision, domain>::test_out_of_place_buffer() {
    if (!init(MemoryAccessModel::buffer)) {
        return test_skipped;
    }

    descriptor_t descriptor{ sizes };
    auto strides_fwd_cpy = strides_fwd;
    auto strides_bwd_cpy = strides_bwd;
    if (strides_fwd_cpy.size()) {
        descriptor.set_value(oneapi::math::dft::config_param::FWD_STRIDES, strides_fwd_cpy.data());
    }
    else {
        strides_fwd_cpy.resize(sizes.size() + 1);
        descriptor.get_value(oneapi::math::dft::config_param::FWD_STRIDES, strides_fwd_cpy.data());
    }
    if (strides_bwd_cpy.size()) {
        descriptor.set_value(oneapi::math::dft::config_param::BWD_STRIDES, strides_bwd_cpy.data());
    }
    else {
        strides_bwd_cpy.resize(sizes.size() + 1);
        descriptor.get_value(oneapi::math::dft::config_param::BWD_STRIDES, strides_bwd_cpy.data());
    }
    auto [forward_distance, backward_distance] =
        get_default_distances<domain>(sizes, strides_fwd_cpy, strides_bwd_cpy);
    auto ref_distance = std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<>());
    descriptor.set_value(oneapi::math::dft::config_param::PLACEMENT,
                         oneapi::math::dft::config_value::NOT_INPLACE);
    if constexpr (domain == oneapi::math::dft::domain::REAL) {
        descriptor.set_value(oneapi::math::dft::config_param::CONJUGATE_EVEN_STORAGE,
                             oneapi::math::dft::config_value::COMPLEX_COMPLEX);
        descriptor.set_value(oneapi::math::dft::config_param::PACKED_FORMAT,
                             oneapi::math::dft::config_value::CCE_FORMAT);
    }
    descriptor.set_value(oneapi::math::dft::config_param::NUMBER_OF_TRANSFORMS, batches);
    descriptor.set_value(oneapi::math::dft::config_param::FWD_DISTANCE, forward_distance);
    descriptor.set_value(oneapi::math::dft::config_param::BWD_DISTANCE, backward_distance);
    commit_descriptor(descriptor, sycl_queue);
    std::vector<FwdInputType> fwd_data(
        strided_copy(input, sizes, strides_fwd_cpy, batches, forward_distance));

    auto tmp = std::vector<FwdOutputType>(
        cast_unsigned(backward_distance * batches + get_default(strides_bwd_cpy, 0, 0L)), 0);
    {
        sycl::buffer<FwdInputType, 1> fwd_buf{ fwd_data };
        sycl::buffer<FwdOutputType, 1> bwd_buf{ tmp };

        oneapi::math::dft::compute_forward<descriptor_t, FwdInputType, FwdOutputType>(
            descriptor, fwd_buf, bwd_buf);

        {
            auto acc_bwd = bwd_buf.get_host_access();
            auto bwd_ptr = acc_bwd.get_pointer();
            for (std::int64_t i = 0; i < batches; i++) {
                EXPECT_TRUE(check_equal_strided < domain ==
                            oneapi::math::dft::domain::REAL >
                                (bwd_ptr + backward_distance * i,
                                 out_host_ref.data() + ref_distance * i, sizes, strides_bwd_cpy,
                                 abs_error_margin, rel_error_margin, std::cout));
            }
        }

        oneapi::math::dft::compute_backward<std::remove_reference_t<decltype(descriptor)>,
                                            FwdOutputType, FwdInputType>(descriptor, bwd_buf,
                                                                         fwd_buf);
    }

    // account for scaling that occurs during DFT
    std::for_each(input.begin(), input.end(),
                  [this](auto& x) { x *= static_cast<PrecisionType>(forward_elements); });

    for (std::int64_t i = 0; i < batches; i++) {
        EXPECT_TRUE(check_equal_strided<false>(
            fwd_data.data() + forward_distance * i, input.data() + ref_distance * i, sizes,
            strides_fwd_cpy, abs_error_margin, rel_error_margin, std::cout));
    }

    return !::testing::Test::HasFailure();
}

template <oneapi::math::dft::precision precision, oneapi::math::dft::domain domain>
int DFT_Test<precision, domain>::test_out_of_place_USM() {
    if (!init(MemoryAccessModel::usm)) {
        return test_skipped;
    }
    const std::vector<sycl::event> no_dependencies;

    descriptor_t descriptor{ sizes };
    auto strides_fwd_cpy = strides_fwd;
    auto strides_bwd_cpy = strides_bwd;
    if (strides_fwd_cpy.size()) {
        descriptor.set_value(oneapi::math::dft::config_param::FWD_STRIDES, strides_fwd_cpy.data());
    }
    else {
        strides_fwd_cpy.resize(sizes.size() + 1);
        descriptor.get_value(oneapi::math::dft::config_param::FWD_STRIDES, strides_fwd_cpy.data());
    }
    if (strides_bwd_cpy.size()) {
        descriptor.set_value(oneapi::math::dft::config_param::BWD_STRIDES, strides_bwd_cpy.data());
    }
    else {
        strides_bwd_cpy.resize(sizes.size() + 1);
        descriptor.get_value(oneapi::math::dft::config_param::BWD_STRIDES, strides_bwd_cpy.data());
    }
    auto [forward_distance, backward_distance] =
        get_default_distances<domain>(sizes, strides_fwd_cpy, strides_bwd_cpy);
    auto ref_distance = std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<>());
    descriptor.set_value(oneapi::math::dft::config_param::PLACEMENT,
                         oneapi::math::dft::config_value::NOT_INPLACE);
    if constexpr (domain == oneapi::math::dft::domain::REAL) {
        descriptor.set_value(oneapi::math::dft::config_param::CONJUGATE_EVEN_STORAGE,
                             oneapi::math::dft::config_value::COMPLEX_COMPLEX);
        descriptor.set_value(oneapi::math::dft::config_param::PACKED_FORMAT,
                             oneapi::math::dft::config_value::CCE_FORMAT);
    }
    descriptor.set_value(oneapi::math::dft::config_param::PLACEMENT,
                         oneapi::math::dft::config_value::NOT_INPLACE);
    if constexpr (domain == oneapi::math::dft::domain::REAL) {
        descriptor.set_value(oneapi::math::dft::config_param::CONJUGATE_EVEN_STORAGE,
                             oneapi::math::dft::config_value::COMPLEX_COMPLEX);
        descriptor.set_value(oneapi::math::dft::config_param::PACKED_FORMAT,
                             oneapi::math::dft::config_value::CCE_FORMAT);
    }
    descriptor.set_value(oneapi::math::dft::config_param::NUMBER_OF_TRANSFORMS, batches);
    descriptor.set_value(oneapi::math::dft::config_param::FWD_DISTANCE, forward_distance);
    descriptor.set_value(oneapi::math::dft::config_param::BWD_DISTANCE, backward_distance);
    commit_descriptor(descriptor, sycl_queue);

    auto ua_input = usm_allocator_t<FwdInputType>(cxt, *dev);
    auto ua_output = usm_allocator_t<FwdOutputType>(cxt, *dev);

    std::vector<FwdInputType, decltype(ua_input)> fwd(
        strided_copy(input, sizes, strides_fwd_cpy, batches, forward_distance, ua_input), ua_input);
    std::vector<FwdOutputType, decltype(ua_output)> bwd(
        cast_unsigned(backward_distance * batches + get_default(strides_bwd_cpy, 0, 0L)),
        ua_output);

    oneapi::math::dft::compute_forward<descriptor_t, FwdInputType, FwdOutputType>(
        descriptor, fwd.data(), bwd.data(), no_dependencies)
        .wait_and_throw();

    auto bwd_ptr = &bwd[0];
    for (std::int64_t i = 0; i < batches; i++) {
        EXPECT_TRUE(check_equal_strided < domain ==
                    oneapi::math::dft::domain::REAL >
                        (bwd_ptr + backward_distance * i, out_host_ref.data() + ref_distance * i,
                         sizes, strides_bwd_cpy, abs_error_margin, rel_error_margin, std::cout));
    }

    oneapi::math::dft::compute_backward<std::remove_reference_t<decltype(descriptor)>,
                                        FwdOutputType, FwdInputType>(descriptor, bwd.data(),
                                                                     fwd.data(), no_dependencies)
        .wait_and_throw();

    // account for scaling that occurs during DFT
    std::for_each(input.begin(), input.end(),
                  [this](auto& x) { x *= static_cast<PrecisionType>(forward_elements); });

    for (std::int64_t i = 0; i < batches; i++) {
        EXPECT_TRUE(check_equal_strided<false>(
            fwd.data() + forward_distance * i, input.data() + ref_distance * i, sizes,
            strides_fwd_cpy, abs_error_margin, rel_error_margin, std::cout));
    }

    return !::testing::Test::HasFailure();
}

#endif //ONEMATH_COMPUTE_OUT_OF_PLACE_HPP
