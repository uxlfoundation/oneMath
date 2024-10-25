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

#ifndef ONEMKL_COMPUTE_INPLACE_HPP
#define ONEMKL_COMPUTE_INPLACE_HPP

#include "compute_tester.hpp"
#include <oneapi/mkl/exceptions.hpp>

template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain>
int DFT_Test<precision, domain>::test_in_place_buffer() {
    if (!init(MemoryAccessModel::buffer)) {
        return test_skipped;
    }

    auto modified_strides_fwd = this->strides_fwd;
    auto modified_strides_bwd = this->strides_bwd;
    if (domain == oneapi::mkl::dft::domain::REAL) {
        // both input and output strides must be set
        auto default_conjuate_strides = get_conjugate_even_complex_strides(sizes);
        std::ptrdiff_t rank = static_cast<std::ptrdiff_t>(sizes.size());

        if (modified_strides_fwd.size() == 0) {
            modified_strides_fwd = std::vector<std::int64_t>(
                default_conjuate_strides.begin(), default_conjuate_strides.begin() + rank + 1);
            std::transform(modified_strides_fwd.begin() + 1, modified_strides_fwd.begin() + rank,
                           modified_strides_fwd.begin() + 1, [](std::int64_t& s) { return 2 * s; });
        }
        if (modified_strides_bwd.size() == 0) {
            modified_strides_bwd = std::vector<std::int64_t>(
                default_conjuate_strides.begin(), default_conjuate_strides.begin() + rank + 1);
        }
    }
    else {
        // General consistency requirements for in-place complex domain transforms require that strides are the same forward and backward.
        modified_strides_fwd = modified_strides_bwd;
    }

    auto [forward_distance, backward_distance] =
        get_default_distances<domain, true>(sizes, modified_strides_fwd, modified_strides_bwd);
    auto ref_distance = std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<>());

    descriptor_t descriptor{ sizes };
    descriptor.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                         oneapi::mkl::dft::config_value::INPLACE);
    if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
        descriptor.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE,
                             oneapi::mkl::dft::config_value::COMPLEX_COMPLEX);
        descriptor.set_value(oneapi::mkl::dft::config_param::PACKED_FORMAT,
                             oneapi::mkl::dft::config_value::CCE_FORMAT);
    }
    descriptor.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, batches);
    descriptor.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, forward_distance);
    descriptor.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, backward_distance);
    if (modified_strides_fwd.size()) {
        descriptor.set_value(oneapi::mkl::dft::config_param::FWD_STRIDES,
                             modified_strides_fwd.data());
    }
    if (modified_strides_bwd.size()) {
        descriptor.set_value(oneapi::mkl::dft::config_param::BWD_STRIDES,
                             modified_strides_bwd.data());
    }
    commit_descriptor(descriptor, sycl_queue);

    std::vector<FwdInputType> inout_host(
        strided_copy(input, sizes, modified_strides_fwd, batches, forward_distance));
    int real_multiplier = (domain == oneapi::mkl::dft::domain::REAL ? 2 : 1);
    inout_host.resize(
        cast_unsigned(std::max(forward_distance, real_multiplier * backward_distance) * batches +
                      get_default(modified_strides_bwd, 0, 0L) * real_multiplier));

    {
        sycl::buffer<FwdInputType, 1> inout_buf{ inout_host };

        oneapi::mkl::dft::compute_forward<descriptor_t, FwdInputType>(descriptor, inout_buf);

        {
            auto acc_host = inout_buf.get_host_access();
            auto ptr_host = reinterpret_cast<FwdOutputType*>(acc_host.get_pointer());
            for (std::int64_t i = 0; i < batches; i++) {
                EXPECT_TRUE(check_equal_strided < domain ==
                            oneapi::mkl::dft::domain::REAL >
                                (ptr_host + backward_distance * i,
                                 out_host_ref.data() + ref_distance * i, sizes,
                                 modified_strides_bwd, abs_error_margin, rel_error_margin,
                                 std::cout));
            }
        }

        oneapi::mkl::dft::compute_backward<std::remove_reference_t<decltype(descriptor)>,
                                           FwdInputType>(descriptor, inout_buf);
    }

    std::vector<FwdInputType> fwd_data_ref = input;
    // account for scaling that occurs during DFT
    std::for_each(fwd_data_ref.begin(), fwd_data_ref.end(),
                  [this](auto& x) { x *= static_cast<PrecisionType>(forward_elements); });

    for (std::int64_t i = 0; i < batches; i++) {
        EXPECT_TRUE(check_equal_strided<false>(
            inout_host.data() + forward_distance * i, fwd_data_ref.data() + ref_distance * i, sizes,
            modified_strides_fwd, abs_error_margin, rel_error_margin, std::cout));
    }

    return !::testing::Test::HasFailure();
}

template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain>
int DFT_Test<precision, domain>::test_in_place_USM() {
    if (!init(MemoryAccessModel::usm)) {
        return test_skipped;
    }

    auto modified_strides_fwd = this->strides_fwd;
    auto modified_strides_bwd = this->strides_bwd;
    if (domain == oneapi::mkl::dft::domain::REAL) {
        // both input and output strides must be set
        auto default_conjuate_strides = get_conjugate_even_complex_strides(sizes);
        std::ptrdiff_t rank = static_cast<std::ptrdiff_t>(sizes.size());

        if (modified_strides_fwd.size() == 0) {
            modified_strides_fwd = std::vector<std::int64_t>(
                default_conjuate_strides.begin(), default_conjuate_strides.begin() + rank + 1);
            std::transform(modified_strides_fwd.begin() + 1, modified_strides_fwd.begin() + rank,
                           modified_strides_fwd.begin() + 1, [](std::int64_t& s) { return 2 * s; });
        }
        if (modified_strides_bwd.size() == 0) {
            modified_strides_bwd = std::vector<std::int64_t>(
                default_conjuate_strides.begin(), default_conjuate_strides.begin() + rank + 1);
        }
    }
    else {
        // General consistency requirements for in-place complex domain transforms require that strides are the same forward and backward.
        modified_strides_fwd = modified_strides_bwd;
    }

    auto [forward_distance, backward_distance] =
        get_default_distances<domain, true>(sizes, modified_strides_fwd, modified_strides_bwd);
    auto ref_distance = std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<>());

    descriptor_t descriptor = { sizes };
    descriptor.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                         oneapi::mkl::dft::config_value::INPLACE);
    if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
        descriptor.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE,
                             oneapi::mkl::dft::config_value::COMPLEX_COMPLEX);
        descriptor.set_value(oneapi::mkl::dft::config_param::PACKED_FORMAT,
                             oneapi::mkl::dft::config_value::CCE_FORMAT);
    }
    descriptor.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, batches);
    descriptor.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, forward_distance);
    descriptor.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, backward_distance);
    if (modified_strides_fwd.size()) {
        descriptor.set_value(oneapi::mkl::dft::config_param::FWD_STRIDES,
                             modified_strides_fwd.data());
    }
    if (modified_strides_bwd.size()) {
        descriptor.set_value(oneapi::mkl::dft::config_param::BWD_STRIDES,
                             modified_strides_bwd.data());
    }
    commit_descriptor(descriptor, sycl_queue);

    auto ua_input = usm_allocator_t<FwdInputType>(cxt, *dev);
    std::vector<FwdInputType, decltype(ua_input)> inout(
        strided_copy(input, sizes, modified_strides_fwd, batches, forward_distance, ua_input),
        ua_input);
    int real_multiplier = (domain == oneapi::mkl::dft::domain::REAL ? 2 : 1);
    inout.resize(
        cast_unsigned(std::max(forward_distance, real_multiplier * backward_distance) * batches +
                      real_multiplier * get_default(modified_strides_bwd, 0, 0L)));

    std::vector<sycl::event> no_dependencies;
    oneapi::mkl::dft::compute_forward<descriptor_t, FwdInputType>(descriptor, inout.data(),
                                                                  no_dependencies)
        .wait_and_throw();

    for (std::int64_t i = 0; i < batches; i++) {
        EXPECT_TRUE(check_equal_strided < domain ==
                    oneapi::mkl::dft::domain::REAL >
                        (reinterpret_cast<FwdOutputType*>(inout.data()) + backward_distance * i,
                         out_host_ref.data() + ref_distance * i, sizes, modified_strides_bwd,
                         abs_error_margin, rel_error_margin, std::cout));
    }

    sycl::event done =
        oneapi::mkl::dft::compute_backward<std::remove_reference_t<decltype(descriptor)>,
                                           FwdInputType>(descriptor, inout.data(), no_dependencies);
    done.wait_and_throw();

    std::for_each(input.begin(), input.end(),
                  [this](auto& x) { x *= static_cast<PrecisionType>(forward_elements); });

    for (std::int64_t i = 0; i < batches; i++) {
        EXPECT_TRUE(check_equal_strided<false>(
            inout.data() + forward_distance * i, input.data() + ref_distance * i, sizes,
            modified_strides_fwd, abs_error_margin, rel_error_margin, std::cout));
    }

    return !::testing::Test::HasFailure();
}

#endif //ONEMKL_COMPUTE_INPLACE_HPP
