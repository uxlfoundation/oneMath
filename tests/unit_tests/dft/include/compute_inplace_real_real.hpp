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

#ifndef ONEMATH_COMPUTE_INPLACE_REAL_REAL_HPP
#define ONEMATH_COMPUTE_INPLACE_REAL_REAL_HPP

#include "compute_tester.hpp"

template <oneapi::math::dft::precision precision, oneapi::math::dft::domain domain>
int DFT_Test<precision, domain>::test_in_place_real_real_USM() {
    if (!init(MemoryAccessModel::usm)) {
        return test_skipped;
    }
    if constexpr (domain == oneapi::math::dft::domain::REAL) {
        std::cout << "skipping real split tests as they are not supported" << std::endl;

        return test_skipped;
    }
    else {
        descriptor_t descriptor{ sizes };
        PrecisionType backward_scale = 1.f / static_cast<PrecisionType>(forward_elements);
        descriptor.set_value(oneapi::math::dft::config_param::PLACEMENT,
                             oneapi::math::dft::config_value::INPLACE);
        descriptor.set_value(oneapi::math::dft::config_param::COMPLEX_STORAGE,
                             oneapi::math::dft::config_value::REAL_REAL);
        descriptor.set_value(oneapi::math::dft::config_param::NUMBER_OF_TRANSFORMS, batches);
        descriptor.set_value(oneapi::math::dft::config_param::FWD_DISTANCE, forward_elements);
        descriptor.set_value(oneapi::math::dft::config_param::BWD_DISTANCE, forward_elements);
        descriptor.set_value(oneapi::math::dft::config_param::BACKWARD_SCALE, backward_scale);

        commit_descriptor(descriptor, sycl_queue);

        auto ua_input = usm_allocator_t<PrecisionType>(cxt, *dev);

        std::vector<PrecisionType, decltype(ua_input)> inout_re(size_total, ua_input);
        std::vector<PrecisionType, decltype(ua_input)> inout_im(size_total, ua_input);
        std::copy(input_re.begin(), input_re.end(), inout_re.begin());
        std::copy(input_im.begin(), input_im.end(), inout_im.begin());

        std::vector<sycl::event> no_dependencies;
        oneapi::math::dft::compute_forward<descriptor_t, PrecisionType>(
            descriptor, inout_re.data(), inout_im.data(), no_dependencies)
            .wait_and_throw();

        std::vector<FwdOutputType> output_data(size_total);
        for (std::size_t i = 0; i < output_data.size(); ++i) {
            output_data[i] = { inout_re[i], inout_im[i] };
        }
        EXPECT_TRUE(check_equal_vector(output_data.data(), out_host_ref.data(), output_data.size(),
                                       abs_error_margin, rel_error_margin, std::cout));

        oneapi::math::dft::compute_backward<std::remove_reference_t<decltype(descriptor)>,
                                            PrecisionType>(descriptor, inout_re.data(),
                                                           inout_im.data(), no_dependencies)
            .wait_and_throw();

        for (std::size_t i = 0; i < output_data.size(); ++i) {
            output_data[i] = { inout_re[i], inout_im[i] };
        }

        EXPECT_TRUE(check_equal_vector(output_data.data(), input.data(), input.size(),
                                       abs_error_margin, rel_error_margin, std::cout));

        return !::testing::Test::HasFailure();
    }
}

template <oneapi::math::dft::precision precision, oneapi::math::dft::domain domain>
int DFT_Test<precision, domain>::test_in_place_real_real_buffer() {
    if (!init(MemoryAccessModel::buffer)) {
        return test_skipped;
    }

    if constexpr (domain == oneapi::math::dft::domain::REAL) {
        std::cout << "skipping real split tests as they are not supported" << std::endl;

        return test_skipped;
    }
    else {
        descriptor_t descriptor{ sizes };

        PrecisionType backward_scale = 1.f / static_cast<PrecisionType>(forward_elements);
        descriptor.set_value(oneapi::math::dft::config_param::PLACEMENT,
                             oneapi::math::dft::config_value::INPLACE);
        descriptor.set_value(oneapi::math::dft::config_param::COMPLEX_STORAGE,
                             oneapi::math::dft::config_value::REAL_REAL);
        descriptor.set_value(oneapi::math::dft::config_param::NUMBER_OF_TRANSFORMS, batches);
        descriptor.set_value(oneapi::math::dft::config_param::FWD_DISTANCE, forward_elements);
        descriptor.set_value(oneapi::math::dft::config_param::BWD_DISTANCE, forward_elements);
        descriptor.set_value(oneapi::math::dft::config_param::BACKWARD_SCALE, backward_scale);

        commit_descriptor(descriptor, sycl_queue);

        std::vector<PrecisionType> host_inout_re(size_total, static_cast<PrecisionType>(0));
        std::vector<PrecisionType> host_inout_im(size_total, static_cast<PrecisionType>(0));
        std::copy(input_re.begin(), input_re.end(), host_inout_re.begin());
        std::copy(input_im.begin(), input_im.end(), host_inout_im.begin());

        sycl::buffer<PrecisionType, 1> inout_re_buf{ host_inout_re.data(),
                                                     sycl::range<1>(size_total) };
        sycl::buffer<PrecisionType, 1> inout_im_buf{ host_inout_im.data(),
                                                     sycl::range<1>(size_total) };

        oneapi::math::dft::compute_forward<descriptor_t, PrecisionType>(descriptor, inout_re_buf,
                                                                        inout_im_buf);

        {
            auto acc_inout_re = inout_re_buf.get_host_access();
            auto acc_inout_im = inout_im_buf.get_host_access();
            std::vector<FwdOutputType> output_data(size_total, static_cast<FwdOutputType>(0));
            for (std::size_t i = 0; i < output_data.size(); ++i) {
                output_data[i] = { acc_inout_re[i], acc_inout_im[i] };
            }
            EXPECT_TRUE(check_equal_vector(output_data.data(), out_host_ref.data(),
                                           output_data.size(), abs_error_margin, rel_error_margin,
                                           std::cout));
        }

        oneapi::math::dft::compute_backward<std::remove_reference_t<decltype(descriptor)>,
                                            PrecisionType>(descriptor, inout_re_buf, inout_im_buf);

        {
            auto acc_inout_re = inout_re_buf.get_host_access();
            auto acc_inout_im = inout_im_buf.get_host_access();
            std::vector<FwdInputType> output_data(size_total, static_cast<FwdInputType>(0));
            for (std::size_t i = 0; i < output_data.size(); ++i) {
                output_data[i] = { acc_inout_re[i], acc_inout_im[i] };
            }
            EXPECT_TRUE(check_equal_vector(output_data.data(), input.data(), input.size(),
                                           abs_error_margin, rel_error_margin, std::cout));
        }
        return !::testing::Test::HasFailure();
    }
}

#endif //ONEMATH_COMPUTE_INPLACE_REAL_REAL_HPP
