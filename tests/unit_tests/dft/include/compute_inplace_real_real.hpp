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

#ifndef ONEMKL_COMPUTE_INPLACE_REAL_REAL_HPP
#define ONEMKL_COMPUTE_INPLACE_REAL_REAL_HPP

#include "compute_tester.hpp"

/* Test is not implemented because currently there are no available dft implementations.
 * These are stubs to make sure that dft::oneapi::mkl::unimplemented exception is thrown */
template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain>
int DFT_Test<precision, domain>::test_in_place_real_real_USM() {
    if (!init(MemoryAccessModel::usm)) {
        return test_skipped;
    }

    try {
        descriptor_t descriptor{ sizes };

        descriptor.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                             oneapi::mkl::dft::config_value::INPLACE);
        descriptor.set_value(oneapi::mkl::dft::config_param::COMPLEX_STORAGE,
                             oneapi::mkl::dft::config_value::REAL_REAL);
        descriptor.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, batches);
        descriptor.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                             static_cast<std::int64_t>(forward_elements));
        descriptor.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                             static_cast<std::int64_t>(forward_elements));
        commit_descriptor(descriptor, sycl_queue);

        auto ua_input = usm_allocator_t<PrecisionType>(cxt, *dev);

        std::vector<PrecisionType, decltype(ua_input)> inout_re(size_total, ua_input);
        std::vector<PrecisionType, decltype(ua_input)> inout_im(size_total, ua_input);
        std::copy(input_re.begin(), input_re.end(), inout_re.begin());
        std::copy(input_im.begin(), input_im.end(), inout_im.begin());

        std::vector<sycl::event> dependencies;
        sycl::event done = oneapi::mkl::dft::compute_forward<descriptor_t, PrecisionType>(
            descriptor, inout_re.data(), inout_im.data(), dependencies);
        done.wait();

        descriptor_t descriptor_back{ sizes };

        descriptor_back.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                                  oneapi::mkl::dft::config_value::INPLACE);
        descriptor_back.set_value(oneapi::mkl::dft::config_param::COMPLEX_STORAGE,
                                  oneapi::mkl::dft::config_value::REAL_REAL);
        descriptor_back.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                                  (1.0 / forward_elements));
        descriptor_back.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, batches);
        descriptor_back.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                                  static_cast<std::int64_t>(forward_elements));
        descriptor_back.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                                  static_cast<std::int64_t>(forward_elements));
        commit_descriptor(descriptor_back, sycl_queue);

        done =
            oneapi::mkl::dft::compute_backward<std::remove_reference_t<decltype(descriptor_back)>,
                                               PrecisionType>(descriptor_back, inout_re.data(),
                                                              inout_im.data(), dependencies);
        done.wait();
    }
    catch (oneapi::mkl::unimplemented &e) {
        std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
        return test_skipped;
    }

    /* Once implementations exist, results will need to be verified */
    EXPECT_TRUE(false);

    return !::testing::Test::HasFailure();
}

/* Test is not implemented because currently there are no available dft implementations.
 * These are stubs to make sure that dft::oneapi::mkl::unimplemented exception is thrown */
template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain>
int DFT_Test<precision, domain>::test_in_place_real_real_buffer() {
    if (!init(MemoryAccessModel::buffer)) {
        return test_skipped;
    }

    try {
        descriptor_t descriptor{ sizes };

        descriptor.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                             oneapi::mkl::dft::config_value::INPLACE);
        descriptor.set_value(oneapi::mkl::dft::config_param::COMPLEX_STORAGE,
                             oneapi::mkl::dft::config_value::REAL_REAL);
        descriptor.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, batches);
        descriptor.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                             static_cast<std::int64_t>(forward_elements));
        descriptor.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                             static_cast<std::int64_t>(forward_elements));
        commit_descriptor(descriptor, sycl_queue);

        sycl::buffer<PrecisionType, 1> inout_re_buf{ input_re.data(), sycl::range<1>(size_total) };
        sycl::buffer<PrecisionType, 1> inout_im_buf{ input_im.data(), sycl::range<1>(size_total) };

        oneapi::mkl::dft::compute_forward<descriptor_t, PrecisionType>(descriptor, inout_re_buf,
                                                                       inout_im_buf);

        descriptor_t descriptor_back{ sizes };

        descriptor_back.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                                  oneapi::mkl::dft::config_value::INPLACE);
        descriptor_back.set_value(oneapi::mkl::dft::config_param::COMPLEX_STORAGE,
                                  oneapi::mkl::dft::config_value::REAL_REAL);
        descriptor_back.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                                  (1.0 / forward_elements));
        descriptor_back.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, batches);
        descriptor_back.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                                  static_cast<std::int64_t>(forward_elements));
        descriptor_back.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                                  static_cast<std::int64_t>(forward_elements));
        commit_descriptor(descriptor_back, sycl_queue);

        oneapi::mkl::dft::compute_backward<std::remove_reference_t<decltype(descriptor_back)>,
                                           PrecisionType>(descriptor_back, inout_re_buf,
                                                          inout_im_buf);
    }
    catch (oneapi::mkl::unimplemented &e) {
        std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
        return test_skipped;
    }

    /* Once implementations exist, results will need to be verified */
    EXPECT_TRUE(false);

    return !::testing::Test::HasFailure();
}

#endif //ONEMKL_COMPUTE_INPLACE_REAL_REAL_HPP
