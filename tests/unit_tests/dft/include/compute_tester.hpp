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

#ifndef ONEMATH_COMPUTE_TESTER_HPP
#define ONEMATH_COMPUTE_TESTER_HPP

#include <algorithm>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include "oneapi/math.hpp"
#include "test_helper.hpp"
#include "test_common.hpp"
#include "reference_dft.hpp"

#include <numeric>

template <oneapi::math::dft::precision precision, oneapi::math::dft::domain domain>
struct DFT_Test {
    using descriptor_t = oneapi::math::dft::descriptor<precision, domain>;

    template <typename ElemT>
    using usm_allocator_t = sycl::usm_allocator<ElemT, sycl::usm::alloc::shared, 64>;

    using PrecisionType =
        typename std::conditional_t<precision == oneapi::math::dft::precision::SINGLE, float,
                                    double>;

    using FwdInputType = typename std::conditional_t<domain == oneapi::math::dft::domain::REAL,
                                                     PrecisionType, std::complex<PrecisionType>>;
    using FwdOutputType = std::complex<PrecisionType>;

    enum class MemoryAccessModel { buffer, usm };

    const std::vector<std::int64_t> sizes;
    const std::vector<std::int64_t> strides_fwd;
    const std::vector<std::int64_t> strides_bwd;
    const std::int64_t batches;
    const std::int64_t forward_elements;
    const std::size_t size_total;
    double abs_error_margin{ 0 };
    double rel_error_margin{ 0 };

    sycl::device* dev;
    sycl::queue sycl_queue;
    sycl::context cxt;

    std::vector<FwdInputType> input;
    std::vector<PrecisionType> input_re;
    std::vector<PrecisionType> input_im;
    std::vector<FwdOutputType> out_host_ref;

    DFT_Test(sycl::device* dev, std::vector<std::int64_t> sizes_,
             std::vector<std::int64_t> strides_fwd, std::vector<std::int64_t> strides_bwd,
             std::int64_t batches_)
            : sizes{ std::move(sizes_) },
              strides_fwd(std::move(strides_fwd)),
              strides_bwd(std::move(strides_bwd)),
              batches{ batches_ },
              forward_elements{ std::accumulate(sizes.begin(), sizes.end(), 1,
                                                std::multiplies<>{}) },
              size_total{ cast_unsigned(forward_elements * batches) },
              dev{ dev },
              sycl_queue{ *dev, exception_handler },
              cxt{ sycl_queue.get_context() } {
        input = std::vector<FwdInputType>(size_total);
        input_re = std::vector<PrecisionType>(size_total);
        input_im = std::vector<PrecisionType>(size_total);

        // out_host_ref contains redundant information for domain::REAL
        // tests. This simplifies the test implementation, but increases
        // storage and computational requirements. There is scope for
        // improvement here if test performance becomes an issue.
        out_host_ref = std::vector<FwdOutputType>(size_total);

        rand_vector(input, size_total);
        if constexpr (domain == oneapi::math::dft::domain::REAL) {
            for (std::size_t i = 0; i < input.size(); ++i) {
                input_re[i] = { input[i] };
                input_im[i] = 0;
            }
        }
        else {
            for (std::size_t i = 0; i < input.size(); ++i) {
                input_re[i] = { input[i].real() };
                input_im[i] = { input[i].imag() };
            }
        }
    }

    bool skip_test(MemoryAccessModel mem_acc) {
        if constexpr (precision == oneapi::math::dft::precision::DOUBLE) {
            if (!sycl_queue.get_device().has(sycl::aspect::fp64)) {
                std::cout << "Device does not support double precision." << std::endl;
                return true;
            }
        }

        if (mem_acc == MemoryAccessModel::usm &&
            !sycl_queue.get_device().has(sycl::aspect::usm_shared_allocations)) {
            std::cout << "Device does not support usm shared allocations." << std::endl;
            return true;
        }

        return false;
    }

    bool init(MemoryAccessModel mem_acc) {
        for (int i = 0; i < batches; i += 1) {
            reference_forward_dft<FwdInputType, FwdOutputType>(
                sizes, input.data() + i * forward_elements,
                out_host_ref.data() + i * forward_elements);
        }
        auto max_norm_ref = *std::max_element(std::begin(out_host_ref), std::end(out_host_ref),
                                              [](const FwdOutputType& a, const FwdOutputType& b) {
                                                  return std::abs(a) < std::abs(b);
                                              });
        // Heuristic for the average-case error margins
        abs_error_margin =
            10 * std::abs(max_norm_ref) * std::log2(static_cast<double>(forward_elements));
        rel_error_margin = 200.0 * std::log2(static_cast<double>(forward_elements));
        return !skip_test(mem_acc);
    }

    int test_in_place_buffer();
    int test_in_place_real_real_buffer();
    int test_out_of_place_buffer();
    int test_out_of_place_real_real_buffer();
    int test_in_place_USM();
    int test_in_place_real_real_USM();
    int test_out_of_place_USM();
    int test_out_of_place_real_real_USM();
};

#endif //ONEMATH_COMPUTE_TESTER_HPP
