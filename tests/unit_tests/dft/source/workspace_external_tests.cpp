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

#include <iostream>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "test_helper.hpp"
#include "test_common.hpp"
#include "parseval_check.hpp"
#include <gtest/gtest.h>

extern std::vector<sycl::device*> devices;

class WorkspaceExternalTests : public ::testing::TestWithParam<sycl::device*> {};

template <oneapi::math::dft::precision prec, oneapi::math::dft::domain dom>
int test_workspace_external_usm_impl(std::size_t dft_size, sycl::device* dev) {
    using namespace oneapi::math::dft;
    using scalar_t = std::conditional_t<prec == precision::DOUBLE, double, float>;
    using forward_t = std::conditional_t<dom == domain::COMPLEX, std::complex<scalar_t>, scalar_t>;
    using backward_t = std::complex<scalar_t>;

    sycl::queue sycl_queue(*dev);
    if (prec == precision::DOUBLE && !sycl_queue.get_device().has(sycl::aspect::fp64)) {
        std::cout << "Device does not support double precision." << std::endl;
        return test_skipped;
    }
    descriptor<prec, dom> desc(static_cast<std::int64_t>(dft_size));

    desc.set_value(config_param::WORKSPACE_PLACEMENT, config_value::WORKSPACE_EXTERNAL);
    desc.set_value(config_param::PLACEMENT, config_value::NOT_INPLACE);
    try {
        commit_descriptor(desc, sycl_queue);
    }
    catch (oneapi::math::unimplemented&) {
        std::cout << "Test configuration not implemented." << std::endl;
        return test_skipped;
    }
    std::int64_t workspace_bytes = -1;
    desc.get_value(config_param::WORKSPACE_EXTERNAL_BYTES, &workspace_bytes);
    if (workspace_bytes < 0) {
        return ::testing::Test::HasFailure();
    }
    scalar_t* workspace = sycl::malloc_device<scalar_t>(
        static_cast<std::size_t>(workspace_bytes) / sizeof(scalar_t), sycl_queue);
    desc.set_workspace(workspace);
    // Generate data
    std::vector<forward_t> host_fwd(static_cast<std::size_t>(dft_size));
    std::size_t bwd_size = dom == domain::COMPLEX ? dft_size : dft_size / 2 + 1;
    std::vector<backward_t> host_bwd(bwd_size);
    rand_vector(host_fwd, dft_size);

    // Allocate enough memory that we don't have to worry about the domain.
    forward_t* device_fwd = sycl::malloc_device<forward_t>(dft_size, sycl_queue);
    backward_t* deviceBwd = sycl::malloc_device<backward_t>(bwd_size, sycl_queue);
    sycl_queue.copy(host_fwd.data(), device_fwd, dft_size);
    sycl_queue.wait_and_throw();

    compute_forward<decltype(desc), forward_t, backward_t>(desc, device_fwd, deviceBwd);
    sycl_queue.wait_and_throw();

    sycl_queue.copy(deviceBwd, host_bwd.data(), bwd_size);
    sycl_queue.wait_and_throw();

    // To see external workspaces, larger sizes of DFT may be needed. Using the reference DFT with larger sizes is slow,
    // so use Parseval's theorum as a sanity check instead.
    bool sanityCheckPasses = parseval_check(dft_size, host_fwd.data(), host_bwd.data());

    if (sanityCheckPasses) {
        sycl_queue.copy(host_fwd.data(), device_fwd, dft_size);
        sycl_queue.wait_and_throw();
        compute_backward<decltype(desc), backward_t, forward_t>(desc, deviceBwd, device_fwd);
        sycl_queue.wait_and_throw();
        sycl_queue.copy(device_fwd, host_fwd.data(), dft_size);
        sycl_queue.wait_and_throw();
        forward_t rescale =
            static_cast<forward_t>(1) / static_cast<forward_t>(static_cast<scalar_t>(dft_size));
        sanityCheckPasses = parseval_check(dft_size, host_fwd.data(), host_bwd.data(), rescale);
    }

    sycl::free(device_fwd, sycl_queue);
    sycl::free(deviceBwd, sycl_queue);
    sycl::free(workspace, sycl_queue);
    return sanityCheckPasses ? !::testing::Test::HasFailure() : ::testing::Test::HasFailure();
}

template <oneapi::math::dft::precision prec, oneapi::math::dft::domain dom>
int test_workspace_external_buffer_impl(std::size_t dft_size, sycl::device* dev) {
    using namespace oneapi::math::dft;
    using scalar_t = std::conditional_t<prec == precision::DOUBLE, double, float>;
    using forward_t = std::conditional_t<dom == domain::COMPLEX, std::complex<scalar_t>, scalar_t>;
    using backward_t = std::complex<scalar_t>;

    sycl::queue sycl_queue(*dev);
    if (prec == precision::DOUBLE && !sycl_queue.get_device().has(sycl::aspect::fp64)) {
        std::cout << "Device does not support double precision." << std::endl;
        return test_skipped;
    }
    descriptor<prec, dom> desc(static_cast<std::int64_t>(dft_size));

    desc.set_value(config_param::WORKSPACE_PLACEMENT, config_value::WORKSPACE_EXTERNAL);
    desc.set_value(config_param::PLACEMENT, config_value::NOT_INPLACE);
    try {
        commit_descriptor(desc, sycl_queue);
    }
    catch (oneapi::math::unimplemented&) {
        std::cout << "Test configuration not implemented." << std::endl;
        return test_skipped;
    }
    std::int64_t workspace_bytes = -1;
    desc.get_value(config_param::WORKSPACE_EXTERNAL_BYTES, &workspace_bytes);
    if (workspace_bytes < 0) {
        return ::testing::Test::HasFailure();
    }
    sycl::buffer<scalar_t> workspace(static_cast<std::size_t>(workspace_bytes) / sizeof(scalar_t));
    desc.set_workspace(workspace);
    // Generate data
    std::vector<forward_t> host_fwd(static_cast<std::size_t>(dft_size));
    std::size_t bwd_size =
        dom == domain::COMPLEX ? dft_size : dft_size / 2 + 1; // TODO: Check this!
    std::vector<backward_t> host_bwd(bwd_size);
    rand_vector(host_fwd, dft_size);
    auto host_fwdCpy = host_fwd; // Some backends modify the input data (rocFFT).

    {
        sycl::buffer<forward_t> buf_fwd(host_fwd);
        sycl::buffer<backward_t> buf_bwd(host_bwd);
        compute_forward<decltype(desc), forward_t, backward_t>(desc, buf_fwd, buf_bwd);
    }

    // To see external workspaces, larger sizes of DFT may be needed. Using the reference DFT with larger sizes is slow,
    // so use Parseval's theorum as a sanity check instead.
    bool sanityCheckPasses = parseval_check(dft_size, host_fwdCpy.data(), host_bwd.data());

    if (sanityCheckPasses) {
        auto host_bwdCpy = host_bwd;
        {
            sycl::buffer<forward_t> buf_fwd(host_fwd);
            sycl::buffer<backward_t> buf_bwd(host_bwd);
            compute_backward<decltype(desc), backward_t, forward_t>(desc, buf_bwd, buf_fwd);
            sycl_queue.wait_and_throw();
        }
        forward_t rescale =
            static_cast<forward_t>(1) / static_cast<forward_t>(static_cast<scalar_t>(dft_size));
        sanityCheckPasses = parseval_check(dft_size, host_fwd.data(), host_bwdCpy.data(), rescale);
    }

    return sanityCheckPasses ? !::testing::Test::HasFailure() : ::testing::Test::HasFailure();
}

template <oneapi::math::dft::precision prec, oneapi::math::dft::domain dom>
void test_workspace_external_usm(sycl::device* dev) {
    EXPECT_TRUEORSKIP((test_workspace_external_usm_impl<prec, dom>(2, dev)));
    EXPECT_TRUEORSKIP((test_workspace_external_usm_impl<prec, dom>(1024 * 3 * 5 * 7 * 16, dev)));
}

template <oneapi::math::dft::precision prec, oneapi::math::dft::domain dom>
void test_workspace_external_buffer(sycl::device* dev) {
    EXPECT_TRUEORSKIP((test_workspace_external_buffer_impl<prec, dom>(2, dev)));
    EXPECT_TRUEORSKIP((test_workspace_external_buffer_impl<prec, dom>(1024 * 3 * 5 * 7 * 16, dev)));
}

TEST_P(WorkspaceExternalTests, TestWorkspaceExternalSingleUsm) {
    using precision = oneapi::math::dft::precision;
    using domain = oneapi::math::dft::domain;
    test_workspace_external_usm<precision::SINGLE, domain::REAL>(GetParam());
    test_workspace_external_usm<precision::SINGLE, domain::COMPLEX>(GetParam());
}

TEST_P(WorkspaceExternalTests, TestWorkspaceExternalDoubleUsm) {
    using precision = oneapi::math::dft::precision;
    using domain = oneapi::math::dft::domain;
    test_workspace_external_usm<precision::DOUBLE, domain::REAL>(GetParam());
    test_workspace_external_usm<precision::DOUBLE, domain::COMPLEX>(GetParam());
}

TEST_P(WorkspaceExternalTests, TestWorkspaceExternalSingleBuffer) {
    using precision = oneapi::math::dft::precision;
    using domain = oneapi::math::dft::domain;
    test_workspace_external_buffer<precision::SINGLE, domain::REAL>(GetParam());
    test_workspace_external_buffer<precision::SINGLE, domain::COMPLEX>(GetParam());
}

TEST_P(WorkspaceExternalTests, TestWorkspaceExternalDoubleBuffer) {
    using precision = oneapi::math::dft::precision;
    using domain = oneapi::math::dft::domain;
    test_workspace_external_buffer<precision::DOUBLE, domain::REAL>(GetParam());
    test_workspace_external_buffer<precision::DOUBLE, domain::COMPLEX>(GetParam());
}

/// A test where set_workspace is called when an external workspace is not set.
TEST_P(WorkspaceExternalTests, SetWorkspaceOnWorkspaceAutomatic) {
    using namespace oneapi::math::dft;
    sycl::queue sycl_queue(*GetParam());
    const int dft_len = 1024 * 3 * 5 * 7 * 16; // A size likely to require an external workspace.
    float* fft_data_usm = sycl::malloc_device<float>(dft_len * 2, sycl_queue);
    sycl::buffer<float> fft_data_buf(dft_len * 2);
    descriptor<precision::SINGLE, domain::COMPLEX> desc_usm(dft_len), desc_buf(dft_len);
    try {
        // WORKSPACE_EXTERNAL is NOT set.
        commit_descriptor(desc_usm, sycl_queue);
        commit_descriptor(desc_buf, sycl_queue);
    }
    catch (oneapi::math::unimplemented&) {
        // The DFT size may not be supported. Use a size that is likely to be supported, even if
        // that means no external workspace is actually used.
        descriptor<precision::SINGLE, domain::COMPLEX> desc_usm2(2), desc_buf2(2);
        desc_usm = std::move(desc_usm2);
        desc_buf = std::move(desc_buf2);
        commit_descriptor(desc_usm, sycl_queue);
        commit_descriptor(desc_buf, sycl_queue);
    }
    std::int64_t workspace_bytes = 0;
    desc_usm.get_value(config_param::WORKSPACE_EXTERNAL_BYTES, &workspace_bytes);

    // No workspace set yet: all of the following should work.
    compute_forward(desc_usm, fft_data_usm);
    compute_forward(desc_buf, fft_data_buf);
    compute_backward(desc_usm, fft_data_usm);
    compute_backward(desc_buf, fft_data_buf);
    compute_forward(desc_usm, fft_data_buf);
    compute_forward(desc_buf, fft_data_usm);
    compute_backward(desc_usm, fft_data_buf);
    compute_backward(desc_buf, fft_data_usm);
    sycl_queue.wait_and_throw();

    // Set workspace
    float* usm_workspace = sycl::malloc_device<float>(
        static_cast<std::size_t>(workspace_bytes) / sizeof(float), sycl_queue);
    sycl::buffer<float> bufferWorkspace(static_cast<std::size_t>(workspace_bytes) / sizeof(float));
    desc_usm.set_workspace(usm_workspace);
    desc_buf.set_workspace(bufferWorkspace);

    // Should work:
    compute_forward(desc_usm, fft_data_usm);
    sycl_queue.wait_and_throw();
    compute_forward(desc_buf, fft_data_buf);
    sycl_queue.wait_and_throw();
    compute_backward(desc_usm, fft_data_usm);
    sycl_queue.wait_and_throw();
    compute_backward(desc_buf, fft_data_buf);
    sycl_queue.wait_and_throw();

    // Should not work:
    EXPECT_THROW(compute_forward(desc_usm, fft_data_buf), oneapi::math::invalid_argument);
    EXPECT_THROW(compute_forward(desc_buf, fft_data_usm), oneapi::math::invalid_argument);
    EXPECT_THROW(compute_backward(desc_usm, fft_data_buf), oneapi::math::invalid_argument);
    EXPECT_THROW(compute_backward(desc_buf, fft_data_usm), oneapi::math::invalid_argument);
    sycl_queue.wait_and_throw();

    // Free any allocations:
    sycl::free(usm_workspace, sycl_queue);
    sycl::free(fft_data_usm, sycl_queue);
}

/// Test that the implementation throws as expected.
TEST_P(WorkspaceExternalTests, ThrowOnBadCalls) {
    using namespace oneapi::math::dft;
    sycl::queue sycl_queue(*GetParam());
    const int dft_len = 1024 * 3 * 5 * 7 * 16; // A size likely to require an external workspace.
    float* fft_data_usm = sycl::malloc_device<float>(dft_len * 2, sycl_queue);
    sycl::buffer<float> fft_data_buf(dft_len * 2);
    descriptor<precision::SINGLE, domain::COMPLEX> desc_usm(dft_len), desc_buf(dft_len);
    desc_usm.set_value(config_param::WORKSPACE_PLACEMENT, config_value::WORKSPACE_EXTERNAL);
    desc_buf.set_value(config_param::WORKSPACE_PLACEMENT, config_value::WORKSPACE_EXTERNAL);
    // We expect the following to throw because the decriptor has not been committed.
    std::int64_t workspace_bytes = -10;
    float* usm_workspace = nullptr;
    EXPECT_THROW(desc_usm.get_value(config_param::WORKSPACE_EXTERNAL_BYTES, &workspace_bytes),
                 oneapi::math::invalid_argument);
    EXPECT_THROW(desc_usm.set_workspace(usm_workspace), oneapi::math::uninitialized);
    try {
        commit_descriptor(desc_usm, sycl_queue);
        commit_descriptor(desc_buf, sycl_queue);
    }
    catch (oneapi::math::unimplemented&) {
        // DFT size may not be supported. Use a DFT size that probably will be, even if it
        // won't actually use an external workspace internally.
        descriptor<precision::SINGLE, domain::COMPLEX> desc_usm2(2), desc_buf2(2);
        desc_usm = std::move(desc_usm2);
        desc_buf = std::move(desc_buf2);
        desc_usm.set_value(config_param::WORKSPACE_PLACEMENT, config_value::WORKSPACE_EXTERNAL);
        desc_buf.set_value(config_param::WORKSPACE_PLACEMENT, config_value::WORKSPACE_EXTERNAL);
        commit_descriptor(desc_usm, sycl_queue);
        commit_descriptor(desc_buf, sycl_queue);
    }

    desc_usm.get_value(config_param::WORKSPACE_EXTERNAL_BYTES, &workspace_bytes);
    EXPECT_GE(workspace_bytes, 0);

    // We haven't set a workspace, so the following should fail;
    EXPECT_THROW(compute_forward(desc_usm, fft_data_usm), oneapi::math::invalid_argument);
    sycl_queue.wait_and_throw();
    EXPECT_THROW(compute_forward(desc_usm, fft_data_buf), oneapi::math::invalid_argument);
    sycl_queue.wait_and_throw();

    if (workspace_bytes > 0) {
        EXPECT_THROW(desc_usm.set_workspace(nullptr), oneapi::math::invalid_argument);
        sycl::buffer<float> undersize_workspace(
            static_cast<std::size_t>(workspace_bytes) / sizeof(float) - 1);
        EXPECT_THROW(desc_buf.set_workspace(undersize_workspace), oneapi::math::invalid_argument);
    }

    usm_workspace = sycl::malloc_device<float>(
        static_cast<std::size_t>(workspace_bytes) / sizeof(float), sycl_queue);
    sycl::buffer<float> bufferWorkspace(static_cast<std::size_t>(workspace_bytes) / sizeof(float));

    desc_usm.set_workspace(usm_workspace);
    desc_buf.set_workspace(bufferWorkspace);

    // Should work:
    compute_forward(desc_usm, fft_data_usm);
    sycl_queue.wait_and_throw();
    compute_forward(desc_buf, fft_data_buf);
    sycl_queue.wait_and_throw();
    compute_backward(desc_usm, fft_data_usm);
    sycl_queue.wait_and_throw();
    compute_backward(desc_buf, fft_data_buf);
    sycl_queue.wait_and_throw();

    // Should not work:
    EXPECT_THROW(compute_forward(desc_usm, fft_data_buf), oneapi::math::invalid_argument);
    EXPECT_THROW(compute_forward(desc_buf, fft_data_usm), oneapi::math::invalid_argument);
    EXPECT_THROW(compute_backward(desc_usm, fft_data_buf), oneapi::math::invalid_argument);
    EXPECT_THROW(compute_backward(desc_buf, fft_data_usm), oneapi::math::invalid_argument);
    sycl_queue.wait_and_throw();

    // Free any allocations:
    sycl::free(usm_workspace, sycl_queue);
    sycl::free(fft_data_usm, sycl_queue);
}

TEST_P(WorkspaceExternalTests, RecommitBehaviour) {
    using namespace oneapi::math::dft;
    sycl::queue sycl_queue(*GetParam());
    const int dft_len = 1024 * 3 * 5 * 7 * 16; // A size likely to require an external workspace.
    float* fft_data_usm = sycl::malloc_device<float>(dft_len * 2, sycl_queue);
    descriptor<precision::SINGLE, domain::COMPLEX> desc_usm(dft_len);
    try {
        // WORKSPACE_EXTERNAL is NOT set.
        commit_descriptor(desc_usm, sycl_queue);
    }
    catch (oneapi::math::unimplemented&) {
        // DFT size may not be supported. Use a DFT size that probably will be, even if it
        // won't actually use an external workspace internally.
        descriptor<precision::SINGLE, domain::COMPLEX> desc_usm2(2);
        desc_usm = std::move(desc_usm2);
        commit_descriptor(desc_usm, sycl_queue);
    }
    std::int64_t workspace_bytes = 0;
    desc_usm.get_value(config_param::WORKSPACE_EXTERNAL_BYTES, &workspace_bytes);
    float* usm_workspace = sycl::malloc_device<float>(
        static_cast<std::size_t>(workspace_bytes) / sizeof(float), sycl_queue);

    // Should work with workspace automatic
    compute_forward(desc_usm, fft_data_usm);
    sycl_queue.wait_and_throw();

    desc_usm.set_value(config_param::WORKSPACE_PLACEMENT, config_value::WORKSPACE_EXTERNAL);
    commit_descriptor(desc_usm, sycl_queue);

    // No workspace, expect throw
    EXPECT_THROW(compute_forward(desc_usm, fft_data_usm), oneapi::math::invalid_argument);

    desc_usm.set_workspace(usm_workspace);

    compute_forward(desc_usm, fft_data_usm);
    sycl_queue.wait_and_throw();

    // Recommitting should require workspace to be set again.
    commit_descriptor(desc_usm, sycl_queue);
    EXPECT_THROW(compute_forward(desc_usm, fft_data_usm), oneapi::math::invalid_argument);
    sycl_queue.wait_and_throw();

    // Free any allocations:
    sycl::free(usm_workspace, sycl_queue);
    sycl::free(fft_data_usm, sycl_queue);
}

INSTANTIATE_TEST_SUITE_P(WorkspaceExternalTestSuite, WorkspaceExternalTests,
                         testing::ValuesIn(devices), ::DeviceNamePrint());
