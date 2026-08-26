/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include "allocator_helper.hpp"
#include "cblas.h"
#include "oneapi/math/detail/config.hpp"
#include "oneapi/math.hpp"
#include "onemath_blas_helper.hpp"
#include "reference_blas_templates.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include <gtest/gtest.h>

using namespace sycl;
using std::vector;

extern std::vector<sycl::device*> devices;

namespace {

template <typename Ta, typename Tb, typename Tc, typename Ts>
int test(device* dev, oneapi::math::layout layout, int64_t batch_size) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during GEMM_BATCH_STRIDE:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    queue main_queue(*dev, exception_handler);
    context cxt = main_queue.get_context();
    event done;
    std::vector<event> dependencies;

    // Prepare data.
    int64_t m, n, k;
    int64_t lda, ldb, ldc;
    oneapi::math::transpose transa, transb;
    Ts alpha, beta;

    int64_t i, tmp;

    batch_size = 1 + std::rand() % 20;
    m = 1 + std::rand() % 500;
    n = 1 + std::rand() % 500;
    k = 1 + std::rand() % 500;
    lda = std::max(m, k);
    ldb = std::max(n, k);
    ldc = std::max(m, n);
    alpha = rand_scalar<Ts>();
    beta = rand_scalar<Ts>();
    if ((std::is_same<Ts, std::complex<float>>::value) ||
        (std::is_same<Ts, std::complex<double>>::value)) {
        tmp = std::rand() % 3;
        if (tmp == 2)
            transa = oneapi::math::transpose::conjtrans;
        else
            transa = (oneapi::math::transpose)tmp;
        tmp = std::rand() % 3;
        if (tmp == 2)
            transb = oneapi::math::transpose::conjtrans;
        else
            transb = (oneapi::math::transpose)tmp;
    }
    else {
        transa = (oneapi::math::transpose)(std::rand() % 2);
        transb = (oneapi::math::transpose)(std::rand() % 2);
    }

    int64_t stride_a, stride_b, stride_c;

    switch (layout) {
        case oneapi::math::layout::col_major:
            stride_a = (transa == oneapi::math::transpose::nontrans) ? lda * k : lda * m;
            stride_b = (transb == oneapi::math::transpose::nontrans) ? ldb * n : ldb * k;
            stride_c = ldc * n;
            break;
        case oneapi::math::layout::row_major:
            stride_a = (transa == oneapi::math::transpose::nontrans) ? lda * m : lda * k;
            stride_b = (transb == oneapi::math::transpose::nontrans) ? ldb * k : ldb * n;
            stride_c = ldc * m;
            break;
        default: break;
    }

    auto ua = usm_allocator<Ta, usm::alloc::shared, 64>(cxt, *dev);
    auto ub = usm_allocator<Tb, usm::alloc::shared, 64>(cxt, *dev);
    auto uc = usm_allocator<Tc, usm::alloc::shared, 64>(cxt, *dev);
    auto us = usm_allocator<Ts, usm::alloc::shared, 64>(cxt, *dev);
    vector<Ta, decltype(ua)> A(ua);
    vector<Tb, decltype(ub)> B(ub);
    vector<Tc, decltype(uc)> C(uc), C_cast_ref(uc);
    vector<Ts, decltype(us)> A_ref(us), B_ref(us), C_ref(us);

    A.resize(stride_a * batch_size);
    B.resize(stride_b * batch_size);
    C.resize(stride_c * batch_size);
    A_ref.resize(stride_c * batch_size);
    B_ref.resize(stride_c * batch_size);
    C_ref.resize(stride_c * batch_size);
    C_cast_ref.resize(stride_c * batch_size);

    Ta** a_array = (Ta**)oneapi::math::malloc_shared(64, sizeof(Ta*) * batch_size, *dev, cxt);
    Tb** b_array = (Tb**)oneapi::math::malloc_shared(64, sizeof(Tb*) * batch_size, *dev, cxt);
    Tc** c_array = (Tc**)oneapi::math::malloc_shared(64, sizeof(Tc*) * batch_size, *dev, cxt);
    Ts** c_ref_array = (Ts**)oneapi::math::malloc_shared(64, sizeof(Ts*) * batch_size, *dev, cxt);

    if ((a_array == NULL) || (b_array == NULL) || (c_array == NULL) || (c_ref_array == NULL)) {
        std::cout << "Error cannot allocate arrays of pointers\n";
        oneapi::math::free_shared(a_array, cxt);
        oneapi::math::free_shared(b_array, cxt);
        oneapi::math::free_shared(c_array, cxt);
        oneapi::math::free_shared(c_ref_array, cxt);
        return false;
    }

    for (i = 0; i < batch_size; i++) {
        a_array[i] = &A[i * stride_a];
        b_array[i] = &B[i * stride_b];
        c_array[i] = &C[i * stride_c];
        c_ref_array[i] = &C_ref[i * stride_c];
    }

    rand_matrix(A, oneapi::math::layout::col_major, oneapi::math::transpose::nontrans,
                stride_a * batch_size, 1, stride_a * batch_size);
    rand_matrix(B, oneapi::math::layout::col_major, oneapi::math::transpose::nontrans,
                stride_b * batch_size, 1, stride_b * batch_size);
    rand_matrix(C, oneapi::math::layout::col_major, oneapi::math::transpose::nontrans,
                stride_c * batch_size, 1, stride_c * batch_size);
    copy_matrix(A, oneapi::math::layout::col_major, oneapi::math::transpose::nontrans,
                stride_a * batch_size, 1, stride_a * batch_size, A_ref);
    copy_matrix(B, oneapi::math::layout::col_major, oneapi::math::transpose::nontrans,
                stride_b * batch_size, 1, stride_b * batch_size, B_ref);
    copy_matrix(C, oneapi::math::layout::col_major, oneapi::math::transpose::nontrans,
                stride_c * batch_size, 1, stride_c * batch_size, C_ref);

    // Call reference GEMM_BATCH_STRIDE.
    using fp_ref = typename ref_type_info<Ts>::type;
    int m_ref = (int)m;
    int n_ref = (int)n;
    int k_ref = (int)k;
    int lda_ref = (int)lda;
    int ldb_ref = (int)ldb;
    int ldc_ref = (int)ldc;
    int batch_size_ref = (int)batch_size;
    for (i = 0; i < batch_size_ref; i++) {
        ::gemm(convert_to_cblas_layout(layout), convert_to_cblas_trans(transa),
               convert_to_cblas_trans(transb), (const int*)&m_ref, (const int*)&n_ref,
               (const int*)&k_ref, (const fp_ref*)&alpha,
               (const fp_ref*)(A_ref.data() + stride_a * i), (const int*)&lda_ref,
               (const fp_ref*)(B_ref.data() + stride_b * i), (const int*)&ldb_ref,
               (const fp_ref*)&beta, (fp_ref*)(C_ref.data() + stride_c * i), (const int*)&ldc_ref);
    }

    // Call DPC++ GEMM_BATCH_STRIDE.

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::math::layout::col_major:
                done = oneapi::math::blas::column_major::gemm_batch(
                    main_queue, transa, transb, m, n, k, alpha, &A[0], lda, stride_a, &B[0], ldb,
                    stride_b, beta, &C[0], ldc, stride_c, batch_size, dependencies);
                break;
            case oneapi::math::layout::row_major:
                done = oneapi::math::blas::row_major::gemm_batch(
                    main_queue, transa, transb, m, n, k, alpha, &A[0], lda, stride_a, &B[0], ldb,
                    stride_b, beta, &C[0], ldc, stride_c, batch_size, dependencies);
                break;
            default: break;
        }
        done.wait_and_throw();
#else
        switch (layout) {
            case oneapi::math::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::column_major::gemm_batch,
                                        transa, transb, m, n, k, alpha, &A[0], lda, stride_a, &B[0],
                                        ldb, stride_b, beta, &C[0], ldc, stride_c, batch_size,
                                        dependencies);
                break;
            case oneapi::math::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::row_major::gemm_batch,
                                        transa, transb, m, n, k, alpha, &A[0], lda, stride_a, &B[0],
                                        ldb, stride_b, beta, &C[0], ldc, stride_c, batch_size,
                                        dependencies);
                break;
            default: break;
        }
        main_queue.wait_and_throw();
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during GEMM_BATCH_STRIDE:\n"
                  << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::math::unimplemented& e) {
        oneapi::math::free_shared(a_array, cxt);
        oneapi::math::free_shared(b_array, cxt);
        oneapi::math::free_shared(c_array, cxt);
        oneapi::math::free_shared(c_ref_array, cxt);
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of GEMM_BATCH_STRIDE:\n"
                  << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    int tol_scalar = 10;
    int error_mag = tol_scalar * k;
    if (std::is_same_v<Tc, int32_t>)
        error_mag = 1;

    // A float output accumulated from int8 inputs is rounded at the magnitude of the terms summed,
    // |alpha| * sum|a*b|, which k * 128 * 128 bounds from above. An entry whose sum cancels is far
    // smaller than that and so cannot meet any relative bound, so allow an absolute error of eps
    // times the accumulated magnitude instead. Int8Int8SinglePrecisionErrorModel checks that the
    // error really does stay inside eps * |alpha| * sum|a*b| on fixed data.
    constexpr bool int8_to_float = std::is_same_v<Ta, std::int8_t> &&
                                   std::is_same_v<Tb, std::int8_t> && std::is_same_v<Tc, float> &&
                                   std::is_same_v<Ts, float>;
    double abs_error_bound = 0.0;
    if constexpr (int8_to_float)
        abs_error_bound = std::numeric_limits<float>::epsilon() * std::abs(double(alpha)) *
                          double(k) * 128.0 * 128.0;

    for (size_t i = 0; i < C_ref.size(); ++i) {
        C_cast_ref[i] = C_ref[i];
    }
    bool good = check_almost_equal_matrix(C, C_cast_ref, oneapi::math::layout::col_major,
                                          stride_c * batch_size, 1, stride_c * batch_size,
                                          error_mag, std::cout, abs_error_bound);

    oneapi::math::free_shared(a_array, cxt);
    oneapi::math::free_shared(b_array, cxt);
    oneapi::math::free_shared(c_array, cxt);
    oneapi::math::free_shared(c_ref_array, cxt);

    return (int)good;
}

// Regression test for the int8-to-float tolerance above. The sizes and data are fixed rather than
// drawn from std::rand(), so this does not depend on the order the tests run in, and the expected
// result is accumulated exactly in integers, so it does not depend on the reference BLAS either.
// The leading rows of A and columns of B are built to cancel exactly, which puts those entries out
// of reach of any relative bound and leaves the absolute bound as the only one that can accept
// them. Every entry is checked against that pair of bounds, and against the accumulation error
// model the absolute bound is calibrated from: eps times the magnitude of the terms summed.
int int8_accumulation_error_model(device* dev, oneapi::math::layout layout) {
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during GEMM_BATCH_STRIDE:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    queue main_queue(*dev, exception_handler);
    context cxt = main_queue.get_context();
    event done;
    std::vector<event> dependencies;

    const auto transa = oneapi::math::transpose::nontrans;
    const auto transb = oneapi::math::transpose::nontrans;
    // Shapes are not interchangeable here: a backend may accumulate some of them exactly, in which
    // case no tolerance is needed and nothing exercises this one. This shape leaves a rounding
    // error large enough for the cancelling entries below to fall back on the absolute bound.
    const int64_t m = 466, n = 15, batch_size = 2;
    const int64_t k = 141; // a multiple of three, for the cancelling triples built below
    const int64_t cancelling = 8; // leading rows of A and columns of B that cancel exactly
    // alpha is not a power of two, so that scaling the terms of the sum rounds.
    const float alpha = 0.3f, beta = 0.25f;

    const bool col = layout == oneapi::math::layout::col_major;
    const int64_t lda = col ? m : k, ldb = col ? k : n, ldc = col ? m : n;
    const int64_t stride_a = col ? lda * k : lda * m;
    const int64_t stride_b = col ? ldb * n : ldb * k;
    const int64_t stride_c = col ? ldc * n : ldc * m;
    auto a_at = [=](int64_t i, int64_t l) {
        return col ? i + l * lda : i * lda + l;
    };
    auto b_at = [=](int64_t l, int64_t j) {
        return col ? l + j * ldb : l * ldb + j;
    };
    auto c_at = [=](int64_t i, int64_t j) {
        return col ? i + j * ldc : i * ldc + j;
    };

    auto ua = usm_allocator<std::int8_t, usm::alloc::shared, 64>(cxt, *dev);
    auto uc = usm_allocator<float, usm::alloc::shared, 64>(cxt, *dev);
    vector<std::int8_t, decltype(ua)> A(stride_a * batch_size, ua), B(stride_b * batch_size, ua);
    vector<float, decltype(uc)> C(stride_c * batch_size, uc);
    std::vector<float> C_in(stride_c * batch_size);

    std::uint32_t seed = 20250814u;
    auto next = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return seed >> 16;
    };
    for (int64_t b = 0; b < batch_size; b++) {
        for (int64_t i = 0; i < m; i++)
            for (int64_t l = 0; l < k; l++)
                A[b * stride_a + a_at(i, l)] = std::int8_t(int(next() % 254) - 127);
        for (int64_t l = 0; l < k; l++)
            for (int64_t j = 0; j < n; j++)
                B[b * stride_b + b_at(l, j)] = std::int8_t(int(next() % 254) - 127);
        // The leading rows of A and columns of B are built from triples whose products are 5x, -3x
        // and -2x, so the exact dot product of any such row with any such column is zero while the
        // terms summed stay large. None of the three is a power of two times another, so rounding
        // the scaled terms does not cancel along with the terms themselves.
        for (int64_t i = 0; i < cancelling; i++)
            for (int64_t l = 0; l < k; l += 3) {
                const int g = int(next() % 25) + 1;
                A[b * stride_a + a_at(i, l)] = std::int8_t(5 * g);
                A[b * stride_a + a_at(i, l + 1)] = std::int8_t(3 * g);
                A[b * stride_a + a_at(i, l + 2)] = std::int8_t(2 * g);
            }
        for (int64_t j = 0; j < cancelling; j++)
            for (int64_t l = 0; l < k; l += 3) {
                const int h = (int(next() % 127) + 1) * (next() % 2 ? 1 : -1);
                B[b * stride_b + b_at(l, j)] = std::int8_t(h);
                B[b * stride_b + b_at(l + 1, j)] = std::int8_t(-h);
                B[b * stride_b + b_at(l + 2, j)] = std::int8_t(-h);
            }
        for (int64_t j = 0; j < n; j++)
            for (int64_t i = 0; i < m; i++) {
                const auto idx = b * stride_c + c_at(i, j);
                C[idx] = float(next() % 1024) / 512.0f - 1.0f;
                C_in[idx] = C[idx];
            }
    }

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::math::layout::col_major:
                done = oneapi::math::blas::column_major::gemm_batch(
                    main_queue, transa, transb, m, n, k, alpha, &A[0], lda, stride_a, &B[0], ldb,
                    stride_b, beta, &C[0], ldc, stride_c, batch_size, dependencies);
                break;
            case oneapi::math::layout::row_major:
                done = oneapi::math::blas::row_major::gemm_batch(
                    main_queue, transa, transb, m, n, k, alpha, &A[0], lda, stride_a, &B[0], ldb,
                    stride_b, beta, &C[0], ldc, stride_c, batch_size, dependencies);
                break;
            default: break;
        }
        done.wait_and_throw();
#else
        switch (layout) {
            case oneapi::math::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::column_major::gemm_batch,
                                        transa, transb, m, n, k, alpha, &A[0], lda, stride_a, &B[0],
                                        ldb, stride_b, beta, &C[0], ldc, stride_c, batch_size,
                                        dependencies);
                break;
            case oneapi::math::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::row_major::gemm_batch,
                                        transa, transb, m, n, k, alpha, &A[0], lda, stride_a, &B[0],
                                        ldb, stride_b, beta, &C[0], ldc, stride_c, batch_size,
                                        dependencies);
                break;
            default: break;
        }
        main_queue.wait_and_throw();
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during GEMM_BATCH_STRIDE:\n"
                  << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::math::unimplemented& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of GEMM_BATCH_STRIDE:\n"
                  << error.what() << std::endl;
    }

    const double eps = std::numeric_limits<float>::epsilon();
    // The same pair of bounds the int8-to-float tests above apply, evaluated here against an exact
    // integer reference: a relative bound of 10 * k * eps, or an absolute one of eps times the
    // bound k * 128 * 128 on the accumulated magnitude.
    const double relative_bound = double(10 * k) * eps;
    const double absolute_bound = eps * std::abs(double(alpha)) * double(k) * 128.0 * 128.0;
    double worst_model_usage = 0.0, worst_absolute_usage = 0.0;
    double worst_cancelling_relative_allowance = 0.0;
    int64_t entries_missing_relative_bound = 0, cancelling_missing_relative_bound = 0;
    bool good = true;
    for (int64_t b = 0; b < batch_size; b++) {
        const std::int8_t* Ab = &A[b * stride_a];
        const std::int8_t* Bb = &B[b * stride_b];
        for (int64_t j = 0; j < n; j++)
            for (int64_t i = 0; i < m; i++) {
                std::int64_t dot = 0, abs_sum = 0;
                for (int64_t l = 0; l < k; l++) {
                    const std::int64_t a = Ab[a_at(i, l)], bb = Bb[b_at(l, j)];
                    dot += a * bb;
                    abs_sum += std::abs(a * bb);
                }
                const auto idx = b * stride_c + c_at(i, j);
                const double expected =
                    double(alpha) * double(dot) + double(beta) * double(C_in[idx]);
                const double error = std::abs(double(C[idx]) - expected);
                const bool cancels = i < cancelling && j < cancelling;
                if (cancels && dot != 0) {
                    std::cout << "test bug: entry (" << i << "," << j
                              << ") was built to cancel but its dot product is " << dot
                              << std::endl;
                    return false;
                }

                // The error the accumulation is allowed: eps times the magnitude of the terms
                // summed, plus the scaling of C and the final addition. Exceeding this means the
                // calibration the absolute tolerance rests on no longer describes the backend.
                const double model_bound =
                    eps * (std::abs(double(alpha)) * double(abs_sum) +
                           std::abs(double(beta) * double(C_in[idx])) + std::abs(expected));
                worst_model_usage =
                    std::max(worst_model_usage, model_bound > 0.0 ? error / model_bound : 0.0);
                worst_absolute_usage = std::max(worst_absolute_usage, error / absolute_bound);
                if (error > model_bound)
                    good = false;

                if (cancels)
                    worst_cancelling_relative_allowance = std::max(
                        worst_cancelling_relative_allowance, relative_bound * std::abs(expected));
                if (error > relative_bound * std::abs(expected)) {
                    entries_missing_relative_bound++;
                    if (cancels)
                        cancelling_missing_relative_bound++;
                    if (error > absolute_bound)
                        good = false;
                }
            }
    }

    // The cancelling entries are the ones the absolute bound exists for: whatever error the
    // backend makes on them, the relative bound can only accept a fraction of what the model
    // permits, so they rest on the absolute bound alone.
    if (worst_cancelling_relative_allowance >= absolute_bound) {
        std::cout << "test bug: the relative bound already covers the cancelling entries, so they "
                     "do not exercise the absolute tolerance"
                  << std::endl;
        return false;
    }

    std::cout << "int8 accumulation error reached " << worst_model_usage
              << " of the accumulated magnitude the model allows and " << worst_absolute_usage
              << " of the absolute tolerance; " << entries_missing_relative_bound
              << " entries missed the relative bound, " << cancelling_missing_relative_bound
              << " of them cancelling" << std::endl;
    if (!good)
        std::cout << "int8 accumulation error exceeded the tolerance the int8-to-float gemm_batch "
                     "tests rely on"
                  << std::endl;
    return good;
}

// The rocBLAS int8-to-float fallback scales its int32 accumulator in a kernel of its own, and
// launches it over a flat range because HIP cannot map every large multi-dimensional range onto
// its grid. A large prime n keeps that flat launch covered.
int int8_flat_range_large_prime(device* dev, oneapi::math::layout layout) {
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during GEMM_BATCH_STRIDE:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    queue main_queue(*dev, exception_handler);
    context cxt = main_queue.get_context();
    std::vector<event> dependencies;
    const int64_t m = 3, n = 65537, k = 1, batch_size = 1;
    const bool col = layout == oneapi::math::layout::col_major;
    const int64_t lda = col ? m : k;
    const int64_t ldb = col ? k : n;
    const int64_t ldc = col ? m : n;
    const int64_t stride_a = col ? lda * k : lda * m;
    const int64_t stride_b = col ? ldb * n : ldb * k;
    const int64_t stride_c = col ? ldc * n : ldc * m;

    auto ua = usm_allocator<std::int8_t, usm::alloc::shared, 64>(cxt, *dev);
    auto uc = usm_allocator<float, usm::alloc::shared, 64>(cxt, *dev);
    vector<std::int8_t, decltype(ua)> A(stride_a, ua), B(stride_b, ua);
    vector<float, decltype(uc)> C(stride_c, uc);
    for (int64_t i = 0; i < m; ++i)
        A[col ? i : i * lda] = std::int8_t(i + 1);
    for (int64_t j = 0; j < n; ++j)
        B[col ? j * ldb : j] = std::int8_t(j % 7 - 3);
    std::fill(C.begin(), C.end(), -1.0f);

    try {
        event done;
#ifdef CALL_RT_API
        if (col) {
            done = oneapi::math::blas::column_major::gemm_batch(
                main_queue, oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans, m,
                n, k, 1.0f, &A[0], lda, stride_a, &B[0], ldb, stride_b, 0.0f, &C[0], ldc, stride_c,
                batch_size, dependencies);
        }
        else {
            done = oneapi::math::blas::row_major::gemm_batch(
                main_queue, oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans, m,
                n, k, 1.0f, &A[0], lda, stride_a, &B[0], ldb, stride_b, 0.0f, &C[0], ldc, stride_c,
                batch_size, dependencies);
        }
        done.wait_and_throw();
#else
        if (col) {
            TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::column_major::gemm_batch,
                                    oneapi::math::transpose::nontrans,
                                    oneapi::math::transpose::nontrans, m, n, k, 1.0f, &A[0], lda,
                                    stride_a, &B[0], ldb, stride_b, 0.0f, &C[0], ldc, stride_c,
                                    batch_size, dependencies);
        }
        else {
            TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::row_major::gemm_batch,
                                    oneapi::math::transpose::nontrans,
                                    oneapi::math::transpose::nontrans, m, n, k, 1.0f, &A[0], lda,
                                    stride_a, &B[0], ldb, stride_b, 0.0f, &C[0], ldc, stride_c,
                                    batch_size, dependencies);
        }
        main_queue.wait_and_throw();
#endif
    }
    catch (const oneapi::math::unimplemented&) {
        return test_skipped;
    }
    catch (const std::exception& error) {
        std::cout << "Error raised during large-prime GEMM_BATCH_STRIDE:\n"
                  << error.what() << std::endl;
        return false;
    }

    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < m; ++i) {
            const int64_t index = col ? i + j * ldc : i * ldc + j;
            const float expected = float((i + 1) * (j % 7 - 3));
            if (C[index] != expected) {
                std::cout << "Difference in entry (" << i << ',' << j << "): DPC++ " << C[index]
                          << " vs. Reference " << expected << std::endl;
                return false;
            }
        }
    return true;
}

class GemmBatchStrideUsmTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::math::layout>> {};

TEST_P(GemmBatchStrideUsmTests, RealHalfPrecision) {
    EXPECT_TRUEORSKIP((test<sycl::half, sycl::half, sycl::half, sycl::half>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), 5)));
}

TEST_P(GemmBatchStrideUsmTests, HalfHalfFloatPrecision) {
    EXPECT_TRUEORSKIP((test<sycl::half, sycl::half, float, float>(std::get<0>(GetParam()),
                                                                  std::get<1>(GetParam()), 5)));
}

TEST_P(GemmBatchStrideUsmTests, Int8Int8SinglePrecision) {
    EXPECT_TRUEORSKIP((test<std::int8_t, std::int8_t, float, float>(std::get<0>(GetParam()),
                                                                    std::get<1>(GetParam()), 5)));
}

TEST_P(GemmBatchStrideUsmTests, Int8Int8SinglePrecisionErrorModel) {
    EXPECT_TRUEORSKIP(
        (int8_accumulation_error_model(std::get<0>(GetParam()), std::get<1>(GetParam()))));
}

TEST_P(GemmBatchStrideUsmTests, Int8Int8SinglePrecisionLargePrimeRange) {
    EXPECT_TRUEORSKIP(
        (int8_flat_range_large_prime(std::get<0>(GetParam()), std::get<1>(GetParam()))));
}

TEST_P(GemmBatchStrideUsmTests, Int8Int8Int32Precision) {
    EXPECT_TRUEORSKIP((test<std::int8_t, std::int8_t, std::int32_t, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), 5)));
}

TEST_P(GemmBatchStrideUsmTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(
        (test<float, float, float, float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5)));
}

TEST_P(GemmBatchStrideUsmTests, RealDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP((
        test<double, double, double, double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5)));
}

TEST_P(GemmBatchStrideUsmTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(
        (test<std::complex<float>, std::complex<float>, std::complex<float>, std::complex<float>>(
            std::get<0>(GetParam()), std::get<1>(GetParam()), 5)));
}

TEST_P(GemmBatchStrideUsmTests, ComplexDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(
        (test<std::complex<double>, std::complex<double>, std::complex<double>,
              std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5)));
}

INSTANTIATE_TEST_SUITE_P(GemmBatchStrideUsmTestSuite, GemmBatchStrideUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::math::layout::col_major,
                                                            oneapi::math::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
