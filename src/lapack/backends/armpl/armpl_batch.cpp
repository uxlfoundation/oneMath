/***************************************************************************
*  Copyright 2025 SiPearl
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
#include "armpl_common.hpp"

#include "oneapi/math/exceptions.hpp"
#include "oneapi/math/lapack/detail/armpl/onemath_lapack_armpl.hpp"

namespace oneapi {
namespace math {
namespace lapack {
namespace armpl {

// BATCH BUFFER API

void geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<float>& a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<float>& tau,
                 std::int64_t stride_tau, std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "geqrf_batch");
}
void geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<double>& a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<double>& tau,
                 std::int64_t stride_tau, std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "geqrf_batch");
}
void geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<std::complex<float>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::complex<float>>& tau, std::int64_t stride_tau,
                 std::int64_t batch_size, sycl::buffer<std::complex<float>>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "geqrf_batch");
}
void geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<std::complex<double>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::complex<double>>& tau, std::int64_t stride_tau,
                 std::int64_t batch_size, sycl::buffer<std::complex<double>>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "geqrf_batch");
}
void getri_batch(sycl::queue& queue, std::int64_t n, sycl::buffer<float>& a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getri_batch");
}
void getri_batch(sycl::queue& queue, std::int64_t n, sycl::buffer<double>& a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getri_batch");
}
void getri_batch(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>>& a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                 std::int64_t stride_ipiv, std::int64_t batch_size,
                 sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getri_batch");
}
void getri_batch(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>>& a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                 std::int64_t stride_ipiv, std::int64_t batch_size,
                 sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getri_batch");
}
void getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                 std::int64_t nrhs, sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv, sycl::buffer<float>& b,
                 std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                 sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getrs_batch");
}
void getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                 std::int64_t nrhs, sycl::buffer<double>& a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 sycl::buffer<double>& b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getrs_batch");
}
void getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                 std::int64_t nrhs, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 sycl::buffer<std::complex<float>>& b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<std::complex<float>>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getrs_batch");
}
void getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                 std::int64_t nrhs, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 sycl::buffer<std::complex<double>>& b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<std::complex<double>>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getrs_batch");
}
void getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<float>& a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                 std::int64_t stride_ipiv, std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getrf_batch");
}
void getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<double>& a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                 std::int64_t stride_ipiv, std::int64_t batch_size,
                 sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getrf_batch");
}
void getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<std::complex<float>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<std::complex<float>>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getrf_batch");
}
void getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<std::complex<double>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<std::complex<double>>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getrf_batch");
}
void orgqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                 sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<float>& tau, std::int64_t stride_tau, std::int64_t batch_size,
                 sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "orgqr_batch");
}
void orgqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                 sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<double>& tau, std::int64_t stride_tau, std::int64_t batch_size,
                 sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "orgqr_batch");
}
void potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                 sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                 std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "potrf_batch");
}
void potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                 sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                 std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "potrf_batch");
}
void potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                 sycl::buffer<std::complex<float>>& a, std::int64_t lda, std::int64_t stride_a,
                 std::int64_t batch_size, sycl::buffer<std::complex<float>>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "potrf_batch");
}
void potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                 sycl::buffer<std::complex<double>>& a, std::int64_t lda, std::int64_t stride_a,
                 std::int64_t batch_size, sycl::buffer<std::complex<double>>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "potrf_batch");
}
void potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,
                 sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<float>& b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "potrs_batch");
}
void potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,
                 sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<double>& b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "potrs_batch");
}
void potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,
                 sycl::buffer<std::complex<float>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::complex<float>>& b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<std::complex<float>>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "potrs_batch");
}
void potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,
                 sycl::buffer<std::complex<double>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::complex<double>>& b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<std::complex<double>>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "potrs_batch");
}
void ungqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                 sycl::buffer<std::complex<float>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::complex<float>>& tau, std::int64_t stride_tau,
                 std::int64_t batch_size, sycl::buffer<std::complex<float>>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "ungqr_batch");
}
void ungqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                 sycl::buffer<std::complex<double>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::complex<double>>& tau, std::int64_t stride_tau,
                 std::int64_t batch_size, sycl::buffer<std::complex<double>>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "ungqr_batch");
}

// BATCH USM API

sycl::event geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, float* a,
                        std::int64_t lda, std::int64_t stride_a, float* tau,
                        std::int64_t stride_tau, std::int64_t batch_size, float* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "geqrf_batch");
}
sycl::event geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, double* a,
                        std::int64_t lda, std::int64_t stride_a, double* tau,
                        std::int64_t stride_tau, std::int64_t batch_size, double* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "geqrf_batch");
}
sycl::event geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<float>* a,
                        std::int64_t lda, std::int64_t stride_a, std::complex<float>* tau,
                        std::int64_t stride_tau, std::int64_t batch_size,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "geqrf_batch");
}
sycl::event geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<double>* a,
                        std::int64_t lda, std::int64_t stride_a, std::complex<double>* tau,
                        std::int64_t stride_tau, std::int64_t batch_size,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "geqrf_batch");
}
sycl::event geqrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, float** a,
                        std::int64_t* lda, float** tau, std::int64_t group_count,
                        std::int64_t* group_sizes, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "geqrf_batch");
}
sycl::event geqrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, double** a,
                        std::int64_t* lda, double** tau, std::int64_t group_count,
                        std::int64_t* group_sizes, double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "geqrf_batch");
}
sycl::event geqrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                        std::complex<float>** a, std::int64_t* lda, std::complex<float>** tau,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "geqrf_batch");
}
sycl::event geqrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                        std::complex<double>** a, std::int64_t* lda, std::complex<double>** tau,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "geqrf_batch");
}
sycl::event getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, float* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size, float* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrf_batch");
}
sycl::event getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, double* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size, double* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrf_batch");
}
sycl::event getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<float>* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrf_batch");
}
sycl::event getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<double>* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrf_batch");
}
sycl::event getrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, float** a,
                        std::int64_t* lda, std::int64_t** ipiv, std::int64_t group_count,
                        std::int64_t* group_sizes, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrf_batch");
}
sycl::event getrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, double** a,
                        std::int64_t* lda, std::int64_t** ipiv, std::int64_t group_count,
                        std::int64_t* group_sizes, double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrf_batch");
}
sycl::event getrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                        std::complex<float>** a, std::int64_t* lda, std::int64_t** ipiv,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrf_batch");
}
sycl::event getrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                        std::complex<double>** a, std::int64_t* lda, std::int64_t** ipiv,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrf_batch");
}
sycl::event getri_batch(sycl::queue& queue, std::int64_t n, float* a, std::int64_t lda,
                        std::int64_t stride_a, std::int64_t* ipiv, std::int64_t stride_ipiv,
                        std::int64_t batch_size, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getri_batch");
}
sycl::event getri_batch(sycl::queue& queue, std::int64_t n, double* a, std::int64_t lda,
                        std::int64_t stride_a, std::int64_t* ipiv, std::int64_t stride_ipiv,
                        std::int64_t batch_size, double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getri_batch");
}
sycl::event getri_batch(sycl::queue& queue, std::int64_t n, std::complex<float>* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getri_batch");
}
sycl::event getri_batch(sycl::queue& queue, std::int64_t n, std::complex<double>* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getri_batch");
}
sycl::event getri_batch(sycl::queue& queue, std::int64_t* n, float** a, std::int64_t* lda,
                        std::int64_t** ipiv, std::int64_t group_count, std::int64_t* group_sizes,
                        float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getri_batch");
}
sycl::event getri_batch(sycl::queue& queue, std::int64_t* n, double** a, std::int64_t* lda,
                        std::int64_t** ipiv, std::int64_t group_count, std::int64_t* group_sizes,
                        double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getri_batch");
}
sycl::event getri_batch(sycl::queue& queue, std::int64_t* n, std::complex<float>** a,
                        std::int64_t* lda, std::int64_t** ipiv, std::int64_t group_count,
                        std::int64_t* group_sizes, std::complex<float>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getri_batch");
}
sycl::event getri_batch(sycl::queue& queue, std::int64_t* n, std::complex<double>** a,
                        std::int64_t* lda, std::int64_t** ipiv, std::int64_t group_count,
                        std::int64_t* group_sizes, std::complex<double>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getri_batch");
}
sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                        std::int64_t nrhs, float* a, std::int64_t lda, std::int64_t stride_a,
                        std::int64_t* ipiv, std::int64_t stride_ipiv, float* b, std::int64_t ldb,
                        std::int64_t stride_b, std::int64_t batch_size, float* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrs_batch");
}
sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                        std::int64_t nrhs, double* a, std::int64_t lda, std::int64_t stride_a,
                        std::int64_t* ipiv, std::int64_t stride_ipiv, double* b, std::int64_t ldb,
                        std::int64_t stride_b, std::int64_t batch_size, double* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrs_batch");
}
sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                        std::int64_t nrhs, std::complex<float>* a, std::int64_t lda,
                        std::int64_t stride_a, std::int64_t* ipiv, std::int64_t stride_ipiv,
                        std::complex<float>* b, std::int64_t ldb, std::int64_t stride_b,
                        std::int64_t batch_size, std::complex<float>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrs_batch");
}
sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                        std::int64_t nrhs, std::complex<double>* a, std::int64_t lda,
                        std::int64_t stride_a, std::int64_t* ipiv, std::int64_t stride_ipiv,
                        std::complex<double>* b, std::int64_t ldb, std::int64_t stride_b,
                        std::int64_t batch_size, std::complex<double>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrs_batch");
}
sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose* trans, std::int64_t* n,
                        std::int64_t* nrhs, float** a, std::int64_t* lda, std::int64_t** ipiv,
                        float** b, std::int64_t* ldb, std::int64_t group_count,
                        std::int64_t* group_sizes, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrs_batch");
}
sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose* trans, std::int64_t* n,
                        std::int64_t* nrhs, double** a, std::int64_t* lda, std::int64_t** ipiv,
                        double** b, std::int64_t* ldb, std::int64_t group_count,
                        std::int64_t* group_sizes, double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrs_batch");
}
sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose* trans, std::int64_t* n,
                        std::int64_t* nrhs, std::complex<float>** a, std::int64_t* lda,
                        std::int64_t** ipiv, std::complex<float>** b, std::int64_t* ldb,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrs_batch");
}
sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose* trans, std::int64_t* n,
                        std::int64_t* nrhs, std::complex<double>** a, std::int64_t* lda,
                        std::int64_t** ipiv, std::complex<double>** b, std::int64_t* ldb,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrs_batch");
}
sycl::event orgqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                        float* a, std::int64_t lda, std::int64_t stride_a, float* tau,
                        std::int64_t stride_tau, std::int64_t batch_size, float* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "orgqr_batch");
}
sycl::event orgqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                        double* a, std::int64_t lda, std::int64_t stride_a, double* tau,
                        std::int64_t stride_tau, std::int64_t batch_size, double* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "orgqr_batch");
}
sycl::event orgqr_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, std::int64_t* k,
                        float** a, std::int64_t* lda, float** tau, std::int64_t group_count,
                        std::int64_t* group_sizes, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "orgqr_batch");
}
sycl::event orgqr_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, std::int64_t* k,
                        double** a, std::int64_t* lda, double** tau, std::int64_t group_count,
                        std::int64_t* group_sizes, double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "orgqr_batch");
}
sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, float* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size,
                        float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "potrf_batch");
}
sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, double* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size,
                        double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "potrf_batch");
}
sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                        std::complex<float>* a, std::int64_t lda, std::int64_t stride_a,
                        std::int64_t batch_size, std::complex<float>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "potrf_batch");
}
sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                        std::complex<double>* a, std::int64_t lda, std::int64_t stride_a,
                        std::int64_t batch_size, std::complex<double>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "potrf_batch");
}

template <typename T>
inline sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n, T** a,
                               std::int64_t* lda, std::int64_t group_count,
                               std::int64_t* group_sizes, T* scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "potrf_batch");
}

#define POTRF_BATCH_LAUNCHER_USM(TYPE)                                                            \
    sycl::event potrf_batch(                                                                      \
        sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n, TYPE** a,                  \
        std::int64_t* lda, std::int64_t group_count, std::int64_t* group_sizes, TYPE* scratchpad, \
        std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {             \
        return potrf_batch(queue, uplo, n, a, lda, group_count, group_sizes, scratchpad,          \
                           scratchpad_size, dependencies);                                        \
    }

POTRF_BATCH_LAUNCHER_USM(float)
POTRF_BATCH_LAUNCHER_USM(double)
POTRF_BATCH_LAUNCHER_USM(std::complex<float>)
POTRF_BATCH_LAUNCHER_USM(std::complex<double>)

#undef POTRF_BATCH_LAUNCHER_USM

sycl::event potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                        std::int64_t nrhs, float* a, std::int64_t lda, std::int64_t stride_a,
                        float* b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                        float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "potrs_batch");
}
sycl::event potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                        std::int64_t nrhs, double* a, std::int64_t lda, std::int64_t stride_a,
                        double* b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                        double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "potrs_batch");
}
sycl::event potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                        std::int64_t nrhs, std::complex<float>* a, std::int64_t lda,
                        std::int64_t stride_a, std::complex<float>* b, std::int64_t ldb,
                        std::int64_t stride_b, std::int64_t batch_size,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "potrs_batch");
}
sycl::event potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                        std::int64_t nrhs, std::complex<double>* a, std::int64_t lda,
                        std::int64_t stride_a, std::complex<double>* b, std::int64_t ldb,
                        std::int64_t stride_b, std::int64_t batch_size,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "potrs_batch");
}

template <typename T>
inline sycl::event potrs_batch(sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n,
                               std::int64_t* nrhs, T** a, std::int64_t* lda, T** b,
                               std::int64_t* ldb, std::int64_t group_count,
                               std::int64_t* group_sizes, T* scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "potrs_batch");
}

#define POTRS_BATCH_LAUNCHER_USM(TYPE)                                                      \
    sycl::event potrs_batch(                                                                \
        sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n, std::int64_t* nrhs,  \
        TYPE** a, std::int64_t* lda, TYPE** b, std::int64_t* ldb, std::int64_t group_count, \
        std::int64_t* group_sizes, TYPE* scratchpad, std::int64_t scratchpad_size,          \
        const std::vector<sycl::event>& dependencies) {                                     \
        return potrs_batch(queue, uplo, n, nrhs, a, lda, b, ldb, group_count, group_sizes,  \
                           scratchpad, scratchpad_size, dependencies);                      \
    }

POTRS_BATCH_LAUNCHER_USM(float)
POTRS_BATCH_LAUNCHER_USM(double)
POTRS_BATCH_LAUNCHER_USM(std::complex<float>)
POTRS_BATCH_LAUNCHER_USM(std::complex<double>)

#undef POTRS_BATCH_LAUNCHER_USM

sycl::event ungqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                        std::complex<float>* a, std::int64_t lda, std::int64_t stride_a,
                        std::complex<float>* tau, std::int64_t stride_tau, std::int64_t batch_size,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "ungqr_batch");
}
sycl::event ungqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                        std::complex<double>* a, std::int64_t lda, std::int64_t stride_a,
                        std::complex<double>* tau, std::int64_t stride_tau, std::int64_t batch_size,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "ungqr_batch");
}
sycl::event ungqr_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, std::int64_t* k,
                        std::complex<float>** a, std::int64_t* lda, std::complex<float>** tau,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "ungqr_batch");
}
sycl::event ungqr_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, std::int64_t* k,
                        std::complex<double>** a, std::int64_t* lda, std::complex<double>** tau,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "ungqr_batch");
}

// BATCH SCRATCHPAD API

template <>
std::int64_t getrf_batch_scratchpad_size<float>(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                std::int64_t lda, std::int64_t stride_a,
                                                std::int64_t stride_ipiv, std::int64_t batch_size) {
    throw unimplemented("lapack", "getrf_batch_scratchpad_size");
}
template <>
std::int64_t getrf_batch_scratchpad_size<double>(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                 std::int64_t lda, std::int64_t stride_a,
                                                 std::int64_t stride_ipiv,
                                                 std::int64_t batch_size) {
    throw unimplemented("lapack", "getrf_batch_scratchpad_size");
}
template <>
std::int64_t getrf_batch_scratchpad_size<std::complex<float>>(sycl::queue& queue, std::int64_t m,
                                                              std::int64_t n, std::int64_t lda,
                                                              std::int64_t stride_a,
                                                              std::int64_t stride_ipiv,
                                                              std::int64_t batch_size) {
    throw unimplemented("lapack", "getrf_batch_scratchpad_size");
}
template <>
std::int64_t getrf_batch_scratchpad_size<std::complex<double>>(sycl::queue& queue, std::int64_t m,
                                                               std::int64_t n, std::int64_t lda,
                                                               std::int64_t stride_a,
                                                               std::int64_t stride_ipiv,
                                                               std::int64_t batch_size) {
    throw unimplemented("lapack", "getrf_batch_scratchpad_size");
}
template <>
std::int64_t getri_batch_scratchpad_size<float>(sycl::queue& queue, std::int64_t n,
                                                std::int64_t lda, std::int64_t stride_a,
                                                std::int64_t stride_ipiv, std::int64_t batch_size) {
    throw unimplemented("lapack", "getri_batch_scratchpad_size");
}
template <>
std::int64_t getri_batch_scratchpad_size<double>(sycl::queue& queue, std::int64_t n,
                                                 std::int64_t lda, std::int64_t stride_a,
                                                 std::int64_t stride_ipiv,
                                                 std::int64_t batch_size) {
    throw unimplemented("lapack", "getri_batch_scratchpad_size");
}
template <>
std::int64_t getri_batch_scratchpad_size<std::complex<float>>(sycl::queue& queue, std::int64_t n,
                                                              std::int64_t lda,
                                                              std::int64_t stride_a,
                                                              std::int64_t stride_ipiv,
                                                              std::int64_t batch_size) {
    throw unimplemented("lapack", "getri_batch_scratchpad_size");
}
template <>
std::int64_t getri_batch_scratchpad_size<std::complex<double>>(sycl::queue& queue, std::int64_t n,
                                                               std::int64_t lda,
                                                               std::int64_t stride_a,
                                                               std::int64_t stride_ipiv,
                                                               std::int64_t batch_size) {
    throw unimplemented("lapack", "getri_batch_scratchpad_size");
}
template <>
std::int64_t getrs_batch_scratchpad_size<float>(sycl::queue& queue, oneapi::math::transpose trans,
                                                std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                                std::int64_t stride_a, std::int64_t stride_ipiv,
                                                std::int64_t ldb, std::int64_t stride_b,
                                                std::int64_t batch_size) {
    throw unimplemented("lapack", "getrs_batch_scratchpad_size");
}
template <>
std::int64_t getrs_batch_scratchpad_size<double>(sycl::queue& queue, oneapi::math::transpose trans,
                                                 std::int64_t n, std::int64_t nrhs,
                                                 std::int64_t lda, std::int64_t stride_a,
                                                 std::int64_t stride_ipiv, std::int64_t ldb,
                                                 std::int64_t stride_b, std::int64_t batch_size) {
    throw unimplemented("lapack", "getrs_batch_scratchpad_size");
}
template <>
std::int64_t getrs_batch_scratchpad_size<std::complex<float>>(
    sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n, std::int64_t nrhs,
    std::int64_t lda, std::int64_t stride_a, std::int64_t stride_ipiv, std::int64_t ldb,
    std::int64_t stride_b, std::int64_t batch_size) {
    throw unimplemented("lapack", "getrs_batch_scratchpad_size");
}
template <>
std::int64_t getrs_batch_scratchpad_size<std::complex<double>>(
    sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n, std::int64_t nrhs,
    std::int64_t lda, std::int64_t stride_a, std::int64_t stride_ipiv, std::int64_t ldb,
    std::int64_t stride_b, std::int64_t batch_size) {
    throw unimplemented("lapack", "getrs_batch_scratchpad_size");
}
template <>
std::int64_t geqrf_batch_scratchpad_size<float>(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                std::int64_t lda, std::int64_t stride_a,
                                                std::int64_t stride_tau, std::int64_t batch_size) {
    throw unimplemented("lapack", "geqrf_batch_scratchpad_size");
}
template <>
std::int64_t geqrf_batch_scratchpad_size<double>(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                 std::int64_t lda, std::int64_t stride_a,
                                                 std::int64_t stride_tau, std::int64_t batch_size) {
    throw unimplemented("lapack", "geqrf_batch_scratchpad_size");
}
template <>
std::int64_t geqrf_batch_scratchpad_size<std::complex<float>>(sycl::queue& queue, std::int64_t m,
                                                              std::int64_t n, std::int64_t lda,
                                                              std::int64_t stride_a,
                                                              std::int64_t stride_tau,
                                                              std::int64_t batch_size) {
    throw unimplemented("lapack", "geqrf_batch_scratchpad_size");
}
template <>
std::int64_t geqrf_batch_scratchpad_size<std::complex<double>>(sycl::queue& queue, std::int64_t m,
                                                               std::int64_t n, std::int64_t lda,
                                                               std::int64_t stride_a,
                                                               std::int64_t stride_tau,
                                                               std::int64_t batch_size) {
    throw unimplemented("lapack", "geqrf_batch_scratchpad_size");
}

template <>
std::int64_t potrf_batch_scratchpad_size<float>(sycl::queue& queue, oneapi::math::uplo uplo,
                                                std::int64_t n, std::int64_t lda,
                                                std::int64_t stride_a, std::int64_t batch_size) {
    throw unimplemented("lapack", "potrf_batch_scratchpad_size");
}
template <>
std::int64_t potrf_batch_scratchpad_size<double>(sycl::queue& queue, oneapi::math::uplo uplo,
                                                 std::int64_t n, std::int64_t lda,
                                                 std::int64_t stride_a, std::int64_t batch_size) {
    throw unimplemented("lapack", "potrf_batch_scratchpad_size");
}
template <>
std::int64_t potrf_batch_scratchpad_size<std::complex<float>>(sycl::queue& queue,
                                                              oneapi::math::uplo uplo,
                                                              std::int64_t n, std::int64_t lda,
                                                              std::int64_t stride_a,
                                                              std::int64_t batch_size) {
    throw unimplemented("lapack", "potrf_batch_scratchpad_size");
}
template <>
std::int64_t potrf_batch_scratchpad_size<std::complex<double>>(sycl::queue& queue,
                                                               oneapi::math::uplo uplo,
                                                               std::int64_t n, std::int64_t lda,
                                                               std::int64_t stride_a,
                                                               std::int64_t batch_size) {
    throw unimplemented("lapack", "potrf_batch_scratchpad_size");
}
template <>
std::int64_t potrs_batch_scratchpad_size<float>(sycl::queue& queue, oneapi::math::uplo uplo,
                                                std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                                std::int64_t stride_a, std::int64_t ldb,
                                                std::int64_t stride_b, std::int64_t batch_size) {
    throw unimplemented("lapack", "potrs_batch_scratchpad_size");
}
template <>
std::int64_t potrs_batch_scratchpad_size<double>(sycl::queue& queue, oneapi::math::uplo uplo,
                                                 std::int64_t n, std::int64_t nrhs,
                                                 std::int64_t lda, std::int64_t stride_a,
                                                 std::int64_t ldb, std::int64_t stride_b,
                                                 std::int64_t batch_size) {
    throw unimplemented("lapack", "potrs_batch_scratchpad_size");
}
template <>
std::int64_t potrs_batch_scratchpad_size<std::complex<float>>(
    sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,
    std::int64_t lda, std::int64_t stride_a, std::int64_t ldb, std::int64_t stride_b,
    std::int64_t batch_size) {
    throw unimplemented("lapack", "potrs_batch_scratchpad_size");
}
template <>
std::int64_t potrs_batch_scratchpad_size<std::complex<double>>(
    sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,
    std::int64_t lda, std::int64_t stride_a, std::int64_t ldb, std::int64_t stride_b,
    std::int64_t batch_size) {
    throw unimplemented("lapack", "potrs_batch_scratchpad_size");
}
template <>
std::int64_t orgqr_batch_scratchpad_size<float>(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                std::int64_t k, std::int64_t lda,
                                                std::int64_t stride_a, std::int64_t stride_tau,
                                                std::int64_t batch_size) {
    throw unimplemented("lapack", "orgqr_batch_scratchpad_size");
}
template <>
std::int64_t orgqr_batch_scratchpad_size<double>(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                 std::int64_t k, std::int64_t lda,
                                                 std::int64_t stride_a, std::int64_t stride_tau,
                                                 std::int64_t batch_size) {
    throw unimplemented("lapack", "orgqr_batch_scratchpad_size");
}
template <>
std::int64_t ungqr_batch_scratchpad_size<std::complex<float>>(
    sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda,
    std::int64_t stride_a, std::int64_t stride_tau, std::int64_t batch_size) {
    throw unimplemented("lapack", "ungqr_batch_scratchpad_size");
}
template <>
std::int64_t ungqr_batch_scratchpad_size<std::complex<double>>(
    sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda,
    std::int64_t stride_a, std::int64_t stride_tau, std::int64_t batch_size) {
    throw unimplemented("lapack", "ungqr_batch_scratchpad_size");
}
template <>
std::int64_t getrf_batch_scratchpad_size<float>(sycl::queue& queue, std::int64_t* m,
                                                std::int64_t* n, std::int64_t* lda,
                                                std::int64_t group_count,
                                                std::int64_t* group_sizes) {
    throw unimplemented("lapack", "getrf_batch_scratchpad_size");
}
template <>
std::int64_t getrf_batch_scratchpad_size<double>(sycl::queue& queue, std::int64_t* m,
                                                 std::int64_t* n, std::int64_t* lda,
                                                 std::int64_t group_count,
                                                 std::int64_t* group_sizes) {
    throw unimplemented("lapack", "getrf_batch_scratchpad_size");
}
template <>
std::int64_t getrf_batch_scratchpad_size<std::complex<float>>(sycl::queue& queue, std::int64_t* m,
                                                              std::int64_t* n, std::int64_t* lda,
                                                              std::int64_t group_count,
                                                              std::int64_t* group_sizes) {
    throw unimplemented("lapack", "getrf_batch_scratchpad_size");
}
template <>
std::int64_t getrf_batch_scratchpad_size<std::complex<double>>(sycl::queue& queue, std::int64_t* m,
                                                               std::int64_t* n, std::int64_t* lda,
                                                               std::int64_t group_count,
                                                               std::int64_t* group_sizes) {
    throw unimplemented("lapack", "getrf_batch_scratchpad_size");
}
template <>
std::int64_t getri_batch_scratchpad_size<float>(sycl::queue& queue, std::int64_t* n,
                                                std::int64_t* lda, std::int64_t group_count,
                                                std::int64_t* group_sizes) {
    throw unimplemented("lapack", "getri_batch_scratchpad_size");
}
template <>
std::int64_t getri_batch_scratchpad_size<double>(sycl::queue& queue, std::int64_t* n,
                                                 std::int64_t* lda, std::int64_t group_count,
                                                 std::int64_t* group_sizes) {
    throw unimplemented("lapack", "getri_batch_scratchpad_size");
}
template <>
std::int64_t getri_batch_scratchpad_size<std::complex<float>>(sycl::queue& queue, std::int64_t* n,
                                                              std::int64_t* lda,
                                                              std::int64_t group_count,
                                                              std::int64_t* group_sizes) {
    throw unimplemented("lapack", "getri_batch_scratchpad_size");
}
template <>
std::int64_t getri_batch_scratchpad_size<std::complex<double>>(sycl::queue& queue, std::int64_t* n,
                                                               std::int64_t* lda,
                                                               std::int64_t group_count,
                                                               std::int64_t* group_sizes) {
    throw unimplemented("lapack", "getri_batch_scratchpad_size");
}
template <>
std::int64_t getrs_batch_scratchpad_size<float>(sycl::queue& queue, oneapi::math::transpose* trans,
                                                std::int64_t* n, std::int64_t* nrhs,
                                                std::int64_t* lda, std::int64_t* ldb,
                                                std::int64_t group_count,
                                                std::int64_t* group_sizes) {
    throw unimplemented("lapack", "getrs_batch_scratchpad_size");
}
template <>
std::int64_t getrs_batch_scratchpad_size<double>(sycl::queue& queue, oneapi::math::transpose* trans,
                                                 std::int64_t* n, std::int64_t* nrhs,
                                                 std::int64_t* lda, std::int64_t* ldb,
                                                 std::int64_t group_count,
                                                 std::int64_t* group_sizes) {
    throw unimplemented("lapack", "getrs_batch_scratchpad_size");
}
template <>
std::int64_t getrs_batch_scratchpad_size<std::complex<float>>(
    sycl::queue& queue, oneapi::math::transpose* trans, std::int64_t* n, std::int64_t* nrhs,
    std::int64_t* lda, std::int64_t* ldb, std::int64_t group_count, std::int64_t* group_sizes) {
    throw unimplemented("lapack", "getrs_batch_scratchpad_size");
}
template <>
std::int64_t getrs_batch_scratchpad_size<std::complex<double>>(
    sycl::queue& queue, oneapi::math::transpose* trans, std::int64_t* n, std::int64_t* nrhs,
    std::int64_t* lda, std::int64_t* ldb, std::int64_t group_count, std::int64_t* group_sizes) {
    throw unimplemented("lapack", "getrs_batch_scratchpad_size");
}
template <>
std::int64_t geqrf_batch_scratchpad_size<float>(sycl::queue& queue, std::int64_t* m,
                                                std::int64_t* n, std::int64_t* lda,
                                                std::int64_t group_count,
                                                std::int64_t* group_sizes) {
    throw unimplemented("lapack", "geqrf_batch_scratchpad_size");
}
template <>
std::int64_t geqrf_batch_scratchpad_size<double>(sycl::queue& queue, std::int64_t* m,
                                                 std::int64_t* n, std::int64_t* lda,
                                                 std::int64_t group_count,
                                                 std::int64_t* group_sizes) {
    throw unimplemented("lapack", "geqrf_batch_scratchpad_size");
}
template <>
std::int64_t geqrf_batch_scratchpad_size<std::complex<float>>(sycl::queue& queue, std::int64_t* m,
                                                              std::int64_t* n, std::int64_t* lda,
                                                              std::int64_t group_count,
                                                              std::int64_t* group_sizes) {
    throw unimplemented("lapack", "geqrf_batch_scratchpad_size");
}
template <>
std::int64_t geqrf_batch_scratchpad_size<std::complex<double>>(sycl::queue& queue, std::int64_t* m,
                                                               std::int64_t* n, std::int64_t* lda,
                                                               std::int64_t group_count,
                                                               std::int64_t* group_sizes) {
    throw unimplemented("lapack", "geqrf_batch_scratchpad_size");
}
template <>
std::int64_t orgqr_batch_scratchpad_size<float>(sycl::queue& queue, std::int64_t* m,
                                                std::int64_t* n, std::int64_t* k, std::int64_t* lda,
                                                std::int64_t group_count,
                                                std::int64_t* group_sizes) {
    throw unimplemented("lapack", "orgqr_batch_scratchpad_size");
}
template <>
std::int64_t orgqr_batch_scratchpad_size<double>(sycl::queue& queue, std::int64_t* m,
                                                 std::int64_t* n, std::int64_t* k,
                                                 std::int64_t* lda, std::int64_t group_count,
                                                 std::int64_t* group_sizes) {
    throw unimplemented("lapack", "orgqr_batch_scratchpad_size");
}
#define POTRF_GROUP_LAUNCHER_SCRATCH(TYPE)                                                  \
    template <>                                                                             \
    std::int64_t potrf_batch_scratchpad_size<TYPE>(                                         \
        sycl::queue & queue, oneapi::math::uplo * uplo, std::int64_t* n, std::int64_t* lda, \
        std::int64_t group_count, std::int64_t* group_sizes) {                              \
        throw unimplemented("lapack", "potrf_batch_scratchpad_size");                       \
    }

POTRF_GROUP_LAUNCHER_SCRATCH(float)
POTRF_GROUP_LAUNCHER_SCRATCH(double)
POTRF_GROUP_LAUNCHER_SCRATCH(std::complex<float>)
POTRF_GROUP_LAUNCHER_SCRATCH(std::complex<double>)

#undef POTRF_GROUP_LAUNCHER_SCRATCH

#define POTRS_GROUP_LAUNCHER_SCRATCH(TYPE)                                                   \
    template <>                                                                              \
    std::int64_t potrs_batch_scratchpad_size<TYPE>(                                          \
        sycl::queue & queue, oneapi::math::uplo * uplo, std::int64_t* n, std::int64_t* nrhs, \
        std::int64_t* lda, std::int64_t* ldb, std::int64_t group_count,                      \
        std::int64_t* group_sizes) {                                                         \
        throw unimplemented("lapack", "potrs_batch_scratchpad_size");                        \
    }

POTRS_GROUP_LAUNCHER_SCRATCH(float)
POTRS_GROUP_LAUNCHER_SCRATCH(double)
POTRS_GROUP_LAUNCHER_SCRATCH(std::complex<float>)
POTRS_GROUP_LAUNCHER_SCRATCH(std::complex<double>)

#undef POTRS_GROUP_LAUNCHER_SCRATCH

template <>
std::int64_t ungqr_batch_scratchpad_size<std::complex<float>>(sycl::queue& queue, std::int64_t* m,
                                                              std::int64_t* n, std::int64_t* k,
                                                              std::int64_t* lda,
                                                              std::int64_t group_count,
                                                              std::int64_t* group_sizes) {
    throw unimplemented("lapack", "ungqr_batch_scratchpad_size");
}
template <>
std::int64_t ungqr_batch_scratchpad_size<std::complex<double>>(sycl::queue& queue, std::int64_t* m,
                                                               std::int64_t* n, std::int64_t* k,
                                                               std::int64_t* lda,
                                                               std::int64_t group_count,
                                                               std::int64_t* group_sizes) {
    throw unimplemented("lapack", "ungqr_batch_scratchpad_size");
}

} // namespace armpl
} // namespace lapack
} // namespace math
} // namespace oneapi
