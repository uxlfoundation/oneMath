/*******************************************************************************
* Copyright Codeplay Software
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

// Buffer APIs

void gemm_bias(sycl::queue& queue, oneapi::math::transpose transa, oneapi::math::transpose transb,
               oneapi::math::offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
               float alpha, sycl::buffer<int8_t, 1>& a, std::int64_t lda, int8_t ao,
               sycl::buffer<uint8_t, 1>& b, std::int64_t ldb, uint8_t bo, float beta,
               sycl::buffer<int32_t, 1>& c, std::int64_t ldc, sycl::buffer<int32_t, 1>& co) {
    throw unimplemented("blas", "gemm_bias", "");
}

void gemm_bias(sycl::queue& queue, oneapi::math::transpose transa, oneapi::math::transpose transb,
               oneapi::math::offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
               float alpha, sycl::buffer<int8_t, 1>& a, std::int64_t lda, int8_t ao,
               sycl::buffer<int8_t, 1>& b, std::int64_t ldb, int8_t bo, float beta,
               sycl::buffer<int32_t, 1>& c, std::int64_t ldc, sycl::buffer<int32_t, 1>& co) {
    throw unimplemented("blas", "gemm_bias", "");
}

void gemm_bias(sycl::queue& queue, oneapi::math::transpose transa, oneapi::math::transpose transb,
               oneapi::math::offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
               float alpha, sycl::buffer<uint8_t, 1>& a, std::int64_t lda, uint8_t ao,
               sycl::buffer<int8_t, 1>& b, std::int64_t ldb, int8_t bo, float beta,
               sycl::buffer<int32_t, 1>& c, std::int64_t ldc, sycl::buffer<int32_t, 1>& co) {
    throw unimplemented("blas", "gemm_bias", "");
}

void gemm_bias(sycl::queue& queue, oneapi::math::transpose transa, oneapi::math::transpose transb,
               oneapi::math::offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
               float alpha, sycl::buffer<uint8_t, 1>& a, std::int64_t lda, uint8_t ao,
               sycl::buffer<uint8_t, 1>& b, std::int64_t ldb, uint8_t bo, float beta,
               sycl::buffer<int32_t, 1>& c, std::int64_t ldc, sycl::buffer<int32_t, 1>& co) {
    throw unimplemented("blas", "gemm_bias", "");
}

// USM APIs

sycl::event gemm_bias(sycl::queue& queue, oneapi::math::transpose transa,
                      oneapi::math::transpose transb, oneapi::math::offset offsetc, std::int64_t m,
                      std::int64_t n, std::int64_t k, float alpha, const std::int8_t* a,
                      std::int64_t lda, std::int8_t ao, const std::uint8_t* b, std::int64_t ldb,
                      std::uint8_t bo, float beta, std::int32_t* c, std::int64_t ldc,
                      const std::int32_t* co, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm_bias", " for USM");
}

sycl::event gemm_bias(sycl::queue& queue, oneapi::math::transpose transa,
                      oneapi::math::transpose transb, oneapi::math::offset offsetc, std::int64_t m,
                      std::int64_t n, std::int64_t k, float alpha, const std::int8_t* a,
                      std::int64_t lda, std::int8_t ao, const std::int8_t* b, std::int64_t ldb,
                      std::int8_t bo, float beta, std::int32_t* c, std::int64_t ldc,
                      const std::int32_t* co, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm_bias", " for USM");
}

sycl::event gemm_bias(sycl::queue& queue, oneapi::math::transpose transa,
                      oneapi::math::transpose transb, oneapi::math::offset offsetc, std::int64_t m,
                      std::int64_t n, std::int64_t k, float alpha, const std::uint8_t* a,
                      std::int64_t lda, std::uint8_t ao, const std::int8_t* b, std::int64_t ldb,
                      std::int8_t bo, float beta, std::int32_t* c, std::int64_t ldc,
                      const std::int32_t* co, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm_bias", " for USM");
}

sycl::event gemm_bias(sycl::queue& queue, oneapi::math::transpose transa,
                      oneapi::math::transpose transb, oneapi::math::offset offsetc, std::int64_t m,
                      std::int64_t n, std::int64_t k, float alpha, const std::uint8_t* a,
                      std::int64_t lda, std::uint8_t ao, const std::uint8_t* b, std::int64_t ldb,
                      std::uint8_t bo, float beta, std::int32_t* c, std::int64_t ldc,
                      const std::int32_t* co, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm_bias", " for USM");
}
