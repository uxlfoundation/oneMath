/*******************************************************************************
* Copyright 2022 Intel Corporation
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

/// level3, buffer

void gemm(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
          std::int64_t k, float alpha, sycl::buffer<float, 1>& a, std::int64_t lda,
          sycl::buffer<float, 1>& b, std::int64_t ldb, float beta, sycl::buffer<float, 1>& c,
          std::int64_t ldc);

void gemm(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
          std::int64_t k, double alpha, sycl::buffer<double, 1>& a, std::int64_t lda,
          sycl::buffer<double, 1>& b, std::int64_t ldb, double beta, sycl::buffer<double, 1>& c,
          std::int64_t ldc);

void gemm(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
          std::int64_t k, std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
          std::int64_t lda, sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb,
          std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc);

void gemm(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
          std::int64_t k, std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
          std::int64_t lda, sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb,
          std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& c, std::int64_t ldc);

void gemm(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
          std::int64_t k, sycl::half alpha, sycl::buffer<sycl::half, 1>& a, std::int64_t lda,
          sycl::buffer<sycl::half, 1>& b, std::int64_t ldb, sycl::half beta,
          sycl::buffer<sycl::half, 1>& c, std::int64_t ldc);

void gemm(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
          std::int64_t k, float alpha, sycl::buffer<sycl::half, 1>& a, std::int64_t lda,
          sycl::buffer<sycl::half, 1>& b, std::int64_t ldb, float beta, sycl::buffer<float, 1>& c,
          std::int64_t ldc);

void gemm(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
          std::int64_t k, float alpha, sycl::buffer<bfloat16, 1>& a, std::int64_t lda,
          sycl::buffer<bfloat16, 1>& b, std::int64_t ldb, float beta, sycl::buffer<float, 1>& c,
          std::int64_t ldc);

void gemm(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
          std::int64_t k, float alpha, sycl::buffer<bfloat16, 1>& a, std::int64_t lda,
          sycl::buffer<bfloat16, 1>& b, std::int64_t ldb, float beta, sycl::buffer<bfloat16, 1>& c,
          std::int64_t ldc);

void gemm(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
          std::int64_t k, float alpha, sycl::buffer<std::int8_t, 1>& a, std::int64_t lda,
          sycl::buffer<std::int8_t, 1>& b, std::int64_t ldb, float beta,
          sycl::buffer<std::int32_t, 1>& c, std::int64_t ldc);

void gemm(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
          std::int64_t k, float alpha, sycl::buffer<std::int8_t, 1>& a, std::int64_t lda,
          sycl::buffer<std::int8_t, 1>& b, std::int64_t ldb, float beta, sycl::buffer<float, 1>& c,
          std::int64_t ldc);

void symm(sycl::queue& queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
          float alpha, sycl::buffer<float, 1>& a, std::int64_t lda, sycl::buffer<float, 1>& b,
          std::int64_t ldb, float beta, sycl::buffer<float, 1>& c, std::int64_t ldc);

void symm(sycl::queue& queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
          double alpha, sycl::buffer<double, 1>& a, std::int64_t lda, sycl::buffer<double, 1>& b,
          std::int64_t ldb, double beta, sycl::buffer<double, 1>& c, std::int64_t ldc);

void symm(sycl::queue& queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc);

void symm(sycl::queue& queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1>& c, std::int64_t ldc);

void hemm(sycl::queue& queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc);

void hemm(sycl::queue& queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1>& c, std::int64_t ldc);

void syrk(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
          float alpha, sycl::buffer<float, 1>& a, std::int64_t lda, float beta,
          sycl::buffer<float, 1>& c, std::int64_t ldc);

void syrk(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
          double alpha, sycl::buffer<double, 1>& a, std::int64_t lda, double beta,
          sycl::buffer<double, 1>& c, std::int64_t ldc);

void syrk(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
          std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc);

void syrk(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
          std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& c, std::int64_t ldc);

void herk(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
          float alpha, sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda, float beta,
          sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc);

void herk(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
          double alpha, sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda, double beta,
          sycl::buffer<std::complex<double>, 1>& c, std::int64_t ldc);

void syr2k(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
           float alpha, sycl::buffer<float, 1>& a, std::int64_t lda, sycl::buffer<float, 1>& b,
           std::int64_t ldb, float beta, sycl::buffer<float, 1>& c, std::int64_t ldc);

void syr2k(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
           double alpha, sycl::buffer<double, 1>& a, std::int64_t lda, sycl::buffer<double, 1>& b,
           std::int64_t ldb, double beta, sycl::buffer<double, 1>& c, std::int64_t ldc);

void syr2k(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
           std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb, std::complex<float> beta,
           sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc);

void syr2k(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
           std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb, std::complex<double> beta,
           sycl::buffer<std::complex<double>, 1>& c, std::int64_t ldc);

void her2k(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
           std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb, float beta,
           sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc);

void her2k(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
           std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb, double beta,
           sycl::buffer<std::complex<double>, 1>& c, std::int64_t ldc);

void trmm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t m, std::int64_t n, float alpha, sycl::buffer<float, 1>& a, std::int64_t lda,
          sycl::buffer<float, 1>& b, std::int64_t ldb);

void trmm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t m, std::int64_t n, double alpha, sycl::buffer<double, 1>& a,
          std::int64_t lda, sycl::buffer<double, 1>& b, std::int64_t ldb);

void trmm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t m, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb);

void trmm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t m, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb);

void trsm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t m, std::int64_t n, float alpha, sycl::buffer<float, 1>& a, std::int64_t lda,
          sycl::buffer<float, 1>& b, std::int64_t ldb);

void trsm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t m, std::int64_t n, double alpha, sycl::buffer<double, 1>& a,
          std::int64_t lda, sycl::buffer<double, 1>& b, std::int64_t ldb);

void trsm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t m, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb);

void trsm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t m, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb);

// level 3, USM

sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                 std::int64_t n, std::int64_t k, value_or_pointer<float> alpha, const float* a,
                 std::int64_t lda, const float* b, std::int64_t ldb, value_or_pointer<float> beta,
                 float* c, std::int64_t ldc, const std::vector<sycl::event>& dependencies = {});

sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                 std::int64_t n, std::int64_t k, value_or_pointer<double> alpha, const double* a,
                 std::int64_t lda, const double* b, std::int64_t ldb, value_or_pointer<double> beta,
                 double* c, std::int64_t ldc, const std::vector<sycl::event>& dependencies = {});

sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                 std::int64_t n, std::int64_t k, value_or_pointer<std::complex<float>> alpha,
                 const std::complex<float>* a, std::int64_t lda, const std::complex<float>* b,
                 std::int64_t ldb, value_or_pointer<std::complex<float>> beta,
                 std::complex<float>* c, std::int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                 std::int64_t n, std::int64_t k, value_or_pointer<std::complex<double>> alpha,
                 const std::complex<double>* a, std::int64_t lda, const std::complex<double>* b,
                 std::int64_t ldb, value_or_pointer<std::complex<double>> beta,
                 std::complex<double>* c, std::int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                 std::int64_t n, std::int64_t k, value_or_pointer<sycl::half> alpha,
                 const sycl::half* a, std::int64_t lda, const sycl::half* b, std::int64_t ldb,
                 value_or_pointer<sycl::half> beta, sycl::half* c, std::int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                 std::int64_t n, std::int64_t k, value_or_pointer<float> alpha, const sycl::half* a,
                 std::int64_t lda, const sycl::half* b, std::int64_t ldb,
                 value_or_pointer<float> beta, float* c, std::int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                 std::int64_t n, std::int64_t k, value_or_pointer<float> alpha, const bfloat16* a,
                 std::int64_t lda, const bfloat16* b, std::int64_t ldb,
                 value_or_pointer<float> beta, float* c, std::int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                 std::int64_t n, std::int64_t k, value_or_pointer<float> alpha, const bfloat16* a,
                 std::int64_t lda, const bfloat16* b, std::int64_t ldb,
                 value_or_pointer<float> beta, bfloat16* c, std::int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                 std::int64_t n, std::int64_t k, value_or_pointer<float> alpha,
                 const std::int8_t* a, std::int64_t lda, const std::int8_t* b, std::int64_t ldb,
                 value_or_pointer<float> beta, std::int32_t* c, std::int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                 std::int64_t n, std::int64_t k, value_or_pointer<float> alpha,
                 const std::int8_t* a, std::int64_t lda, const std::int8_t* b, std::int64_t ldb,
                 value_or_pointer<float> beta, float* c, std::int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event symm(sycl::queue& queue, side left_right, uplo upper_lower, std::int64_t m,
                 std::int64_t n, value_or_pointer<float> alpha, const float* a, std::int64_t lda,
                 const float* b, std::int64_t ldb, value_or_pointer<float> beta, float* c,
                 std::int64_t ldc, const std::vector<sycl::event>& dependencies = {});

sycl::event symm(sycl::queue& queue, side left_right, uplo upper_lower, std::int64_t m,
                 std::int64_t n, value_or_pointer<double> alpha, const double* a, std::int64_t lda,
                 const double* b, std::int64_t ldb, value_or_pointer<double> beta, double* c,
                 std::int64_t ldc, const std::vector<sycl::event>& dependencies = {});

sycl::event symm(sycl::queue& queue, side left_right, uplo upper_lower, std::int64_t m,
                 std::int64_t n, value_or_pointer<std::complex<float>> alpha,
                 const std::complex<float>* a, std::int64_t lda, const std::complex<float>* b,
                 std::int64_t ldb, value_or_pointer<std::complex<float>> beta,
                 std::complex<float>* c, std::int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event symm(sycl::queue& queue, side left_right, uplo upper_lower, std::int64_t m,
                 std::int64_t n, value_or_pointer<std::complex<double>> alpha,
                 const std::complex<double>* a, std::int64_t lda, const std::complex<double>* b,
                 std::int64_t ldb, value_or_pointer<std::complex<double>> beta,
                 std::complex<double>* c, std::int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event hemm(sycl::queue& queue, side left_right, uplo upper_lower, std::int64_t m,
                 std::int64_t n, value_or_pointer<std::complex<float>> alpha,
                 const std::complex<float>* a, std::int64_t lda, const std::complex<float>* b,
                 std::int64_t ldb, value_or_pointer<std::complex<float>> beta,
                 std::complex<float>* c, std::int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event hemm(sycl::queue& queue, side left_right, uplo upper_lower, std::int64_t m,
                 std::int64_t n, value_or_pointer<std::complex<double>> alpha,
                 const std::complex<double>* a, std::int64_t lda, const std::complex<double>* b,
                 std::int64_t ldb, value_or_pointer<std::complex<double>> beta,
                 std::complex<double>* c, std::int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event syrk(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                 std::int64_t k, value_or_pointer<float> alpha, const float* a, std::int64_t lda,
                 value_or_pointer<float> beta, float* c, std::int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event syrk(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                 std::int64_t k, value_or_pointer<double> alpha, const double* a, std::int64_t lda,
                 value_or_pointer<double> beta, double* c, std::int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event syrk(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                 std::int64_t k, value_or_pointer<std::complex<float>> alpha,
                 const std::complex<float>* a, std::int64_t lda,
                 value_or_pointer<std::complex<float>> beta, std::complex<float>* c,
                 std::int64_t ldc, const std::vector<sycl::event>& dependencies = {});

sycl::event syrk(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                 std::int64_t k, value_or_pointer<std::complex<double>> alpha,
                 const std::complex<double>* a, std::int64_t lda,
                 value_or_pointer<std::complex<double>> beta, std::complex<double>* c,
                 std::int64_t ldc, const std::vector<sycl::event>& dependencies = {});

sycl::event herk(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                 std::int64_t k, value_or_pointer<float> alpha, const std::complex<float>* a,
                 std::int64_t lda, value_or_pointer<float> beta, std::complex<float>* c,
                 std::int64_t ldc, const std::vector<sycl::event>& dependencies = {});

sycl::event herk(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                 std::int64_t k, value_or_pointer<double> alpha, const std::complex<double>* a,
                 std::int64_t lda, value_or_pointer<double> beta, std::complex<double>* c,
                 std::int64_t ldc, const std::vector<sycl::event>& dependencies = {});

sycl::event syr2k(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                  std::int64_t k, value_or_pointer<float> alpha, const float* a, std::int64_t lda,
                  const float* b, std::int64_t ldb, value_or_pointer<float> beta, float* c,
                  std::int64_t ldc, const std::vector<sycl::event>& dependencies = {});

sycl::event syr2k(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                  std::int64_t k, value_or_pointer<double> alpha, const double* a, std::int64_t lda,
                  const double* b, std::int64_t ldb, value_or_pointer<double> beta, double* c,
                  std::int64_t ldc, const std::vector<sycl::event>& dependencies = {});

sycl::event syr2k(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                  std::int64_t k, value_or_pointer<std::complex<float>> alpha,
                  const std::complex<float>* a, std::int64_t lda, const std::complex<float>* b,
                  std::int64_t ldb, value_or_pointer<std::complex<float>> beta,
                  std::complex<float>* c, std::int64_t ldc,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event syr2k(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                  std::int64_t k, value_or_pointer<std::complex<double>> alpha,
                  const std::complex<double>* a, std::int64_t lda, const std::complex<double>* b,
                  std::int64_t ldb, value_or_pointer<std::complex<double>> beta,
                  std::complex<double>* c, std::int64_t ldc,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event her2k(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                  std::int64_t k, value_or_pointer<std::complex<float>> alpha,
                  const std::complex<float>* a, std::int64_t lda, const std::complex<float>* b,
                  std::int64_t ldb, value_or_pointer<float> beta, std::complex<float>* c,
                  std::int64_t ldc, const std::vector<sycl::event>& dependencies = {});

sycl::event her2k(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                  std::int64_t k, value_or_pointer<std::complex<double>> alpha,
                  const std::complex<double>* a, std::int64_t lda, const std::complex<double>* b,
                  std::int64_t ldb, value_or_pointer<double> beta, std::complex<double>* c,
                  std::int64_t ldc, const std::vector<sycl::event>& dependencies = {});

sycl::event trmm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                 diag unit_diag, std::int64_t m, std::int64_t n, value_or_pointer<float> alpha,
                 const float* a, std::int64_t lda, float* b, std::int64_t ldb,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event trmm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                 diag unit_diag, std::int64_t m, std::int64_t n, value_or_pointer<double> alpha,
                 const double* a, std::int64_t lda, double* b, std::int64_t ldb,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event trmm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                 diag unit_diag, std::int64_t m, std::int64_t n,
                 value_or_pointer<std::complex<float>> alpha, const std::complex<float>* a,
                 std::int64_t lda, std::complex<float>* b, std::int64_t ldb,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event trmm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                 diag unit_diag, std::int64_t m, std::int64_t n,
                 value_or_pointer<std::complex<double>> alpha, const std::complex<double>* a,
                 std::int64_t lda, std::complex<double>* b, std::int64_t ldb,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event trsm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                 diag unit_diag, std::int64_t m, std::int64_t n, value_or_pointer<float> alpha,
                 const float* a, std::int64_t lda, float* b, std::int64_t ldb,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event trsm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                 diag unit_diag, std::int64_t m, std::int64_t n, value_or_pointer<double> alpha,
                 const double* a, std::int64_t lda, double* b, std::int64_t ldb,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event trsm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                 diag unit_diag, std::int64_t m, std::int64_t n,
                 value_or_pointer<std::complex<float>> alpha, const std::complex<float>* a,
                 std::int64_t lda, std::complex<float>* b, std::int64_t ldb,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event trsm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                 diag unit_diag, std::int64_t m, std::int64_t n,
                 value_or_pointer<std::complex<double>> alpha, const std::complex<double>* a,
                 std::int64_t lda, std::complex<double>* b, std::int64_t ldb,
                 const std::vector<sycl::event>& dependencies = {});

// level 2, buffer

void gemv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n, float alpha,
          sycl::buffer<float, 1>& a, std::int64_t lda, sycl::buffer<float, 1>& x, std::int64_t incx,
          float beta, sycl::buffer<float, 1>& y, std::int64_t incy);

void gemv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n, double alpha,
          sycl::buffer<double, 1>& a, std::int64_t lda, sycl::buffer<double, 1>& x,
          std::int64_t incx, double beta, sycl::buffer<double, 1>& y, std::int64_t incy);

void gemv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy);

void gemv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy);

void gbmv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n, std::int64_t kl,
          std::int64_t ku, float alpha, sycl::buffer<float, 1>& a, std::int64_t lda,
          sycl::buffer<float, 1>& x, std::int64_t incx, float beta, sycl::buffer<float, 1>& y,
          std::int64_t incy);

void gbmv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n, std::int64_t kl,
          std::int64_t ku, double alpha, sycl::buffer<double, 1>& a, std::int64_t lda,
          sycl::buffer<double, 1>& x, std::int64_t incx, double beta, sycl::buffer<double, 1>& y,
          std::int64_t incy);

void gbmv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n, std::int64_t kl,
          std::int64_t ku, std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
          std::int64_t lda, sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
          std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy);

void gbmv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n, std::int64_t kl,
          std::int64_t ku, std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
          std::int64_t lda, sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
          std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy);

void ger(sycl::queue& queue, std::int64_t m, std::int64_t n, float alpha, sycl::buffer<float, 1>& x,
         std::int64_t incx, sycl::buffer<float, 1>& y, std::int64_t incy, sycl::buffer<float, 1>& a,
         std::int64_t lda);

void ger(sycl::queue& queue, std::int64_t m, std::int64_t n, double alpha,
         sycl::buffer<double, 1>& x, std::int64_t incx, sycl::buffer<double, 1>& y,
         std::int64_t incy, sycl::buffer<double, 1>& a, std::int64_t lda);

void gerc(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda);

void gerc(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda);

void geru(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda);

void geru(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda);

void hbmv(sycl::queue& queue, uplo upper_lower, std::int64_t n, std::int64_t k,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy);

void hbmv(sycl::queue& queue, uplo upper_lower, std::int64_t n, std::int64_t k,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy);

void hemv(sycl::queue& queue, uplo upper_lower, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy);

void hemv(sycl::queue& queue, uplo upper_lower, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy);

void her(sycl::queue& queue, uplo upper_lower, std::int64_t n, float alpha,
         sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
         sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda);

void her(sycl::queue& queue, uplo upper_lower, std::int64_t n, double alpha,
         sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
         sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda);

void her2(sycl::queue& queue, uplo upper_lower, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda);

void her2(sycl::queue& queue, uplo upper_lower, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda);

void hpmv(sycl::queue& queue, uplo upper_lower, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& a, sycl::buffer<std::complex<float>, 1>& x,
          std::int64_t incx, std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& y,
          std::int64_t incy);

void hpmv(sycl::queue& queue, uplo upper_lower, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& a, sycl::buffer<std::complex<double>, 1>& x,
          std::int64_t incx, std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& y,
          std::int64_t incy);

void hpr(sycl::queue& queue, uplo upper_lower, std::int64_t n, float alpha,
         sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
         sycl::buffer<std::complex<float>, 1>& a);

void hpr(sycl::queue& queue, uplo upper_lower, std::int64_t n, double alpha,
         sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
         sycl::buffer<std::complex<double>, 1>& a);

void hpr2(sycl::queue& queue, uplo upper_lower, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1>& a);

void hpr2(sycl::queue& queue, uplo upper_lower, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1>& a);

void sbmv(sycl::queue& queue, uplo upper_lower, std::int64_t n, std::int64_t k, float alpha,
          sycl::buffer<float, 1>& a, std::int64_t lda, sycl::buffer<float, 1>& x, std::int64_t incx,
          float beta, sycl::buffer<float, 1>& y, std::int64_t incy);

void sbmv(sycl::queue& queue, uplo upper_lower, std::int64_t n, std::int64_t k, double alpha,
          sycl::buffer<double, 1>& a, std::int64_t lda, sycl::buffer<double, 1>& x,
          std::int64_t incx, double beta, sycl::buffer<double, 1>& y, std::int64_t incy);

void symv(sycl::queue& queue, uplo upper_lower, std::int64_t n, float alpha,
          sycl::buffer<float, 1>& a, std::int64_t lda, sycl::buffer<float, 1>& x, std::int64_t incx,
          float beta, sycl::buffer<float, 1>& y, std::int64_t incy);

void symv(sycl::queue& queue, uplo upper_lower, std::int64_t n, double alpha,
          sycl::buffer<double, 1>& a, std::int64_t lda, sycl::buffer<double, 1>& x,
          std::int64_t incx, double beta, sycl::buffer<double, 1>& y, std::int64_t incy);

void syr(sycl::queue& queue, uplo upper_lower, std::int64_t n, float alpha,
         sycl::buffer<float, 1>& x, std::int64_t incx, sycl::buffer<float, 1>& a, std::int64_t lda);

void syr(sycl::queue& queue, uplo upper_lower, std::int64_t n, double alpha,
         sycl::buffer<double, 1>& x, std::int64_t incx, sycl::buffer<double, 1>& a,
         std::int64_t lda);

void syr2(sycl::queue& queue, uplo upper_lower, std::int64_t n, float alpha,
          sycl::buffer<float, 1>& x, std::int64_t incx, sycl::buffer<float, 1>& y,
          std::int64_t incy, sycl::buffer<float, 1>& a, std::int64_t lda);

void syr2(sycl::queue& queue, uplo upper_lower, std::int64_t n, double alpha,
          sycl::buffer<double, 1>& x, std::int64_t incx, sycl::buffer<double, 1>& y,
          std::int64_t incy, sycl::buffer<double, 1>& a, std::int64_t lda);

void spmv(sycl::queue& queue, uplo upper_lower, std::int64_t n, float alpha,
          sycl::buffer<float, 1>& a, sycl::buffer<float, 1>& x, std::int64_t incx, float beta,
          sycl::buffer<float, 1>& y, std::int64_t incy);

void spmv(sycl::queue& queue, uplo upper_lower, std::int64_t n, double alpha,
          sycl::buffer<double, 1>& a, sycl::buffer<double, 1>& x, std::int64_t incx, double beta,
          sycl::buffer<double, 1>& y, std::int64_t incy);

void spr(sycl::queue& queue, uplo upper_lower, std::int64_t n, float alpha,
         sycl::buffer<float, 1>& x, std::int64_t incx, sycl::buffer<float, 1>& a);

void spr(sycl::queue& queue, uplo upper_lower, std::int64_t n, double alpha,
         sycl::buffer<double, 1>& x, std::int64_t incx, sycl::buffer<double, 1>& a);

void spr2(sycl::queue& queue, uplo upper_lower, std::int64_t n, float alpha,
          sycl::buffer<float, 1>& x, std::int64_t incx, sycl::buffer<float, 1>& y,
          std::int64_t incy, sycl::buffer<float, 1>& a);

void spr2(sycl::queue& queue, uplo upper_lower, std::int64_t n, double alpha,
          sycl::buffer<double, 1>& x, std::int64_t incx, sycl::buffer<double, 1>& y,
          std::int64_t incy, sycl::buffer<double, 1>& a);

void tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          std::int64_t k, sycl::buffer<float, 1>& a, std::int64_t lda, sycl::buffer<float, 1>& x,
          std::int64_t incx);

void tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          std::int64_t k, sycl::buffer<double, 1>& a, std::int64_t lda, sycl::buffer<double, 1>& x,
          std::int64_t incx);

void tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          std::int64_t k, sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx);

void tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          std::int64_t k, sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx);

void tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          std::int64_t k, sycl::buffer<float, 1>& a, std::int64_t lda, sycl::buffer<float, 1>& x,
          std::int64_t incx);

void tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          std::int64_t k, sycl::buffer<double, 1>& a, std::int64_t lda, sycl::buffer<double, 1>& x,
          std::int64_t incx);

void tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          std::int64_t k, sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx);

void tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          std::int64_t k, sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx);

void tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          sycl::buffer<float, 1>& a, sycl::buffer<float, 1>& x, std::int64_t incx);

void tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          sycl::buffer<double, 1>& a, sycl::buffer<double, 1>& x, std::int64_t incx);

void tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          sycl::buffer<std::complex<float>, 1>& a, sycl::buffer<std::complex<float>, 1>& x,
          std::int64_t incx);

void tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          sycl::buffer<std::complex<double>, 1>& a, sycl::buffer<std::complex<double>, 1>& x,
          std::int64_t incx);

void tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          sycl::buffer<float, 1>& a, sycl::buffer<float, 1>& x, std::int64_t incx);

void tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          sycl::buffer<double, 1>& a, sycl::buffer<double, 1>& x, std::int64_t incx);

void tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          sycl::buffer<std::complex<float>, 1>& a, sycl::buffer<std::complex<float>, 1>& x,
          std::int64_t incx);

void tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          sycl::buffer<std::complex<double>, 1>& a, sycl::buffer<std::complex<double>, 1>& x,
          std::int64_t incx);

void trmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          sycl::buffer<float, 1>& a, std::int64_t lda, sycl::buffer<float, 1>& x,
          std::int64_t incx);

void trmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          sycl::buffer<double, 1>& a, std::int64_t lda, sycl::buffer<double, 1>& x,
          std::int64_t incx);

void trmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx);

void trmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx);

void trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          sycl::buffer<float, 1>& a, std::int64_t lda, sycl::buffer<float, 1>& x,
          std::int64_t incx);

void trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          sycl::buffer<double, 1>& a, std::int64_t lda, sycl::buffer<double, 1>& x,
          std::int64_t incx);

void trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx);

void trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx);

// level 2, USM

sycl::event gemv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                 value_or_pointer<float> alpha, const float* a, std::int64_t lda, const float* x,
                 std::int64_t incx, value_or_pointer<float> beta, float* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event gemv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                 value_or_pointer<double> alpha, const double* a, std::int64_t lda, const double* x,
                 std::int64_t incx, value_or_pointer<double> beta, double* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event gemv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                 value_or_pointer<std::complex<float>> alpha, const std::complex<float>* a,
                 std::int64_t lda, const std::complex<float>* x, std::int64_t incx,
                 value_or_pointer<std::complex<float>> beta, std::complex<float>* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies = {});

sycl::event gemv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                 value_or_pointer<std::complex<double>> alpha, const std::complex<double>* a,
                 std::int64_t lda, const std::complex<double>* x, std::int64_t incx,
                 value_or_pointer<std::complex<double>> beta, std::complex<double>* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies = {});

sycl::event gbmv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                 std::int64_t kl, std::int64_t ku, value_or_pointer<float> alpha, const float* a,
                 std::int64_t lda, const float* x, std::int64_t incx, value_or_pointer<float> beta,
                 float* y, std::int64_t incy, const std::vector<sycl::event>& dependencies = {});

sycl::event gbmv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                 std::int64_t kl, std::int64_t ku, value_or_pointer<double> alpha, const double* a,
                 std::int64_t lda, const double* x, std::int64_t incx,
                 value_or_pointer<double> beta, double* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event gbmv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                 std::int64_t kl, std::int64_t ku, value_or_pointer<std::complex<float>> alpha,
                 const std::complex<float>* a, std::int64_t lda, const std::complex<float>* x,
                 std::int64_t incx, value_or_pointer<std::complex<float>> beta,
                 std::complex<float>* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event gbmv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                 std::int64_t kl, std::int64_t ku, value_or_pointer<std::complex<double>> alpha,
                 const std::complex<double>* a, std::int64_t lda, const std::complex<double>* x,
                 std::int64_t incx, value_or_pointer<std::complex<double>> beta,
                 std::complex<double>* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event ger(sycl::queue& queue, std::int64_t m, std::int64_t n, value_or_pointer<float> alpha,
                const float* x, std::int64_t incx, const float* y, std::int64_t incy, float* a,
                std::int64_t lda, const std::vector<sycl::event>& dependencies = {});

sycl::event ger(sycl::queue& queue, std::int64_t m, std::int64_t n, value_or_pointer<double> alpha,
                const double* x, std::int64_t incx, const double* y, std::int64_t incy, double* a,
                std::int64_t lda, const std::vector<sycl::event>& dependencies = {});

sycl::event gerc(sycl::queue& queue, std::int64_t m, std::int64_t n,
                 value_or_pointer<std::complex<float>> alpha, const std::complex<float>* x,
                 std::int64_t incx, const std::complex<float>* y, std::int64_t incy,
                 std::complex<float>* a, std::int64_t lda,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event gerc(sycl::queue& queue, std::int64_t m, std::int64_t n,
                 value_or_pointer<std::complex<double>> alpha, const std::complex<double>* x,
                 std::int64_t incx, const std::complex<double>* y, std::int64_t incy,
                 std::complex<double>* a, std::int64_t lda,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event geru(sycl::queue& queue, std::int64_t m, std::int64_t n,
                 value_or_pointer<std::complex<float>> alpha, const std::complex<float>* x,
                 std::int64_t incx, const std::complex<float>* y, std::int64_t incy,
                 std::complex<float>* a, std::int64_t lda,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event geru(sycl::queue& queue, std::int64_t m, std::int64_t n,
                 value_or_pointer<std::complex<double>> alpha, const std::complex<double>* x,
                 std::int64_t incx, const std::complex<double>* y, std::int64_t incy,
                 std::complex<double>* a, std::int64_t lda,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event hbmv(sycl::queue& queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                 value_or_pointer<std::complex<float>> alpha, const std::complex<float>* a,
                 std::int64_t lda, const std::complex<float>* x, std::int64_t incx,
                 value_or_pointer<std::complex<float>> beta, std::complex<float>* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies = {});

sycl::event hbmv(sycl::queue& queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                 value_or_pointer<std::complex<double>> alpha, const std::complex<double>* a,
                 std::int64_t lda, const std::complex<double>* x, std::int64_t incx,
                 value_or_pointer<std::complex<double>> beta, std::complex<double>* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies = {});

sycl::event hemv(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                 value_or_pointer<std::complex<float>> alpha, const std::complex<float>* a,
                 std::int64_t lda, const std::complex<float>* x, std::int64_t incx,
                 value_or_pointer<std::complex<float>> beta, std::complex<float>* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies = {});

sycl::event hemv(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                 value_or_pointer<std::complex<double>> alpha, const std::complex<double>* a,
                 std::int64_t lda, const std::complex<double>* x, std::int64_t incx,
                 value_or_pointer<std::complex<double>> beta, std::complex<double>* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies = {});

sycl::event her(sycl::queue& queue, uplo upper_lower, std::int64_t n, value_or_pointer<float> alpha,
                const std::complex<float>* x, std::int64_t incx, std::complex<float>* a,
                std::int64_t lda, const std::vector<sycl::event>& dependencies = {});

sycl::event her(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                value_or_pointer<double> alpha, const std::complex<double>* x, std::int64_t incx,
                std::complex<double>* a, std::int64_t lda,
                const std::vector<sycl::event>& dependencies = {});

sycl::event her2(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                 value_or_pointer<std::complex<float>> alpha, const std::complex<float>* x,
                 std::int64_t incx, const std::complex<float>* y, std::int64_t incy,
                 std::complex<float>* a, std::int64_t lda,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event her2(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                 value_or_pointer<std::complex<double>> alpha, const std::complex<double>* x,
                 std::int64_t incx, const std::complex<double>* y, std::int64_t incy,
                 std::complex<double>* a, std::int64_t lda,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event hpmv(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                 value_or_pointer<std::complex<float>> alpha, const std::complex<float>* a,
                 const std::complex<float>* x, std::int64_t incx,
                 value_or_pointer<std::complex<float>> beta, std::complex<float>* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies = {});

sycl::event hpmv(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                 value_or_pointer<std::complex<double>> alpha, const std::complex<double>* a,
                 const std::complex<double>* x, std::int64_t incx,
                 value_or_pointer<std::complex<double>> beta, std::complex<double>* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies = {});

sycl::event hpr(sycl::queue& queue, uplo upper_lower, std::int64_t n, value_or_pointer<float> alpha,
                const std::complex<float>* x, std::int64_t incx, std::complex<float>* a,
                const std::vector<sycl::event>& dependencies = {});

sycl::event hpr(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                value_or_pointer<double> alpha, const std::complex<double>* x, std::int64_t incx,
                std::complex<double>* a, const std::vector<sycl::event>& dependencies = {});

sycl::event hpr2(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                 value_or_pointer<std::complex<float>> alpha, const std::complex<float>* x,
                 std::int64_t incx, const std::complex<float>* y, std::int64_t incy,
                 std::complex<float>* a, const std::vector<sycl::event>& dependencies = {});

sycl::event hpr2(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                 value_or_pointer<std::complex<double>> alpha, const std::complex<double>* x,
                 std::int64_t incx, const std::complex<double>* y, std::int64_t incy,
                 std::complex<double>* a, const std::vector<sycl::event>& dependencies = {});

sycl::event sbmv(sycl::queue& queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                 value_or_pointer<float> alpha, const float* a, std::int64_t lda, const float* x,
                 std::int64_t incx, value_or_pointer<float> beta, float* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event sbmv(sycl::queue& queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                 value_or_pointer<double> alpha, const double* a, std::int64_t lda, const double* x,
                 std::int64_t incx, value_or_pointer<double> beta, double* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event symv(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                 value_or_pointer<float> alpha, const float* a, std::int64_t lda, const float* x,
                 std::int64_t incx, value_or_pointer<float> beta, float* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event symv(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                 value_or_pointer<double> alpha, const double* a, std::int64_t lda, const double* x,
                 std::int64_t incx, value_or_pointer<double> beta, double* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event syr(sycl::queue& queue, uplo upper_lower, std::int64_t n, value_or_pointer<float> alpha,
                const float* x, std::int64_t incx, float* a, std::int64_t lda,
                const std::vector<sycl::event>& dependencies = {});

sycl::event syr(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                value_or_pointer<double> alpha, const double* x, std::int64_t incx, double* a,
                std::int64_t lda, const std::vector<sycl::event>& dependencies = {});

sycl::event syr2(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                 value_or_pointer<float> alpha, const float* x, std::int64_t incx, const float* y,
                 std::int64_t incy, float* a, std::int64_t lda,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event syr2(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                 value_or_pointer<double> alpha, const double* x, std::int64_t incx,
                 const double* y, std::int64_t incy, double* a, std::int64_t lda,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event spmv(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                 value_or_pointer<float> alpha, const float* a, const float* x, std::int64_t incx,
                 value_or_pointer<float> beta, float* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event spmv(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                 value_or_pointer<double> alpha, const double* a, const double* x,
                 std::int64_t incx, value_or_pointer<double> beta, double* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event spr(sycl::queue& queue, uplo upper_lower, std::int64_t n, value_or_pointer<float> alpha,
                const float* x, std::int64_t incx, float* a,
                const std::vector<sycl::event>& dependencies = {});

sycl::event spr(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                value_or_pointer<double> alpha, const double* x, std::int64_t incx, double* a,
                const std::vector<sycl::event>& dependencies = {});

sycl::event spr2(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                 value_or_pointer<float> alpha, const float* x, std::int64_t incx, const float* y,
                 std::int64_t incy, float* a, const std::vector<sycl::event>& dependencies = {});

sycl::event spr2(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                 value_or_pointer<double> alpha, const double* x, std::int64_t incx,
                 const double* y, std::int64_t incy, double* a,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                 std::int64_t n, std::int64_t k, const float* a, std::int64_t lda, float* x,
                 std::int64_t incx, const std::vector<sycl::event>& dependencies = {});

sycl::event tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                 std::int64_t n, std::int64_t k, const double* a, std::int64_t lda, double* x,
                 std::int64_t incx, const std::vector<sycl::event>& dependencies = {});

sycl::event tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                 std::int64_t n, std::int64_t k, const std::complex<float>* a, std::int64_t lda,
                 std::complex<float>* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                 std::int64_t n, std::int64_t k, const std::complex<double>* a, std::int64_t lda,
                 std::complex<double>* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                 std::int64_t n, std::int64_t k, const float* a, std::int64_t lda, float* x,
                 std::int64_t incx, const std::vector<sycl::event>& dependencies = {});

sycl::event tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                 std::int64_t n, std::int64_t k, const double* a, std::int64_t lda, double* x,
                 std::int64_t incx, const std::vector<sycl::event>& dependencies = {});

sycl::event tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                 std::int64_t n, std::int64_t k, const std::complex<float>* a, std::int64_t lda,
                 std::complex<float>* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                 std::int64_t n, std::int64_t k, const std::complex<double>* a, std::int64_t lda,
                 std::complex<double>* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                 std::int64_t n, const float* a, float* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                 std::int64_t n, const double* a, double* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                 std::int64_t n, const std::complex<float>* a, std::complex<float>* x,
                 std::int64_t incx, const std::vector<sycl::event>& dependencies = {});

sycl::event tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                 std::int64_t n, const std::complex<double>* a, std::complex<double>* x,
                 std::int64_t incx, const std::vector<sycl::event>& dependencies = {});

sycl::event tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                 std::int64_t n, const float* a, float* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                 std::int64_t n, const double* a, double* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                 std::int64_t n, const std::complex<float>* a, std::complex<float>* x,
                 std::int64_t incx, const std::vector<sycl::event>& dependencies = {});

sycl::event tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                 std::int64_t n, const std::complex<double>* a, std::complex<double>* x,
                 std::int64_t incx, const std::vector<sycl::event>& dependencies = {});

sycl::event trmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                 std::int64_t n, const float* a, std::int64_t lda, float* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event trmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                 std::int64_t n, const double* a, std::int64_t lda, double* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event trmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                 std::int64_t n, const std::complex<float>* a, std::int64_t lda,
                 std::complex<float>* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event trmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                 std::int64_t n, const std::complex<double>* a, std::int64_t lda,
                 std::complex<double>* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                 std::int64_t n, const float* a, std::int64_t lda, float* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                 std::int64_t n, const double* a, std::int64_t lda, double* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                 std::int64_t n, const std::complex<float>* a, std::int64_t lda,
                 std::complex<float>* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                 std::int64_t n, const std::complex<double>* a, std::int64_t lda,
                 std::complex<double>* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

// level 1, buffer

void dotc(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
          std::int64_t incx, sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1>& result);

void dotc(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
          std::int64_t incx, sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1>& result);

void dotu(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
          std::int64_t incx, sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1>& result);

void dotu(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
          std::int64_t incx, sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1>& result);

void iamax(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
           sycl::buffer<std::int64_t, 1>& result, index_base base = index_base::zero);

void iamax(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
           sycl::buffer<std::int64_t, 1>& result, index_base base = index_base::zero);

void iamax(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
           std::int64_t incx, sycl::buffer<std::int64_t, 1>& result,
           index_base base = index_base::zero);

void iamax(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
           std::int64_t incx, sycl::buffer<std::int64_t, 1>& result,
           index_base base = index_base::zero);

void iamin(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
           sycl::buffer<std::int64_t, 1>& result, index_base base = index_base::zero);

void iamin(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
           sycl::buffer<std::int64_t, 1>& result, index_base base = index_base::zero);

void iamin(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
           std::int64_t incx, sycl::buffer<std::int64_t, 1>& result,
           index_base base = index_base::zero);

void iamin(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
           std::int64_t incx, sycl::buffer<std::int64_t, 1>& result,
           index_base base = index_base::zero);

void asum(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
          std::int64_t incx, sycl::buffer<float, 1>& result);

void asum(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
          std::int64_t incx, sycl::buffer<double, 1>& result);

void asum(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
          sycl::buffer<float, 1>& result);

void asum(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
          sycl::buffer<double, 1>& result);

void axpy(sycl::queue& queue, std::int64_t n, float alpha, sycl::buffer<float, 1>& x,
          std::int64_t incx, sycl::buffer<float, 1>& y, std::int64_t incy);

void axpy(sycl::queue& queue, std::int64_t n, double alpha, sycl::buffer<double, 1>& x,
          std::int64_t incx, sycl::buffer<double, 1>& y, std::int64_t incy);

void axpy(sycl::queue& queue, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy);

void axpy(sycl::queue& queue, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy);

void axpby(sycl::queue& queue, std::int64_t n, float alpha, sycl::buffer<float, 1>& x,
           std::int64_t incx, float beta, sycl::buffer<float, 1>& y, std::int64_t incy);

void axpby(sycl::queue& queue, std::int64_t n, double alpha, sycl::buffer<double, 1>& x,
           std::int64_t incx, double beta, sycl::buffer<double, 1>& y, std::int64_t incy);

void axpby(sycl::queue& queue, std::int64_t n, std::complex<float> alpha,
           sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx, std::complex<float> beta,
           sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy);

void axpby(sycl::queue& queue, std::int64_t n, std::complex<double> alpha,
           sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx, std::complex<double> beta,
           sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy);

void copy(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
          sycl::buffer<float, 1>& y, std::int64_t incy);

void copy(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
          sycl::buffer<double, 1>& y, std::int64_t incy);

void copy(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
          std::int64_t incx, sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy);

void copy(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
          std::int64_t incx, sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy);

void dot(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
         sycl::buffer<float, 1>& y, std::int64_t incy, sycl::buffer<float, 1>& result);

void dot(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
         sycl::buffer<double, 1>& y, std::int64_t incy, sycl::buffer<double, 1>& result);

void sdsdot(sycl::queue& queue, std::int64_t n, float sb, sycl::buffer<float, 1>& x,
            std::int64_t incx, sycl::buffer<float, 1>& y, std::int64_t incy,
            sycl::buffer<float, 1>& result);

void dot(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
         sycl::buffer<float, 1>& y, std::int64_t incy, sycl::buffer<double, 1>& result);

void nrm2(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
          std::int64_t incx, sycl::buffer<float, 1>& result);

void nrm2(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
          std::int64_t incx, sycl::buffer<double, 1>& result);

void nrm2(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
          sycl::buffer<float, 1>& result);

void nrm2(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
          sycl::buffer<double, 1>& result);

void rot(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
         std::int64_t incx, sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy, float c,
         float s);

void rot(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
         std::int64_t incx, sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy, double c,
         double s);

void rot(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
         sycl::buffer<float, 1>& y, std::int64_t incy, float c, float s);

void rot(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
         sycl::buffer<double, 1>& y, std::int64_t incy, double c, double s);

void rotg(sycl::queue& queue, sycl::buffer<float, 1>& a, sycl::buffer<float, 1>& b,
          sycl::buffer<float, 1>& c, sycl::buffer<float, 1>& s);

void rotg(sycl::queue& queue, sycl::buffer<double, 1>& a, sycl::buffer<double, 1>& b,
          sycl::buffer<double, 1>& c, sycl::buffer<double, 1>& s);

void rotg(sycl::queue& queue, sycl::buffer<std::complex<float>, 1>& a,
          sycl::buffer<std::complex<float>, 1>& b, sycl::buffer<float, 1>& c,
          sycl::buffer<std::complex<float>, 1>& s);

void rotg(sycl::queue& queue, sycl::buffer<std::complex<double>, 1>& a,
          sycl::buffer<std::complex<double>, 1>& b, sycl::buffer<double, 1>& c,
          sycl::buffer<std::complex<double>, 1>& s);

void rotm(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
          sycl::buffer<float, 1>& y, std::int64_t incy, sycl::buffer<float, 1>& param);

void rotm(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
          sycl::buffer<double, 1>& y, std::int64_t incy, sycl::buffer<double, 1>& param);

void rotmg(sycl::queue& queue, sycl::buffer<float, 1>& d1, sycl::buffer<float, 1>& d2,
           sycl::buffer<float, 1>& x1, float y1, sycl::buffer<float, 1>& param);

void rotmg(sycl::queue& queue, sycl::buffer<double, 1>& d1, sycl::buffer<double, 1>& d2,
           sycl::buffer<double, 1>& x1, double y1, sycl::buffer<double, 1>& param);

void scal(sycl::queue& queue, std::int64_t n, float alpha, sycl::buffer<float, 1>& x,
          std::int64_t incx);

void scal(sycl::queue& queue, std::int64_t n, double alpha, sycl::buffer<double, 1>& x,
          std::int64_t incx);

void scal(sycl::queue& queue, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx);

void scal(sycl::queue& queue, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx);

void scal(sycl::queue& queue, std::int64_t n, float alpha, sycl::buffer<std::complex<float>, 1>& x,
          std::int64_t incx);

void scal(sycl::queue& queue, std::int64_t n, double alpha,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx);

void swap(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
          sycl::buffer<float, 1>& y, std::int64_t incy);

void swap(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
          sycl::buffer<double, 1>& y, std::int64_t incy);

void swap(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
          std::int64_t incx, sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy);

void swap(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
          std::int64_t incx, sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy);

// level 1, USM

sycl::event dotc(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                 std::int64_t incx, const std::complex<float>* y, std::int64_t incy,
                 std::complex<float>* result, const std::vector<sycl::event>& dependencies = {});

sycl::event dotc(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                 std::int64_t incx, const std::complex<double>* y, std::int64_t incy,
                 std::complex<double>* result, const std::vector<sycl::event>& dependencies = {});

sycl::event dotu(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                 std::int64_t incx, const std::complex<float>* y, std::int64_t incy,
                 std::complex<float>* result, const std::vector<sycl::event>& dependencies = {});

sycl::event dotu(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                 std::int64_t incx, const std::complex<double>* y, std::int64_t incy,
                 std::complex<double>* result, const std::vector<sycl::event>& dependencies = {});

sycl::event iamax(sycl::queue& queue, std::int64_t n, const float* x, std::int64_t incx,
                  std::int64_t* result, const std::vector<sycl::event>& dependencies = {});

sycl::event iamax(sycl::queue& queue, std::int64_t n, const double* x, std::int64_t incx,
                  std::int64_t* result, const std::vector<sycl::event>& dependencies = {});

sycl::event iamax(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                  std::int64_t incx, std::int64_t* result,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event iamax(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                  std::int64_t incx, std::int64_t* result,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event iamin(sycl::queue& queue, std::int64_t n, const float* x, std::int64_t incx,
                  std::int64_t* result, const std::vector<sycl::event>& dependencies = {});

sycl::event iamin(sycl::queue& queue, std::int64_t n, const double* x, std::int64_t incx,
                  std::int64_t* result, const std::vector<sycl::event>& dependencies = {});

sycl::event iamin(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                  std::int64_t incx, std::int64_t* result,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event iamin(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                  std::int64_t incx, std::int64_t* result,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event asum(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                 std::int64_t incx, float* result,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event asum(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                 std::int64_t incx, double* result,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event asum(sycl::queue& queue, std::int64_t n, const float* x, std::int64_t incx,
                 float* result, const std::vector<sycl::event>& dependencies = {});

sycl::event asum(sycl::queue& queue, std::int64_t n, const double* x, std::int64_t incx,
                 double* result, const std::vector<sycl::event>& dependencies = {});

sycl::event axpy(sycl::queue& queue, std::int64_t n, value_or_pointer<float> alpha, const float* x,
                 std::int64_t incx, float* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event axpy(sycl::queue& queue, std::int64_t n, value_or_pointer<double> alpha,
                 const double* x, std::int64_t incx, double* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event axpy(sycl::queue& queue, std::int64_t n, value_or_pointer<std::complex<float>> alpha,
                 const std::complex<float>* x, std::int64_t incx, std::complex<float>* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies = {});

sycl::event axpy(sycl::queue& queue, std::int64_t n, value_or_pointer<std::complex<double>> alpha,
                 const std::complex<double>* x, std::int64_t incx, std::complex<double>* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies = {});

sycl::event axpby(sycl::queue& queue, std::int64_t n, value_or_pointer<float> alpha, const float* x,
                  std::int64_t incx, value_or_pointer<float> beta, float* y, std::int64_t incy,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event axpby(sycl::queue& queue, std::int64_t n, value_or_pointer<double> alpha,
                  const double* x, std::int64_t incx, value_or_pointer<double> beta, double* y,
                  std::int64_t incy, const std::vector<sycl::event>& dependencies = {});

sycl::event axpby(sycl::queue& queue, std::int64_t n, value_or_pointer<std::complex<float>> alpha,
                  const std::complex<float>* x, std::int64_t incx,
                  value_or_pointer<std::complex<float>> beta, std::complex<float>* y,
                  std::int64_t incy, const std::vector<sycl::event>& dependencies = {});

sycl::event axpby(sycl::queue& queue, std::int64_t n, value_or_pointer<std::complex<double>> alpha,
                  const std::complex<double>* x, std::int64_t incx,
                  value_or_pointer<std::complex<double>> beta, std::complex<double>* y,
                  std::int64_t incy, const std::vector<sycl::event>& dependencies = {});

sycl::event copy(sycl::queue& queue, std::int64_t n, const float* x, std::int64_t incx, float* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies = {});

sycl::event copy(sycl::queue& queue, std::int64_t n, const double* x, std::int64_t incx, double* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies = {});

sycl::event copy(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                 std::int64_t incx, std::complex<float>* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event copy(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                 std::int64_t incx, std::complex<double>* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event dot(sycl::queue& queue, std::int64_t n, const float* x, std::int64_t incx,
                const float* y, std::int64_t incy, float* result,
                const std::vector<sycl::event>& dependencies = {});

sycl::event dot(sycl::queue& queue, std::int64_t n, const double* x, std::int64_t incx,
                const double* y, std::int64_t incy, double* result,
                const std::vector<sycl::event>& dependencies = {});

sycl::event sdsdot(sycl::queue& queue, std::int64_t n, float sb, const float* x, std::int64_t incx,
                   const float* y, std::int64_t incy, float* result,
                   const std::vector<sycl::event>& dependencies = {});

sycl::event dot(sycl::queue& queue, std::int64_t n, const float* x, std::int64_t incx,
                const float* y, std::int64_t incy, double* result,
                const std::vector<sycl::event>& dependencies = {});

sycl::event nrm2(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                 std::int64_t incx, float* result,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event nrm2(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                 std::int64_t incx, double* result,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event nrm2(sycl::queue& queue, std::int64_t n, const float* x, std::int64_t incx,
                 float* result, const std::vector<sycl::event>& dependencies = {});

sycl::event nrm2(sycl::queue& queue, std::int64_t n, const double* x, std::int64_t incx,
                 double* result, const std::vector<sycl::event>& dependencies = {});

sycl::event rot(sycl::queue& queue, std::int64_t n, std::complex<float>* x, std::int64_t incx,
                std::complex<float>* y, std::int64_t incy, value_or_pointer<float> c,
                value_or_pointer<float> s, const std::vector<sycl::event>& dependencies = {});

sycl::event rot(sycl::queue& queue, std::int64_t n, std::complex<double>* x, std::int64_t incx,
                std::complex<double>* y, std::int64_t incy, value_or_pointer<double> c,
                value_or_pointer<double> s, const std::vector<sycl::event>& dependencies = {});

sycl::event rot(sycl::queue& queue, std::int64_t n, float* x, std::int64_t incx, float* y,
                std::int64_t incy, value_or_pointer<float> c, value_or_pointer<float> s,
                const std::vector<sycl::event>& dependencies = {});

sycl::event rot(sycl::queue& queue, std::int64_t n, double* x, std::int64_t incx, double* y,
                std::int64_t incy, value_or_pointer<double> c, value_or_pointer<double> s,
                const std::vector<sycl::event>& dependencies = {});

sycl::event rotg(sycl::queue& queue, float* a, float* b, float* c, float* s,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event rotg(sycl::queue& queue, double* a, double* b, double* c, double* s,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event rotg(sycl::queue& queue, std::complex<float>* a, std::complex<float>* b, float* c,
                 std::complex<float>* s, const std::vector<sycl::event>& dependencies = {});

sycl::event rotg(sycl::queue& queue, std::complex<double>* a, std::complex<double>* b, double* c,
                 std::complex<double>* s, const std::vector<sycl::event>& dependencies = {});

sycl::event rotm(sycl::queue& queue, std::int64_t n, float* x, std::int64_t incx, float* y,
                 std::int64_t incy, const float* param,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event rotm(sycl::queue& queue, std::int64_t n, double* x, std::int64_t incx, double* y,
                 std::int64_t incy, const double* param,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event rotmg(sycl::queue& queue, float* d1, float* d2, float* x1, value_or_pointer<float> y1,
                  float* param, const std::vector<sycl::event>& dependencies = {});

sycl::event rotmg(sycl::queue& queue, double* d1, double* d2, double* x1,
                  value_or_pointer<double> y1, double* param,
                  const std::vector<sycl::event>& dependencies = {});

#define ONEMKL_DECLARE_SCAL(T, Ts)                                                         \
    sycl::event scal(sycl::queue& queue, std::int64_t n, value_or_pointer<Ts> alpha, T* x, \
                     std::int64_t incx, const std::vector<sycl::event>& dependencies = {});

ONEMKL_DECLARE_SCAL(float, float)
ONEMKL_DECLARE_SCAL(double, double)
ONEMKL_DECLARE_SCAL(std::complex<float>, std::complex<float>)
ONEMKL_DECLARE_SCAL(std::complex<double>, std::complex<double>)
ONEMKL_DECLARE_SCAL(std::complex<float>, float)
ONEMKL_DECLARE_SCAL(std::complex<double>, double)
sycl::event scal(sycl::queue& queue, std::int64_t n, float alpha, std::complex<float>* x,
                 std::int64_t incx, const std::vector<sycl::event>& dependencies = {});
sycl::event scal(sycl::queue& queue, std::int64_t n, double alpha, std::complex<double>* x,
                 std::int64_t incx, const std::vector<sycl::event>& dependencies = {});

#undef ONEMKL_DECLARE_SCAL

sycl::event swap(sycl::queue& queue, std::int64_t n, float* x, std::int64_t incx, float* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies = {});

sycl::event swap(sycl::queue& queue, std::int64_t n, double* x, std::int64_t incx, double* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies = {});

sycl::event swap(sycl::queue& queue, std::int64_t n, std::complex<float>* x, std::int64_t incx,
                 std::complex<float>* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event swap(sycl::queue& queue, std::int64_t n, std::complex<double>* x, std::int64_t incx,
                 std::complex<double>* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

// extensions, buffer

void gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb, std::int64_t n,
           std::int64_t k, float alpha, sycl::buffer<float, 1>& a, std::int64_t lda,
           sycl::buffer<float, 1>& b, std::int64_t ldb, float beta, sycl::buffer<float, 1>& c,
           std::int64_t ldc);

void gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb, std::int64_t n,
           std::int64_t k, double alpha, sycl::buffer<double, 1>& a, std::int64_t lda,
           sycl::buffer<double, 1>& b, std::int64_t ldb, double beta, sycl::buffer<double, 1>& c,
           std::int64_t ldc);

void gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb, std::int64_t n,
           std::int64_t k, std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
           std::int64_t lda, sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb,
           std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc);

void gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb, std::int64_t n,
           std::int64_t k, std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
           std::int64_t lda, sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb,
           std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& c, std::int64_t ldc);

void gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc,
               std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
               sycl::buffer<std::int8_t, 1>& a, std::int64_t lda, std::int8_t ao,
               sycl::buffer<std::uint8_t, 1>& b, std::int64_t ldb, std::uint8_t bo, float beta,
               sycl::buffer<std::int32_t, 1>& c, std::int64_t ldc,
               sycl::buffer<std::int32_t, 1>& co);

void gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc,
               std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
               sycl::buffer<std::int8_t, 1>& a, std::int64_t lda, std::int8_t ao,
               sycl::buffer<std::int8_t, 1>& b, std::int64_t ldb, std::int8_t bo, float beta,
               sycl::buffer<std::int32_t, 1>& c, std::int64_t ldc,
               sycl::buffer<std::int32_t, 1>& co);

void gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc,
               std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
               sycl::buffer<std::uint8_t, 1>& a, std::int64_t lda, std::uint8_t ao,
               sycl::buffer<std::int8_t, 1>& b, std::int64_t ldb, std::int8_t bo, float beta,
               sycl::buffer<std::int32_t, 1>& c, std::int64_t ldc,
               sycl::buffer<std::int32_t, 1>& co);

void gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc,
               std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
               sycl::buffer<std::uint8_t, 1>& a, std::int64_t lda, std::uint8_t ao,
               sycl::buffer<std::uint8_t, 1>& b, std::int64_t ldb, std::uint8_t bo, float beta,
               sycl::buffer<std::int32_t, 1>& c, std::int64_t ldc,
               sycl::buffer<std::int32_t, 1>& co);

// extensions, USM

sycl::event gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb,
                  std::int64_t n, std::int64_t k, value_or_pointer<float> alpha, const float* a,
                  std::int64_t lda, const float* b, std::int64_t ldb, value_or_pointer<float> beta,
                  float* c, std::int64_t ldc, const std::vector<sycl::event>& dependencies = {});

sycl::event gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb,
                  std::int64_t n, std::int64_t k, value_or_pointer<double> alpha, const double* a,
                  std::int64_t lda, const double* b, std::int64_t ldb,
                  value_or_pointer<double> beta, double* c, std::int64_t ldc,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb,
                  std::int64_t n, std::int64_t k, value_or_pointer<std::complex<float>> alpha,
                  const std::complex<float>* a, std::int64_t lda, const std::complex<float>* b,
                  std::int64_t ldb, value_or_pointer<std::complex<float>> beta,
                  std::complex<float>* c, std::int64_t ldc,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb,
                  std::int64_t n, std::int64_t k, value_or_pointer<std::complex<double>> alpha,
                  const std::complex<double>* a, std::int64_t lda, const std::complex<double>* b,
                  std::int64_t ldb, value_or_pointer<std::complex<double>> beta,
                  std::complex<double>* c, std::int64_t ldc,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc,
                      std::int64_t m, std::int64_t n, std::int64_t k, value_or_pointer<float> alpha,
                      const std::int8_t* a, std::int64_t lda, std::int8_t ao, const std::uint8_t* b,
                      std::int64_t ldb, std::uint8_t bo, value_or_pointer<float> beta,
                      std::int32_t* c, std::int64_t ldc, const std::int32_t* co,
                      const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc,
                      std::int64_t m, std::int64_t n, std::int64_t k, value_or_pointer<float> alpha,
                      const std::int8_t* a, std::int64_t lda, std::int8_t ao, const std::int8_t* b,
                      std::int64_t ldb, std::int8_t bo, value_or_pointer<float> beta,
                      std::int32_t* c, std::int64_t ldc, const std::int32_t* co,
                      const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc,
                      std::int64_t m, std::int64_t n, std::int64_t k, value_or_pointer<float> alpha,
                      const std::uint8_t* a, std::int64_t lda, std::uint8_t ao,
                      const std::int8_t* b, std::int64_t ldb, std::int8_t bo,
                      value_or_pointer<float> beta, std::int32_t* c, std::int64_t ldc,
                      const std::int32_t* co, const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc,
                      std::int64_t m, std::int64_t n, std::int64_t k, value_or_pointer<float> alpha,
                      const std::uint8_t* a, std::int64_t lda, std::uint8_t ao,
                      const std::uint8_t* b, std::int64_t ldb, std::uint8_t bo,
                      value_or_pointer<float> beta, std::int32_t* c, std::int64_t ldc,
                      const std::int32_t* co, const std::vector<sycl::event>& dependencies = {});

// batch, buffer

void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                float alpha, sycl::buffer<float, 1>& a, int64_t lda, int64_t stride_a, float beta,
                sycl::buffer<float, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size);
void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                double alpha, sycl::buffer<double, 1>& a, int64_t lda, int64_t stride_a,
                double beta, sycl::buffer<double, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size);
void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
                int64_t stride_a, std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& c,
                int64_t ldc, int64_t stride_c, int64_t batch_size);
void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
                int64_t stride_a, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size);

void copy_batch(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
                std::int64_t stridex, sycl::buffer<float, 1>& y, std::int64_t incy,
                std::int64_t stridey, std::int64_t batch_size);

void copy_batch(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
                std::int64_t stridex, sycl::buffer<double, 1>& y, std::int64_t incy,
                std::int64_t stridey, std::int64_t batch_size);

void copy_batch(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
                std::int64_t incx, std::int64_t stridex, sycl::buffer<std::complex<float>, 1>& y,
                std::int64_t incy, std::int64_t stridey, std::int64_t batch_size);

void copy_batch(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
                std::int64_t incx, std::int64_t stridex, sycl::buffer<std::complex<double>, 1>& y,
                std::int64_t incy, std::int64_t stridey, std::int64_t batch_size);

void gemv_batch(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n, float alpha,
                sycl::buffer<float, 1>& a, std::int64_t lda, std::int64_t stridea,
                sycl::buffer<float, 1>& x, std::int64_t incx, std::int64_t stridex, float beta,
                sycl::buffer<float, 1>& y, std::int64_t incy, std::int64_t stridey,
                std::int64_t batch_size);

void gemv_batch(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n, double alpha,
                sycl::buffer<double, 1>& a, std::int64_t lda, std::int64_t stridea,
                sycl::buffer<double, 1>& x, std::int64_t incx, std::int64_t stridex, double beta,
                sycl::buffer<double, 1>& y, std::int64_t incy, std::int64_t stridey,
                std::int64_t batch_size);

void gemv_batch(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
                std::int64_t lda, std::int64_t stridea, sycl::buffer<std::complex<float>, 1>& x,
                std::int64_t incx, std::int64_t stridex, std::complex<float> beta,
                sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy, std::int64_t stridey,
                std::int64_t batch_size);

void gemv_batch(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                std::int64_t lda, std::int64_t stridea, sycl::buffer<std::complex<double>, 1>& x,
                std::int64_t incx, std::int64_t stridex, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy, std::int64_t stridey,
                std::int64_t batch_size);

void dgmm_batch(sycl::queue& queue, side left_right, std::int64_t m, std::int64_t n,
                sycl::buffer<float, 1>& a, std::int64_t lda, std::int64_t stridea,
                sycl::buffer<float, 1>& x, std::int64_t incx, std::int64_t stridex,
                sycl::buffer<float, 1>& c, std::int64_t ldc, std::int64_t stridec,
                std::int64_t batch_size);

void dgmm_batch(sycl::queue& queue, side left_right, std::int64_t m, std::int64_t n,
                sycl::buffer<double, 1>& a, std::int64_t lda, std::int64_t stridea,
                sycl::buffer<double, 1>& x, std::int64_t incx, std::int64_t stridex,
                sycl::buffer<double, 1>& c, std::int64_t ldc, std::int64_t stridec,
                std::int64_t batch_size);

void dgmm_batch(sycl::queue& queue, side left_right, std::int64_t m, std::int64_t n,
                sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda, std::int64_t stridea,
                sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx, std::int64_t stridex,
                sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc, std::int64_t stridec,
                std::int64_t batch_size);

void dgmm_batch(sycl::queue& queue, side left_right, std::int64_t m, std::int64_t n,
                sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda, std::int64_t stridea,
                sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx, std::int64_t stridex,
                sycl::buffer<std::complex<double>, 1>& c, std::int64_t ldc, std::int64_t stridec,
                std::int64_t batch_size);

void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                sycl::buffer<float, 1>& a, std::int64_t lda, std::int64_t stride_a,
                sycl::buffer<float, 1>& b, std::int64_t ldb, std::int64_t stride_b,
                std::int64_t batch_size);

void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
                sycl::buffer<double, 1>& a, std::int64_t lda, std::int64_t stride_a,
                sycl::buffer<double, 1>& b, std::int64_t ldb, std::int64_t stride_b,
                std::int64_t batch_size);

void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda, std::int64_t stride_a,
                sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb, std::int64_t stride_b,
                std::int64_t batch_size);

void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda, std::int64_t stride_a,
                sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb, std::int64_t stride_b,
                std::int64_t batch_size);

void axpy_batch(sycl::queue& queue, std::int64_t n, float alpha, sycl::buffer<float, 1>& x,
                std::int64_t incx, std::int64_t stridex, sycl::buffer<float, 1>& y,
                std::int64_t incy, std::int64_t stridey, std::int64_t batch_size);

void axpy_batch(sycl::queue& queue, std::int64_t n, double alpha, sycl::buffer<double, 1>& x,
                std::int64_t incx, std::int64_t stridex, sycl::buffer<double, 1>& y,
                std::int64_t incy, std::int64_t stridey, std::int64_t batch_size);

void axpy_batch(sycl::queue& queue, std::int64_t n, std::complex<float> alpha,
                sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx, std::int64_t stridex,
                sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy, std::int64_t stridey,
                std::int64_t batch_size);

void axpy_batch(sycl::queue& queue, std::int64_t n, std::complex<double> alpha,
                sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx, std::int64_t stridex,
                sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy, std::int64_t stridey,
                std::int64_t batch_size);

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                std::int64_t n, std::int64_t k, float alpha, sycl::buffer<float, 1>& a,
                std::int64_t lda, std::int64_t stride_a, sycl::buffer<float, 1>& b,
                std::int64_t ldb, std::int64_t stride_b, float beta, sycl::buffer<float, 1>& c,
                std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size);

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                std::int64_t n, std::int64_t k, double alpha, sycl::buffer<double, 1>& a,
                std::int64_t lda, std::int64_t stride_a, sycl::buffer<double, 1>& b,
                std::int64_t ldb, std::int64_t stride_b, double beta, sycl::buffer<double, 1>& c,
                std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size);

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                std::int64_t n, std::int64_t k, std::complex<float> alpha,
                sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda, std::int64_t stride_a,
                sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb, std::int64_t stride_b,
                std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size);

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                std::int64_t n, std::int64_t k, std::complex<double> alpha,
                sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda, std::int64_t stride_a,
                sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb, std::int64_t stride_b,
                std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& c,
                std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size);

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                std::int64_t n, std::int64_t k, sycl::half alpha, sycl::buffer<sycl::half, 1>& a,
                std::int64_t lda, std::int64_t stride_a, sycl::buffer<sycl::half, 1>& b,
                std::int64_t ldb, std::int64_t stride_b, sycl::half beta,
                sycl::buffer<sycl::half, 1>& c, std::int64_t ldc, std::int64_t stride_c,
                std::int64_t batch_size);

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                std::int64_t n, std::int64_t k, float alpha, sycl::buffer<sycl::half, 1>& a,
                std::int64_t lda, std::int64_t stride_a, sycl::buffer<sycl::half, 1>& b,
                std::int64_t ldb, std::int64_t stride_b, float beta, sycl::buffer<float, 1>& c,
                std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size);

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                std::int64_t n, std::int64_t k, float alpha, sycl::buffer<bfloat16, 1>& a,
                std::int64_t lda, std::int64_t stride_a, sycl::buffer<bfloat16, 1>& b,
                std::int64_t ldb, std::int64_t stride_b, float beta, sycl::buffer<bfloat16, 1>& c,
                std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size);

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                std::int64_t n, std::int64_t k, float alpha, sycl::buffer<bfloat16, 1>& a,
                std::int64_t lda, std::int64_t stride_a, sycl::buffer<bfloat16, 1>& b,
                std::int64_t ldb, std::int64_t stride_b, float beta, sycl::buffer<float, 1>& c,
                std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size);

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                std::int64_t n, std::int64_t k, float alpha, sycl::buffer<std::int8_t, 1>& a,
                std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::int8_t, 1>& b,
                std::int64_t ldb, std::int64_t stride_b, float beta,
                sycl::buffer<std::int32_t, 1>& c, std::int64_t ldc, std::int64_t stride_c,
                std::int64_t batch_size);

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                std::int64_t n, std::int64_t k, float alpha, sycl::buffer<std::int8_t, 1>& a,
                std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::int8_t, 1>& b,
                std::int64_t ldb, std::int64_t stride_b, float beta, sycl::buffer<float, 1>& c,
                std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size);

void omatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                    float alpha, sycl::buffer<float, 1>& a, std::int64_t lda, std::int64_t stride_a,
                    sycl::buffer<float, 1>& b, std::int64_t ldb, std::int64_t stride_b,
                    std::int64_t batch_size);

void omatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                    double alpha, sycl::buffer<double, 1>& a, std::int64_t lda,
                    std::int64_t stride_a, sycl::buffer<double, 1>& b, std::int64_t ldb,
                    std::int64_t stride_b, std::int64_t batch_size);

void omatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                    std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
                    std::int64_t lda, std::int64_t stride_a,
                    sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb,
                    std::int64_t stride_b, std::int64_t batch_size);

void omatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                    std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                    std::int64_t lda, std::int64_t stride_a,
                    sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb,
                    std::int64_t stride_b, std::int64_t batch_size);

void imatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                    float alpha, sycl::buffer<float, 1>& ab, std::int64_t lda, std::int64_t ldb,
                    std::int64_t stride, std::int64_t batch_size);

void imatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                    double alpha, sycl::buffer<double, 1>& ab, std::int64_t lda, std::int64_t ldb,
                    std::int64_t stride, std::int64_t batch_size);

void imatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                    std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& ab,
                    std::int64_t lda, std::int64_t ldb, std::int64_t stride,
                    std::int64_t batch_size);

void imatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                    std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& ab,
                    std::int64_t lda, std::int64_t ldb, std::int64_t stride,
                    std::int64_t batch_size);

void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                   std::int64_t n, float alpha, sycl::buffer<float, 1>& a, std::int64_t lda,
                   std::int64_t stride_a, float beta, sycl::buffer<float, 1>& b, std::int64_t ldb,
                   std::int64_t stride_b, sycl::buffer<float, 1>& c, std::int64_t ldc,
                   std::int64_t stride_c, std::int64_t batch_size);

void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                   std::int64_t n, double alpha, sycl::buffer<double, 1>& a, std::int64_t lda,
                   std::int64_t stride_a, double beta, sycl::buffer<double, 1>& b, std::int64_t ldb,
                   std::int64_t stride_b, sycl::buffer<double, 1>& c, std::int64_t ldc,
                   std::int64_t stride_c, std::int64_t batch_size);

void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                   std::int64_t n, std::complex<float> alpha,
                   sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda, std::int64_t stride_a,
                   std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& b,
                   std::int64_t ldb, std::int64_t stride_b, sycl::buffer<std::complex<float>, 1>& c,
                   std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size);

void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                   std::int64_t n, std::complex<double> alpha,
                   sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                   std::int64_t stride_a, std::complex<double> beta,
                   sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb,
                   std::int64_t stride_b, sycl::buffer<std::complex<double>, 1>& c,
                   std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size);

void omatcopy(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n, float alpha,
              sycl::buffer<float, 1>& a, std::int64_t lda, sycl::buffer<float, 1>& b,
              std::int64_t ldb);

void omatcopy(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n, double alpha,
              sycl::buffer<double, 1>& a, std::int64_t lda, sycl::buffer<double, 1>& b,
              std::int64_t ldb);

void omatcopy(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
              std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
              sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb);

void omatcopy(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
              std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
              std::int64_t lda, sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb);

void omatcopy2(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n, float alpha,
               sycl::buffer<float, 1>& a, std::int64_t lda, std::int64_t stridea,
               sycl::buffer<float, 1>& b, std::int64_t ldb, std::int64_t strideb);

void omatcopy2(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n, double alpha,
               sycl::buffer<double, 1>& a, std::int64_t lda, std::int64_t stridea,
               sycl::buffer<double, 1>& b, std::int64_t ldb, std::int64_t strideb);

void omatcopy2(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
               std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
               std::int64_t stridea, sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb,
               std::int64_t strideb);

void omatcopy2(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
               std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
               std::int64_t lda, std::int64_t stridea, sycl::buffer<std::complex<double>, 1>& b,
               std::int64_t ldb, std::int64_t strideb);

void imatcopy(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n, float alpha,
              sycl::buffer<float, 1>& ab, std::int64_t lda, std::int64_t ldb);

void imatcopy(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n, double alpha,
              sycl::buffer<double, 1>& ab, std::int64_t lda, std::int64_t ldb);

void imatcopy(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
              std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& ab, std::int64_t lda,
              std::int64_t ldb);

void imatcopy(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
              std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& ab,
              std::int64_t lda, std::int64_t ldb);

void omatadd(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
             float alpha, sycl::buffer<float, 1>& a, std::int64_t lda, float beta,
             sycl::buffer<float, 1>& b, std::int64_t ldb, sycl::buffer<float, 1>& c,
             std::int64_t ldc);

void omatadd(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
             double alpha, sycl::buffer<double, 1>& a, std::int64_t lda, double beta,
             sycl::buffer<double, 1>& b, std::int64_t ldb, sycl::buffer<double, 1>& c,
             std::int64_t ldc);

void omatadd(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
             std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
             std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb,
             sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc);

void omatadd(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
             std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
             std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb,
             sycl::buffer<std::complex<double>, 1>& c, std::int64_t ldc);

// batch, usm

sycl::event syrk_batch(sycl::queue& queue, const uplo* upper_lower, const transpose* trans,
                       const std::int64_t* n, const std::int64_t* k, const float* alpha,
                       const float** a, const std::int64_t* lda, const float* beta, float** c,
                       const std::int64_t* ldc, std::int64_t group_count,
                       const std::int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies = {});
sycl::event syrk_batch(sycl::queue& queue, const uplo* upper_lower, const transpose* trans,
                       const std::int64_t* n, const std::int64_t* k, const double* alpha,
                       const double** a, const std::int64_t* lda, const double* beta, double** c,
                       const std::int64_t* ldc, std::int64_t group_count,
                       const std::int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies = {});
sycl::event syrk_batch(sycl::queue& queue, const uplo* upper_lower, const transpose* trans,
                       const std::int64_t* n, const std::int64_t* k,
                       const std::complex<float>* alpha, const std::complex<float>** a,
                       const std::int64_t* lda, const std::complex<float>* beta,
                       std::complex<float>** c, const std::int64_t* ldc, std::int64_t group_count,
                       const std::int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies = {});
sycl::event syrk_batch(sycl::queue& queue, const uplo* upper_lower, const transpose* trans,
                       const std::int64_t* n, const std::int64_t* k,
                       const std::complex<double>* alpha, const std::complex<double>** a,
                       const std::int64_t* lda, const std::complex<double>* beta,
                       std::complex<double>** c, const std::int64_t* ldc, std::int64_t group_count,
                       const std::int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                       std::int64_t k, value_or_pointer<float> alpha, const float* a,
                       std::int64_t lda, std::int64_t stride_a, value_or_pointer<float> beta,
                       float* c, std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});
sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                       std::int64_t k, value_or_pointer<double> alpha, const double* a,
                       std::int64_t lda, std::int64_t stride_a, value_or_pointer<double> beta,
                       double* c, std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});
sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                       std::int64_t k, value_or_pointer<std::complex<float>> alpha,
                       const std::complex<float>* a, std::int64_t lda, std::int64_t stride_a,
                       value_or_pointer<std::complex<float>> beta, std::complex<float>* c,
                       std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});
sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                       std::int64_t k, value_or_pointer<std::complex<double>> alpha,
                       const std::complex<double>* a, std::int64_t lda, std::int64_t stride_a,
                       value_or_pointer<std::complex<double>> beta, std::complex<double>* c,
                       std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event copy_batch(sycl::queue& queue, std::int64_t n, const float* x, std::int64_t incx,
                       std::int64_t stridex, float* y, std::int64_t incy, std::int64_t stridey,
                       std::int64_t batch_size, const std::vector<sycl::event>& dependencies = {});

sycl::event copy_batch(sycl::queue& queue, std::int64_t n, const double* x, std::int64_t incx,
                       std::int64_t stridex, double* y, std::int64_t incy, std::int64_t stridey,
                       std::int64_t batch_size, const std::vector<sycl::event>& dependencies = {});

sycl::event copy_batch(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                       std::int64_t incx, std::int64_t stridex, std::complex<float>* y,
                       std::int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event copy_batch(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                       std::int64_t incx, std::int64_t stridex, std::complex<double>* y,
                       std::int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event copy_batch(sycl::queue& queue, const std::int64_t* n, const float** x,
                       const std::int64_t* incx, float** y, const std::int64_t* incy,
                       std::int64_t group_count, const std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event copy_batch(sycl::queue& queue, const std::int64_t* n, const double** x,
                       const std::int64_t* incx, double** y, const std::int64_t* incy,
                       std::int64_t group_count, const std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event copy_batch(sycl::queue& queue, const std::int64_t* n, const std::complex<float>** x,
                       const std::int64_t* incx, std::complex<float>** y, const std::int64_t* incy,
                       std::int64_t group_count, const std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event copy_batch(sycl::queue& queue, const std::int64_t* n, const std::complex<double>** x,
                       const std::int64_t* incx, std::complex<double>** y, const std::int64_t* incy,
                       std::int64_t group_count, const std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event dgmm_batch(sycl::queue& queue, side left_right, std::int64_t m, std::int64_t n,
                       const float* a, std::int64_t lda, std::int64_t stridea, const float* x,
                       std::int64_t incx, std::int64_t stridex, float* c, std::int64_t ldc,
                       std::int64_t stridec, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event dgmm_batch(sycl::queue& queue, side left_right, std::int64_t m, std::int64_t n,
                       const double* a, std::int64_t lda, std::int64_t stridea, const double* x,
                       std::int64_t incx, std::int64_t stridex, double* c, std::int64_t ldc,
                       std::int64_t stridec, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event dgmm_batch(sycl::queue& queue, side left_right, std::int64_t m, std::int64_t n,
                       const std::complex<float>* a, std::int64_t lda, std::int64_t stridea,
                       const std::complex<float>* x, std::int64_t incx, std::int64_t stridex,
                       std::complex<float>* c, std::int64_t ldc, std::int64_t stridec,
                       std::int64_t batch_size, const std::vector<sycl::event>& dependencies = {});

sycl::event dgmm_batch(sycl::queue& queue, side left_right, std::int64_t m, std::int64_t n,
                       const std::complex<double>* a, std::int64_t lda, std::int64_t stridea,
                       const std::complex<double>* x, std::int64_t incx, std::int64_t stridex,
                       std::complex<double>* c, std::int64_t ldc, std::int64_t stridec,
                       std::int64_t batch_size, const std::vector<sycl::event>& dependencies = {});

sycl::event dgmm_batch(sycl::queue& queue, const side* left_right, const std::int64_t* m,
                       const std::int64_t* n, const float** a, const std::int64_t* lda,
                       const float** x, const std::int64_t* incx, float** c,
                       const std::int64_t* ldc, std::int64_t group_count,
                       const std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event dgmm_batch(sycl::queue& queue, const side* left_right, const std::int64_t* m,
                       const std::int64_t* n, const double** a, const std::int64_t* lda,
                       const double** x, const std::int64_t* incx, double** c,
                       const std::int64_t* ldc, std::int64_t group_count,
                       const std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event dgmm_batch(sycl::queue& queue, const side* left_right, const std::int64_t* m,
                       const std::int64_t* n, const std::complex<float>** a,
                       const std::int64_t* lda, const std::complex<float>** x,
                       const std::int64_t* incx, std::complex<float>** c, const std::int64_t* ldc,
                       std::int64_t group_count, const std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event dgmm_batch(sycl::queue& queue, const side* left_right, const std::int64_t* m,
                       const std::int64_t* n, const std::complex<double>** a,
                       const std::int64_t* lda, const std::complex<double>** x,
                       const std::int64_t* incx, std::complex<double>** c, const std::int64_t* ldc,
                       std::int64_t group_count, const std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemv_batch(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                       value_or_pointer<float> alpha, const float* a, std::int64_t lda,
                       std::int64_t stridea, const float* x, std::int64_t incx,
                       std::int64_t stridex, value_or_pointer<float> beta, float* y,
                       std::int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemv_batch(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                       value_or_pointer<double> alpha, const double* a, std::int64_t lda,
                       std::int64_t stridea, const double* x, std::int64_t incx,
                       std::int64_t stridex, value_or_pointer<double> beta, double* y,
                       std::int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemv_batch(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                       value_or_pointer<std::complex<float>> alpha, const std::complex<float>* a,
                       std::int64_t lda, std::int64_t stridea, const std::complex<float>* x,
                       std::int64_t incx, std::int64_t stridex,
                       value_or_pointer<std::complex<float>> beta, std::complex<float>* y,
                       std::int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemv_batch(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                       value_or_pointer<std::complex<double>> alpha, const std::complex<double>* a,
                       std::int64_t lda, std::int64_t stridea, const std::complex<double>* x,
                       std::int64_t incx, std::int64_t stridex,
                       value_or_pointer<std::complex<double>> beta, std::complex<double>* y,
                       std::int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemv_batch(sycl::queue& queue, const transpose* trans, const std::int64_t* m,
                       const std::int64_t* n, const float* alpha, const float** a,
                       const std::int64_t* lda, const float** x, const std::int64_t* incx,
                       const float* beta, float** y, const std::int64_t* incy,
                       std::int64_t group_count, const std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemv_batch(sycl::queue& queue, const transpose* trans, const std::int64_t* m,
                       const std::int64_t* n, const double* alpha, const double** a,
                       const std::int64_t* lda, const double** x, const std::int64_t* incx,
                       const double* beta, double** y, const std::int64_t* incy,
                       std::int64_t group_count, const std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemv_batch(sycl::queue& queue, const transpose* trans, const std::int64_t* m,
                       const std::int64_t* n, const std::complex<float>* alpha,
                       const std::complex<float>** a, const std::int64_t* lda,
                       const std::complex<float>** x, const std::int64_t* incx,
                       const std::complex<float>* beta, std::complex<float>** y,
                       const std::int64_t* incy, std::int64_t group_count,
                       const std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemv_batch(sycl::queue& queue, const transpose* trans, const std::int64_t* m,
                       const std::int64_t* n, const std::complex<double>* alpha,
                       const std::complex<double>** a, const std::int64_t* lda,
                       const std::complex<double>** x, const std::int64_t* incx,
                       const std::complex<double>* beta, std::complex<double>** y,
                       const std::int64_t* incy, std::int64_t group_count,
                       const std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event axpy_batch(sycl::queue& queue, const std::int64_t* n, const double* alpha,
                       const double** x, const std::int64_t* incx, double** y,
                       const std::int64_t* incy, std::int64_t group_count,
                       const std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event axpy_batch(sycl::queue& queue, const std::int64_t* n, const float* alpha,
                       const float** x, const std::int64_t* incx, float** y,
                       const std::int64_t* incy, std::int64_t group_count,
                       const std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event axpy_batch(sycl::queue& queue, const std::int64_t* n, const std::complex<double>* alpha,
                       const std::complex<double>** x, const std::int64_t* incx,
                       std::complex<double>** y, const std::int64_t* incy, std::int64_t group_count,
                       const std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event axpy_batch(sycl::queue& queue, const std::int64_t* n, const std::complex<float>* alpha,
                       const std::complex<float>** x, const std::int64_t* incx,
                       std::complex<float>** y, const std::int64_t* incy, std::int64_t group_count,
                       const std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event axpy_batch(sycl::queue& queue, std::int64_t n, value_or_pointer<float> alpha,
                       const float* x, std::int64_t incx, std::int64_t stridex, float* y,
                       std::int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event axpy_batch(sycl::queue& queue, std::int64_t n, value_or_pointer<double> alpha,
                       const double* x, std::int64_t incx, std::int64_t stridex, double* y,
                       std::int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event axpy_batch(sycl::queue& queue, std::int64_t n,
                       value_or_pointer<std::complex<float>> alpha, const std::complex<float>* x,
                       std::int64_t incx, std::int64_t stridex, std::complex<float>* y,
                       std::int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event axpy_batch(sycl::queue& queue, std::int64_t n,
                       value_or_pointer<std::complex<double>> alpha, const std::complex<double>* x,
                       std::int64_t incx, std::int64_t stridex, std::complex<double>* y,
                       std::int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, const transpose* transa, const transpose* transb,
                       const std::int64_t* m, const std::int64_t* n, const std::int64_t* k,
                       const float* alpha, const float** a, const std::int64_t* lda,
                       const float** b, const std::int64_t* ldb, const float* beta, float** c,
                       const std::int64_t* ldc, std::int64_t group_count,
                       const std::int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, const transpose* transa, const transpose* transb,
                       const std::int64_t* m, const std::int64_t* n, const std::int64_t* k,
                       const double* alpha, const double** a, const std::int64_t* lda,
                       const double** b, const std::int64_t* ldb, const double* beta, double** c,
                       const std::int64_t* ldc, std::int64_t group_count,
                       const std::int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, const transpose* transa, const transpose* transb,
                       const std::int64_t* m, const std::int64_t* n, const std::int64_t* k,
                       const std::complex<float>* alpha, const std::complex<float>** a,
                       const std::int64_t* lda, const std::complex<float>** b,
                       const std::int64_t* ldb, const std::complex<float>* beta,
                       std::complex<float>** c, const std::int64_t* ldc, std::int64_t group_count,
                       const std::int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, const transpose* transa, const transpose* transb,
                       const std::int64_t* m, const std::int64_t* n, const std::int64_t* k,
                       const std::complex<double>* alpha, const std::complex<double>** a,
                       const std::int64_t* lda, const std::complex<double>** b,
                       const std::int64_t* ldb, const std::complex<double>* beta,
                       std::complex<double>** c, const std::int64_t* ldc, std::int64_t group_count,
                       const std::int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, const transpose* transa, const transpose* transb,
                       const std::int64_t* m, const std::int64_t* n, const std::int64_t* k,
                       const sycl::half* alpha, const sycl::half** a, const std::int64_t* lda,
                       const sycl::half** b, const std::int64_t* ldb, const sycl::half* beta,
                       sycl::half** c, const std::int64_t* ldc, std::int64_t group_count,
                       const std::int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, const transpose* transa, const transpose* transb,
                       const std::int64_t* m, const std::int64_t* n, const std::int64_t* k,
                       const float* alpha, const sycl::half** a, const std::int64_t* lda,
                       const sycl::half** b, const std::int64_t* ldb, const float* beta, float** c,
                       const std::int64_t* ldc, std::int64_t group_count,
                       const std::int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, const transpose* transa, const transpose* transb,
                       const std::int64_t* m, const std::int64_t* n, const std::int64_t* k,
                       const float* alpha, const bfloat16** a, const std::int64_t* lda,
                       const bfloat16** b, const std::int64_t* ldb, const float* beta, bfloat16** c,
                       const std::int64_t* ldc, std::int64_t group_count,
                       const std::int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, const transpose* transa, const transpose* transb,
                       const std::int64_t* m, const std::int64_t* n, const std::int64_t* k,
                       const float* alpha, const bfloat16** a, const std::int64_t* lda,
                       const bfloat16** b, const std::int64_t* ldb, const float* beta, float** c,
                       const std::int64_t* ldc, std::int64_t group_count,
                       const std::int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, const transpose* transa, const transpose* transb,
                       const std::int64_t* m, const std::int64_t* n, const std::int64_t* k,
                       const float* alpha, const std::int8_t** a, const std::int64_t* lda,
                       const std::int8_t** b, const std::int64_t* ldb, const float* beta,
                       std::int32_t** c, const std::int64_t* ldc, std::int64_t group_count,
                       const std::int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, const transpose* transa, const transpose* transb,
                       const std::int64_t* m, const std::int64_t* n, const std::int64_t* k,
                       const float* alpha, const std::int8_t** a, const std::int64_t* lda,
                       const std::int8_t** b, const std::int64_t* ldb, const float* beta, float** c,
                       const std::int64_t* ldc, std::int64_t group_count,
                       const std::int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                       std::int64_t n, std::int64_t k, value_or_pointer<float> alpha,
                       const float* a, std::int64_t lda, std::int64_t stride_a, const float* b,
                       std::int64_t ldb, std::int64_t stride_b, value_or_pointer<float> beta,
                       float* c, std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                       std::int64_t n, std::int64_t k, value_or_pointer<double> alpha,
                       const double* a, std::int64_t lda, std::int64_t stride_a, const double* b,
                       std::int64_t ldb, std::int64_t stride_b, value_or_pointer<double> beta,
                       double* c, std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                       std::int64_t n, std::int64_t k, value_or_pointer<std::complex<float>> alpha,
                       const std::complex<float>* a, std::int64_t lda, std::int64_t stride_a,
                       const std::complex<float>* b, std::int64_t ldb, std::int64_t stride_b,
                       value_or_pointer<std::complex<float>> beta, std::complex<float>* c,
                       std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                       std::int64_t n, std::int64_t k, value_or_pointer<std::complex<double>> alpha,
                       const std::complex<double>* a, std::int64_t lda, std::int64_t stride_a,
                       const std::complex<double>* b, std::int64_t ldb, std::int64_t stride_b,
                       value_or_pointer<std::complex<double>> beta, std::complex<double>* c,
                       std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                       std::int64_t n, std::int64_t k, value_or_pointer<sycl::half> alpha,
                       const sycl::half* a, std::int64_t lda, std::int64_t stride_a,
                       const sycl::half* b, std::int64_t ldb, std::int64_t stride_b,
                       value_or_pointer<sycl::half> beta, sycl::half* c, std::int64_t ldc,
                       std::int64_t stride_c, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                       std::int64_t n, std::int64_t k, value_or_pointer<float> alpha,
                       const sycl::half* a, std::int64_t lda, std::int64_t stride_a,
                       const sycl::half* b, std::int64_t ldb, std::int64_t stride_b,
                       value_or_pointer<float> beta, float* c, std::int64_t ldc,
                       std::int64_t stride_c, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                       std::int64_t n, std::int64_t k, value_or_pointer<float> alpha,
                       const bfloat16* a, std::int64_t lda, std::int64_t stride_a,
                       const bfloat16* b, std::int64_t ldb, std::int64_t stride_b,
                       value_or_pointer<float> beta, bfloat16* c, std::int64_t ldc,
                       std::int64_t stride_c, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                       std::int64_t n, std::int64_t k, value_or_pointer<float> alpha,
                       const bfloat16* a, std::int64_t lda, std::int64_t stride_a,
                       const bfloat16* b, std::int64_t ldb, std::int64_t stride_b,
                       value_or_pointer<float> beta, float* c, std::int64_t ldc,
                       std::int64_t stride_c, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                       std::int64_t n, std::int64_t k, value_or_pointer<float> alpha,
                       const std::int8_t* a, std::int64_t lda, std::int64_t stride_a,
                       const std::int8_t* b, std::int64_t ldb, std::int64_t stride_b,
                       value_or_pointer<float> beta, std::int32_t* c, std::int64_t ldc,
                       std::int64_t stride_c, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                       std::int64_t n, std::int64_t k, value_or_pointer<float> alpha,
                       const std::int8_t* a, std::int64_t lda, std::int64_t stride_a,
                       const std::int8_t* b, std::int64_t ldb, std::int64_t stride_b,
                       value_or_pointer<float> beta, float* c, std::int64_t ldc,
                       std::int64_t stride_c, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, std::int64_t m, std::int64_t n,
                       value_or_pointer<float> alpha, const float* a, std::int64_t lda,
                       std::int64_t stride_a, float* b, std::int64_t ldb, std::int64_t stride_b,
                       std::int64_t batch_size, const std::vector<sycl::event>& dependencies = {});

sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, std::int64_t m, std::int64_t n,
                       value_or_pointer<double> alpha, const double* a, std::int64_t lda,
                       std::int64_t stride_a, double* b, std::int64_t ldb, std::int64_t stride_b,
                       std::int64_t batch_size, const std::vector<sycl::event>& dependencies = {});

sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, std::int64_t m, std::int64_t n,
                       value_or_pointer<std::complex<float>> alpha, const std::complex<float>* a,
                       std::int64_t lda, std::int64_t stride_a, std::complex<float>* b,
                       std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, std::int64_t m, std::int64_t n,
                       value_or_pointer<std::complex<double>> alpha, const std::complex<double>* a,
                       std::int64_t lda, std::int64_t stride_a, std::complex<double>* b,
                       std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event trsm_batch(sycl::queue& queue, const side* left_right, const uplo* upper_lower,
                       const transpose* trans, const diag* unit_diag, const std::int64_t* m,
                       const std::int64_t* n, const float* alpha, const float** a,
                       const std::int64_t* lda, float** b, const std::int64_t* ldb,
                       std::int64_t group_count, const std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event trsm_batch(sycl::queue& queue, const side* left_right, const uplo* upper_lower,
                       const transpose* trans, const diag* unit_diag, const std::int64_t* m,
                       const std::int64_t* n, const double* alpha, const double** a,
                       const std::int64_t* lda, double** b, const std::int64_t* ldb,
                       std::int64_t group_count, const std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event trsm_batch(sycl::queue& queue, const side* left_right, const uplo* upper_lower,
                       const transpose* trans, const diag* unit_diag, const std::int64_t* m,
                       const std::int64_t* n, const std::complex<float>* alpha,
                       const std::complex<float>** a, const std::int64_t* lda,
                       std::complex<float>** b, const std::int64_t* ldb, std::int64_t group_count,
                       const std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event trsm_batch(sycl::queue& queue, const side* left_right, const uplo* upper_lower,
                       const transpose* trans, const diag* unit_diag, const std::int64_t* m,
                       const std::int64_t* n, const std::complex<double>* alpha,
                       const std::complex<double>** a, const std::int64_t* lda,
                       std::complex<double>** b, const std::int64_t* ldb, std::int64_t group_count,
                       const std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                           value_or_pointer<float> alpha, const float* a, std::int64_t lda,
                           std::int64_t stride_a, float* b, std::int64_t ldb, std::int64_t stride_b,
                           std::int64_t batch_size,
                           const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                           value_or_pointer<double> alpha, const double* a, std::int64_t lda,
                           std::int64_t stride_a, double* b, std::int64_t ldb,
                           std::int64_t stride_b, std::int64_t batch_size,
                           const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                           value_or_pointer<std::complex<float>> alpha,
                           const std::complex<float>* a, std::int64_t lda, std::int64_t stride_a,
                           std::complex<float>* b, std::int64_t ldb, std::int64_t stride_b,
                           std::int64_t batch_size,
                           const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                           value_or_pointer<std::complex<double>> alpha,
                           const std::complex<double>* a, std::int64_t lda, std::int64_t stride_a,
                           std::complex<double>* b, std::int64_t ldb, std::int64_t stride_b,
                           std::int64_t batch_size,
                           const std::vector<sycl::event>& dependencies = {});

sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                           value_or_pointer<float> alpha, float* ab, std::int64_t lda,
                           std::int64_t ldb, std::int64_t stride, std::int64_t batch_size,
                           const std::vector<sycl::event>& dependencies = {});

sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                           value_or_pointer<double> alpha, double* ab, std::int64_t lda,
                           std::int64_t ldb, std::int64_t stride, std::int64_t batch_size,
                           const std::vector<sycl::event>& dependencies = {});

sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                           value_or_pointer<std::complex<float>> alpha, std::complex<float>* ab,
                           std::int64_t lda, std::int64_t ldb, std::int64_t stride,
                           std::int64_t batch_size,
                           const std::vector<sycl::event>& dependencies = {});

sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                           value_or_pointer<std::complex<double>> alpha, std::complex<double>* ab,
                           std::int64_t lda, std::int64_t ldb, std::int64_t stride,
                           std::int64_t batch_size,
                           const std::vector<sycl::event>& dependencies = {});

sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                          std::int64_t n, value_or_pointer<float> alpha, const float* a,
                          std::int64_t lda, std::int64_t stride_a, value_or_pointer<float> beta,
                          const float* b, std::int64_t ldb, std::int64_t stride_b, float* c,
                          std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
                          const std::vector<sycl::event>& dependencies = {});

sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                          std::int64_t n, value_or_pointer<double> alpha, const double* a,
                          std::int64_t lda, std::int64_t stride_a, value_or_pointer<double> beta,
                          const double* b, std::int64_t ldb, std::int64_t stride_b, double* c,
                          std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
                          const std::vector<sycl::event>& dependencies = {});

sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                          std::int64_t n, value_or_pointer<std::complex<float>> alpha,
                          const std::complex<float>* a, std::int64_t lda, std::int64_t stride_a,
                          value_or_pointer<std::complex<float>> beta, const std::complex<float>* b,
                          std::int64_t ldb, std::int64_t stride_b, std::complex<float>* c,
                          std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
                          const std::vector<sycl::event>& dependencies = {});

sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                          std::int64_t n, value_or_pointer<std::complex<double>> alpha,
                          const std::complex<double>* a, std::int64_t lda, std::int64_t stride_a,
                          value_or_pointer<std::complex<double>> beta,
                          const std::complex<double>* b, std::int64_t ldb, std::int64_t stride_b,
                          std::complex<double>* c, std::int64_t ldc, std::int64_t stride_c,
                          std::int64_t batch_size,
                          const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                     value_or_pointer<float> alpha, const float* a, std::int64_t lda, float* b,
                     std::int64_t ldb, const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                     value_or_pointer<double> alpha, const double* a, std::int64_t lda, double* b,
                     std::int64_t ldb, const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                     value_or_pointer<std::complex<float>> alpha, const std::complex<float>* a,
                     std::int64_t lda, std::complex<float>* b, std::int64_t ldb,
                     const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                     value_or_pointer<std::complex<double>> alpha, const std::complex<double>* a,
                     std::int64_t lda, std::complex<double>* b, std::int64_t ldb,
                     const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy2(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                      value_or_pointer<float> alpha, const float* a, std::int64_t lda,
                      std::int64_t stridea, float* b, std::int64_t ldb, std::int64_t strideb,
                      const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy2(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                      value_or_pointer<double> alpha, const double* a, std::int64_t lda,
                      std::int64_t stridea, double* b, std::int64_t ldb, std::int64_t strideb,
                      const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy2(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                      value_or_pointer<std::complex<float>> alpha, const std::complex<float>* a,
                      std::int64_t lda, std::int64_t stridea, std::complex<float>* b,
                      std::int64_t ldb, std::int64_t strideb,
                      const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy2(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                      value_or_pointer<std::complex<double>> alpha, const std::complex<double>* a,
                      std::int64_t lda, std::int64_t stridea, std::complex<double>* b,
                      std::int64_t ldb, std::int64_t strideb,
                      const std::vector<sycl::event>& dependencies = {});

sycl::event imatcopy(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                     value_or_pointer<float> alpha, float* ab, std::int64_t lda, std::int64_t ldb,
                     const std::vector<sycl::event>& dependencies = {});

sycl::event imatcopy(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                     value_or_pointer<double> alpha, double* ab, std::int64_t lda, std::int64_t ldb,
                     const std::vector<sycl::event>& dependencies = {});

sycl::event imatcopy(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                     value_or_pointer<std::complex<float>> alpha, std::complex<float>* ab,
                     std::int64_t lda, std::int64_t ldb,
                     const std::vector<sycl::event>& dependencies = {});

sycl::event imatcopy(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                     value_or_pointer<std::complex<double>> alpha, std::complex<double>* ab,
                     std::int64_t lda, std::int64_t ldb,
                     const std::vector<sycl::event>& dependencies = {});

sycl::event omatadd(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                    std::int64_t n, value_or_pointer<float> alpha, const float* a, std::int64_t lda,
                    value_or_pointer<float> beta, const float* b, std::int64_t ldb, float* c,
                    std::int64_t ldc, const std::vector<sycl::event>& dependencies = {});

sycl::event omatadd(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                    std::int64_t n, value_or_pointer<double> alpha, const double* a,
                    std::int64_t lda, value_or_pointer<double> beta, const double* b,
                    std::int64_t ldb, double* c, std::int64_t ldc,
                    const std::vector<sycl::event>& dependencies = {});

sycl::event omatadd(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                    std::int64_t n, value_or_pointer<std::complex<float>> alpha,
                    const std::complex<float>* a, std::int64_t lda,
                    value_or_pointer<std::complex<float>> beta, const std::complex<float>* b,
                    std::int64_t ldb, std::complex<float>* c, std::int64_t ldc,
                    const std::vector<sycl::event>& dependencies = {});

sycl::event omatadd(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                    std::int64_t n, value_or_pointer<std::complex<double>> alpha,
                    const std::complex<double>* a, std::int64_t lda,
                    value_or_pointer<std::complex<double>> beta, const std::complex<double>* b,
                    std::int64_t ldb, std::complex<double>* c, std::int64_t ldc,
                    const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy_batch(sycl::queue& queue, const transpose* trans, const std::int64_t* m,
                           const std::int64_t* n, const float* alpha, const float** a,
                           const std::int64_t* lda, float** b, const std::int64_t* ldb,
                           std::int64_t group_count, const std::int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy_batch(sycl::queue& queue, const transpose* trans, const std::int64_t* m,
                           const std::int64_t* n, const double* alpha, const double** a,
                           const std::int64_t* lda, double** b, const std::int64_t* ldb,
                           std::int64_t group_count, const std::int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy_batch(sycl::queue& queue, const transpose* trans, const std::int64_t* m,
                           const std::int64_t* n, const std::complex<float>* alpha,
                           const std::complex<float>** a, const std::int64_t* lda,
                           std::complex<float>** b, const std::int64_t* ldb,
                           std::int64_t group_count, const std::int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy_batch(sycl::queue& queue, const transpose* trans, const std::int64_t* m,
                           const std::int64_t* n, const std::complex<double>* alpha,
                           const std::complex<double>** a, const std::int64_t* lda,
                           std::complex<double>** b, const std::int64_t* ldb,
                           std::int64_t group_count, const std::int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies = {});

sycl::event imatcopy_batch(sycl::queue& queue, const transpose* trans, const std::int64_t* m,
                           const std::int64_t* n, const float* alpha, float** ab,
                           const std::int64_t* lda, const std::int64_t* ldb,
                           std::int64_t group_count, const std::int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies = {});

sycl::event imatcopy_batch(sycl::queue& queue, const transpose* trans, const std::int64_t* m,
                           const std::int64_t* n, const double* alpha, double** ab,
                           const std::int64_t* lda, const std::int64_t* ldb,
                           std::int64_t group_count, const std::int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies = {});

sycl::event imatcopy_batch(sycl::queue& queue, const transpose* trans, const std::int64_t* m,
                           const std::int64_t* n, const std::complex<float>* alpha,
                           std::complex<float>** ab, const std::int64_t* lda,
                           const std::int64_t* ldb, std::int64_t group_count,
                           const std::int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies = {});

sycl::event imatcopy_batch(sycl::queue& queue, const transpose* trans, const std::int64_t* m,
                           const std::int64_t* n, const std::complex<double>* alpha,
                           std::complex<double>** ab, const std::int64_t* lda,
                           const std::int64_t* ldb, std::int64_t group_count,
                           const std::int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies = {});
