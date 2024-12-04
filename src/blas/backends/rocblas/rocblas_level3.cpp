/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Copyright (C) 2022 Heidelberg University, Engineering Mathematics and Computing Lab (EMCL) and Computing Centre (URZ)
*
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

#include "rocblas_helper.hpp"
#include "rocblas_task.hpp"

#include "oneapi/math/exceptions.hpp"
#include "oneapi/math/blas/detail/rocblas/onemath_blas_rocblas.hpp"

namespace oneapi {
namespace math {
namespace blas {
namespace rocblas {
namespace column_major {

// Buffer APIs

template <typename Func, typename T>
inline void gemm(Func func, sycl::queue& queue, transpose transa, transpose transb, int64_t m,
                 int64_t n, int64_t k, T alpha, sycl::buffer<T, 1>& a, int64_t lda,
                 sycl::buffer<T, 1>& b, int64_t ldb, T beta, sycl::buffer<T, 1>& c, int64_t ldc) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, ldb, ldc);

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType*>(a_acc);
            auto b_ = sc.get_mem<rocDataType*>(b_acc);
            auto c_ = sc.get_mem<rocDataType*>(c_acc);
            rocblas_status err;
            rocblas_native_func(func, err, handle, get_rocblas_operation(transa),
                                get_rocblas_operation(transb), m, n, k, (rocDataType*)&alpha, a_,
                                lda, b_, ldb, (rocDataType*)&beta, c_, ldc);
        });
    });
}

#define GEMM_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                  \
    void gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,   \
              int64_t k, TYPE alpha, sycl::buffer<TYPE, 1>& a, int64_t lda,                   \
              sycl::buffer<TYPE, 1>& b, int64_t ldb, TYPE beta, sycl::buffer<TYPE, 1>& c,     \
              int64_t ldc) {                                                                  \
        gemm(ROCBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, \
             ldc);                                                                            \
    }

GEMM_LAUNCHER(float, rocblas_sgemm)
GEMM_LAUNCHER(double, rocblas_dgemm)
GEMM_LAUNCHER(std::complex<float>, rocblas_cgemm)
GEMM_LAUNCHER(std::complex<double>, rocblas_zgemm)

#undef GEMM_LAUNCHER

template <typename Func, typename T_A, typename T_B, typename T_C, typename T_S,
          typename DATATYPE_A, typename DATATYPE_B, typename DATATYPE_C, typename COMPUTETYPE>
inline void gemm_ex(Func func, DATATYPE_A DT_A, DATATYPE_B DT_B, DATATYPE_C DT_C, COMPUTETYPE CT,
                    sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                    int64_t k, T_S alpha, sycl::buffer<T_A, 1>& a, int64_t lda,
                    sycl::buffer<T_B, 1>& b, int64_t ldb, T_S beta, sycl::buffer<T_C, 1>& c,
                    int64_t ldc) {
    using rocDataType_A = typename RocEquivalentType<T_A>::Type;
    using rocDataType_B = typename RocEquivalentType<T_B>::Type;
    using rocDataType_C = typename RocEquivalentType<T_C>::Type;
    using rocDataType_S = typename RocEquivalentType<T_S>::Type;
    overflow_check(m, n, k, lda, ldb, ldc);

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType_A*>(a_acc);
            auto b_ = sc.get_mem<rocDataType_B*>(b_acc);
            auto c_ = sc.get_mem<rocDataType_C*>(c_acc);
            rocblas_status err;
            rocblas_native_func(func, err, handle, get_rocblas_operation(transa),
                                get_rocblas_operation(transb), m, n, k, (rocDataType_S*)&alpha, a_,
                                DT_A, lda, b_, DT_B, ldb, (rocDataType_S*)&beta, c_, DT_C, ldc, c_,
                                DT_C, ldc, CT, rocblas_gemm_algo_standard, 0, 0);
        });
    });
}

#define GEMM_EX_LAUNCHER(TYPE_A, TYPE_B, TYPE_C, TYPE_S, ROCBLAS_ROUTINE, ROCMDATATYPE_A,         \
                         ROCMDATATYPE_B, ROCMDATATYPE_C, ROCMCOMPUTETYPE)                         \
    void gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,       \
              int64_t k, TYPE_S alpha, sycl::buffer<TYPE_A, 1>& a, int64_t lda,                   \
              sycl::buffer<TYPE_B, 1>& b, int64_t ldb, TYPE_S beta, sycl::buffer<TYPE_C, 1>& c,   \
              int64_t ldc) {                                                                      \
        gemm_ex(ROCBLAS_ROUTINE, ROCMDATATYPE_A, ROCMDATATYPE_B, ROCMDATATYPE_C, ROCMCOMPUTETYPE, \
                queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);             \
    }

GEMM_EX_LAUNCHER(sycl::half, sycl::half, float, float, rocblas_gemm_ex, rocblas_datatype_f16_r,
                 rocblas_datatype_f16_r, rocblas_datatype_f32_r, rocblas_datatype_f32_r)
GEMM_EX_LAUNCHER(sycl::half, sycl::half, sycl::half, sycl::half, rocblas_gemm_ex,
                 rocblas_datatype_f16_r, rocblas_datatype_f16_r, rocblas_datatype_f16_r,
                 rocblas_datatype_f16_r)

GEMM_EX_LAUNCHER(bfloat16, bfloat16, float, float, rocblas_gemm_ex, rocblas_datatype_bf16_r,
                 rocblas_datatype_bf16_r, rocblas_datatype_f32_r, rocblas_datatype_f32_r)
GEMM_EX_LAUNCHER(bfloat16, bfloat16, bfloat16, float, rocblas_gemm_ex, rocblas_datatype_bf16_r,
                 rocblas_datatype_bf16_r, rocblas_datatype_bf16_r, rocblas_datatype_f32_r)

#undef GEMM_EX_LAUNCHER

template <typename Func, typename T>
inline void symm(Func func, sycl::queue& queue, side left_right, uplo upper_lower, int64_t m,
                 int64_t n, T alpha, sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& b,
                 int64_t ldb, T beta, sycl::buffer<T, 1>& c, int64_t ldc) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb, ldc);

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType*>(a_acc);
            auto b_ = sc.get_mem<rocDataType*>(b_acc);
            auto c_ = sc.get_mem<rocDataType*>(c_acc);
            rocblas_status err;
            rocblas_native_func(func, err, handle, get_rocblas_side_mode(left_right),
                                get_rocblas_fill_mode(upper_lower), m, n, (rocDataType*)&alpha, a_,
                                lda, b_, ldb, (rocDataType*)&beta, c_, ldc);
        });
    });
}

#define SYMM_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                     \
    void symm(sycl::queue& queue, side left_right, uplo upper_lower, int64_t m, int64_t n,       \
              TYPE alpha, sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& b,       \
              int64_t ldb, TYPE beta, sycl::buffer<TYPE, 1>& c, int64_t ldc) {                   \
        symm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, \
             c, ldc);                                                                            \
    }

SYMM_LAUNCHER(float, rocblas_ssymm)
SYMM_LAUNCHER(double, rocblas_dsymm)
SYMM_LAUNCHER(std::complex<float>, rocblas_csymm)
SYMM_LAUNCHER(std::complex<double>, rocblas_zsymm)

#undef SYMM_LAUNCHER

template <typename Func, typename T>
inline void hemm(Func func, sycl::queue& queue, side left_right, uplo upper_lower, int64_t m,
                 int64_t n, T alpha, sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& b,
                 int64_t ldb, T beta, sycl::buffer<T, 1>& c, int64_t ldc) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb, ldc);

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType*>(a_acc);
            auto b_ = sc.get_mem<rocDataType*>(b_acc);
            auto c_ = sc.get_mem<rocDataType*>(c_acc);
            rocblas_status err;
            rocblas_native_func(func, err, handle, get_rocblas_side_mode(left_right),
                                get_rocblas_fill_mode(upper_lower), m, n, (rocDataType*)&alpha, a_,
                                lda, b_, ldb, (rocDataType*)&beta, c_, ldc);
        });
    });
}

#define HEMM_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                     \
    void hemm(sycl::queue& queue, side left_right, uplo upper_lower, int64_t m, int64_t n,       \
              TYPE alpha, sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& b,       \
              int64_t ldb, TYPE beta, sycl::buffer<TYPE, 1>& c, int64_t ldc) {                   \
        hemm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, \
             c, ldc);                                                                            \
    }

HEMM_LAUNCHER(std::complex<float>, rocblas_chemm)
HEMM_LAUNCHER(std::complex<double>, rocblas_zhemm)

#undef HEMM_LAUNCHER

template <typename Func, typename T>
inline void syrk(Func func, sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n,
                 int64_t k, T alpha, sycl::buffer<T, 1>& a, int64_t lda, T beta,
                 sycl::buffer<T, 1>& c, int64_t ldc) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, k, lda, ldc);

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType*>(a_acc);
            auto c_ = sc.get_mem<rocDataType*>(c_acc);
            rocblas_status err;
            rocblas_native_func(func, err, handle, get_rocblas_fill_mode(upper_lower),
                                get_rocblas_operation(trans), n, k, (rocDataType*)&alpha, a_, lda,
                                (rocDataType*)&beta, c_, ldc);
        });
    });
}

#define SYRK_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                 \
    void syrk(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,   \
              TYPE alpha, sycl::buffer<TYPE, 1>& a, int64_t lda, TYPE beta,                  \
              sycl::buffer<TYPE, 1>& c, int64_t ldc) {                                       \
        syrk(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc); \
    }

SYRK_LAUNCHER(float, rocblas_ssyrk)
SYRK_LAUNCHER(double, rocblas_dsyrk)
SYRK_LAUNCHER(std::complex<float>, rocblas_csyrk)
SYRK_LAUNCHER(std::complex<double>, rocblas_zsyrk)

#undef SYRK_LAUNCHER

template <typename Func, typename DataType, typename ScalarType>
inline void herk(Func func, sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n,
                 int64_t k, ScalarType alpha, sycl::buffer<DataType, 1>& a, int64_t lda,
                 ScalarType beta, sycl::buffer<DataType, 1>& c, int64_t ldc) {
    using rocDataType = typename RocEquivalentType<DataType>::Type;
    using rocScalarType = typename RocEquivalentType<ScalarType>::Type;
    overflow_check(n, k, lda, ldc);

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType*>(a_acc);
            auto c_ = sc.get_mem<rocDataType*>(c_acc);
            rocblas_status err;
            rocblas_native_func(func, err, handle, get_rocblas_fill_mode(upper_lower),
                                get_rocblas_operation(trans), n, k, (rocScalarType*)&alpha, a_, lda,
                                (rocScalarType*)&beta, c_, ldc);
        });
    });
}

#define HERK_LAUNCHER(DATA_TYPE, SCALAR_TYPE, ROCBLAS_ROUTINE)                                 \
    void herk(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,     \
              SCALAR_TYPE alpha, sycl::buffer<DATA_TYPE, 1>& a, int64_t lda, SCALAR_TYPE beta, \
              sycl::buffer<DATA_TYPE, 1>& c, int64_t ldc) {                                    \
        herk(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);   \
    }

HERK_LAUNCHER(std::complex<float>, float, rocblas_cherk)
HERK_LAUNCHER(std::complex<double>, double, rocblas_zherk)

#undef HERK_LAUNCHER

template <typename Func, typename T>
inline void syr2k(Func func, sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n,
                  int64_t k, T alpha, sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& b,
                  int64_t ldb, T beta, sycl::buffer<T, 1>& c, int64_t ldc) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, k, lda, ldb, ldc);

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType*>(a_acc);
            auto b_ = sc.get_mem<rocDataType*>(b_acc);
            auto c_ = sc.get_mem<rocDataType*>(c_acc);
            rocblas_status err;
            rocblas_native_func(func, err, handle, get_rocblas_fill_mode(upper_lower),
                                get_rocblas_operation(trans), n, k, (rocDataType*)&alpha, a_, lda,
                                b_, ldb, (rocDataType*)&beta, c_, ldc);
        });
    });
}

#define SYR2K_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                   \
    void syr2k(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,     \
               TYPE alpha, sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& b,     \
               int64_t ldb, TYPE beta, sycl::buffer<TYPE, 1>& c, int64_t ldc) {                 \
        syr2k(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, \
              ldc);                                                                             \
    }

SYR2K_LAUNCHER(float, rocblas_ssyr2k)
SYR2K_LAUNCHER(double, rocblas_dsyr2k)
SYR2K_LAUNCHER(std::complex<float>, rocblas_csyr2k)
SYR2K_LAUNCHER(std::complex<double>, rocblas_zsyr2k)

#undef SYR2K_LAUNCHER

template <typename Func, typename DataType, typename ScalarType>
inline void her2k(Func func, sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n,
                  int64_t k, DataType alpha, sycl::buffer<DataType, 1>& a, int64_t lda,
                  sycl::buffer<DataType, 1>& b, int64_t ldb, ScalarType beta,
                  sycl::buffer<DataType, 1>& c, int64_t ldc) {
    using rocDataType = typename RocEquivalentType<DataType>::Type;
    using rocScalarType = typename RocEquivalentType<ScalarType>::Type;
    overflow_check(n, k, lda, ldb, ldc);

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType*>(a_acc);
            auto b_ = sc.get_mem<rocDataType*>(b_acc);
            auto c_ = sc.get_mem<rocDataType*>(c_acc);
            rocblas_status err;
            rocblas_native_func(func, err, handle, get_rocblas_fill_mode(upper_lower),
                                get_rocblas_operation(trans), n, k, (rocDataType*)&alpha, a_, lda,
                                b_, ldb, (rocScalarType*)&beta, c_, ldc);
        });
    });
}

#define HER2K_LAUNCHER(DATA_TYPE, SCALAR_TYPE, ROCBLAS_ROUTINE)                                 \
    void her2k(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,     \
               DATA_TYPE alpha, sycl::buffer<DATA_TYPE, 1>& a, int64_t lda,                     \
               sycl::buffer<DATA_TYPE, 1>& b, int64_t ldb, SCALAR_TYPE beta,                    \
               sycl::buffer<DATA_TYPE, 1>& c, int64_t ldc) {                                    \
        her2k(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, \
              ldc);                                                                             \
    }

HER2K_LAUNCHER(std::complex<float>, float, rocblas_cher2k)
HER2K_LAUNCHER(std::complex<double>, double, rocblas_zher2k)

#undef HER2K_LAUNCHER

// NOTE: In rocblas TRMM diverted from the netlib blas and for performance
// reason it requires the C matrix to be
// separated from the B matrix. It is possible to use B instead of C, but this
// will slow-down the code.
template <typename Func, typename T>
inline void trmm(Func func, sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                 diag unit_diag, int64_t m, int64_t n, T alpha, sycl::buffer<T, 1>& a, int64_t lda,
                 sycl::buffer<T, 1>& b, int64_t ldb) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb);

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType*>(a_acc);
            auto b_ = sc.get_mem<rocDataType*>(b_acc);
            rocblas_status err;
#if ROCBLAS_VERSION_MAJOR >= 4
            rocblas_native_func(func, err, handle, get_rocblas_side_mode(left_right),
                                get_rocblas_fill_mode(upper_lower), get_rocblas_operation(trans),
                                get_rocblas_diag_type(unit_diag), m, n, (rocDataType*)&alpha, a_,
                                lda, b_, ldb, b_, ldb);
#else
            rocblas_native_func(func, err, handle, get_rocblas_side_mode(left_right),
                                    get_rocblas_fill_mode(upper_lower),
                                    get_rocblas_operation(trans), get_rocblas_diag_type(unit_diag),
                                    m, n, (rocDataType *)&alpha, a_, lda, b_, ldb);
#endif
        });
    });
}

#define TRMM_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                    \
    void trmm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,           \
              diag unit_diag, int64_t m, int64_t n, TYPE alpha, sycl::buffer<TYPE, 1>& a,       \
              int64_t lda, sycl::buffer<TYPE, 1>& b, int64_t ldb) {                             \
        trmm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, \
             lda, b, ldb);                                                                      \
    }

TRMM_LAUNCHER(float, rocblas_strmm)
TRMM_LAUNCHER(double, rocblas_dtrmm)
TRMM_LAUNCHER(std::complex<float>, rocblas_ctrmm)
TRMM_LAUNCHER(std::complex<double>, rocblas_ztrmm)

#undef TRMM_LAUNCHER

template <typename Func, typename T>
inline void trsm(Func func, sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                 diag unit_diag, int64_t m, int64_t n, T alpha, sycl::buffer<T, 1>& a, int64_t lda,
                 sycl::buffer<T, 1>& b, int64_t ldb) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb);

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<rocDataType*>(a_acc);
            auto b_ = sc.get_mem<rocDataType*>(b_acc);
            rocblas_status err;
            rocblas_native_func(func, err, handle, get_rocblas_side_mode(left_right),
                                get_rocblas_fill_mode(upper_lower), get_rocblas_operation(trans),
                                get_rocblas_diag_type(unit_diag), m, n, (rocDataType*)&alpha, a_,
                                lda, b_, ldb);
        });
    });
}

#define TRSM_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                    \
    void trsm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,           \
              diag unit_diag, int64_t m, int64_t n, TYPE alpha, sycl::buffer<TYPE, 1>& a,       \
              int64_t lda, sycl::buffer<TYPE, 1>& b, int64_t ldb) {                             \
        trsm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, \
             lda, b, ldb);                                                                      \
    }

TRSM_LAUNCHER(float, rocblas_strsm)
TRSM_LAUNCHER(double, rocblas_dtrsm)
TRSM_LAUNCHER(std::complex<float>, rocblas_ctrsm)
TRSM_LAUNCHER(std::complex<double>, rocblas_ztrsm)

#undef TRSM_LAUNCHER

// USM APIs

template <typename Func, typename T>
inline sycl::event gemm(Func func, sycl::queue& queue, transpose transa, transpose transb,
                        int64_t m, int64_t n, int64_t k, T alpha, const T* a, int64_t lda,
                        const T* b, int64_t ldb, T beta, T* c, int64_t ldc,
                        const std::vector<sycl::event>& dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, ldb, ldc);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType*>(a);
            auto b_ = reinterpret_cast<const rocDataType*>(b);
            auto c_ = reinterpret_cast<rocDataType*>(c);
            rocblas_status err;
            rocblas_native_func(func, err, handle, get_rocblas_operation(transa),
                                get_rocblas_operation(transb), m, n, k, (rocDataType*)&alpha, a_,
                                lda, b_, ldb, (rocDataType*)&beta, c_, ldc);
        });
    });

    return done;
}

#define GEMM_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                   \
    sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n, \
                     int64_t k, TYPE alpha, const TYPE* a, int64_t lda, const TYPE* b,             \
                     int64_t ldb, TYPE beta, TYPE* c, int64_t ldc,                                 \
                     const std::vector<sycl::event>& dependencies) {                               \
        return gemm(ROCBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,  \
                    c, ldc, dependencies);                                                         \
    }

GEMM_LAUNCHER_USM(float, rocblas_sgemm)
GEMM_LAUNCHER_USM(double, rocblas_dgemm)
GEMM_LAUNCHER_USM(std::complex<float>, rocblas_cgemm)
GEMM_LAUNCHER_USM(std::complex<double>, rocblas_zgemm)

#undef GEMM_LAUNCHER_USM

template <typename Func, typename T_A, typename T_B, typename T_C, typename T_S,
          typename DATATYPE_A, typename DATATYPE_B, typename DATATYPE_C, typename COMPUTETYPE>
inline sycl::event gemm_ex(Func func, DATATYPE_A DT_A, DATATYPE_B DT_B, DATATYPE_C DT_C,
                           COMPUTETYPE CT, sycl::queue& queue, transpose transa, transpose transb,
                           int64_t m, int64_t n, int64_t k, T_S alpha, const T_A* a, int64_t lda,
                           const T_B* b, int64_t ldb, T_S beta, T_C* c, int64_t ldc,
                           const std::vector<sycl::event>& dependencies) {
    using rocDataType_A = typename RocEquivalentType<T_A>::Type;
    using rocDataType_B = typename RocEquivalentType<T_B>::Type;
    using rocDataType_C = typename RocEquivalentType<T_C>::Type;
    using rocDataType_S = typename RocEquivalentType<T_S>::Type;
    overflow_check(m, n, k, lda, ldb, ldc);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType_A*>(a);
            auto b_ = reinterpret_cast<const rocDataType_B*>(b);
            auto c_ = reinterpret_cast<rocDataType_C*>(c);
            rocblas_status err;
            rocblas_native_func(func, err, handle, get_rocblas_operation(transa),
                                get_rocblas_operation(transb), m, n, k, (rocDataType_S*)&alpha, a_,
                                DT_A, lda, b_, DT_B, ldb, (rocDataType_S*)&beta, c_, DT_C, ldc, c_,
                                DT_C, ldc, CT, rocblas_gemm_algo_standard, 0, 0);
        });
    });

    return done;
}

#define GEMM_EX_LAUNCHER_USM(TYPE_A, TYPE_B, TYPE_C, TYPE_S, ROCBLAS_ROUTINE, ROCMDATATYPE_A,      \
                             ROCMDATATYPE_B, ROCMDATATYPE_C, ROCMCOMPUTETYPE)                      \
    sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n, \
                     int64_t k, TYPE_S alpha, const TYPE_A* a, int64_t lda, const TYPE_B* b,       \
                     int64_t ldb, TYPE_S beta, TYPE_C* c, int64_t ldc,                             \
                     const std::vector<sycl::event>& dependencies) {                               \
        return gemm_ex(ROCBLAS_ROUTINE, ROCMDATATYPE_A, ROCMDATATYPE_B, ROCMDATATYPE_C,            \
                       ROCMCOMPUTETYPE, queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,     \
                       beta, c, ldc, dependencies);                                                \
    }

GEMM_EX_LAUNCHER_USM(sycl::half, sycl::half, float, float, rocblas_gemm_ex, rocblas_datatype_f16_r,
                     rocblas_datatype_f16_r, rocblas_datatype_f32_r, rocblas_datatype_f32_r)
GEMM_EX_LAUNCHER_USM(sycl::half, sycl::half, sycl::half, sycl::half, rocblas_gemm_ex,
                     rocblas_datatype_f16_r, rocblas_datatype_f16_r, rocblas_datatype_f16_r,
                     rocblas_datatype_f16_r)

GEMM_EX_LAUNCHER_USM(bfloat16, bfloat16, float, float, rocblas_gemm_ex, rocblas_datatype_bf16_r,
                     rocblas_datatype_bf16_r, rocblas_datatype_f32_r, rocblas_datatype_f32_r)
GEMM_EX_LAUNCHER_USM(bfloat16, bfloat16, bfloat16, float, rocblas_gemm_ex, rocblas_datatype_bf16_r,
                     rocblas_datatype_bf16_r, rocblas_datatype_bf16_r, rocblas_datatype_f32_r)

#undef GEMM_EX_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event symm(Func func, sycl::queue& queue, side left_right, uplo upper_lower, int64_t m,
                        int64_t n, T alpha, const T* a, int64_t lda, const T* b, int64_t ldb,
                        T beta, T* c, int64_t ldc, const std::vector<sycl::event>& dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb, ldc);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType*>(a);
            auto b_ = reinterpret_cast<const rocDataType*>(b);
            auto c_ = reinterpret_cast<rocDataType*>(c);
            rocblas_status err;
            rocblas_native_func(func, err, handle, get_rocblas_side_mode(left_right),
                                get_rocblas_fill_mode(upper_lower), m, n, (rocDataType*)&alpha, a_,
                                lda, b_, ldb, (rocDataType*)&beta, c_, ldc);
        });
    });

    return done;
}

#define SYMM_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                  \
    sycl::event symm(sycl::queue& queue, side left_right, uplo upper_lower, int64_t m, int64_t n, \
                     TYPE alpha, const TYPE* a, int64_t lda, const TYPE* b, int64_t ldb,          \
                     TYPE beta, TYPE* c, int64_t ldc,                                             \
                     const std::vector<sycl::event>& dependencies) {                              \
        return symm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, \
                    beta, c, ldc, dependencies);                                                  \
    }

SYMM_LAUNCHER_USM(float, rocblas_ssymm)
SYMM_LAUNCHER_USM(double, rocblas_dsymm)
SYMM_LAUNCHER_USM(std::complex<float>, rocblas_csymm)
SYMM_LAUNCHER_USM(std::complex<double>, rocblas_zsymm)

#undef SYMM_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event hemm(Func func, sycl::queue& queue, side left_right, uplo upper_lower, int64_t m,
                        int64_t n, T alpha, const T* a, int64_t lda, const T* b, int64_t ldb,
                        T beta, T* c, int64_t ldc, const std::vector<sycl::event>& dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb, ldc);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType*>(a);
            auto b_ = reinterpret_cast<const rocDataType*>(b);
            auto c_ = reinterpret_cast<rocDataType*>(c);
            rocblas_status err;
            rocblas_native_func(func, err, handle, get_rocblas_side_mode(left_right),
                                get_rocblas_fill_mode(upper_lower), m, n, (rocDataType*)&alpha, a_,
                                lda, b_, ldb, (rocDataType*)&beta, c_, ldc);
        });
    });

    return done;
}

#define HEMM_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                  \
    sycl::event hemm(sycl::queue& queue, side left_right, uplo upper_lower, int64_t m, int64_t n, \
                     TYPE alpha, const TYPE* a, int64_t lda, const TYPE* b, int64_t ldb,          \
                     TYPE beta, TYPE* c, int64_t ldc,                                             \
                     const std::vector<sycl::event>& dependencies) {                              \
        return hemm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, \
                    beta, c, ldc, dependencies);                                                  \
    }

HEMM_LAUNCHER_USM(std::complex<float>, rocblas_chemm)
HEMM_LAUNCHER_USM(std::complex<double>, rocblas_zhemm)

#undef HEMM_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event syrk(Func func, sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n,
                        int64_t k, T alpha, const T* a, int64_t lda, T beta, T* c, int64_t ldc,
                        const std::vector<sycl::event>& dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, k, lda, ldc);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType*>(a);
            auto c_ = reinterpret_cast<rocDataType*>(c);
            rocblas_status err;
            rocblas_native_func(func, err, handle, get_rocblas_fill_mode(upper_lower),
                                get_rocblas_operation(trans), n, k, (rocDataType*)&alpha, a_, lda,
                                (rocDataType*)&beta, c_, ldc);
        });
    });

    return done;
}

#define SYRK_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                   \
    sycl::event syrk(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,  \
                     TYPE alpha, const TYPE* a, int64_t lda, TYPE beta, TYPE* c, int64_t ldc,      \
                     const std::vector<sycl::event>& dependencies) {                               \
        return syrk(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, \
                    dependencies);                                                                 \
    }

SYRK_LAUNCHER_USM(float, rocblas_ssyrk)
SYRK_LAUNCHER_USM(double, rocblas_dsyrk)
SYRK_LAUNCHER_USM(std::complex<float>, rocblas_csyrk)
SYRK_LAUNCHER_USM(std::complex<double>, rocblas_zsyrk)

#undef SYRK_LAUNCHER_USM

template <typename Func, typename DataType, typename ScalarType>
inline sycl::event herk(Func func, sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n,
                        int64_t k, const ScalarType alpha, const DataType* a, int64_t lda,
                        const ScalarType beta, DataType* c, int64_t ldc,
                        const std::vector<sycl::event>& dependencies) {
    using rocDataType = typename RocEquivalentType<DataType>::Type;
    using rocScalarType = typename RocEquivalentType<ScalarType>::Type;
    overflow_check(n, k, lda, ldc);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType*>(a);
            auto c_ = reinterpret_cast<rocDataType*>(c);
            rocblas_status err;
            rocblas_native_func(func, err, handle, get_rocblas_fill_mode(upper_lower),
                                get_rocblas_operation(trans), n, k, (rocScalarType*)&alpha, a_, lda,
                                (rocScalarType*)&beta, c_, ldc);
        });
    });

    return done;
}

#define HERK_LAUNCHER_USM(DATA_TYPE, SCALAR_TYPE, ROCBLAS_ROUTINE)                                 \
    sycl::event herk(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,  \
                     const SCALAR_TYPE alpha, const DATA_TYPE* a, int64_t lda,                     \
                     const SCALAR_TYPE beta, DATA_TYPE* c, int64_t ldc,                            \
                     const std::vector<sycl::event>& dependencies) {                               \
        return herk(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, \
                    dependencies);                                                                 \
    }

HERK_LAUNCHER_USM(std::complex<float>, float, rocblas_cherk)
HERK_LAUNCHER_USM(std::complex<double>, double, rocblas_zherk)

#undef HERK_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event syr2k(Func func, sycl::queue& queue, uplo upper_lower, transpose trans,
                         int64_t n, int64_t k, T alpha, const T* a, int64_t lda, const T* b,
                         int64_t ldb, T beta, T* c, int64_t ldc,
                         const std::vector<sycl::event>& dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, k, lda, ldb, ldc);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType*>(a);
            auto b_ = reinterpret_cast<const rocDataType*>(b);
            auto c_ = reinterpret_cast<rocDataType*>(c);
            rocblas_status err;
            rocblas_native_func(func, err, handle, get_rocblas_fill_mode(upper_lower),
                                get_rocblas_operation(trans), n, k, (rocDataType*)&alpha, a_, lda,
                                b_, ldb, (rocDataType*)&beta, c_, ldc);
        });
    });

    return done;
}

#define SYR2K_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                  \
    sycl::event syr2k(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k, \
                      TYPE alpha, const TYPE* a, int64_t lda, const TYPE* b, int64_t ldb,          \
                      TYPE beta, TYPE* c, int64_t ldc,                                             \
                      const std::vector<sycl::event>& dependencies) {                              \
        return syr2k(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,      \
                     beta, c, ldc, dependencies);                                                  \
    }

SYR2K_LAUNCHER_USM(float, rocblas_ssyr2k)
SYR2K_LAUNCHER_USM(double, rocblas_dsyr2k)
SYR2K_LAUNCHER_USM(std::complex<float>, rocblas_csyr2k)
SYR2K_LAUNCHER_USM(std::complex<double>, rocblas_zsyr2k)

#undef SYR2K_LAUNCHER_USM

template <typename Func, typename DataType, typename ScalarType>
inline sycl::event her2k(Func func, sycl::queue& queue, uplo upper_lower, transpose trans,
                         int64_t n, int64_t k, const DataType alpha, const DataType* a, int64_t lda,
                         const DataType* b, int64_t ldb, const ScalarType beta, DataType* c,
                         int64_t ldc, const std::vector<sycl::event>& dependencies) {
    using rocDataType = typename RocEquivalentType<DataType>::Type;
    using rocScalarType = typename RocEquivalentType<ScalarType>::Type;
    overflow_check(n, k, lda, ldb, ldc);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType*>(a);
            auto b_ = reinterpret_cast<const rocDataType*>(b);
            auto c_ = reinterpret_cast<rocDataType*>(c);
            rocblas_status err;
            rocblas_native_func(func, err, handle, get_rocblas_fill_mode(upper_lower),
                                get_rocblas_operation(trans), n, k, (rocDataType*)&alpha, a_, lda,
                                b_, ldb, (rocScalarType*)&beta, c_, ldc);
        });
    });

    return done;
}

#define HER2K_LAUNCHER_USM(DATA_TYPE, SCALAR_TYPE, ROCBLAS_ROUTINE)                                \
    sycl::event her2k(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k, \
                      const DATA_TYPE alpha, const DATA_TYPE* a, int64_t lda, const DATA_TYPE* b,  \
                      int64_t ldb, const SCALAR_TYPE beta, DATA_TYPE* c, int64_t ldc,              \
                      const std::vector<sycl::event>& dependencies) {                              \
        return her2k(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,      \
                     beta, c, ldc, dependencies);                                                  \
    }

HER2K_LAUNCHER_USM(std::complex<float>, float, rocblas_cher2k)
HER2K_LAUNCHER_USM(std::complex<double>, double, rocblas_zher2k)

#undef HER2K_LAUNCHER_USM

// NOTE: In rocblas TRMM diverted from the netlib blas and for performance
// reason it requires the C matrix to be
// separated from the B matrix. It is possible to use B instead of C, but this
// will slow-down the code.
template <typename Func, typename T>
inline sycl::event trmm(Func func, sycl::queue& queue, side left_right, uplo upper_lower,
                        transpose trans, diag unit_diag, int64_t m, int64_t n, T alpha, const T* a,
                        int64_t lda, T* b, int64_t ldb,
                        const std::vector<sycl::event>& dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType*>(a);
            auto b_ = reinterpret_cast<rocDataType*>(b);
            rocblas_status err;
#if ROCBLAS_VERSION_MAJOR >= 4
            rocblas_native_func(func, err, handle, get_rocblas_side_mode(left_right),
                                get_rocblas_fill_mode(upper_lower), get_rocblas_operation(trans),
                                get_rocblas_diag_type(unit_diag), m, n, (rocDataType*)&alpha, a_,
                                lda, b_, ldb, b_, ldb);
#else
            rocblas_native_func(func, err, handle, get_rocblas_side_mode(left_right),
                                    get_rocblas_fill_mode(upper_lower),
                                    get_rocblas_operation(trans), get_rocblas_diag_type(unit_diag),
                                    m, n, (rocDataType *)&alpha, a_, lda, b_, ldb);
#endif
        });
    });

    return done;
}

#define TRMM_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                   \
    sycl::event trmm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,       \
                     diag unit_diag, int64_t m, int64_t n, TYPE alpha, const TYPE* a, int64_t lda, \
                     TYPE* b, int64_t ldb, const std::vector<sycl::event>& dependencies) {         \
        return trmm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, trans, unit_diag, m, n,       \
                    alpha, a, lda, b, ldb, dependencies);                                          \
    }

TRMM_LAUNCHER_USM(float, rocblas_strmm)
TRMM_LAUNCHER_USM(double, rocblas_dtrmm)
TRMM_LAUNCHER_USM(std::complex<float>, rocblas_ctrmm)
TRMM_LAUNCHER_USM(std::complex<double>, rocblas_ztrmm)

#undef TRMM_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event trsm(Func func, sycl::queue& queue, side left_right, uplo upper_lower,
                        transpose trans, diag unit_diag, int64_t m, int64_t n, T alpha, const T* a,
                        int64_t lda, T* b, int64_t ldb,
                        const std::vector<sycl::event>& dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const rocDataType*>(a);
            auto b_ = reinterpret_cast<rocDataType*>(b);
            rocblas_status err;
            rocblas_native_func(func, err, handle, get_rocblas_side_mode(left_right),
                                get_rocblas_fill_mode(upper_lower), get_rocblas_operation(trans),
                                get_rocblas_diag_type(unit_diag), m, n, (rocDataType*)&alpha, a_,
                                lda, b_, ldb);
        });
    });

    return done;
}

#define TRSM_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                   \
    sycl::event trsm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,       \
                     diag unit_diag, int64_t m, int64_t n, TYPE alpha, const TYPE* a, int64_t lda, \
                     TYPE* b, int64_t ldb, const std::vector<sycl::event>& dependencies) {         \
        return trsm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, trans, unit_diag, m, n,       \
                    alpha, a, lda, b, ldb, dependencies);                                          \
    }

TRSM_LAUNCHER_USM(float, rocblas_strsm)
TRSM_LAUNCHER_USM(double, rocblas_dtrsm)
TRSM_LAUNCHER_USM(std::complex<float>, rocblas_ctrsm)
TRSM_LAUNCHER_USM(std::complex<double>, rocblas_ztrsm)

#undef TRSM_LAUNCHER_USM

} // namespace column_major

namespace row_major {

// Buffer APIs

template <typename Func, typename T>
inline void gemm(Func func, sycl::queue& queue, transpose transa, transpose transb, int64_t m,
                 int64_t n, int64_t k, T alpha, sycl::buffer<T, 1>& a, int64_t lda,
                 sycl::buffer<T, 1>& b, int64_t ldb, T beta, sycl::buffer<T, 1>& c, int64_t ldc) {
    auto new_transa = transb;
    auto new_transb = transa;

    column_major::gemm(func, queue, new_transa, new_transb, n, m, k, alpha, b, ldb, a, lda, beta, c,
                       ldc);
}

#define GEMM_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                  \
    void gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,   \
              int64_t k, TYPE alpha, sycl::buffer<TYPE, 1>& a, int64_t lda,                   \
              sycl::buffer<TYPE, 1>& b, int64_t ldb, TYPE beta, sycl::buffer<TYPE, 1>& c,     \
              int64_t ldc) {                                                                  \
        gemm(ROCBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, \
             ldc);                                                                            \
    }

GEMM_LAUNCHER(float, rocblas_sgemm)
GEMM_LAUNCHER(double, rocblas_dgemm)
GEMM_LAUNCHER(std::complex<float>, rocblas_cgemm)
GEMM_LAUNCHER(std::complex<double>, rocblas_zgemm)

#undef GEMM_LAUNCHER

template <typename Func, typename T_A, typename T_B, typename T_C, typename T_S,
          typename DATATYPE_A, typename DATATYPE_B, typename DATATYPE_C, typename COMPUTETYPE>
inline void gemm_ex(Func func, DATATYPE_A DT_A, DATATYPE_B DT_B, DATATYPE_C DT_C, COMPUTETYPE CT,
                    sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                    int64_t k, T_S alpha, sycl::buffer<T_A, 1>& a, int64_t lda,
                    sycl::buffer<T_B, 1>& b, int64_t ldb, T_S beta, sycl::buffer<T_C, 1>& c,
                    int64_t ldc) {
    auto new_transa = transb;
    auto new_transb = transa;

    column_major::gemm_ex(func, DT_A, DT_B, DT_C, CT, queue, new_transa, new_transb, n, m, k, alpha,
                          b, ldb, a, lda, beta, c, ldc);
}

#define GEMM_EX_LAUNCHER(TYPE_A, TYPE_B, TYPE_C, TYPE_S, ROCBLAS_ROUTINE, ROCMDATATYPE_A,         \
                         ROCMDATATYPE_B, ROCMDATATYPE_C, ROCMCOMPUTETYPE)                         \
    void gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,       \
              int64_t k, TYPE_S alpha, sycl::buffer<TYPE_A, 1>& a, int64_t lda,                   \
              sycl::buffer<TYPE_B, 1>& b, int64_t ldb, TYPE_S beta, sycl::buffer<TYPE_C, 1>& c,   \
              int64_t ldc) {                                                                      \
        gemm_ex(ROCBLAS_ROUTINE, ROCMDATATYPE_A, ROCMDATATYPE_B, ROCMDATATYPE_C, ROCMCOMPUTETYPE, \
                queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);             \
    }

GEMM_EX_LAUNCHER(sycl::half, sycl::half, float, float, rocblas_gemm_ex, rocblas_datatype_f16_r,
                 rocblas_datatype_f16_r, rocblas_datatype_f32_r, rocblas_datatype_f32_r)
GEMM_EX_LAUNCHER(sycl::half, sycl::half, sycl::half, sycl::half, rocblas_gemm_ex,
                 rocblas_datatype_f16_r, rocblas_datatype_f16_r, rocblas_datatype_f16_r,
                 rocblas_datatype_f16_r)

GEMM_EX_LAUNCHER(bfloat16, bfloat16, float, float, rocblas_gemm_ex, rocblas_datatype_bf16_r,
                 rocblas_datatype_bf16_r, rocblas_datatype_f32_r, rocblas_datatype_f32_r)
GEMM_EX_LAUNCHER(bfloat16, bfloat16, bfloat16, float, rocblas_gemm_ex, rocblas_datatype_bf16_r,
                 rocblas_datatype_bf16_r, rocblas_datatype_bf16_r, rocblas_datatype_f32_r)

#undef GEMM_EX_LAUNCHER

template <typename Func, typename T>
inline void symm(Func func, sycl::queue& queue, side left_right, uplo upper_lower, int64_t m,
                 int64_t n, T alpha, sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& b,
                 int64_t ldb, T beta, sycl::buffer<T, 1>& c, int64_t ldc) {
    auto new_side = left_right == oneapi::math::side::left ? oneapi::math::side::right
                                                           : oneapi::math::side::left;
    auto new_uplo = upper_lower == oneapi::math::uplo::lower ? oneapi::math::uplo::upper
                                                             : oneapi::math::uplo::lower;

    column_major::symm(func, queue, new_side, new_uplo, n, m, alpha, a, lda, b, ldb, beta, c, ldc);
}

#define SYMM_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                     \
    void symm(sycl::queue& queue, side left_right, uplo upper_lower, int64_t m, int64_t n,       \
              TYPE alpha, sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& b,       \
              int64_t ldb, TYPE beta, sycl::buffer<TYPE, 1>& c, int64_t ldc) {                   \
        symm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, \
             c, ldc);                                                                            \
    }

SYMM_LAUNCHER(float, rocblas_ssymm)
SYMM_LAUNCHER(double, rocblas_dsymm)
SYMM_LAUNCHER(std::complex<float>, rocblas_csymm)
SYMM_LAUNCHER(std::complex<double>, rocblas_zsymm)

#undef SYMM_LAUNCHER

template <typename Func, typename T>
inline void hemm(Func func, sycl::queue& queue, side left_right, uplo upper_lower, int64_t m,
                 int64_t n, T alpha, sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& b,
                 int64_t ldb, T beta, sycl::buffer<T, 1>& c, int64_t ldc) {
    auto new_side = left_right == oneapi::math::side::left ? oneapi::math::side::right
                                                           : oneapi::math::side::left;
    auto new_uplo = upper_lower == oneapi::math::uplo::lower ? oneapi::math::uplo::upper
                                                             : oneapi::math::uplo::lower;

    column_major::hemm(func, queue, new_side, new_uplo, n, m, alpha, a, lda, b, ldb, beta, c, ldc);
}

#define HEMM_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                     \
    void hemm(sycl::queue& queue, side left_right, uplo upper_lower, int64_t m, int64_t n,       \
              TYPE alpha, sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& b,       \
              int64_t ldb, TYPE beta, sycl::buffer<TYPE, 1>& c, int64_t ldc) {                   \
        hemm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, \
             c, ldc);                                                                            \
    }

HEMM_LAUNCHER(std::complex<float>, rocblas_chemm)
HEMM_LAUNCHER(std::complex<double>, rocblas_zhemm)

#undef HEMM_LAUNCHER

template <typename Func, typename T>
inline void syrk(Func func, sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n,
                 int64_t k, T alpha, sycl::buffer<T, 1>& a, int64_t lda, T beta,
                 sycl::buffer<T, 1>& c, int64_t ldc) {
    auto new_uplo = upper_lower == oneapi::math::uplo::lower ? oneapi::math::uplo::upper
                                                             : oneapi::math::uplo::lower;
    auto new_trans = trans == oneapi::math::transpose::nontrans ? oneapi::math::transpose::trans
                                                                : oneapi::math::transpose::nontrans;

    column_major::syrk(func, queue, new_uplo, new_trans, n, k, alpha, a, lda, beta, c, ldc);
}

#define SYRK_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                 \
    void syrk(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,   \
              TYPE alpha, sycl::buffer<TYPE, 1>& a, int64_t lda, TYPE beta,                  \
              sycl::buffer<TYPE, 1>& c, int64_t ldc) {                                       \
        syrk(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc); \
    }

SYRK_LAUNCHER(float, rocblas_ssyrk)
SYRK_LAUNCHER(double, rocblas_dsyrk)
SYRK_LAUNCHER(std::complex<float>, rocblas_csyrk)
SYRK_LAUNCHER(std::complex<double>, rocblas_zsyrk)

#undef SYRK_LAUNCHER

template <typename Func, typename DataType, typename ScalarType>
inline void herk(Func func, sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n,
                 int64_t k, ScalarType alpha, sycl::buffer<DataType, 1>& a, int64_t lda,
                 ScalarType beta, sycl::buffer<DataType, 1>& c, int64_t ldc) {
    auto new_uplo = upper_lower == oneapi::math::uplo::lower ? oneapi::math::uplo::upper
                                                             : oneapi::math::uplo::lower;
    auto new_trans = trans == oneapi::math::transpose::nontrans ? oneapi::math::transpose::conjtrans
                                                                : oneapi::math::transpose::nontrans;

    column_major::herk(func, queue, new_uplo, new_trans, n, k, alpha, a, lda, beta, c, ldc);
}

#define HERK_LAUNCHER(DATA_TYPE, SCALAR_TYPE, ROCBLAS_ROUTINE)                                 \
    void herk(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,     \
              SCALAR_TYPE alpha, sycl::buffer<DATA_TYPE, 1>& a, int64_t lda, SCALAR_TYPE beta, \
              sycl::buffer<DATA_TYPE, 1>& c, int64_t ldc) {                                    \
        herk(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);   \
    }

HERK_LAUNCHER(std::complex<float>, float, rocblas_cherk)
HERK_LAUNCHER(std::complex<double>, double, rocblas_zherk)

#undef HERK_LAUNCHER

template <typename Func, typename T>
inline void syr2k(Func func, sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n,
                  int64_t k, T alpha, sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& b,
                  int64_t ldb, T beta, sycl::buffer<T, 1>& c, int64_t ldc) {
    auto new_uplo = upper_lower == oneapi::math::uplo::lower ? oneapi::math::uplo::upper
                                                             : oneapi::math::uplo::lower;
    auto new_trans = trans == oneapi::math::transpose::nontrans ? oneapi::math::transpose::trans
                                                                : oneapi::math::transpose::nontrans;

    column_major::syr2k(func, queue, new_uplo, new_trans, n, k, alpha, a, lda, b, ldb, beta, c,
                        ldc);
}

#define SYR2K_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                   \
    void syr2k(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,     \
               TYPE alpha, sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& b,     \
               int64_t ldb, TYPE beta, sycl::buffer<TYPE, 1>& c, int64_t ldc) {                 \
        syr2k(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, \
              ldc);                                                                             \
    }

SYR2K_LAUNCHER(float, rocblas_ssyr2k)
SYR2K_LAUNCHER(double, rocblas_dsyr2k)
SYR2K_LAUNCHER(std::complex<float>, rocblas_csyr2k)
SYR2K_LAUNCHER(std::complex<double>, rocblas_zsyr2k)

#undef SYR2K_LAUNCHER

template <typename Func, typename DataType, typename ScalarType>
inline void her2k(Func func, sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n,
                  int64_t k, DataType alpha, sycl::buffer<DataType, 1>& a, int64_t lda,
                  sycl::buffer<DataType, 1>& b, int64_t ldb, ScalarType beta,
                  sycl::buffer<DataType, 1>& c, int64_t ldc) {
    auto new_uplo = upper_lower == oneapi::math::uplo::lower ? oneapi::math::uplo::upper
                                                             : oneapi::math::uplo::lower;
    auto new_trans = trans == oneapi::math::transpose::nontrans ? oneapi::math::transpose::conjtrans
                                                                : oneapi::math::transpose::nontrans;
    auto new_alpha = std::conj(alpha);

    column_major::her2k(func, queue, new_uplo, new_trans, n, k, new_alpha, a, lda, b, ldb, beta, c,
                        ldc);
}

#define HER2K_LAUNCHER(DATA_TYPE, SCALAR_TYPE, ROCBLAS_ROUTINE)                                 \
    void her2k(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,     \
               DATA_TYPE alpha, sycl::buffer<DATA_TYPE, 1>& a, int64_t lda,                     \
               sycl::buffer<DATA_TYPE, 1>& b, int64_t ldb, SCALAR_TYPE beta,                    \
               sycl::buffer<DATA_TYPE, 1>& c, int64_t ldc) {                                    \
        her2k(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, \
              ldc);                                                                             \
    }

HER2K_LAUNCHER(std::complex<float>, float, rocblas_cher2k)
HER2K_LAUNCHER(std::complex<double>, double, rocblas_zher2k)

#undef HER2K_LAUNCHER

// NOTE: In rocblas TRMM diverted from the netlib blas and for performance
// reason it requires the C matrix to be
// separated from the B matrix. It is possible to use B instead of C, but this
// will slow-down the code.
template <typename Func, typename T>
inline void trmm(Func func, sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                 diag unit_diag, int64_t m, int64_t n, T alpha, sycl::buffer<T, 1>& a, int64_t lda,
                 sycl::buffer<T, 1>& b, int64_t ldb) {
    auto new_side = left_right == oneapi::math::side::left ? oneapi::math::side::right
                                                           : oneapi::math::side::left;
    auto new_uplo = upper_lower == oneapi::math::uplo::lower ? oneapi::math::uplo::upper
                                                             : oneapi::math::uplo::lower;

    column_major::trmm(func, queue, new_side, new_uplo, trans, unit_diag, n, m, alpha, a, lda, b,
                       ldb);
}

#define TRMM_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                    \
    void trmm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,           \
              diag unit_diag, int64_t m, int64_t n, TYPE alpha, sycl::buffer<TYPE, 1>& a,       \
              int64_t lda, sycl::buffer<TYPE, 1>& b, int64_t ldb) {                             \
        trmm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, \
             lda, b, ldb);                                                                      \
    }

TRMM_LAUNCHER(float, rocblas_strmm)
TRMM_LAUNCHER(double, rocblas_dtrmm)
TRMM_LAUNCHER(std::complex<float>, rocblas_ctrmm)
TRMM_LAUNCHER(std::complex<double>, rocblas_ztrmm)

#undef TRMM_LAUNCHER

template <typename Func, typename T>
inline void trsm(Func func, sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                 diag unit_diag, int64_t m, int64_t n, T alpha, sycl::buffer<T, 1>& a, int64_t lda,
                 sycl::buffer<T, 1>& b, int64_t ldb) {
    auto new_side = left_right == oneapi::math::side::left ? oneapi::math::side::right
                                                           : oneapi::math::side::left;
    auto new_uplo = upper_lower == oneapi::math::uplo::lower ? oneapi::math::uplo::upper
                                                             : oneapi::math::uplo::lower;

    column_major::trsm(func, queue, new_side, new_uplo, trans, unit_diag, n, m, alpha, a, lda, b,
                       ldb);
}

#define TRSM_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                    \
    void trsm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,           \
              diag unit_diag, int64_t m, int64_t n, TYPE alpha, sycl::buffer<TYPE, 1>& a,       \
              int64_t lda, sycl::buffer<TYPE, 1>& b, int64_t ldb) {                             \
        trsm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, \
             lda, b, ldb);                                                                      \
    }

TRSM_LAUNCHER(float, rocblas_strsm)
TRSM_LAUNCHER(double, rocblas_dtrsm)
TRSM_LAUNCHER(std::complex<float>, rocblas_ctrsm)
TRSM_LAUNCHER(std::complex<double>, rocblas_ztrsm)

#undef TRSM_LAUNCHER

// USM APIs

template <typename Func, typename T>
inline sycl::event gemm(Func func, sycl::queue& queue, transpose transa, transpose transb,
                        int64_t m, int64_t n, int64_t k, T alpha, const T* a, int64_t lda,
                        const T* b, int64_t ldb, T beta, T* c, int64_t ldc,
                        const std::vector<sycl::event>& dependencies) {
    auto new_transa = transb;
    auto new_transb = transa;

    return column_major::gemm(func, queue, new_transa, new_transb, n, m, k, alpha, b, ldb, a, lda,
                              beta, c, ldc, dependencies);
}

#define GEMM_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                   \
    sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n, \
                     int64_t k, TYPE alpha, const TYPE* a, int64_t lda, const TYPE* b,             \
                     int64_t ldb, TYPE beta, TYPE* c, int64_t ldc,                                 \
                     const std::vector<sycl::event>& dependencies) {                               \
        return gemm(ROCBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,  \
                    c, ldc, dependencies);                                                         \
    }

GEMM_LAUNCHER_USM(float, rocblas_sgemm)
GEMM_LAUNCHER_USM(double, rocblas_dgemm)
GEMM_LAUNCHER_USM(std::complex<float>, rocblas_cgemm)
GEMM_LAUNCHER_USM(std::complex<double>, rocblas_zgemm)

#undef GEMM_LAUNCHER_USM

template <typename Func, typename T_A, typename T_B, typename T_C, typename T_S,
          typename DATATYPE_A, typename DATATYPE_B, typename DATATYPE_C, typename COMPUTETYPE>
inline sycl::event gemm_ex(Func func, DATATYPE_A DT_A, DATATYPE_B DT_B, DATATYPE_C DT_C,
                           COMPUTETYPE CT, sycl::queue& queue, transpose transa, transpose transb,
                           int64_t m, int64_t n, int64_t k, T_S alpha, const T_A* a, int64_t lda,
                           const T_B* b, int64_t ldb, T_S beta, T_C* c, int64_t ldc,
                           const std::vector<sycl::event>& dependencies) {
    auto new_transa = transb;
    auto new_transb = transa;

    return column_major::gemm_ex(func, DT_A, DT_B, DT_C, CT, queue, new_transa, new_transb, n, m, k,
                                 alpha, b, ldb, a, lda, beta, c, ldc, dependencies);
}

#define GEMM_EX_LAUNCHER_USM(TYPE_A, TYPE_B, TYPE_C, TYPE_S, ROCBLAS_ROUTINE, ROCMDATATYPE_A,      \
                             ROCMDATATYPE_B, ROCMDATATYPE_C, ROCMCOMPUTETYPE)                      \
    sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n, \
                     int64_t k, TYPE_S alpha, const TYPE_A* a, int64_t lda, const TYPE_B* b,       \
                     int64_t ldb, TYPE_S beta, TYPE_C* c, int64_t ldc,                             \
                     const std::vector<sycl::event>& dependencies) {                               \
        return gemm_ex(ROCBLAS_ROUTINE, ROCMDATATYPE_A, ROCMDATATYPE_B, ROCMDATATYPE_C,            \
                       ROCMCOMPUTETYPE, queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,     \
                       beta, c, ldc, dependencies);                                                \
    }

GEMM_EX_LAUNCHER_USM(sycl::half, sycl::half, float, float, rocblas_gemm_ex, rocblas_datatype_f16_r,
                     rocblas_datatype_f16_r, rocblas_datatype_f32_r, rocblas_datatype_f32_r)
GEMM_EX_LAUNCHER_USM(sycl::half, sycl::half, sycl::half, sycl::half, rocblas_gemm_ex,
                     rocblas_datatype_f16_r, rocblas_datatype_f16_r, rocblas_datatype_f16_r,
                     rocblas_datatype_f16_r)

GEMM_EX_LAUNCHER_USM(bfloat16, bfloat16, float, float, rocblas_gemm_ex, rocblas_datatype_bf16_r,
                     rocblas_datatype_bf16_r, rocblas_datatype_f32_r, rocblas_datatype_f32_r)
GEMM_EX_LAUNCHER_USM(bfloat16, bfloat16, bfloat16, float, rocblas_gemm_ex, rocblas_datatype_bf16_r,
                     rocblas_datatype_bf16_r, rocblas_datatype_bf16_r, rocblas_datatype_f32_r)

#undef GEMM_EX_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event symm(Func func, sycl::queue& queue, side left_right, uplo upper_lower, int64_t m,
                        int64_t n, T alpha, const T* a, int64_t lda, const T* b, int64_t ldb,
                        T beta, T* c, int64_t ldc, const std::vector<sycl::event>& dependencies) {
    auto new_side = left_right == oneapi::math::side::left ? oneapi::math::side::right
                                                           : oneapi::math::side::left;
    auto new_uplo = upper_lower == oneapi::math::uplo::lower ? oneapi::math::uplo::upper
                                                             : oneapi::math::uplo::lower;

    return column_major::symm(func, queue, new_side, new_uplo, n, m, alpha, a, lda, b, ldb, beta, c,
                              ldc, dependencies);
}

#define SYMM_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                  \
    sycl::event symm(sycl::queue& queue, side left_right, uplo upper_lower, int64_t m, int64_t n, \
                     TYPE alpha, const TYPE* a, int64_t lda, const TYPE* b, int64_t ldb,          \
                     TYPE beta, TYPE* c, int64_t ldc,                                             \
                     const std::vector<sycl::event>& dependencies) {                              \
        return symm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, \
                    beta, c, ldc, dependencies);                                                  \
    }

SYMM_LAUNCHER_USM(float, rocblas_ssymm)
SYMM_LAUNCHER_USM(double, rocblas_dsymm)
SYMM_LAUNCHER_USM(std::complex<float>, rocblas_csymm)
SYMM_LAUNCHER_USM(std::complex<double>, rocblas_zsymm)

#undef SYMM_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event hemm(Func func, sycl::queue& queue, side left_right, uplo upper_lower, int64_t m,
                        int64_t n, T alpha, const T* a, int64_t lda, const T* b, int64_t ldb,
                        T beta, T* c, int64_t ldc, const std::vector<sycl::event>& dependencies) {
    auto new_side = left_right == oneapi::math::side::left ? oneapi::math::side::right
                                                           : oneapi::math::side::left;
    auto new_uplo = upper_lower == oneapi::math::uplo::lower ? oneapi::math::uplo::upper
                                                             : oneapi::math::uplo::lower;

    return column_major::hemm(func, queue, new_side, new_uplo, n, m, alpha, a, lda, b, ldb, beta, c,
                              ldc, dependencies);
}

#define HEMM_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                  \
    sycl::event hemm(sycl::queue& queue, side left_right, uplo upper_lower, int64_t m, int64_t n, \
                     TYPE alpha, const TYPE* a, int64_t lda, const TYPE* b, int64_t ldb,          \
                     TYPE beta, TYPE* c, int64_t ldc,                                             \
                     const std::vector<sycl::event>& dependencies) {                              \
        return hemm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, \
                    beta, c, ldc, dependencies);                                                  \
    }

HEMM_LAUNCHER_USM(std::complex<float>, rocblas_chemm)
HEMM_LAUNCHER_USM(std::complex<double>, rocblas_zhemm)

#undef HEMM_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event syrk(Func func, sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n,
                        int64_t k, T alpha, const T* a, int64_t lda, T beta, T* c, int64_t ldc,
                        const std::vector<sycl::event>& dependencies) {
    auto new_uplo = upper_lower == oneapi::math::uplo::lower ? oneapi::math::uplo::upper
                                                             : oneapi::math::uplo::lower;
    auto new_trans = trans == oneapi::math::transpose::nontrans ? oneapi::math::transpose::trans
                                                                : oneapi::math::transpose::nontrans;

    return column_major::syrk(func, queue, new_uplo, new_trans, n, k, alpha, a, lda, beta, c, ldc,
                              dependencies);
}

#define SYRK_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                   \
    sycl::event syrk(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,  \
                     TYPE alpha, const TYPE* a, int64_t lda, TYPE beta, TYPE* c, int64_t ldc,      \
                     const std::vector<sycl::event>& dependencies) {                               \
        return syrk(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, \
                    dependencies);                                                                 \
    }

SYRK_LAUNCHER_USM(float, rocblas_ssyrk)
SYRK_LAUNCHER_USM(double, rocblas_dsyrk)
SYRK_LAUNCHER_USM(std::complex<float>, rocblas_csyrk)
SYRK_LAUNCHER_USM(std::complex<double>, rocblas_zsyrk)

#undef SYRK_LAUNCHER_USM

template <typename Func, typename DataType, typename ScalarType>
inline sycl::event herk(Func func, sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n,
                        int64_t k, const ScalarType alpha, const DataType* a, int64_t lda,
                        const ScalarType beta, DataType* c, int64_t ldc,
                        const std::vector<sycl::event>& dependencies) {
    auto new_uplo = upper_lower == oneapi::math::uplo::lower ? oneapi::math::uplo::upper
                                                             : oneapi::math::uplo::lower;
    auto new_trans = trans == oneapi::math::transpose::nontrans ? oneapi::math::transpose::conjtrans
                                                                : oneapi::math::transpose::nontrans;

    return column_major::herk(func, queue, new_uplo, new_trans, n, k, alpha, a, lda, beta, c, ldc,
                              dependencies);
}

#define HERK_LAUNCHER_USM(DATA_TYPE, SCALAR_TYPE, ROCBLAS_ROUTINE)                                 \
    sycl::event herk(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,  \
                     const SCALAR_TYPE alpha, const DATA_TYPE* a, int64_t lda,                     \
                     const SCALAR_TYPE beta, DATA_TYPE* c, int64_t ldc,                            \
                     const std::vector<sycl::event>& dependencies) {                               \
        return herk(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, \
                    dependencies);                                                                 \
    }

HERK_LAUNCHER_USM(std::complex<float>, float, rocblas_cherk)
HERK_LAUNCHER_USM(std::complex<double>, double, rocblas_zherk)

#undef HERK_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event syr2k(Func func, sycl::queue& queue, uplo upper_lower, transpose trans,
                         int64_t n, int64_t k, T alpha, const T* a, int64_t lda, const T* b,
                         int64_t ldb, T beta, T* c, int64_t ldc,
                         const std::vector<sycl::event>& dependencies) {
    auto new_uplo = upper_lower == oneapi::math::uplo::lower ? oneapi::math::uplo::upper
                                                             : oneapi::math::uplo::lower;
    auto new_trans = trans == oneapi::math::transpose::nontrans ? oneapi::math::transpose::trans
                                                                : oneapi::math::transpose::nontrans;

    return column_major::syr2k(func, queue, new_uplo, new_trans, n, k, alpha, a, lda, b, ldb, beta,
                               c, ldc, dependencies);
}

#define SYR2K_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                  \
    sycl::event syr2k(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k, \
                      TYPE alpha, const TYPE* a, int64_t lda, const TYPE* b, int64_t ldb,          \
                      TYPE beta, TYPE* c, int64_t ldc,                                             \
                      const std::vector<sycl::event>& dependencies) {                              \
        return syr2k(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,      \
                     beta, c, ldc, dependencies);                                                  \
    }

SYR2K_LAUNCHER_USM(float, rocblas_ssyr2k)
SYR2K_LAUNCHER_USM(double, rocblas_dsyr2k)
SYR2K_LAUNCHER_USM(std::complex<float>, rocblas_csyr2k)
SYR2K_LAUNCHER_USM(std::complex<double>, rocblas_zsyr2k)

#undef SYR2K_LAUNCHER_USM

template <typename Func, typename DataType, typename ScalarType>
inline sycl::event her2k(Func func, sycl::queue& queue, uplo upper_lower, transpose trans,
                         int64_t n, int64_t k, const DataType alpha, const DataType* a, int64_t lda,
                         const DataType* b, int64_t ldb, const ScalarType beta, DataType* c,
                         int64_t ldc, const std::vector<sycl::event>& dependencies) {
    auto new_uplo = upper_lower == oneapi::math::uplo::lower ? oneapi::math::uplo::upper
                                                             : oneapi::math::uplo::lower;
    auto new_trans = trans == oneapi::math::transpose::nontrans ? oneapi::math::transpose::conjtrans
                                                                : oneapi::math::transpose::nontrans;
    auto new_alpha = std::conj(alpha);

    return column_major::her2k(func, queue, new_uplo, new_trans, n, k, new_alpha, a, lda, b, ldb,
                               beta, c, ldc, dependencies);
}

#define HER2K_LAUNCHER_USM(DATA_TYPE, SCALAR_TYPE, ROCBLAS_ROUTINE)                                \
    sycl::event her2k(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k, \
                      const DATA_TYPE alpha, const DATA_TYPE* a, int64_t lda, const DATA_TYPE* b,  \
                      int64_t ldb, const SCALAR_TYPE beta, DATA_TYPE* c, int64_t ldc,              \
                      const std::vector<sycl::event>& dependencies) {                              \
        return her2k(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,      \
                     beta, c, ldc, dependencies);                                                  \
    }

HER2K_LAUNCHER_USM(std::complex<float>, float, rocblas_cher2k)
HER2K_LAUNCHER_USM(std::complex<double>, double, rocblas_zher2k)

#undef HER2K_LAUNCHER_USM

// NOTE: In rocblas TRMM diverted from the netlib blas and for performance
// reason it requires the C matrix to be
// separated from the B matrix. It is possible to use B instead of C, but this
// will slow-down the code.
template <typename Func, typename T>
inline sycl::event trmm(Func func, sycl::queue& queue, side left_right, uplo upper_lower,
                        transpose trans, diag unit_diag, int64_t m, int64_t n, T alpha, const T* a,
                        int64_t lda, T* b, int64_t ldb,
                        const std::vector<sycl::event>& dependencies) {
    auto new_side = left_right == oneapi::math::side::left ? oneapi::math::side::right
                                                           : oneapi::math::side::left;
    auto new_uplo = upper_lower == oneapi::math::uplo::lower ? oneapi::math::uplo::upper
                                                             : oneapi::math::uplo::lower;

    return column_major::trmm(func, queue, new_side, new_uplo, trans, unit_diag, n, m, alpha, a,
                              lda, b, ldb, dependencies);
}

#define TRMM_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                   \
    sycl::event trmm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,       \
                     diag unit_diag, int64_t m, int64_t n, TYPE alpha, const TYPE* a, int64_t lda, \
                     TYPE* b, int64_t ldb, const std::vector<sycl::event>& dependencies) {         \
        return trmm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, trans, unit_diag, m, n,       \
                    alpha, a, lda, b, ldb, dependencies);                                          \
    }

TRMM_LAUNCHER_USM(float, rocblas_strmm)
TRMM_LAUNCHER_USM(double, rocblas_dtrmm)
TRMM_LAUNCHER_USM(std::complex<float>, rocblas_ctrmm)
TRMM_LAUNCHER_USM(std::complex<double>, rocblas_ztrmm)

#undef TRMM_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event trsm(Func func, sycl::queue& queue, side left_right, uplo upper_lower,
                        transpose trans, diag unit_diag, int64_t m, int64_t n, T alpha, const T* a,
                        int64_t lda, T* b, int64_t ldb,
                        const std::vector<sycl::event>& dependencies) {
    auto new_side = left_right == oneapi::math::side::left ? oneapi::math::side::right
                                                           : oneapi::math::side::left;
    auto new_uplo = upper_lower == oneapi::math::uplo::lower ? oneapi::math::uplo::upper
                                                             : oneapi::math::uplo::lower;

    return column_major::trsm(func, queue, new_side, new_uplo, trans, unit_diag, n, m, alpha, a,
                              lda, b, ldb, dependencies);
}

#define TRSM_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                   \
    sycl::event trsm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,       \
                     diag unit_diag, int64_t m, int64_t n, TYPE alpha, const TYPE* a, int64_t lda, \
                     TYPE* b, int64_t ldb, const std::vector<sycl::event>& dependencies) {         \
        return trsm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, trans, unit_diag, m, n,       \
                    alpha, a, lda, b, ldb, dependencies);                                          \
    }

TRSM_LAUNCHER_USM(float, rocblas_strsm)
TRSM_LAUNCHER_USM(double, rocblas_dtrsm)
TRSM_LAUNCHER_USM(std::complex<float>, rocblas_ctrsm)
TRSM_LAUNCHER_USM(std::complex<double>, rocblas_ztrsm)

#undef TRSM_LAUNCHER_USM

} // namespace row_major
} // namespace rocblas
} // namespace blas
} // namespace math
} // namespace oneapi
