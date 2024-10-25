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
#include "cublas_helper.hpp"
#include "cublas_task.hpp"
#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/blas/detail/cublas/onemkl_blas_cublas.hpp"

namespace oneapi {
namespace mkl {
namespace blas {
namespace cublas {
namespace column_major {

// Buffer APIs

void gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc, int64_t m,
               int64_t n, int64_t k, float alpha, sycl::buffer<int8_t, 1>& a, int64_t lda,
               int8_t ao, sycl::buffer<int8_t, 1>& b, int64_t ldb, int8_t bo, float beta,
               sycl::buffer<int32_t, 1>& c, int64_t ldc, sycl::buffer<int32_t, 1>& co) {
    throw unimplemented("blas", "gemm_bias", "for column_major layout");
}

void gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc, int64_t m,
               int64_t n, int64_t k, float alpha, sycl::buffer<int8_t, 1>& a, int64_t lda,
               int8_t ao, sycl::buffer<uint8_t, 1>& b, int64_t ldb, uint8_t bo, float beta,
               sycl::buffer<int32_t, 1>& c, int64_t ldc, sycl::buffer<int32_t, 1>& co) {
    throw unimplemented("blas", "gemm_bias", "for column_major layout");
}

void gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc, int64_t m,
               int64_t n, int64_t k, float alpha, sycl::buffer<uint8_t, 1>& a, int64_t lda,
               uint8_t ao, sycl::buffer<int8_t, 1>& b, int64_t ldb, int8_t bo, float beta,
               sycl::buffer<int32_t, 1>& c, int64_t ldc, sycl::buffer<int32_t, 1>& co) {
    throw unimplemented("blas", "gemm_bias", "for column_major layout");
}

void gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc, int64_t m,
               int64_t n, int64_t k, float alpha, sycl::buffer<uint8_t, 1>& a, int64_t lda,
               uint8_t ao, sycl::buffer<uint8_t, 1>& b, int64_t ldb, uint8_t bo, float beta,
               sycl::buffer<int32_t, 1>& c, int64_t ldc, sycl::buffer<int32_t, 1>& co) {
    throw unimplemented("blas", "gemm_bias", "for column_major layout");
}

void gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, float alpha, sycl::buffer<float, 1>& a, int64_t lda,
           sycl::buffer<float, 1>& b, int64_t ldb, float beta, sycl::buffer<float, 1>& c,
           int64_t ldc) {
    throw unimplemented("blas", "gemmt", "for column_major layout");
}

void gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, double alpha, sycl::buffer<double, 1>& a, int64_t lda,
           sycl::buffer<double, 1>& b, int64_t ldb, double beta, sycl::buffer<double, 1>& c,
           int64_t ldc) {
    throw unimplemented("blas", "gemmt", "for column_major layout");
}

void gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
           int64_t lda, sycl::buffer<std::complex<float>, 1>& b, int64_t ldb,
           std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& c, int64_t ldc) {
    throw unimplemented("blas", "gemmt", "for column_major layout");
}

void gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
           int64_t lda, sycl::buffer<std::complex<double>, 1>& b, int64_t ldb,
           std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& c, int64_t ldc) {
    throw unimplemented("blas", "gemmt", "for column_major layout");
}

template <typename Func, typename T>
void omatcopy(const char* func_name, Func func, sycl::queue& queue, transpose trans, int64_t m,
              int64_t n, T alpha, sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& b,
              int64_t ldb) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read_write>(cgh);
        const int64_t logical_m = (trans == oneapi::mkl::transpose::nontrans ? m : n);
        const int64_t logical_n = (trans == oneapi::mkl::transpose::nontrans ? n : m);
        onemkl_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto b_ = sc.get_mem<cuDataType*>(b_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC_T_SYNC(func_name, func, err, handle, get_cublas_operation(trans),
                                     get_cublas_operation(trans), logical_m, logical_n,
                                     (cuDataType*)&alpha, a_, lda, nullptr, nullptr, lda, b_, ldb);
        });
    });
}

#define OMATCOPY_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                   \
    void omatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, TYPE alpha,          \
                  sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& b, int64_t ldb) { \
        omatcopy(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, trans, m, n, alpha, a, lda, b, ldb);     \
    }

OMATCOPY_LAUNCHER(float, cublasSgeam)
OMATCOPY_LAUNCHER(double, cublasDgeam)
OMATCOPY_LAUNCHER(std::complex<float>, cublasCgeam)
OMATCOPY_LAUNCHER(std::complex<double>, cublasZgeam)

#undef OMATCOPY_LAUNCHER

template <typename Func, typename T>
void omatcopy2(const char* func_name, Func func, sycl::queue& queue, transpose trans, int64_t m,
               int64_t n, T alpha, sycl::buffer<T, 1>& a, int64_t lda, std::int64_t stridea,
               sycl::buffer<T, 1>& b, int64_t ldb, std::int64_t strideb) {
    throw unimplemented("blas", "omatcopy2", "");
}

#define OMATCOPY2_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                  \
    void omatcopy2(sycl::queue& queue, transpose trans, int64_t m, int64_t n, TYPE alpha,         \
                   sycl::buffer<TYPE, 1>& a, int64_t lda, int64_t stridea,                        \
                   sycl::buffer<TYPE, 1>& b, int64_t ldb, int64_t strideb) {                      \
        omatcopy2(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, trans, m, n, alpha, a, stridea, lda, b, \
                  ldb, strideb);                                                                  \
    }

OMATCOPY2_LAUNCHER(float, "unimplemented")
OMATCOPY2_LAUNCHER(double, "unimplemented")
OMATCOPY2_LAUNCHER(std::complex<float>, "unimplemented")
OMATCOPY2_LAUNCHER(std::complex<double>, "unimplemented")
#undef OMATCOPY2_LAUNCHER

void imatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
              sycl::buffer<float, 1>& ab, int64_t lda, int64_t ldb) {
    throw unimplemented("blas", "imatcopy", "for column_major layout");
}

void imatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
              sycl::buffer<double, 1>& ab, int64_t lda, int64_t ldb) {
    throw unimplemented("blas", "imatcopy", "for column_major layout");
}

void imatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, std::complex<float> alpha,
              sycl::buffer<std::complex<float>, 1>& ab, int64_t lda, int64_t ldb) {
    throw unimplemented("blas", "imatcopy", "for column_major layout");
}

void imatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, std::complex<double> alpha,
              sycl::buffer<std::complex<double>, 1>& ab, int64_t lda, int64_t ldb) {
    throw unimplemented("blas", "imatcopy", "for column_major layout");
}

template <typename Func, typename T>
void omatadd(const char* func_name, Func func, sycl::queue& queue, transpose transa,
             transpose transb, int64_t m, int64_t n, T alpha, sycl::buffer<T, 1>& a, int64_t lda,
             T beta, sycl::buffer<T, 1>& b, int64_t ldb, sycl::buffer<T, 1>& c, int64_t ldc) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb, ldc);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto b_ = sc.get_mem<cuDataType*>(b_acc);
            auto c_ = sc.get_mem<cuDataType*>(c_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC_T_SYNC(func_name, func, err, handle, get_cublas_operation(transa),
                                     get_cublas_operation(transb), m, n, (cuDataType*)&alpha, a_,
                                     lda, (cuDataType*)&beta, b_, ldb, c_, ldc);
        });
    });
}

#define OMATADD_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                     \
    void omatadd(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,     \
                 TYPE alpha, sycl::buffer<TYPE, 1>& a, int64_t lda, TYPE beta,                     \
                 sycl::buffer<TYPE, 1>& b, int64_t ldb, sycl::buffer<TYPE, 1>& c, int64_t ldc) {   \
        omatadd(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, transa, transb, m, n, alpha, a, lda, beta, \
                b, ldb, c, ldc);                                                                   \
    }

OMATADD_LAUNCHER(float, cublasSgeam)
OMATADD_LAUNCHER(double, cublasDgeam)
OMATADD_LAUNCHER(std::complex<float>, cublasCgeam)
OMATADD_LAUNCHER(std::complex<double>, cublasZgeam)

#undef OMATADD_LAUNCHER

// USM APIs

sycl::event gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc,
                      int64_t m, int64_t n, int64_t k, float alpha, const int8_t* a, int64_t lda,
                      int8_t ao, const int8_t* b, int64_t ldb, int8_t bo, float beta, int32_t* c,
                      int64_t ldc, const int32_t* co,
                      const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm_bias", "for column_major layout");
}

sycl::event gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc,
                      int64_t m, int64_t n, int64_t k, float alpha, const int8_t* a, int64_t lda,
                      int8_t ao, const uint8_t* b, int64_t ldb, uint8_t bo, float beta, int32_t* c,
                      int64_t ldc, const int32_t* co,
                      const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm_bias", "for column_major layout");
}

sycl::event gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc,
                      int64_t m, int64_t n, int64_t k, float alpha, const uint8_t* a, int64_t lda,
                      uint8_t ao, const int8_t* b, int64_t ldb, int8_t bo, float beta, int32_t* c,
                      int64_t ldc, const int32_t* co,
                      const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm_bias", "for column_major layout");
}

sycl::event gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc,
                      int64_t m, int64_t n, int64_t k, float alpha, const uint8_t* a, int64_t lda,
                      uint8_t ao, const uint8_t* b, int64_t ldb, uint8_t bo, float beta, int32_t* c,
                      int64_t ldc, const int32_t* co,
                      const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm_bias", "for column_major layout");
}

sycl::event gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb,
                  int64_t n, int64_t k, float alpha, const float* a, int64_t lda, const float* b,
                  int64_t ldb, float beta, float* c, int64_t ldc,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemmt", "for column_major layout");
}

sycl::event gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb,
                  int64_t n, int64_t k, double alpha, const double* a, int64_t lda, const double* b,
                  int64_t ldb, double beta, double* c, int64_t ldc,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemmt", "for column_major layout");
}

sycl::event gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb,
                  int64_t n, int64_t k, std::complex<float> alpha, const std::complex<float>* a,
                  int64_t lda, const std::complex<float>* b, int64_t ldb, std::complex<float> beta,
                  std::complex<float>* c, int64_t ldc,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemmt", "for column_major layout");
}

sycl::event gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb,
                  int64_t n, int64_t k, std::complex<double> alpha, const std::complex<double>* a,
                  int64_t lda, const std::complex<double>* b, int64_t ldb,
                  std::complex<double> beta, std::complex<double>* c, int64_t ldc,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemmt", "for column_major layout");
}

template <typename Func, typename T>
sycl::event omatcopy(const char* func_name, Func func, sycl::queue& queue, transpose trans,
                     int64_t m, int64_t n, T alpha, const T* a, int64_t lda, T* b, int64_t ldb,
                     const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb);
    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        const int64_t logical_m = (trans == oneapi::mkl::transpose::nontrans ? m : n);
        const int64_t logical_n = (trans == oneapi::mkl::transpose::nontrans ? n : m);
        onemkl_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<const cuDataType*>(a);
            auto b_ = reinterpret_cast<cuDataType*>(b);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC_T_SYNC(func_name, func, err, handle, get_cublas_operation(trans),
                                     get_cublas_operation(trans), logical_m, logical_n,
                                     (cuDataType*)&alpha, a_, lda, nullptr, nullptr, lda, b_, ldb);
        });
    });
    return done;
}

#define OMATCOPY_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                             \
    sycl::event omatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, TYPE alpha, \
                         const TYPE* a, int64_t lda, TYPE* b, int64_t ldb,                      \
                         const std::vector<sycl::event>& dependencies) {                        \
        return omatcopy(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, trans, m, n, alpha, a, lda, b,  \
                        ldb, dependencies);                                                     \
    }

OMATCOPY_LAUNCHER_USM(float, cublasSgeam)
OMATCOPY_LAUNCHER_USM(double, cublasDgeam)
OMATCOPY_LAUNCHER_USM(std::complex<float>, cublasCgeam)
OMATCOPY_LAUNCHER_USM(std::complex<double>, cublasZgeam)

#undef OMATCOPY_LAUNCHER_USM

template <typename Func, typename T>
sycl::event omatcopy2(const char* func_name, Func func, sycl::queue& queue, transpose trans,
                      int64_t m, int64_t n, T alpha, const T* a, int64_t lda, int64_t stridea, T* b,
                      int64_t ldb, int64_t strideb, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatcopy2", "");
}

#define OMATCOPY2_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                             \
    sycl::event omatcopy2(sycl::queue& queue, transpose trans, int64_t m, int64_t n, TYPE alpha, \
                          const TYPE* a, int64_t lda, int64_t stridea, TYPE* b, int64_t ldb,     \
                          int64_t strideb, const std::vector<sycl::event>& dependencies) {       \
        return omatcopy2(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, trans, m, n, alpha, a, stridea, \
                         lda, b, ldb, strideb, dependencies);                                    \
    }

OMATCOPY2_LAUNCHER_USM(float, "unimplemented")
OMATCOPY2_LAUNCHER_USM(double, "unimplemented")
OMATCOPY2_LAUNCHER_USM(std::complex<float>, "unimplemented")
OMATCOPY2_LAUNCHER_USM(std::complex<double>, "unimplemented")
#undef OMATCOPY2_LAUNCHER_USM

sycl::event imatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                     float* ab, int64_t lda, int64_t ldb,
                     const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "imatcopy", "for column_major layout");
}

sycl::event imatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                     double* ab, int64_t lda, int64_t ldb,
                     const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "imatcopy", "for column_major layout");
}

sycl::event imatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                     std::complex<float> alpha, std::complex<float>* ab, int64_t lda, int64_t ldb,
                     const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "imatcopy", "for column_major layout");
}

sycl::event imatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                     std::complex<double> alpha, std::complex<double>* ab, int64_t lda, int64_t ldb,
                     const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "imatcopy", "for column_major layout");
}

template <typename Func, typename T>
inline sycl::event omatadd(const char* func_name, Func func, sycl::queue& queue, transpose transa,
                           transpose transb, int64_t m, int64_t n, T alpha, const T* a, int64_t lda,
                           T beta, const T* b, int64_t ldb, T* c, int64_t ldc,
                           const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb, ldc);
    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemkl_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<const cuDataType*>(a);
            auto b_ = reinterpret_cast<const cuDataType*>(b);
            auto c_ = reinterpret_cast<cuDataType*>(c);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC_T_SYNC(func_name, func, err, handle, get_cublas_operation(transa),
                                     get_cublas_operation(transb), m, n, (cuDataType*)&alpha, a_,
                                     lda, (cuDataType*)&beta, b_, ldb, c_, ldc);
        });
    });
    return done;
}

#define OMATADD_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                             \
    sycl::event omatadd(sycl::queue& queue, transpose transa, transpose transb, int64_t m,     \
                        int64_t n, TYPE alpha, const TYPE* a, int64_t lda, TYPE beta,          \
                        const TYPE* b, int64_t ldb, TYPE* c, int64_t ldc,                      \
                        const std::vector<sycl::event>& dependencies) {                        \
        return omatadd(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, transa, transb, m, n, alpha, a, \
                       lda, beta, b, ldb, c, ldc, dependencies);                               \
    }

OMATADD_LAUNCHER_USM(float, cublasSgeam)
OMATADD_LAUNCHER_USM(double, cublasDgeam)
OMATADD_LAUNCHER_USM(std::complex<float>, cublasCgeam)
OMATADD_LAUNCHER_USM(std::complex<double>, cublasZgeam)

#undef OMATADD_LAUNCHER_USM

} // namespace column_major

namespace row_major {

// Buffer APIs

void gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc, int64_t m,
               int64_t n, int64_t k, float alpha, sycl::buffer<int8_t, 1>& a, int64_t lda,
               int8_t ao, sycl::buffer<int8_t, 1>& b, int64_t ldb, int8_t bo, float beta,
               sycl::buffer<int32_t, 1>& c, int64_t ldc, sycl::buffer<int32_t, 1>& co) {
    throw unimplemented("blas", "gemm_bias", "for row_major layout");
}

void gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc, int64_t m,
               int64_t n, int64_t k, float alpha, sycl::buffer<int8_t, 1>& a, int64_t lda,
               int8_t ao, sycl::buffer<uint8_t, 1>& b, int64_t ldb, uint8_t bo, float beta,
               sycl::buffer<int32_t, 1>& c, int64_t ldc, sycl::buffer<int32_t, 1>& co) {
    throw unimplemented("blas", "gemm_bias", "for row_major layout");
}

void gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc, int64_t m,
               int64_t n, int64_t k, float alpha, sycl::buffer<uint8_t, 1>& a, int64_t lda,
               uint8_t ao, sycl::buffer<int8_t, 1>& b, int64_t ldb, int8_t bo, float beta,
               sycl::buffer<int32_t, 1>& c, int64_t ldc, sycl::buffer<int32_t, 1>& co) {
    throw unimplemented("blas", "gemm_bias", "for row_major layout");
}

void gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc, int64_t m,
               int64_t n, int64_t k, float alpha, sycl::buffer<uint8_t, 1>& a, int64_t lda,
               uint8_t ao, sycl::buffer<uint8_t, 1>& b, int64_t ldb, uint8_t bo, float beta,
               sycl::buffer<int32_t, 1>& c, int64_t ldc, sycl::buffer<int32_t, 1>& co) {
    throw unimplemented("blas", "gemm_bias", "for row_major layout");
}

void gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, float alpha, sycl::buffer<float, 1>& a, int64_t lda,
           sycl::buffer<float, 1>& b, int64_t ldb, float beta, sycl::buffer<float, 1>& c,
           int64_t ldc) {
    throw unimplemented("blas", "gemmt", "for row_major layout");
}

void gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, double alpha, sycl::buffer<double, 1>& a, int64_t lda,
           sycl::buffer<double, 1>& b, int64_t ldb, double beta, sycl::buffer<double, 1>& c,
           int64_t ldc) {
    throw unimplemented("blas", "gemmt", "for row_major layout");
}

void gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
           int64_t lda, sycl::buffer<std::complex<float>, 1>& b, int64_t ldb,
           std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& c, int64_t ldc) {
    throw unimplemented("blas", "gemmt", "for row_major layout");
}

void gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
           int64_t lda, sycl::buffer<std::complex<double>, 1>& b, int64_t ldb,
           std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& c, int64_t ldc) {
    throw unimplemented("blas", "gemmt", "for row_major layout");
}

template <typename Func, typename T>
void omatcopy(const char* func_name, Func func, sycl::queue& queue, transpose trans, int64_t m,
              int64_t n, T alpha, sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& b,
              int64_t ldb) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read_write>(cgh);
        const int64_t logical_m = (trans == oneapi::mkl::transpose::nontrans ? n : m);
        const int64_t logical_n = (trans == oneapi::mkl::transpose::nontrans ? m : n);
        onemkl_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto b_ = sc.get_mem<cuDataType*>(b_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC_T_SYNC(func_name, func, err, handle, get_cublas_operation(trans),
                                     get_cublas_operation(trans), logical_m, logical_n,
                                     (cuDataType*)&alpha, a_, lda, nullptr, nullptr, lda, b_, ldb);
        });
    });
}

#define OMATCOPY_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                   \
    void omatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, TYPE alpha,          \
                  sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& b, int64_t ldb) { \
        omatcopy(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, trans, m, n, alpha, a, lda, b, ldb);     \
    }

OMATCOPY_LAUNCHER(float, cublasSgeam)
OMATCOPY_LAUNCHER(double, cublasDgeam)
OMATCOPY_LAUNCHER(std::complex<float>, cublasCgeam)
OMATCOPY_LAUNCHER(std::complex<double>, cublasZgeam)

#undef OMATCOPY_LAUNCHER

template <typename Func, typename T>
void omatcopy2(const char* func_name, Func func, sycl::queue& queue, transpose trans, int64_t m,
               int64_t n, T alpha, sycl::buffer<T, 1>& a, int64_t lda, std::int64_t stridea,
               sycl::buffer<T, 1>& b, int64_t ldb, std::int64_t strideb) {
    throw unimplemented("blas", "omatcopy2", "");
}

#define OMATCOPY2_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                  \
    void omatcopy2(sycl::queue& queue, transpose trans, int64_t m, int64_t n, TYPE alpha,         \
                   sycl::buffer<TYPE, 1>& a, int64_t lda, int64_t stridea,                        \
                   sycl::buffer<TYPE, 1>& b, int64_t ldb, int64_t strideb) {                      \
        omatcopy2(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, trans, m, n, alpha, a, stridea, lda, b, \
                  ldb, strideb);                                                                  \
    }

OMATCOPY2_LAUNCHER(float, "unimplemented")
OMATCOPY2_LAUNCHER(double, "unimplemented")
OMATCOPY2_LAUNCHER(std::complex<float>, "unimplemented")
OMATCOPY2_LAUNCHER(std::complex<double>, "unimplemented")
#undef OMATCOPY2_LAUNCHER

void imatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
              sycl::buffer<float, 1>& ab, int64_t lda, int64_t ldb) {
    throw unimplemented("blas", "imatcopy", "for row_major layout");
}

void imatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
              sycl::buffer<double, 1>& ab, int64_t lda, int64_t ldb) {
    throw unimplemented("blas", "imatcopy", "for row_major layout");
}

void imatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, std::complex<float> alpha,
              sycl::buffer<std::complex<float>, 1>& ab, int64_t lda, int64_t ldb) {
    throw unimplemented("blas", "imatcopy", "for row_major layout");
}

void imatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, std::complex<double> alpha,
              sycl::buffer<std::complex<double>, 1>& ab, int64_t lda, int64_t ldb) {
    throw unimplemented("blas", "imatcopy", "for row_major layout");
}

template <typename Func, typename T>
void omatadd(const char* func_name, Func func, sycl::queue& queue, transpose transa,
             transpose transb, int64_t m, int64_t n, T alpha, sycl::buffer<T, 1>& a, int64_t lda,
             T beta, sycl::buffer<T, 1>& b, int64_t ldb, sycl::buffer<T, 1>& c, int64_t ldc) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb, ldc);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto b_ = sc.get_mem<cuDataType*>(b_acc);
            auto c_ = sc.get_mem<cuDataType*>(c_acc);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC_T_SYNC(func_name, func, err, handle, get_cublas_operation(transa),
                                     get_cublas_operation(transb), n, m, (cuDataType*)&alpha, a_,
                                     lda, (cuDataType*)&beta, b_, ldb, c_, ldc);
        });
    });
}

#define OMATADD_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                     \
    void omatadd(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,     \
                 TYPE alpha, sycl::buffer<TYPE, 1>& a, int64_t lda, TYPE beta,                     \
                 sycl::buffer<TYPE, 1>& b, int64_t ldb, sycl::buffer<TYPE, 1>& c, int64_t ldc) {   \
        omatadd(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, transa, transb, m, n, alpha, a, lda, beta, \
                b, ldb, c, ldc);                                                                   \
    }

OMATADD_LAUNCHER(float, cublasSgeam)
OMATADD_LAUNCHER(double, cublasDgeam)
OMATADD_LAUNCHER(std::complex<float>, cublasCgeam)
OMATADD_LAUNCHER(std::complex<double>, cublasZgeam)

#undef OMATADD_LAUNCHER

// USM APIs

sycl::event gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc,
                      int64_t m, int64_t n, int64_t k, float alpha, const int8_t* a, int64_t lda,
                      int8_t ao, const int8_t* b, int64_t ldb, int8_t bo, float beta, int32_t* c,
                      int64_t ldc, const int32_t* co,
                      const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm_bias", "for row_major layout");
}

sycl::event gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc,
                      int64_t m, int64_t n, int64_t k, float alpha, const int8_t* a, int64_t lda,
                      int8_t ao, const uint8_t* b, int64_t ldb, uint8_t bo, float beta, int32_t* c,
                      int64_t ldc, const int32_t* co,
                      const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm_bias", "for row_major layout");
}

sycl::event gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc,
                      int64_t m, int64_t n, int64_t k, float alpha, const uint8_t* a, int64_t lda,
                      uint8_t ao, const int8_t* b, int64_t ldb, int8_t bo, float beta, int32_t* c,
                      int64_t ldc, const int32_t* co,
                      const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm_bias", "for row_major layout");
}

sycl::event gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc,
                      int64_t m, int64_t n, int64_t k, float alpha, const uint8_t* a, int64_t lda,
                      uint8_t ao, const uint8_t* b, int64_t ldb, uint8_t bo, float beta, int32_t* c,
                      int64_t ldc, const int32_t* co,
                      const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm_bias", "for row_major layout");
}

sycl::event gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb,
                  int64_t n, int64_t k, float alpha, const float* a, int64_t lda, const float* b,
                  int64_t ldb, float beta, float* c, int64_t ldc,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemmt", "for row_major layout");
}

sycl::event gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb,
                  int64_t n, int64_t k, double alpha, const double* a, int64_t lda, const double* b,
                  int64_t ldb, double beta, double* c, int64_t ldc,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemmt", "for row_major layout");
}

sycl::event gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb,
                  int64_t n, int64_t k, std::complex<float> alpha, const std::complex<float>* a,
                  int64_t lda, const std::complex<float>* b, int64_t ldb, std::complex<float> beta,
                  std::complex<float>* c, int64_t ldc,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemmt", "for row_major layout");
}

sycl::event gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb,
                  int64_t n, int64_t k, std::complex<double> alpha, const std::complex<double>* a,
                  int64_t lda, const std::complex<double>* b, int64_t ldb,
                  std::complex<double> beta, std::complex<double>* c, int64_t ldc,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemmt", "for row_major layout");
}

template <typename Func, typename T>
sycl::event omatcopy(const char* func_name, Func func, sycl::queue& queue, transpose trans,
                     int64_t m, int64_t n, T alpha, const T* a, int64_t lda, T* b, int64_t ldb,
                     const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb);
    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        const int64_t logical_m = (trans == oneapi::mkl::transpose::nontrans ? n : m);
        const int64_t logical_n = (trans == oneapi::mkl::transpose::nontrans ? m : n);
        onemkl_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<const cuDataType*>(a);
            auto b_ = reinterpret_cast<cuDataType*>(b);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC_T_SYNC(func_name, func, err, handle, get_cublas_operation(trans),
                                     get_cublas_operation(trans), logical_m, logical_n,
                                     (cuDataType*)&alpha, a_, lda, nullptr, nullptr, ldb, b_, ldb);
        });
    });
    return done;
}

#define OMATCOPY_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                             \
    sycl::event omatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, TYPE alpha, \
                         const TYPE* a, int64_t lda, TYPE* b, int64_t ldb,                      \
                         const std::vector<sycl::event>& dependencies) {                        \
        return omatcopy(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, trans, m, n, alpha, a, lda, b,  \
                        ldb, dependencies);                                                     \
    }

OMATCOPY_LAUNCHER_USM(float, cublasSgeam)
OMATCOPY_LAUNCHER_USM(double, cublasDgeam)
OMATCOPY_LAUNCHER_USM(std::complex<float>, cublasCgeam)
OMATCOPY_LAUNCHER_USM(std::complex<double>, cublasZgeam)

#undef OMATCOPY_LAUNCHER_USM

template <typename Func, typename T>
sycl::event omatcopy2(const char* func_name, Func func, sycl::queue& queue, transpose trans,
                      int64_t m, int64_t n, T alpha, const T* a, int64_t lda, int64_t stridea, T* b,
                      int64_t ldb, int64_t strideb, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatcopy2", "");
}

#define OMATCOPY2_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                             \
    sycl::event omatcopy2(sycl::queue& queue, transpose trans, int64_t m, int64_t n, TYPE alpha, \
                          const TYPE* a, int64_t lda, int64_t stridea, TYPE* b, int64_t ldb,     \
                          int64_t strideb, const std::vector<sycl::event>& dependencies) {       \
        return omatcopy2(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, trans, m, n, alpha, a, stridea, \
                         lda, b, ldb, strideb, dependencies);                                    \
    }

OMATCOPY2_LAUNCHER_USM(float, "unimplemented")
OMATCOPY2_LAUNCHER_USM(double, "unimplemented")
OMATCOPY2_LAUNCHER_USM(std::complex<float>, "unimplemented")
OMATCOPY2_LAUNCHER_USM(std::complex<double>, "unimplemented")
#undef OMATCOPY2_LAUNCHER_USM

sycl::event imatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                     float* ab, int64_t lda, int64_t ldb,
                     const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "imatcopy", "for row_major layout");
}

sycl::event imatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                     double* ab, int64_t lda, int64_t ldb,
                     const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "imatcopy", "for row_major layout");
}

sycl::event imatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                     std::complex<float> alpha, std::complex<float>* ab, int64_t lda, int64_t ldb,
                     const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "imatcopy", "for row_major layout");
}

sycl::event imatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                     std::complex<double> alpha, std::complex<double>* ab, int64_t lda, int64_t ldb,
                     const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "imatcopy", "for row_major layout");
}

template <typename Func, typename T>
inline sycl::event omatadd(const char* func_name, Func func, sycl::queue& queue, transpose transa,
                           transpose transb, int64_t m, int64_t n, T alpha, const T* a, int64_t lda,
                           T beta, const T* b, int64_t ldb, T* c, int64_t ldc,
                           const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb, ldc);
    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemkl_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<const cuDataType*>(a);
            auto b_ = reinterpret_cast<const cuDataType*>(b);
            auto c_ = reinterpret_cast<cuDataType*>(c);
            cublasStatus_t err;
            CUBLAS_ERROR_FUNC_T_SYNC(func_name, func, err, handle, get_cublas_operation(transa),
                                     get_cublas_operation(transb), n, m, (cuDataType*)&alpha, a_,
                                     lda, (cuDataType*)&beta, b_, ldb, c_, ldc);
        });
    });
    return done;
}

#define OMATADD_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                             \
    sycl::event omatadd(sycl::queue& queue, transpose transa, transpose transb, int64_t m,     \
                        int64_t n, TYPE alpha, const TYPE* a, int64_t lda, TYPE beta,          \
                        const TYPE* b, int64_t ldb, TYPE* c, int64_t ldc,                      \
                        const std::vector<sycl::event>& dependencies) {                        \
        return omatadd(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, transa, transb, m, n, alpha, a, \
                       lda, beta, b, ldb, c, ldc, dependencies);                               \
    }

OMATADD_LAUNCHER_USM(float, cublasSgeam)
OMATADD_LAUNCHER_USM(double, cublasDgeam)
OMATADD_LAUNCHER_USM(std::complex<float>, cublasCgeam)
OMATADD_LAUNCHER_USM(std::complex<double>, cublasZgeam)

#undef OMATADD_LAUNCHER_USM

} // namespace row_major
} // namespace cublas
} // namespace blas
} // namespace mkl
} // namespace oneapi
