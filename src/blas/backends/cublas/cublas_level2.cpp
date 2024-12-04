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

#include "oneapi/math/exceptions.hpp"
#include "oneapi/math/blas/detail/cublas/onemath_blas_cublas.hpp"

namespace oneapi {
namespace math {
namespace blas {
namespace cublas {
namespace column_major {

// Buffer APIs

template <typename Func, typename T>
inline void gemv(const char* func_name, Func func, sycl::queue& queue, transpose trans, int64_t m,
                 int64_t n, T alpha, sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& x,
                 int64_t incx, T beta, sycl::buffer<T, 1>& y, int64_t incy) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, m, lda, incx, incy);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            auto y_ = sc.get_mem<cuDataType*>(y_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle, get_cublas_operation(trans), m,
                                     n, (cuDataType*)&alpha, a_, lda, x_, incx, (cuDataType*)&beta,
                                     y_, incy);
        });
    });
}

#define GEMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                        \
    void gemv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, TYPE alpha,               \
              sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x, int64_t incx,       \
              TYPE beta, sycl::buffer<TYPE, 1>& y, int64_t incy) {                                 \
        gemv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, trans, m, n, alpha, a, lda, x, incx, beta, y, \
             incy);                                                                                \
    }

GEMV_LAUNCHER(float, cublasSgemv)
GEMV_LAUNCHER(double, cublasDgemv)
GEMV_LAUNCHER(std::complex<float>, cublasCgemv)
GEMV_LAUNCHER(std::complex<double>, cublasZgemv)
#undef GEMV_LAUNCHER

template <typename Func, typename T>
inline void gbmv(const char* func_name, Func func, sycl::queue& queue, transpose trans, int64_t m,
                 int64_t n, int64_t kl, int64_t ku, T alpha, sycl::buffer<T, 1>& a, int64_t lda,
                 sycl::buffer<T, 1>& x, int64_t incx, T beta, sycl::buffer<T, 1>& y, int64_t incy) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, m, lda, kl, ku, incx, incy);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            auto y_ = sc.get_mem<cuDataType*>(y_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle, get_cublas_operation(trans), m,
                                     n, kl, ku, (cuDataType*)&alpha, a_, lda, x_, incx,
                                     (cuDataType*)&beta, y_, incy);
        });
    });
}

#define GBMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                       \
    void gbmv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,  \
              TYPE alpha, sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x,        \
              int64_t incx, TYPE beta, sycl::buffer<TYPE, 1>& y, int64_t incy) {                  \
        gbmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, \
             beta, y, incy);                                                                      \
    }

GBMV_LAUNCHER(float, cublasSgbmv)
GBMV_LAUNCHER(double, cublasDgbmv)
GBMV_LAUNCHER(std::complex<float>, cublasCgbmv)
GBMV_LAUNCHER(std::complex<double>, cublasZgbmv)
#undef GBMV_LAUNCHER

template <typename Func, typename T>
inline void ger(const char* func_name, Func func, sycl::queue& queue, int64_t m, int64_t n, T alpha,
                sycl::buffer<T, 1>& x, int64_t incx, sycl::buffer<T, 1>& y, int64_t incy,
                sycl::buffer<T, 1>& a, int64_t lda) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, m, lda, incx, incy);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            auto y_ = sc.get_mem<cuDataType*>(y_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle, m, n, (cuDataType*)&alpha, x_,
                                     incx, y_, incy, a_, lda);
        });
    });
}

#define GER_LAUNCHER(EXT, TYPE, CUBLAS_ROUTINE)                                                   \
    void ger##EXT(sycl::queue& queue, int64_t m, int64_t n, TYPE alpha, sycl::buffer<TYPE, 1>& x, \
                  int64_t incx, sycl::buffer<TYPE, 1>& y, int64_t incy, sycl::buffer<TYPE, 1>& a, \
                  int64_t lda) {                                                                  \
        ger(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, m, n, alpha, x, incx, y, incy, a, lda);       \
    }

GER_LAUNCHER(, float, cublasSger)
GER_LAUNCHER(, double, cublasDger)
GER_LAUNCHER(u, std::complex<float>, cublasCgeru)
GER_LAUNCHER(u, std::complex<double>, cublasZgeru)
GER_LAUNCHER(c, std::complex<float>, cublasCgerc)
GER_LAUNCHER(c, std::complex<double>, cublasZgerc)
#undef GER_LAUNCHER

template <typename Func, typename T>
inline void hbmv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower, int64_t n,
                 int64_t k, T alpha, sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& x,
                 int64_t incx, T beta, sycl::buffer<T, 1>& y, int64_t incy) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, k, lda, incx, incy);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            auto y_ = sc.get_mem<cuDataType*>(y_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), n, k, (cuDataType*)&alpha,
                                     a_, lda, x_, incx, (cuDataType*)&beta, y_, incy);
        });
    });
}

#define HBMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                     \
    void hbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k, TYPE alpha,           \
              sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x, int64_t incx,    \
              TYPE beta, sycl::buffer<TYPE, 1>& y, int64_t incy) {                              \
        hbmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, k, alpha, a, lda, x, incx, \
             beta, y, incy);                                                                    \
    }

HBMV_LAUNCHER(std::complex<float>, cublasChbmv)
HBMV_LAUNCHER(std::complex<double>, cublasZhbmv)
#undef HBMV_LAUNCHER

template <typename Func, typename T>
inline void hemv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower, int64_t n,
                 T alpha, sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& x, int64_t incx,
                 T beta, sycl::buffer<T, 1>& y, int64_t incy) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, incx, incy);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            auto y_ = sc.get_mem<cuDataType*>(y_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), n, (cuDataType*)&alpha, a_,
                                     lda, x_, incx, (cuDataType*)&beta, y_, incy);
        });
    });
}

#define HEMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                        \
    void hemv(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,                         \
              sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x, int64_t incx,       \
              TYPE beta, sycl::buffer<TYPE, 1>& y, int64_t incy) {                                 \
        hemv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, a, lda, x, incx, beta, \
             y, incy);                                                                             \
    }

HEMV_LAUNCHER(std::complex<float>, cublasChemv)
HEMV_LAUNCHER(std::complex<double>, cublasZhemv)
#undef HEMV_LAUNCHER

template <typename Func, typename ScalarType, typename DataType>
inline void her(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower, int64_t n,
                ScalarType alpha, sycl::buffer<DataType, 1>& x, int64_t incx,
                sycl::buffer<DataType, 1>& a, int64_t lda) {
    using cuScalarType = typename CudaEquivalentType<ScalarType>::Type;
    using cuDataType = typename CudaEquivalentType<DataType>::Type;
    overflow_check(n, lda, incx);

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), n, (cuScalarType*)&alpha,
                                     x_, incx, a_, lda);
        });
    });
}

#define HER_LAUNCHER(SCALAR_TYPE, DATA_TYPE, CUBLAS_ROUTINE)                                 \
    void her(sycl::queue& queue, uplo upper_lower, int64_t n, SCALAR_TYPE alpha,             \
             sycl::buffer<DATA_TYPE, 1>& x, int64_t incx, sycl::buffer<DATA_TYPE, 1>& a,     \
             int64_t lda) {                                                                  \
        her(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a, lda); \
    }

HER_LAUNCHER(float, std::complex<float>, cublasCher)
HER_LAUNCHER(double, std::complex<double>, cublasZher)

#undef HER_LAUNCHER

template <typename Func, typename T>
inline void her2(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower, int64_t n,
                 T alpha, sycl::buffer<T, 1>& x, int64_t incx, sycl::buffer<T, 1>& y, int64_t incy,
                 sycl::buffer<T, 1>& a, int64_t lda) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, incx, incy);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            auto y_ = sc.get_mem<cuDataType*>(y_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), n, (cuDataType*)&alpha, x_,
                                     incx, y_, incy, a_, lda);
        });
    });
}

#define HER2_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                      \
    void her2(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,                       \
              sycl::buffer<TYPE, 1>& x, int64_t incx, sycl::buffer<TYPE, 1>& y, int64_t incy,    \
              sycl::buffer<TYPE, 1>& a, int64_t lda) {                                           \
        her2(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y, incy, a, \
             lda);                                                                               \
    }

HER2_LAUNCHER(std::complex<float>, cublasCher2)
HER2_LAUNCHER(std::complex<double>, cublasZher2)

#undef HER2_LAUNCHER

template <typename Func, typename T>
inline void hpmv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower, int64_t n,
                 T alpha, sycl::buffer<T, 1>& a, sycl::buffer<T, 1>& x, int64_t incx, T beta,
                 sycl::buffer<T, 1>& y, int64_t incy) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            auto y_ = sc.get_mem<cuDataType*>(y_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), n, (cuDataType*)&alpha, a_,
                                     x_, incx, (cuDataType*)&beta, y_, incy);
        });
    });
}

#define HPMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                      \
    void hpmv(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,                       \
              sycl::buffer<TYPE, 1>& a, sycl::buffer<TYPE, 1>& x, int64_t incx, TYPE beta,       \
              sycl::buffer<TYPE, 1>& y, int64_t incy) {                                          \
        hpmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, a, x, incx, beta, y, \
             incy);                                                                              \
    }

HPMV_LAUNCHER(std::complex<float>, cublasChpmv)
HPMV_LAUNCHER(std::complex<double>, cublasZhpmv)

#undef HPMV_LAUNCHER

template <typename Func, typename ScalarType, typename DataType>
inline void hpr(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower, int64_t n,
                ScalarType alpha, sycl::buffer<DataType, 1>& x, int64_t incx,
                sycl::buffer<DataType, 1>& a) {
    using cuScalarType = typename CudaEquivalentType<ScalarType>::Type;
    using cuDataType = typename CudaEquivalentType<DataType>::Type;
    overflow_check(n, incx);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), n, (cuScalarType*)&alpha,
                                     x_, incx, a_);
        });
    });
}

#define HPR_LAUNCHER(SCALAR_TYPE, DATA_TYPE, CUBLAS_ROUTINE)                               \
    void hpr(sycl::queue& queue, uplo upper_lower, int64_t n, SCALAR_TYPE alpha,           \
             sycl::buffer<DATA_TYPE, 1>& x, int64_t incx, sycl::buffer<DATA_TYPE, 1>& a) { \
        hpr(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a);    \
    }

HPR_LAUNCHER(float, std::complex<float>, cublasChpr)
HPR_LAUNCHER(double, std::complex<double>, cublasZhpr)

#undef HPR_LAUNCHER

template <typename Func, typename T>
inline void hpr2(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower, int64_t n,
                 T alpha, sycl::buffer<T, 1>& x, int64_t incx, sycl::buffer<T, 1>& y, int64_t incy,
                 sycl::buffer<T, 1>& a) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            auto y_ = sc.get_mem<cuDataType*>(y_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), n, (cuDataType*)&alpha, x_,
                                     incx, y_, incy, a_);
        });
    });
}

#define HPR2_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                       \
    void hpr2(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,                        \
              sycl::buffer<TYPE, 1>& x, int64_t incx, sycl::buffer<TYPE, 1>& y, int64_t incy,     \
              sycl::buffer<TYPE, 1>& a) {                                                         \
        hpr2(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y, incy, a); \
    }

HPR2_LAUNCHER(std::complex<float>, cublasChpr2)
HPR2_LAUNCHER(std::complex<double>, cublasZhpr2)

#undef HPR2_LAUNCHER

template <typename Func, typename T>
inline void sbmv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower, int64_t n,
                 int64_t k, T alpha, sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& x,
                 int64_t incx, T beta, sycl::buffer<T, 1>& y, int64_t incy) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, k, lda, incx, incy);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            auto y_ = sc.get_mem<cuDataType*>(y_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), n, k, (cuDataType*)&alpha,
                                     a_, lda, x_, incx, (cuDataType*)&beta, y_, incy);
        });
    });
}

#define SBMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                     \
    void sbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k, TYPE alpha,           \
              sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x, int64_t incx,    \
              TYPE beta, sycl::buffer<TYPE, 1>& y, int64_t incy) {                              \
        sbmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, k, alpha, a, lda, x, incx, \
             beta, y, incy);                                                                    \
    }

SBMV_LAUNCHER(float, cublasSsbmv)
SBMV_LAUNCHER(double, cublasDsbmv)

#undef SBMV_LAUNCHER

template <typename Func, typename T>
inline void symv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower, int64_t n,
                 T alpha, sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& x, int64_t incx,
                 T beta, sycl::buffer<T, 1>& y, int64_t incy) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, incx, incy);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            auto y_ = sc.get_mem<cuDataType*>(y_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), n, (cuDataType*)&alpha, a_,
                                     lda, x_, incx, (cuDataType*)&beta, y_, incy);
        });
    });
}

#define SYMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                        \
    void symv(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,                         \
              sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x, int64_t incx,       \
              TYPE beta, sycl::buffer<TYPE, 1>& y, int64_t incy) {                                 \
        symv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, a, lda, x, incx, beta, \
             y, incy);                                                                             \
    }

SYMV_LAUNCHER(float, cublasSsymv)
SYMV_LAUNCHER(double, cublasDsymv)

#undef SYMV_LAUNCHER

template <typename Func, typename T>
inline void syr(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower, int64_t n,
                T alpha, sycl::buffer<T, 1>& x, int64_t incx, sycl::buffer<T, 1>& a, int64_t lda) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, incx);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), n, (cuDataType*)&alpha, x_,
                                     incx, a_, lda);
        });
    });
}

#define SYR_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                    \
    void syr(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,                     \
             sycl::buffer<TYPE, 1>& x, int64_t incx, sycl::buffer<TYPE, 1>& a, int64_t lda) { \
        syr(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a, lda);  \
    }

SYR_LAUNCHER(float, cublasSsyr)
SYR_LAUNCHER(double, cublasDsyr)
// Intel does not support the following two
SYR_LAUNCHER(std::complex<float>, cublasCsyr)
SYR_LAUNCHER(std::complex<double>, cublasZsyr)
#undef SYR_LAUNCHER

template <typename Func, typename T>
inline void syr2(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower, int64_t n,
                 T alpha, sycl::buffer<T, 1>& x, int64_t incx, sycl::buffer<T, 1>& y, int64_t incy,
                 sycl::buffer<T, 1>& a, int64_t lda) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, incx, incy);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            auto y_ = sc.get_mem<cuDataType*>(y_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), n, (cuDataType*)&alpha, x_,
                                     incx, y_, incy, a_, lda);
        });
    });
}

#define SYR2_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                      \
    void syr2(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,                       \
              sycl::buffer<TYPE, 1>& x, int64_t incx, sycl::buffer<TYPE, 1>& y, int64_t incy,    \
              sycl::buffer<TYPE, 1>& a, int64_t lda) {                                           \
        syr2(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y, incy, a, \
             lda);                                                                               \
    }

SYR2_LAUNCHER(float, cublasSsyr2)
SYR2_LAUNCHER(double, cublasDsyr2)
// Intel does not support the following two
SYR2_LAUNCHER(std::complex<float>, cublasCsyr2)
SYR2_LAUNCHER(std::complex<double>, cublasZsyr2)

#undef SYR2_LAUNCHER

template <typename Func, typename T>
inline void spmv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower, int64_t n,
                 T alpha, sycl::buffer<T, 1>& a, sycl::buffer<T, 1>& x, int64_t incx, T beta,
                 sycl::buffer<T, 1>& y, int64_t incy) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            auto y_ = sc.get_mem<cuDataType*>(y_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), n, (cuDataType*)&alpha, a_,
                                     x_, incx, (cuDataType*)&beta, y_, incy);
        });
    });
}

#define SPMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                      \
    void spmv(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,                       \
              sycl::buffer<TYPE, 1>& a, sycl::buffer<TYPE, 1>& x, int64_t incx, TYPE beta,       \
              sycl::buffer<TYPE, 1>& y, int64_t incy) {                                          \
        spmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, a, x, incx, beta, y, \
             incy);                                                                              \
    }

SPMV_LAUNCHER(float, cublasSspmv)
SPMV_LAUNCHER(double, cublasDspmv)

#undef SPMV_LAUNCHER

template <typename Func, typename T>
inline void spr(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower, int64_t n,
                T alpha, sycl::buffer<T, 1>& x, int64_t incx, sycl::buffer<T, 1>& a) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), n, (cuDataType*)&alpha, x_,
                                     incx, a_);
        });
    });
}

#define SPR_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                              \
    void spr(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,               \
             sycl::buffer<TYPE, 1>& x, int64_t incx, sycl::buffer<TYPE, 1>& a) {        \
        spr(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a); \
    }

SPR_LAUNCHER(float, cublasSspr)
SPR_LAUNCHER(double, cublasDspr)

#undef SPR_LAUNCHER

template <typename Func, typename T>
inline void spr2(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower, int64_t n,
                 T alpha, sycl::buffer<T, 1>& x, int64_t incx, sycl::buffer<T, 1>& y, int64_t incy,
                 sycl::buffer<T, 1>& a) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            auto y_ = sc.get_mem<cuDataType*>(y_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), n, (cuDataType*)&alpha, x_,
                                     incx, y_, incy, a_);
        });
    });
}

#define SPR2_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                       \
    void spr2(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,                        \
              sycl::buffer<TYPE, 1>& x, int64_t incx, sycl::buffer<TYPE, 1>& y, int64_t incy,     \
              sycl::buffer<TYPE, 1>& a) {                                                         \
        spr2(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y, incy, a); \
    }

SPR2_LAUNCHER(float, cublasSspr2)
SPR2_LAUNCHER(double, cublasDspr2)

#undef SPR2_LAUNCHER

template <typename Func, typename T>
inline void tbmv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                 transpose trans, diag unit_diag, int64_t n, int64_t k, sycl::buffer<T, 1>& a,
                 int64_t lda, sycl::buffer<T, 1>& x, int64_t incx) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, k, lda, incx);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), get_cublas_operation(trans),
                                     get_cublas_diag_type(unit_diag), n, k, a_, lda, x_, incx);
        });
    });
}

#define TBMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                       \
    void tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,   \
              int64_t k, sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x,         \
              int64_t incx) {                                                                     \
        tbmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, k, a, lda, \
             x, incx);                                                                            \
    }

TBMV_LAUNCHER(float, cublasStbmv)
TBMV_LAUNCHER(double, cublasDtbmv)
TBMV_LAUNCHER(std::complex<float>, cublasCtbmv)
TBMV_LAUNCHER(std::complex<double>, cublasZtbmv)

#undef TBMV_LAUNCHER

template <typename Func, typename T>
inline void tbsv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                 transpose trans, diag unit_diag, int64_t n, int64_t k, sycl::buffer<T, 1>& a,
                 int64_t lda, sycl::buffer<T, 1>& x, int64_t incx) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, k, lda, incx);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), get_cublas_operation(trans),
                                     get_cublas_diag_type(unit_diag), n, k, a_, lda, x_, incx);
        });
    });
}

#define TBSV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                       \
    void tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,   \
              int64_t k, sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x,         \
              int64_t incx) {                                                                     \
        tbsv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, k, a, lda, \
             x, incx);                                                                            \
    }

TBSV_LAUNCHER(float, cublasStbsv)
TBSV_LAUNCHER(double, cublasDtbsv)
TBSV_LAUNCHER(std::complex<float>, cublasCtbsv)
TBSV_LAUNCHER(std::complex<double>, cublasZtbsv)

#undef TBSV_LAUNCHER

template <typename Func, typename T>
inline void tpmv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                 transpose trans, diag unit_diag, int64_t n, sycl::buffer<T, 1>& a,
                 sycl::buffer<T, 1>& x, int64_t incx) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), get_cublas_operation(trans),
                                     get_cublas_diag_type(unit_diag), n, a_, x_, incx);
        });
    });
}

#define TPMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                     \
    void tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n, \
              sycl::buffer<TYPE, 1>& a, sycl::buffer<TYPE, 1>& x, int64_t incx) {               \
        tpmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, x,    \
             incx);                                                                             \
    }

TPMV_LAUNCHER(float, cublasStpmv)
TPMV_LAUNCHER(double, cublasDtpmv)
TPMV_LAUNCHER(std::complex<float>, cublasCtpmv)
TPMV_LAUNCHER(std::complex<double>, cublasZtpmv)

#undef TPMV_LAUNCHER

template <typename Func, typename T>
inline void tpsv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                 transpose trans, diag unit_diag, int64_t n, sycl::buffer<T, 1>& a,
                 sycl::buffer<T, 1>& x, int64_t incx) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), get_cublas_operation(trans),
                                     get_cublas_diag_type(unit_diag), n, a_, x_, incx);
        });
    });
}

#define TPSV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                     \
    void tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n, \
              sycl::buffer<TYPE, 1>& a, sycl::buffer<TYPE, 1>& x, int64_t incx) {               \
        tpsv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, x,    \
             incx);                                                                             \
    }

TPSV_LAUNCHER(float, cublasStpsv)
TPSV_LAUNCHER(double, cublasDtpsv)
TPSV_LAUNCHER(std::complex<float>, cublasCtpsv)
TPSV_LAUNCHER(std::complex<double>, cublasZtpsv)

#undef TPSV_LAUNCHER

template <typename Func, typename T>
inline void trmv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                 transpose trans, diag unit_diag, int64_t n, sycl::buffer<T, 1>& a, int64_t lda,
                 sycl::buffer<T, 1>& x, int64_t incx) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, incx);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), get_cublas_operation(trans),
                                     get_cublas_diag_type(unit_diag), n, a_, lda, x_, incx);
        });
    });
}

#define TRMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                       \
    void trmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,   \
              sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x, int64_t incx) {    \
        trmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, lda, x, \
             incx);                                                                               \
    }

TRMV_LAUNCHER(float, cublasStrmv)
TRMV_LAUNCHER(double, cublasDtrmv)
TRMV_LAUNCHER(std::complex<float>, cublasCtrmv)
TRMV_LAUNCHER(std::complex<double>, cublasZtrmv)

#undef TRMV_LAUNCHER

template <typename Func, typename T>
inline void trsv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                 transpose trans, diag unit_diag, int64_t n, sycl::buffer<T, 1>& a, int64_t lda,
                 sycl::buffer<T, 1>& x, int64_t incx) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, incx);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), get_cublas_operation(trans),
                                     get_cublas_diag_type(unit_diag), n, a_, lda, x_, incx);
        });
    });
}

#define TRSV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                       \
    void trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,   \
              sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x, int64_t incx) {    \
        trsv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, lda, x, \
             incx);                                                                               \
    }

TRSV_LAUNCHER(float, cublasStrsv)
TRSV_LAUNCHER(double, cublasDtrsv)
TRSV_LAUNCHER(std::complex<float>, cublasCtrsv)
TRSV_LAUNCHER(std::complex<double>, cublasZtrsv)

#undef TRSV_LAUNCHER

// USM APIs

template <typename Func, typename T>
inline sycl::event gemv(const char* func_name, Func func, sycl::queue& queue, transpose trans,
                        int64_t m, int64_t n, T alpha, const T* a, int64_t lda, const T* x,
                        int64_t incx, T beta, T* y, int64_t incy,
                        const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, m, lda, incx, incy);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<const cuDataType*>(a);
            auto x_ = reinterpret_cast<const cuDataType*>(x);
            auto y_ = reinterpret_cast<cuDataType*>(y);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle, get_cublas_operation(trans), m,
                                     n, (cuDataType*)&alpha, a_, lda, x_, incx, (cuDataType*)&beta,
                                     y_, incy);
        });
    });
    return done;
}

#define GEMV_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                   \
    sycl::event gemv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, TYPE alpha,       \
                     const TYPE* a, int64_t lda, const TYPE* x, int64_t incx, TYPE beta, TYPE* y, \
                     int64_t incy, const std::vector<sycl::event>& dependencies) {                \
        return gemv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, trans, m, n, alpha, a, lda, x, incx,  \
                    beta, y, incy, dependencies);                                                 \
    }

GEMV_LAUNCHER_USM(float, cublasSgemv)
GEMV_LAUNCHER_USM(double, cublasDgemv)
GEMV_LAUNCHER_USM(std::complex<float>, cublasCgemv)
GEMV_LAUNCHER_USM(std::complex<double>, cublasZgemv)
#undef GEMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event gbmv(const char* func_name, Func func, sycl::queue& queue, transpose trans,
                        int64_t m, int64_t n, int64_t kl, int64_t ku, T alpha, const T* a,
                        int64_t lda, const T* x, int64_t incx, T beta, T* y, int64_t incy,
                        const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, m, lda, kl, ku, incx, incy);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<const cuDataType*>(a);
            auto x_ = reinterpret_cast<const cuDataType*>(x);
            auto y_ = reinterpret_cast<cuDataType*>(y);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle, get_cublas_operation(trans), m,
                                     n, kl, ku, (cuDataType*)&alpha, a_, lda, x_, incx,
                                     (cuDataType*)&beta, y_, incy);
        });
    });
    return done;
}

#define GBMV_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                    \
    sycl::event gbmv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, int64_t kl,        \
                     int64_t ku, TYPE alpha, const TYPE* a, int64_t lda, const TYPE* x,            \
                     int64_t incx, TYPE beta, TYPE* y, int64_t incy,                               \
                     const std::vector<sycl::event>& dependencies) {                               \
        return gbmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, trans, m, n, kl, ku, alpha, a, lda, x, \
                    incx, beta, y, incy, dependencies);                                            \
    }

GBMV_LAUNCHER_USM(float, cublasSgbmv)
GBMV_LAUNCHER_USM(double, cublasDgbmv)
GBMV_LAUNCHER_USM(std::complex<float>, cublasCgbmv)
GBMV_LAUNCHER_USM(std::complex<double>, cublasZgbmv)
#undef GBMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event ger(const char* func_name, Func func, sycl::queue& queue, int64_t m, int64_t n,
                       T alpha, const T* x, int64_t incx, const T* y, int64_t incy, T* a,
                       int64_t lda, const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, m, lda, incx, incy);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto x_ = reinterpret_cast<const cuDataType*>(x);
            auto y_ = reinterpret_cast<const cuDataType*>(y);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle, m, n, (cuDataType*)&alpha, x_,
                                     incx, y_, incy, a_, lda);
        });
    });
    return done;
}

#define GER_LAUNCHER_USM(EXT, TYPE, CUBLAS_ROUTINE)                                               \
    sycl::event ger##EXT(sycl::queue& queue, int64_t m, int64_t n, TYPE alpha, const TYPE* x,     \
                         int64_t incx, const TYPE* y, int64_t incy, TYPE* a, int64_t lda,         \
                         const std::vector<sycl::event>& dependencies) {                          \
        return ger(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, m, n, alpha, x, incx, y, incy, a, lda, \
                   dependencies);                                                                 \
    }

GER_LAUNCHER_USM(, float, cublasSger)
GER_LAUNCHER_USM(, double, cublasDger)
GER_LAUNCHER_USM(u, std::complex<float>, cublasCgeru)
GER_LAUNCHER_USM(u, std::complex<double>, cublasZgeru)
GER_LAUNCHER_USM(c, std::complex<float>, cublasCgerc)
GER_LAUNCHER_USM(c, std::complex<double>, cublasZgerc)
#undef GER_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event hbmv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        int64_t n, int64_t k, T alpha, const T* a, int64_t lda, const T* x,
                        int64_t incx, T beta, T* y, int64_t incy,
                        const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, k, lda, incx, incy);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<const cuDataType*>(a);
            auto x_ = reinterpret_cast<const cuDataType*>(x);
            auto y_ = reinterpret_cast<cuDataType*>(y);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), n, k, (cuDataType*)&alpha,
                                     a_, lda, x_, incx, (cuDataType*)&beta, y_, incy);
        });
    });
    return done;
}

#define HBMV_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                   \
    sycl::event hbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k, TYPE alpha,      \
                     const TYPE* a, int64_t lda, const TYPE* x, int64_t incx, TYPE beta, TYPE* y, \
                     int64_t incy, const std::vector<sycl::event>& dependencies) {                \
        return hbmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, k, alpha, a, lda, x,  \
                    incx, beta, y, incy, dependencies);                                           \
    }

HBMV_LAUNCHER_USM(std::complex<float>, cublasChbmv)
HBMV_LAUNCHER_USM(std::complex<double>, cublasZhbmv)
#undef HBMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event hemv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        int64_t n, T alpha, const T* a, int64_t lda, const T* x, int64_t incx,
                        T beta, T* y, int64_t incy, const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, incx, incy);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<const cuDataType*>(a);
            auto x_ = reinterpret_cast<const cuDataType*>(x);
            auto y_ = reinterpret_cast<cuDataType*>(y);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), n, (cuDataType*)&alpha, a_,
                                     lda, x_, incx, (cuDataType*)&beta, y_, incy);
        });
    });
    return done;
}

#define HEMV_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event hemv(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* a, \
                     int64_t lda, const TYPE* x, int64_t incx, TYPE beta, TYPE* y, int64_t incy, \
                     const std::vector<sycl::event>& dependencies) {                             \
        return hemv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, a, lda, x,    \
                    incx, beta, y, incy, dependencies);                                          \
    }

HEMV_LAUNCHER_USM(std::complex<float>, cublasChemv)
HEMV_LAUNCHER_USM(std::complex<double>, cublasZhemv)
#undef HEMV_LAUNCHER_USM

template <typename Func, typename ScalarType, typename DataType>
inline sycl::event her(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                       int64_t n, const ScalarType alpha, const DataType* x, int64_t incx,
                       DataType* a, int64_t lda, const std::vector<sycl::event>& dependencies) {
    using cuScalarType = typename CudaEquivalentType<ScalarType>::Type;
    using cuDataType = typename CudaEquivalentType<DataType>::Type;
    overflow_check(n, lda, incx);

    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto x_ = reinterpret_cast<const cuDataType*>(x);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), n, (cuScalarType*)&alpha,
                                     x_, incx, a_, lda);
        });
    });
    return done;
}

#define HER_LAUNCHER_USM(SCALAR_TYPE, DATA_TYPE, CUBLAS_ROUTINE)                                   \
    sycl::event her(sycl::queue& queue, uplo upper_lower, int64_t n, const SCALAR_TYPE alpha,      \
                    const DATA_TYPE* x, int64_t incx, DATA_TYPE* a, int64_t lda,                   \
                    const std::vector<sycl::event>& dependencies) {                                \
        return her(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a, lda, \
                   dependencies);                                                                  \
    }

HER_LAUNCHER_USM(float, std::complex<float>, cublasCher)
HER_LAUNCHER_USM(double, std::complex<double>, cublasZher)

#undef HER_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event her2(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        int64_t n, T alpha, const T* x, int64_t incx, const T* y, int64_t incy,
                        T* a, int64_t lda, const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, incx, incy);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto x_ = reinterpret_cast<const cuDataType*>(x);
            auto y_ = reinterpret_cast<const cuDataType*>(y);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), n, (cuDataType*)&alpha, x_,
                                     incx, y_, incy, a_, lda);
        });
    });
    return done;
}

#define HER2_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event her2(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* x, \
                     int64_t incx, const TYPE* y, int64_t incy, TYPE* a, int64_t lda,            \
                     const std::vector<sycl::event>& dependencies) {                             \
        return her2(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y,   \
                    incy, a, lda, dependencies);                                                 \
    }

HER2_LAUNCHER_USM(std::complex<float>, cublasCher2)
HER2_LAUNCHER_USM(std::complex<double>, cublasZher2)

#undef HER2_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event hpmv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        int64_t n, T alpha, const T* a, const T* x, int64_t incx, T beta, T* y,
                        int64_t incy, const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<const cuDataType*>(a);
            auto x_ = reinterpret_cast<const cuDataType*>(x);
            auto y_ = reinterpret_cast<cuDataType*>(y);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), n, (cuDataType*)&alpha, a_,
                                     x_, incx, (cuDataType*)&beta, y_, incy);
        });
    });
    return done;
}

#define HPMV_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event hpmv(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* a, \
                     const TYPE* x, int64_t incx, TYPE beta, TYPE* y, int64_t incy,              \
                     const std::vector<sycl::event>& dependencies) {                             \
        return hpmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, a, x, incx,   \
                    beta, y, incy, dependencies);                                                \
    }

HPMV_LAUNCHER_USM(std::complex<float>, cublasChpmv)
HPMV_LAUNCHER_USM(std::complex<double>, cublasZhpmv)

#undef HPMV_LAUNCHER_USM

template <typename Func, typename ScalarType, typename DataType>
inline sycl::event hpr(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                       int64_t n, const ScalarType alpha, const DataType* x, int64_t incx,
                       DataType* a, const std::vector<sycl::event>& dependencies) {
    using cuScalarType = typename CudaEquivalentType<ScalarType>::Type;
    using cuDataType = typename CudaEquivalentType<DataType>::Type;
    overflow_check(n, incx);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto x_ = reinterpret_cast<const cuDataType*>(x);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), n, (cuScalarType*)&alpha,
                                     x_, incx, a_);
        });
    });
    return done;
}

#define HPR_LAUNCHER_USM(SCALAR_TYPE, DATA_TYPE, CUBLAS_ROUTINE)                              \
    sycl::event hpr(sycl::queue& queue, uplo upper_lower, int64_t n, const SCALAR_TYPE alpha, \
                    const DATA_TYPE* x, int64_t incx, DATA_TYPE* a,                           \
                    const std::vector<sycl::event>& dependencies) {                           \
        return hpr(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a, \
                   dependencies);                                                             \
    }

HPR_LAUNCHER_USM(float, std::complex<float>, cublasChpr)
HPR_LAUNCHER_USM(double, std::complex<double>, cublasZhpr)

#undef HPR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event hpr2(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        int64_t n, T alpha, const T* x, int64_t incx, const T* y, int64_t incy,
                        T* a, const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto x_ = reinterpret_cast<const cuDataType*>(x);
            auto y_ = reinterpret_cast<const cuDataType*>(y);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), n, (cuDataType*)&alpha, x_,
                                     incx, y_, incy, a_);
        });
    });
    return done;
}

#define HPR2_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event hpr2(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* x, \
                     int64_t incx, const TYPE* y, int64_t incy, TYPE* a,                         \
                     const std::vector<sycl::event>& dependencies) {                             \
        return hpr2(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y,   \
                    incy, a, dependencies);                                                      \
    }

HPR2_LAUNCHER_USM(std::complex<float>, cublasChpr2)
HPR2_LAUNCHER_USM(std::complex<double>, cublasZhpr2)

#undef HPR2_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event sbmv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        int64_t n, int64_t k, T alpha, const T* a, int64_t lda, const T* x,
                        int64_t incx, T beta, T* y, int64_t incy,
                        const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, k, lda, incx, incy);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<const cuDataType*>(a);
            auto x_ = reinterpret_cast<const cuDataType*>(x);
            auto y_ = reinterpret_cast<cuDataType*>(y);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), n, k, (cuDataType*)&alpha,
                                     a_, lda, x_, incx, (cuDataType*)&beta, y_, incy);
        });
    });
    return done;
}

#define SBMV_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                   \
    sycl::event sbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k, TYPE alpha,      \
                     const TYPE* a, int64_t lda, const TYPE* x, int64_t incx, TYPE beta, TYPE* y, \
                     int64_t incy, const std::vector<sycl::event>& dependencies) {                \
        return sbmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, k, alpha, a, lda, x,  \
                    incx, beta, y, incy, dependencies);                                           \
    }

SBMV_LAUNCHER_USM(float, cublasSsbmv)
SBMV_LAUNCHER_USM(double, cublasDsbmv)

#undef SBMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event symv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        int64_t n, T alpha, const T* a, int64_t lda, const T* x, int64_t incx,
                        T beta, T* y, int64_t incy, const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, incx, incy);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<const cuDataType*>(a);
            auto x_ = reinterpret_cast<const cuDataType*>(x);
            auto y_ = reinterpret_cast<cuDataType*>(y);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), n, (cuDataType*)&alpha, a_,
                                     lda, x_, incx, (cuDataType*)&beta, y_, incy);
        });
    });
    return done;
}

#define SYMV_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event symv(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* a, \
                     int64_t lda, const TYPE* x, int64_t incx, TYPE beta, TYPE* y, int64_t incy, \
                     const std::vector<sycl::event>& dependencies) {                             \
        return symv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, a, lda, x,    \
                    incx, beta, y, incy, dependencies);                                          \
    }

SYMV_LAUNCHER_USM(float, cublasSsymv)
SYMV_LAUNCHER_USM(double, cublasDsymv)

#undef SYMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event syr(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                       int64_t n, T alpha, const T* x, int64_t incx, T* a, int64_t lda,
                       const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, incx);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto x_ = reinterpret_cast<const cuDataType*>(x);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), n, (cuDataType*)&alpha, x_,
                                     incx, a_, lda);
        });
    });
    return done;
}

#define SYR_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                     \
    sycl::event syr(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* x,    \
                    int64_t incx, TYPE* a, int64_t lda,                                            \
                    const std::vector<sycl::event>& dependencies) {                                \
        return syr(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a, lda, \
                   dependencies);                                                                  \
    }

SYR_LAUNCHER_USM(float, cublasSsyr)
SYR_LAUNCHER_USM(double, cublasDsyr)
// Intel does not support the following two
SYR_LAUNCHER_USM(std::complex<float>, cublasCsyr)
SYR_LAUNCHER_USM(std::complex<double>, cublasZsyr)
#undef SYR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event syr2(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        int64_t n, T alpha, const T* x, int64_t incx, const T* y, int64_t incy,
                        T* a, int64_t lda, const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, incx, incy);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto x_ = reinterpret_cast<const cuDataType*>(x);
            auto y_ = reinterpret_cast<const cuDataType*>(y);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), n, (cuDataType*)&alpha, x_,
                                     incx, y_, incy, a_, lda);
        });
    });
    return done;
}

#define SYR2_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event syr2(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* x, \
                     int64_t incx, const TYPE* y, int64_t incy, TYPE* a, int64_t lda,            \
                     const std::vector<sycl::event>& dependencies) {                             \
        return syr2(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y,   \
                    incy, a, lda, dependencies);                                                 \
    }

SYR2_LAUNCHER_USM(float, cublasSsyr2)
SYR2_LAUNCHER_USM(double, cublasDsyr2)
// Intel does not support the following two
SYR2_LAUNCHER_USM(std::complex<float>, cublasCsyr2)
SYR2_LAUNCHER_USM(std::complex<double>, cublasZsyr2)

#undef SYR2_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event spmv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        int64_t n, T alpha, const T* a, const T* x, int64_t incx, T beta, T* y,
                        int64_t incy, const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<const cuDataType*>(a);
            auto x_ = reinterpret_cast<const cuDataType*>(x);
            auto y_ = reinterpret_cast<cuDataType*>(y);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), n, (cuDataType*)&alpha, a_,
                                     x_, incx, (cuDataType*)&beta, y_, incy);
        });
    });
    return done;
}

#define SPMV_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event spmv(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* a, \
                     const TYPE* x, int64_t incx, TYPE beta, TYPE* y, int64_t incy,              \
                     const std::vector<sycl::event>& dependencies) {                             \
        return spmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, a, x, incx,   \
                    beta, y, incy, dependencies);                                                \
    }

SPMV_LAUNCHER_USM(float, cublasSspmv)
SPMV_LAUNCHER_USM(double, cublasDspmv)

#undef SPMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event spr(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                       int64_t n, T alpha, const T* x, int64_t incx, T* a,
                       const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto x_ = reinterpret_cast<const cuDataType*>(x);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), n, (cuDataType*)&alpha, x_,
                                     incx, a_);
        });
    });
    return done;
}

#define SPR_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event spr(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* x, \
                    int64_t incx, TYPE* a, const std::vector<sycl::event>& dependencies) {      \
        return spr(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a,   \
                   dependencies);                                                               \
    }

SPR_LAUNCHER_USM(float, cublasSspr)
SPR_LAUNCHER_USM(double, cublasDspr)

#undef SPR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event spr2(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        int64_t n, T alpha, const T* x, int64_t incx, const T* y, int64_t incy,
                        T* a, const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto x_ = reinterpret_cast<const cuDataType*>(x);
            auto y_ = reinterpret_cast<const cuDataType*>(y);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), n, (cuDataType*)&alpha, x_,
                                     incx, y_, incy, a_);
        });
    });
    return done;
}

#define SPR2_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event spr2(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* x, \
                     int64_t incx, const TYPE* y, int64_t incy, TYPE* a,                         \
                     const std::vector<sycl::event>& dependencies) {                             \
        return spr2(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y,   \
                    incy, a, dependencies);                                                      \
    }

SPR2_LAUNCHER_USM(float, cublasSspr2)
SPR2_LAUNCHER_USM(double, cublasDspr2)

#undef SPR2_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event tbmv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        transpose trans, diag unit_diag, int64_t n, int64_t k, const T* a,
                        int64_t lda, T* x, int64_t incx,
                        const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, k, lda, incx);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<const cuDataType*>(a);
            auto x_ = reinterpret_cast<cuDataType*>(x);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), get_cublas_operation(trans),
                                     get_cublas_diag_type(unit_diag), n, k, a_, lda, x_, incx);
        });
    });
    return done;
}

#define TBMV_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,      \
                     int64_t n, int64_t k, const TYPE* a, int64_t lda, TYPE* x, int64_t incx,    \
                     const std::vector<sycl::event>& dependencies) {                             \
        return tbmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, k, \
                    a, lda, x, incx, dependencies);                                              \
    }

TBMV_LAUNCHER_USM(float, cublasStbmv)
TBMV_LAUNCHER_USM(double, cublasDtbmv)
TBMV_LAUNCHER_USM(std::complex<float>, cublasCtbmv)
TBMV_LAUNCHER_USM(std::complex<double>, cublasZtbmv)

#undef TBMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event tbsv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        transpose trans, diag unit_diag, int64_t n, int64_t k, const T* a,
                        int64_t lda, T* x, int64_t incx,
                        const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, k, lda, incx);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<const cuDataType*>(a);
            auto x_ = reinterpret_cast<cuDataType*>(x);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), get_cublas_operation(trans),
                                     get_cublas_diag_type(unit_diag), n, k, a_, lda, x_, incx);
        });
    });
    return done;
}

#define TBSV_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,      \
                     int64_t n, int64_t k, const TYPE* a, int64_t lda, TYPE* x, int64_t incx,    \
                     const std::vector<sycl::event>& dependencies) {                             \
        return tbsv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, k, \
                    a, lda, x, incx, dependencies);                                              \
    }

TBSV_LAUNCHER_USM(float, cublasStbsv)
TBSV_LAUNCHER_USM(double, cublasDtbsv)
TBSV_LAUNCHER_USM(std::complex<float>, cublasCtbsv)
TBSV_LAUNCHER_USM(std::complex<double>, cublasZtbsv)

#undef TBSV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event tpmv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        transpose trans, diag unit_diag, int64_t n, const T* a, T* x, int64_t incx,
                        const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<const cuDataType*>(a);
            auto x_ = reinterpret_cast<cuDataType*>(x);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), get_cublas_operation(trans),
                                     get_cublas_diag_type(unit_diag), n, a_, x_, incx);
        });
    });
    return done;
}

#define TPMV_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,      \
                     int64_t n, const TYPE* a, TYPE* x, int64_t incx,                            \
                     const std::vector<sycl::event>& dependencies) {                             \
        return tpmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, \
                    x, incx, dependencies);                                                      \
    }

TPMV_LAUNCHER_USM(float, cublasStpmv)
TPMV_LAUNCHER_USM(double, cublasDtpmv)
TPMV_LAUNCHER_USM(std::complex<float>, cublasCtpmv)
TPMV_LAUNCHER_USM(std::complex<double>, cublasZtpmv)

#undef TPMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event tpsv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        transpose trans, diag unit_diag, int64_t n, const T* a, T* x, int64_t incx,
                        const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<const cuDataType*>(a);
            auto x_ = reinterpret_cast<cuDataType*>(x);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), get_cublas_operation(trans),
                                     get_cublas_diag_type(unit_diag), n, a_, x_, incx);
        });
    });
    return done;
}

#define TPSV_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,      \
                     int64_t n, const TYPE* a, TYPE* x, int64_t incx,                            \
                     const std::vector<sycl::event>& dependencies) {                             \
        return tpsv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, \
                    x, incx, dependencies);                                                      \
    }

TPSV_LAUNCHER_USM(float, cublasStpsv)
TPSV_LAUNCHER_USM(double, cublasDtpsv)
TPSV_LAUNCHER_USM(std::complex<float>, cublasCtpsv)
TPSV_LAUNCHER_USM(std::complex<double>, cublasZtpsv)

#undef TPSV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event trmv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        transpose trans, diag unit_diag, int64_t n, const T* a, int64_t lda, T* x,
                        int64_t incx, const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, incx);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<const cuDataType*>(a);
            auto x_ = reinterpret_cast<cuDataType*>(x);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), get_cublas_operation(trans),
                                     get_cublas_diag_type(unit_diag), n, a_, lda, x_, incx);
        });
    });
    return done;
}

#define TRMV_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event trmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,      \
                     int64_t n, const TYPE* a, int64_t lda, TYPE* x, int64_t incx,               \
                     const std::vector<sycl::event>& dependencies) {                             \
        return trmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, \
                    lda, x, incx, dependencies);                                                 \
    }

TRMV_LAUNCHER_USM(float, cublasStrmv)
TRMV_LAUNCHER_USM(double, cublasDtrmv)
TRMV_LAUNCHER_USM(std::complex<float>, cublasCtrmv)
TRMV_LAUNCHER_USM(std::complex<double>, cublasZtrmv)

#undef TRMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event trsv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        transpose trans, diag unit_diag, int64_t n, const T* a, int64_t lda, T* x,
                        int64_t incx, const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, incx);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<const cuDataType*>(a);
            auto x_ = reinterpret_cast<cuDataType*>(x);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle,
                                     get_cublas_fill_mode(upper_lower), get_cublas_operation(trans),
                                     get_cublas_diag_type(unit_diag), n, a_, lda, x_, incx);
        });
    });
    return done;
}

#define TRSV_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,      \
                     int64_t n, const TYPE* a, int64_t lda, TYPE* x, int64_t incx,               \
                     const std::vector<sycl::event>& dependencies) {                             \
        return trsv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, \
                    lda, x, incx, dependencies);                                                 \
    }

TRSV_LAUNCHER_USM(float, cublasStrsv)
TRSV_LAUNCHER_USM(double, cublasDtrsv)
TRSV_LAUNCHER_USM(std::complex<float>, cublasCtrsv)
TRSV_LAUNCHER_USM(std::complex<double>, cublasZtrsv)

#undef TRSV_LAUNCHER_USM

} // namespace column_major
namespace row_major {

// Buffer APIs

template <typename Func, typename T>
inline void gemv(const char* func_name, Func func, sycl::queue& queue, transpose trans, int64_t m,
                 int64_t n, T alpha, sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& x,
                 int64_t incx, T beta, sycl::buffer<T, 1>& y, int64_t incy) {
    throw unimplemented("blas", "gemv", "for row_major layout");
}

#define GEMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                        \
    void gemv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, TYPE alpha,               \
              sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x, int64_t incx,       \
              TYPE beta, sycl::buffer<TYPE, 1>& y, int64_t incy) {                                 \
        gemv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, trans, m, n, alpha, a, lda, x, incx, beta, y, \
             incy);                                                                                \
    }

GEMV_LAUNCHER(float, cublasSgemv)
GEMV_LAUNCHER(double, cublasDgemv)
GEMV_LAUNCHER(std::complex<float>, cublasCgemv)
GEMV_LAUNCHER(std::complex<double>, cublasZgemv)
#undef GEMV_LAUNCHER

template <typename Func, typename T>
inline void gbmv(const char* func_name, Func func, sycl::queue& queue, transpose trans, int64_t m,
                 int64_t n, int64_t kl, int64_t ku, T alpha, sycl::buffer<T, 1>& a, int64_t lda,
                 sycl::buffer<T, 1>& x, int64_t incx, T beta, sycl::buffer<T, 1>& y, int64_t incy) {
    throw unimplemented("blas", "gbmv", "for row_major layout");
}

#define GBMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                       \
    void gbmv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,  \
              TYPE alpha, sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x,        \
              int64_t incx, TYPE beta, sycl::buffer<TYPE, 1>& y, int64_t incy) {                  \
        gbmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, \
             beta, y, incy);                                                                      \
    }

GBMV_LAUNCHER(float, cublasSgbmv)
GBMV_LAUNCHER(double, cublasDgbmv)
GBMV_LAUNCHER(std::complex<float>, cublasCgbmv)
GBMV_LAUNCHER(std::complex<double>, cublasZgbmv)
#undef GBMV_LAUNCHER

template <typename Func, typename T>
inline void ger(const char* func_name, Func func, sycl::queue& queue, int64_t m, int64_t n, T alpha,
                sycl::buffer<T, 1>& x, int64_t incx, sycl::buffer<T, 1>& y, int64_t incy,
                sycl::buffer<T, 1>& a, int64_t lda) {
    throw unimplemented("blas", "ger", "for row_major layout");
}

#define GER_LAUNCHER(EXT, TYPE, CUBLAS_ROUTINE)                                                   \
    void ger##EXT(sycl::queue& queue, int64_t m, int64_t n, TYPE alpha, sycl::buffer<TYPE, 1>& x, \
                  int64_t incx, sycl::buffer<TYPE, 1>& y, int64_t incy, sycl::buffer<TYPE, 1>& a, \
                  int64_t lda) {                                                                  \
        ger(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, m, n, alpha, x, incx, y, incy, a, lda);       \
    }

GER_LAUNCHER(, float, cublasSger)
GER_LAUNCHER(, double, cublasDger)
GER_LAUNCHER(u, std::complex<float>, cublasCgeru)
GER_LAUNCHER(u, std::complex<double>, cublasZgeru)
GER_LAUNCHER(c, std::complex<float>, cublasCgerc)
GER_LAUNCHER(c, std::complex<double>, cublasZgerc)
#undef GER_LAUNCHER

template <typename Func, typename T>
inline void hbmv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower, int64_t n,
                 int64_t k, T alpha, sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& x,
                 int64_t incx, T beta, sycl::buffer<T, 1>& y, int64_t incy) {
    throw unimplemented("blas", "hbmv", "for row_major layout");
}

#define HBMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                     \
    void hbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k, TYPE alpha,           \
              sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x, int64_t incx,    \
              TYPE beta, sycl::buffer<TYPE, 1>& y, int64_t incy) {                              \
        hbmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, k, alpha, a, lda, x, incx, \
             beta, y, incy);                                                                    \
    }

HBMV_LAUNCHER(std::complex<float>, cublasChbmv)
HBMV_LAUNCHER(std::complex<double>, cublasZhbmv)
#undef HBMV_LAUNCHER

template <typename Func, typename T>
inline void hemv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower, int64_t n,
                 T alpha, sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& x, int64_t incx,
                 T beta, sycl::buffer<T, 1>& y, int64_t incy) {
    throw unimplemented("blas", "hemv", "for row_major layout");
}

#define HEMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                        \
    void hemv(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,                         \
              sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x, int64_t incx,       \
              TYPE beta, sycl::buffer<TYPE, 1>& y, int64_t incy) {                                 \
        hemv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, a, lda, x, incx, beta, \
             y, incy);                                                                             \
    }

HEMV_LAUNCHER(std::complex<float>, cublasChemv)
HEMV_LAUNCHER(std::complex<double>, cublasZhemv)
#undef HEMV_LAUNCHER

template <typename Func, typename ScalarType, typename DataType>
inline void her(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower, int64_t n,
                ScalarType alpha, sycl::buffer<DataType, 1>& x, int64_t incx,
                sycl::buffer<DataType, 1>& a, int64_t lda) {
    throw unimplemented("blas", "her", "for row_major layout");
}

#define HER_LAUNCHER(SCALAR_TYPE, DATA_TYPE, CUBLAS_ROUTINE)                                 \
    void her(sycl::queue& queue, uplo upper_lower, int64_t n, SCALAR_TYPE alpha,             \
             sycl::buffer<DATA_TYPE, 1>& x, int64_t incx, sycl::buffer<DATA_TYPE, 1>& a,     \
             int64_t lda) {                                                                  \
        her(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a, lda); \
    }

HER_LAUNCHER(float, std::complex<float>, cublasCher)
HER_LAUNCHER(double, std::complex<double>, cublasZher)

#undef HER_LAUNCHER

template <typename Func, typename T>
inline void her2(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower, int64_t n,
                 T alpha, sycl::buffer<T, 1>& x, int64_t incx, sycl::buffer<T, 1>& y, int64_t incy,
                 sycl::buffer<T, 1>& a, int64_t lda) {
    throw unimplemented("blas", "her2", "for row_major layout");
}

#define HER2_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                      \
    void her2(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,                       \
              sycl::buffer<TYPE, 1>& x, int64_t incx, sycl::buffer<TYPE, 1>& y, int64_t incy,    \
              sycl::buffer<TYPE, 1>& a, int64_t lda) {                                           \
        her2(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y, incy, a, \
             lda);                                                                               \
    }

HER2_LAUNCHER(std::complex<float>, cublasCher2)
HER2_LAUNCHER(std::complex<double>, cublasZher2)

#undef HER2_LAUNCHER

template <typename Func, typename T>
inline void hpmv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower, int64_t n,
                 T alpha, sycl::buffer<T, 1>& a, sycl::buffer<T, 1>& x, int64_t incx, T beta,
                 sycl::buffer<T, 1>& y, int64_t incy) {
    throw unimplemented("blas", "hpmv", "for row_major layout");
}

#define HPMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                      \
    void hpmv(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,                       \
              sycl::buffer<TYPE, 1>& a, sycl::buffer<TYPE, 1>& x, int64_t incx, TYPE beta,       \
              sycl::buffer<TYPE, 1>& y, int64_t incy) {                                          \
        hpmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, a, x, incx, beta, y, \
             incy);                                                                              \
    }

HPMV_LAUNCHER(std::complex<float>, cublasChpmv)
HPMV_LAUNCHER(std::complex<double>, cublasZhpmv)

#undef HPMV_LAUNCHER

template <typename Func, typename ScalarType, typename DataType>
inline void hpr(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower, int64_t n,
                ScalarType alpha, sycl::buffer<DataType, 1>& x, int64_t incx,
                sycl::buffer<DataType, 1>& a) {
    throw unimplemented("blas", "hpr", "for row_major layout");
}

#define HPR_LAUNCHER(SCALAR_TYPE, DATA_TYPE, CUBLAS_ROUTINE)                               \
    void hpr(sycl::queue& queue, uplo upper_lower, int64_t n, SCALAR_TYPE alpha,           \
             sycl::buffer<DATA_TYPE, 1>& x, int64_t incx, sycl::buffer<DATA_TYPE, 1>& a) { \
        hpr(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a);    \
    }

HPR_LAUNCHER(float, std::complex<float>, cublasChpr)
HPR_LAUNCHER(double, std::complex<double>, cublasZhpr)

#undef HPR_LAUNCHER

template <typename Func, typename T>
inline void hpr2(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower, int64_t n,
                 T alpha, sycl::buffer<T, 1>& x, int64_t incx, sycl::buffer<T, 1>& y, int64_t incy,
                 sycl::buffer<T, 1>& a) {
    throw unimplemented("blas", "hpr2", "for row_major layout");
}

#define HPR2_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                       \
    void hpr2(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,                        \
              sycl::buffer<TYPE, 1>& x, int64_t incx, sycl::buffer<TYPE, 1>& y, int64_t incy,     \
              sycl::buffer<TYPE, 1>& a) {                                                         \
        hpr2(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y, incy, a); \
    }

HPR2_LAUNCHER(std::complex<float>, cublasChpr2)
HPR2_LAUNCHER(std::complex<double>, cublasZhpr2)

#undef HPR2_LAUNCHER

template <typename Func, typename T>
inline void sbmv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower, int64_t n,
                 int64_t k, T alpha, sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& x,
                 int64_t incx, T beta, sycl::buffer<T, 1>& y, int64_t incy) {
    throw unimplemented("blas", "sbmv", "for row_major layout");
}

#define SBMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                     \
    void sbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k, TYPE alpha,           \
              sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x, int64_t incx,    \
              TYPE beta, sycl::buffer<TYPE, 1>& y, int64_t incy) {                              \
        sbmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, k, alpha, a, lda, x, incx, \
             beta, y, incy);                                                                    \
    }

SBMV_LAUNCHER(float, cublasSsbmv)
SBMV_LAUNCHER(double, cublasDsbmv)

#undef SBMV_LAUNCHER

template <typename Func, typename T>
inline void symv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower, int64_t n,
                 T alpha, sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& x, int64_t incx,
                 T beta, sycl::buffer<T, 1>& y, int64_t incy) {
    throw unimplemented("blas", "symv", "for row_major layout");
}

#define SYMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                        \
    void symv(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,                         \
              sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x, int64_t incx,       \
              TYPE beta, sycl::buffer<TYPE, 1>& y, int64_t incy) {                                 \
        symv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, a, lda, x, incx, beta, \
             y, incy);                                                                             \
    }

SYMV_LAUNCHER(float, cublasSsymv)
SYMV_LAUNCHER(double, cublasDsymv)

#undef SYMV_LAUNCHER

template <typename Func, typename T>
inline void syr(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower, int64_t n,
                T alpha, sycl::buffer<T, 1>& x, int64_t incx, sycl::buffer<T, 1>& a, int64_t lda) {
    throw unimplemented("blas", "syr", "for row_major layout");
}

#define SYR_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                    \
    void syr(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,                     \
             sycl::buffer<TYPE, 1>& x, int64_t incx, sycl::buffer<TYPE, 1>& a, int64_t lda) { \
        syr(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a, lda);  \
    }

SYR_LAUNCHER(float, cublasSsyr)
SYR_LAUNCHER(double, cublasDsyr)
// Intel does not support the following two
SYR_LAUNCHER(std::complex<float>, cublasCsyr)
SYR_LAUNCHER(std::complex<double>, cublasZsyr)
#undef SYR_LAUNCHER

template <typename Func, typename T>
inline void syr2(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower, int64_t n,
                 T alpha, sycl::buffer<T, 1>& x, int64_t incx, sycl::buffer<T, 1>& y, int64_t incy,
                 sycl::buffer<T, 1>& a, int64_t lda) {
    throw unimplemented("blas", "syr2", "for row_major layout");
}

#define SYR2_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                      \
    void syr2(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,                       \
              sycl::buffer<TYPE, 1>& x, int64_t incx, sycl::buffer<TYPE, 1>& y, int64_t incy,    \
              sycl::buffer<TYPE, 1>& a, int64_t lda) {                                           \
        syr2(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y, incy, a, \
             lda);                                                                               \
    }

SYR2_LAUNCHER(float, cublasSsyr2)
SYR2_LAUNCHER(double, cublasDsyr2)
// Intel does not support the following two
SYR2_LAUNCHER(std::complex<float>, cublasCsyr2)
SYR2_LAUNCHER(std::complex<double>, cublasZsyr2)

#undef SYR2_LAUNCHER

template <typename Func, typename T>
inline void spmv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower, int64_t n,
                 T alpha, sycl::buffer<T, 1>& a, sycl::buffer<T, 1>& x, int64_t incx, T beta,
                 sycl::buffer<T, 1>& y, int64_t incy) {
    throw unimplemented("blas", "spmv", "for row_major layout");
}

#define SPMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                      \
    void spmv(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,                       \
              sycl::buffer<TYPE, 1>& a, sycl::buffer<TYPE, 1>& x, int64_t incx, TYPE beta,       \
              sycl::buffer<TYPE, 1>& y, int64_t incy) {                                          \
        spmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, a, x, incx, beta, y, \
             incy);                                                                              \
    }

SPMV_LAUNCHER(float, cublasSspmv)
SPMV_LAUNCHER(double, cublasDspmv)

#undef SPMV_LAUNCHER

template <typename Func, typename T>
inline void spr(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower, int64_t n,
                T alpha, sycl::buffer<T, 1>& x, int64_t incx, sycl::buffer<T, 1>& a) {
    throw unimplemented("blas", "spr", "for row_major layout");
}

#define SPR_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                              \
    void spr(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,               \
             sycl::buffer<TYPE, 1>& x, int64_t incx, sycl::buffer<TYPE, 1>& a) {        \
        spr(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a); \
    }

SPR_LAUNCHER(float, cublasSspr)
SPR_LAUNCHER(double, cublasDspr)

#undef SPR_LAUNCHER

template <typename Func, typename T>
inline void spr2(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower, int64_t n,
                 T alpha, sycl::buffer<T, 1>& x, int64_t incx, sycl::buffer<T, 1>& y, int64_t incy,
                 sycl::buffer<T, 1>& a) {
    throw unimplemented("blas", "spr2", "for row_major layout");
}

#define SPR2_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                       \
    void spr2(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,                        \
              sycl::buffer<TYPE, 1>& x, int64_t incx, sycl::buffer<TYPE, 1>& y, int64_t incy,     \
              sycl::buffer<TYPE, 1>& a) {                                                         \
        spr2(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y, incy, a); \
    }

SPR2_LAUNCHER(float, cublasSspr2)
SPR2_LAUNCHER(double, cublasDspr2)

#undef SPR2_LAUNCHER

template <typename Func, typename T>
inline void tbmv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                 transpose trans, diag unit_diag, int64_t n, int64_t k, sycl::buffer<T, 1>& a,
                 int64_t lda, sycl::buffer<T, 1>& x, int64_t incx) {
    throw unimplemented("blas", "tbmv", "for row_major layout");
}

#define TBMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                       \
    void tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,   \
              int64_t k, sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x,         \
              int64_t incx) {                                                                     \
        tbmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, k, a, lda, \
             x, incx);                                                                            \
    }

TBMV_LAUNCHER(float, cublasStbmv)
TBMV_LAUNCHER(double, cublasDtbmv)
TBMV_LAUNCHER(std::complex<float>, cublasCtbmv)
TBMV_LAUNCHER(std::complex<double>, cublasZtbmv)

#undef TBMV_LAUNCHER

template <typename Func, typename T>
inline void tbsv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                 transpose trans, diag unit_diag, int64_t n, int64_t k, sycl::buffer<T, 1>& a,
                 int64_t lda, sycl::buffer<T, 1>& x, int64_t incx) {
    throw unimplemented("blas", "tbsv", "for row_major layout");
}

#define TBSV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                       \
    void tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,   \
              int64_t k, sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x,         \
              int64_t incx) {                                                                     \
        tbsv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, k, a, lda, \
             x, incx);                                                                            \
    }

TBSV_LAUNCHER(float, cublasStbsv)
TBSV_LAUNCHER(double, cublasDtbsv)
TBSV_LAUNCHER(std::complex<float>, cublasCtbsv)
TBSV_LAUNCHER(std::complex<double>, cublasZtbsv)

#undef TBSV_LAUNCHER

template <typename Func, typename T>
inline void tpmv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                 transpose trans, diag unit_diag, int64_t n, sycl::buffer<T, 1>& a,
                 sycl::buffer<T, 1>& x, int64_t incx) {
    throw unimplemented("blas", "tpmv", "for row_major layout");
}

#define TPMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                     \
    void tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n, \
              sycl::buffer<TYPE, 1>& a, sycl::buffer<TYPE, 1>& x, int64_t incx) {               \
        tpmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, x,    \
             incx);                                                                             \
    }

TPMV_LAUNCHER(float, cublasStpmv)
TPMV_LAUNCHER(double, cublasDtpmv)
TPMV_LAUNCHER(std::complex<float>, cublasCtpmv)
TPMV_LAUNCHER(std::complex<double>, cublasZtpmv)

#undef TPMV_LAUNCHER

template <typename Func, typename T>
inline void tpsv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                 transpose trans, diag unit_diag, int64_t n, sycl::buffer<T, 1>& a,
                 sycl::buffer<T, 1>& x, int64_t incx) {
    throw unimplemented("blas", "tpsv", "for row_major layout");
}

#define TPSV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                     \
    void tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n, \
              sycl::buffer<TYPE, 1>& a, sycl::buffer<TYPE, 1>& x, int64_t incx) {               \
        tpsv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, x,    \
             incx);                                                                             \
    }

TPSV_LAUNCHER(float, cublasStpsv)
TPSV_LAUNCHER(double, cublasDtpsv)
TPSV_LAUNCHER(std::complex<float>, cublasCtpsv)
TPSV_LAUNCHER(std::complex<double>, cublasZtpsv)

#undef TPSV_LAUNCHER

template <typename Func, typename T>
inline void trmv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                 transpose trans, diag unit_diag, int64_t n, sycl::buffer<T, 1>& a, int64_t lda,
                 sycl::buffer<T, 1>& x, int64_t incx) {
    throw unimplemented("blas", "trmv", "for row_major layout");
}

#define TRMV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                       \
    void trmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,   \
              sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x, int64_t incx) {    \
        trmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, lda, x, \
             incx);                                                                               \
    }

TRMV_LAUNCHER(float, cublasStrmv)
TRMV_LAUNCHER(double, cublasDtrmv)
TRMV_LAUNCHER(std::complex<float>, cublasCtrmv)
TRMV_LAUNCHER(std::complex<double>, cublasZtrmv)

#undef TRMV_LAUNCHER

template <typename Func, typename T>
inline void trsv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                 transpose trans, diag unit_diag, int64_t n, sycl::buffer<T, 1>& a, int64_t lda,
                 sycl::buffer<T, 1>& x, int64_t incx) {
    throw unimplemented("blas", "trsv", "for row_major layout");
}

#define TRSV_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                       \
    void trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,   \
              sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x, int64_t incx) {    \
        trsv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, lda, x, \
             incx);                                                                               \
    }

TRSV_LAUNCHER(float, cublasStrsv)
TRSV_LAUNCHER(double, cublasDtrsv)
TRSV_LAUNCHER(std::complex<float>, cublasCtrsv)
TRSV_LAUNCHER(std::complex<double>, cublasZtrsv)

#undef TRSV_LAUNCHER

// USM APIs

template <typename Func, typename T>
inline sycl::event gemv(const char* func_name, Func func, sycl::queue& queue, transpose trans,
                        int64_t m, int64_t n, T alpha, const T* a, int64_t lda, const T* x,
                        int64_t incx, T beta, T* y, int64_t incy,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemv", "for row_major layout");
}

#define GEMV_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                   \
    sycl::event gemv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, TYPE alpha,       \
                     const TYPE* a, int64_t lda, const TYPE* x, int64_t incx, TYPE beta, TYPE* y, \
                     int64_t incy, const std::vector<sycl::event>& dependencies) {                \
        return gemv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, trans, m, n, alpha, a, lda, x, incx,  \
                    beta, y, incy, dependencies);                                                 \
    }

GEMV_LAUNCHER_USM(float, cublasSgemv)
GEMV_LAUNCHER_USM(double, cublasDgemv)
GEMV_LAUNCHER_USM(std::complex<float>, cublasCgemv)
GEMV_LAUNCHER_USM(std::complex<double>, cublasZgemv)
#undef GEMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event gbmv(const char* func_name, Func func, sycl::queue& queue, transpose trans,
                        int64_t m, int64_t n, int64_t kl, int64_t ku, T alpha, const T* a,
                        int64_t lda, const T* x, int64_t incx, T beta, T* y, int64_t incy,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gbmv", "for row_major layout");
}

#define GBMV_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                    \
    sycl::event gbmv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, int64_t kl,        \
                     int64_t ku, TYPE alpha, const TYPE* a, int64_t lda, const TYPE* x,            \
                     int64_t incx, TYPE beta, TYPE* y, int64_t incy,                               \
                     const std::vector<sycl::event>& dependencies) {                               \
        return gbmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, trans, m, n, kl, ku, alpha, a, lda, x, \
                    incx, beta, y, incy, dependencies);                                            \
    }

GBMV_LAUNCHER_USM(float, cublasSgbmv)
GBMV_LAUNCHER_USM(double, cublasDgbmv)
GBMV_LAUNCHER_USM(std::complex<float>, cublasCgbmv)
GBMV_LAUNCHER_USM(std::complex<double>, cublasZgbmv)
#undef GBMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event ger(const char* func_name, Func func, sycl::queue& queue, int64_t m, int64_t n,
                       T alpha, const T* x, int64_t incx, const T* y, int64_t incy, T* a,
                       int64_t lda, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "ger", "for row_major layout");
}

#define GER_LAUNCHER_USM(EXT, TYPE, CUBLAS_ROUTINE)                                               \
    sycl::event ger##EXT(sycl::queue& queue, int64_t m, int64_t n, TYPE alpha, const TYPE* x,     \
                         int64_t incx, const TYPE* y, int64_t incy, TYPE* a, int64_t lda,         \
                         const std::vector<sycl::event>& dependencies) {                          \
        return ger(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, m, n, alpha, x, incx, y, incy, a, lda, \
                   dependencies);                                                                 \
    }

GER_LAUNCHER_USM(, float, cublasSger)
GER_LAUNCHER_USM(, double, cublasDger)
GER_LAUNCHER_USM(u, std::complex<float>, cublasCgeru)
GER_LAUNCHER_USM(u, std::complex<double>, cublasZgeru)
GER_LAUNCHER_USM(c, std::complex<float>, cublasCgerc)
GER_LAUNCHER_USM(c, std::complex<double>, cublasZgerc)
#undef GER_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event hbmv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        int64_t n, int64_t k, T alpha, const T* a, int64_t lda, const T* x,
                        int64_t incx, T beta, T* y, int64_t incy,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "hbmv", "for row_major layout");
}

#define HBMV_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                   \
    sycl::event hbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k, TYPE alpha,      \
                     const TYPE* a, int64_t lda, const TYPE* x, int64_t incx, TYPE beta, TYPE* y, \
                     int64_t incy, const std::vector<sycl::event>& dependencies) {                \
        return hbmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, k, alpha, a, lda, x,  \
                    incx, beta, y, incy, dependencies);                                           \
    }

HBMV_LAUNCHER_USM(std::complex<float>, cublasChbmv)
HBMV_LAUNCHER_USM(std::complex<double>, cublasZhbmv)
#undef HBMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event hemv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        int64_t n, T alpha, const T* a, int64_t lda, const T* x, int64_t incx,
                        T beta, T* y, int64_t incy, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "hemv", "for row_major layout");
}

#define HEMV_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event hemv(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* a, \
                     int64_t lda, const TYPE* x, int64_t incx, TYPE beta, TYPE* y, int64_t incy, \
                     const std::vector<sycl::event>& dependencies) {                             \
        return hemv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, a, lda, x,    \
                    incx, beta, y, incy, dependencies);                                          \
    }

HEMV_LAUNCHER_USM(std::complex<float>, cublasChemv)
HEMV_LAUNCHER_USM(std::complex<double>, cublasZhemv)
#undef HEMV_LAUNCHER_USM

template <typename Func, typename ScalarType, typename DataType>
inline sycl::event her(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                       int64_t n, const ScalarType alpha, const DataType* x, int64_t incx,
                       DataType* a, int64_t lda, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "her", "for row_major layout");
}

#define HER_LAUNCHER_USM(SCALAR_TYPE, DATA_TYPE, CUBLAS_ROUTINE)                                   \
    sycl::event her(sycl::queue& queue, uplo upper_lower, int64_t n, const SCALAR_TYPE alpha,      \
                    const DATA_TYPE* x, int64_t incx, DATA_TYPE* a, int64_t lda,                   \
                    const std::vector<sycl::event>& dependencies) {                                \
        return her(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a, lda, \
                   dependencies);                                                                  \
    }

HER_LAUNCHER_USM(float, std::complex<float>, cublasCher)
HER_LAUNCHER_USM(double, std::complex<double>, cublasZher)

#undef HER_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event her2(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        int64_t n, T alpha, const T* x, int64_t incx, const T* y, int64_t incy,
                        T* a, int64_t lda, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "her2", "for row_major layout");
}

#define HER2_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event her2(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* x, \
                     int64_t incx, const TYPE* y, int64_t incy, TYPE* a, int64_t lda,            \
                     const std::vector<sycl::event>& dependencies) {                             \
        return her2(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y,   \
                    incy, a, lda, dependencies);                                                 \
    }

HER2_LAUNCHER_USM(std::complex<float>, cublasCher2)
HER2_LAUNCHER_USM(std::complex<double>, cublasZher2)

#undef HER2_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event hpmv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        int64_t n, T alpha, const T* a, const T* x, int64_t incx, T beta, T* y,
                        int64_t incy, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "hpmv", "for row_major layout");
}

#define HPMV_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event hpmv(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* a, \
                     const TYPE* x, int64_t incx, TYPE beta, TYPE* y, int64_t incy,              \
                     const std::vector<sycl::event>& dependencies) {                             \
        return hpmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, a, x, incx,   \
                    beta, y, incy, dependencies);                                                \
    }

HPMV_LAUNCHER_USM(std::complex<float>, cublasChpmv)
HPMV_LAUNCHER_USM(std::complex<double>, cublasZhpmv)

#undef HPMV_LAUNCHER_USM

template <typename Func, typename ScalarType, typename DataType>
inline sycl::event hpr(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                       int64_t n, const ScalarType alpha, const DataType* x, int64_t incx,
                       DataType* a, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "hpr", "for row_major layout");
}

#define HPR_LAUNCHER_USM(SCALAR_TYPE, DATA_TYPE, CUBLAS_ROUTINE)                              \
    sycl::event hpr(sycl::queue& queue, uplo upper_lower, int64_t n, const SCALAR_TYPE alpha, \
                    const DATA_TYPE* x, int64_t incx, DATA_TYPE* a,                           \
                    const std::vector<sycl::event>& dependencies) {                           \
        return hpr(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a, \
                   dependencies);                                                             \
    }

HPR_LAUNCHER_USM(float, std::complex<float>, cublasChpr)
HPR_LAUNCHER_USM(double, std::complex<double>, cublasZhpr)

#undef HPR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event hpr2(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        int64_t n, T alpha, const T* x, int64_t incx, const T* y, int64_t incy,
                        T* a, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "hpr2", "for row_major layout");
}

#define HPR2_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event hpr2(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* x, \
                     int64_t incx, const TYPE* y, int64_t incy, TYPE* a,                         \
                     const std::vector<sycl::event>& dependencies) {                             \
        return hpr2(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y,   \
                    incy, a, dependencies);                                                      \
    }

HPR2_LAUNCHER_USM(std::complex<float>, cublasChpr2)
HPR2_LAUNCHER_USM(std::complex<double>, cublasZhpr2)

#undef HPR2_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event sbmv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        int64_t n, int64_t k, T alpha, const T* a, int64_t lda, const T* x,
                        int64_t incx, T beta, T* y, int64_t incy,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "sbmv", "for row_major layout");
}

#define SBMV_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                   \
    sycl::event sbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k, TYPE alpha,      \
                     const TYPE* a, int64_t lda, const TYPE* x, int64_t incx, TYPE beta, TYPE* y, \
                     int64_t incy, const std::vector<sycl::event>& dependencies) {                \
        return sbmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, k, alpha, a, lda, x,  \
                    incx, beta, y, incy, dependencies);                                           \
    }

SBMV_LAUNCHER_USM(float, cublasSsbmv)
SBMV_LAUNCHER_USM(double, cublasDsbmv)

#undef SBMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event symv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        int64_t n, T alpha, const T* a, int64_t lda, const T* x, int64_t incx,
                        T beta, T* y, int64_t incy, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "symv", "for row_major layout");
}

#define SYMV_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event symv(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* a, \
                     int64_t lda, const TYPE* x, int64_t incx, TYPE beta, TYPE* y, int64_t incy, \
                     const std::vector<sycl::event>& dependencies) {                             \
        return symv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, a, lda, x,    \
                    incx, beta, y, incy, dependencies);                                          \
    }

SYMV_LAUNCHER_USM(float, cublasSsymv)
SYMV_LAUNCHER_USM(double, cublasDsymv)

#undef SYMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event syr(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                       int64_t n, T alpha, const T* x, int64_t incx, T* a, int64_t lda,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "syr", "for row_major layout");
}

#define SYR_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                     \
    sycl::event syr(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* x,    \
                    int64_t incx, TYPE* a, int64_t lda,                                            \
                    const std::vector<sycl::event>& dependencies) {                                \
        return syr(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a, lda, \
                   dependencies);                                                                  \
    }

SYR_LAUNCHER_USM(float, cublasSsyr)
SYR_LAUNCHER_USM(double, cublasDsyr)
// Intel does not support the following two
SYR_LAUNCHER_USM(std::complex<float>, cublasCsyr)
SYR_LAUNCHER_USM(std::complex<double>, cublasZsyr)
#undef SYR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event syr2(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        int64_t n, T alpha, const T* x, int64_t incx, const T* y, int64_t incy,
                        T* a, int64_t lda, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "syr2", "for row_major layout");
}

#define SYR2_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event syr2(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* x, \
                     int64_t incx, const TYPE* y, int64_t incy, TYPE* a, int64_t lda,            \
                     const std::vector<sycl::event>& dependencies) {                             \
        return syr2(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y,   \
                    incy, a, lda, dependencies);                                                 \
    }

SYR2_LAUNCHER_USM(float, cublasSsyr2)
SYR2_LAUNCHER_USM(double, cublasDsyr2)
// Intel does not support the following two
SYR2_LAUNCHER_USM(std::complex<float>, cublasCsyr2)
SYR2_LAUNCHER_USM(std::complex<double>, cublasZsyr2)

#undef SYR2_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event spmv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        int64_t n, T alpha, const T* a, const T* x, int64_t incx, T beta, T* y,
                        int64_t incy, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "spmv", "for row_major layout");
}

#define SPMV_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event spmv(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* a, \
                     const TYPE* x, int64_t incx, TYPE beta, TYPE* y, int64_t incy,              \
                     const std::vector<sycl::event>& dependencies) {                             \
        return spmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, a, x, incx,   \
                    beta, y, incy, dependencies);                                                \
    }

SPMV_LAUNCHER_USM(float, cublasSspmv)
SPMV_LAUNCHER_USM(double, cublasDspmv)

#undef SPMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event spr(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                       int64_t n, T alpha, const T* x, int64_t incx, T* a,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "spr", "for row_major layout");
}

#define SPR_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event spr(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* x, \
                    int64_t incx, TYPE* a, const std::vector<sycl::event>& dependencies) {      \
        return spr(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, a,   \
                   dependencies);                                                               \
    }

SPR_LAUNCHER_USM(float, cublasSspr)
SPR_LAUNCHER_USM(double, cublasDspr)

#undef SPR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event spr2(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        int64_t n, T alpha, const T* x, int64_t incx, const T* y, int64_t incy,
                        T* a, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "spr2", "for row_major layout");
}

#define SPR2_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event spr2(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* x, \
                     int64_t incx, const TYPE* y, int64_t incy, TYPE* a,                         \
                     const std::vector<sycl::event>& dependencies) {                             \
        return spr2(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, n, alpha, x, incx, y,   \
                    incy, a, dependencies);                                                      \
    }

SPR2_LAUNCHER_USM(float, cublasSspr2)
SPR2_LAUNCHER_USM(double, cublasDspr2)

#undef SPR2_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event tbmv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        transpose trans, diag unit_diag, int64_t n, int64_t k, const T* a,
                        int64_t lda, T* x, int64_t incx,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "tbmv", "for row_major layout");
}

#define TBMV_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,      \
                     int64_t n, int64_t k, const TYPE* a, int64_t lda, TYPE* x, int64_t incx,    \
                     const std::vector<sycl::event>& dependencies) {                             \
        return tbmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, k, \
                    a, lda, x, incx, dependencies);                                              \
    }

TBMV_LAUNCHER_USM(float, cublasStbmv)
TBMV_LAUNCHER_USM(double, cublasDtbmv)
TBMV_LAUNCHER_USM(std::complex<float>, cublasCtbmv)
TBMV_LAUNCHER_USM(std::complex<double>, cublasZtbmv)

#undef TBMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event tbsv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        transpose trans, diag unit_diag, int64_t n, int64_t k, const T* a,
                        int64_t lda, T* x, int64_t incx,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "tbsv", "for row_major layout");
}

#define TBSV_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,      \
                     int64_t n, int64_t k, const TYPE* a, int64_t lda, TYPE* x, int64_t incx,    \
                     const std::vector<sycl::event>& dependencies) {                             \
        return tbsv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, k, \
                    a, lda, x, incx, dependencies);                                              \
    }

TBSV_LAUNCHER_USM(float, cublasStbsv)
TBSV_LAUNCHER_USM(double, cublasDtbsv)
TBSV_LAUNCHER_USM(std::complex<float>, cublasCtbsv)
TBSV_LAUNCHER_USM(std::complex<double>, cublasZtbsv)

#undef TBSV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event tpmv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        transpose trans, diag unit_diag, int64_t n, const T* a, T* x, int64_t incx,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "tpmv", "for row_major layout");
}

#define TPMV_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,      \
                     int64_t n, const TYPE* a, TYPE* x, int64_t incx,                            \
                     const std::vector<sycl::event>& dependencies) {                             \
        return tpmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, \
                    x, incx, dependencies);                                                      \
    }

TPMV_LAUNCHER_USM(float, cublasStpmv)
TPMV_LAUNCHER_USM(double, cublasDtpmv)
TPMV_LAUNCHER_USM(std::complex<float>, cublasCtpmv)
TPMV_LAUNCHER_USM(std::complex<double>, cublasZtpmv)

#undef TPMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event tpsv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        transpose trans, diag unit_diag, int64_t n, const T* a, T* x, int64_t incx,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "tpsv", "for row_major layout");
}

#define TPSV_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,      \
                     int64_t n, const TYPE* a, TYPE* x, int64_t incx,                            \
                     const std::vector<sycl::event>& dependencies) {                             \
        return tpsv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, \
                    x, incx, dependencies);                                                      \
    }

TPSV_LAUNCHER_USM(float, cublasStpsv)
TPSV_LAUNCHER_USM(double, cublasDtpsv)
TPSV_LAUNCHER_USM(std::complex<float>, cublasCtpsv)
TPSV_LAUNCHER_USM(std::complex<double>, cublasZtpsv)

#undef TPSV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event trmv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        transpose trans, diag unit_diag, int64_t n, const T* a, int64_t lda, T* x,
                        int64_t incx, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "trmv", "for row_major layout");
}

#define TRMV_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event trmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,      \
                     int64_t n, const TYPE* a, int64_t lda, TYPE* x, int64_t incx,               \
                     const std::vector<sycl::event>& dependencies) {                             \
        return trmv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, \
                    lda, x, incx, dependencies);                                                 \
    }

TRMV_LAUNCHER_USM(float, cublasStrmv)
TRMV_LAUNCHER_USM(double, cublasDtrmv)
TRMV_LAUNCHER_USM(std::complex<float>, cublasCtrmv)
TRMV_LAUNCHER_USM(std::complex<double>, cublasZtrmv)

#undef TRMV_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event trsv(const char* func_name, Func func, sycl::queue& queue, uplo upper_lower,
                        transpose trans, diag unit_diag, int64_t n, const T* a, int64_t lda, T* x,
                        int64_t incx, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "trsv", "for row_major layout");
}

#define TRSV_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                  \
    sycl::event trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,      \
                     int64_t n, const TYPE* a, int64_t lda, TYPE* x, int64_t incx,               \
                     const std::vector<sycl::event>& dependencies) {                             \
        return trsv(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, upper_lower, trans, unit_diag, n, a, \
                    lda, x, incx, dependencies);                                                 \
    }

TRSV_LAUNCHER_USM(float, cublasStrsv)
TRSV_LAUNCHER_USM(double, cublasDtrsv)
TRSV_LAUNCHER_USM(std::complex<float>, cublasCtrsv)
TRSV_LAUNCHER_USM(std::complex<double>, cublasZtrsv)

#undef TRSV_LAUNCHER_USM

} // namespace row_major
} // namespace cublas
} // namespace blas
} // namespace math
} // namespace oneapi
