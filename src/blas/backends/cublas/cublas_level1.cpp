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

// Level 1
template <typename Func, typename T1, typename T2>
inline void asum(const char* func_name, Func func, sycl::queue& queue, int64_t n,
                 sycl::buffer<T1, 1>& x, const int64_t incx, sycl::buffer<T2, 1>& result) {
    using cuDataType1 = typename CudaEquivalentType<T1>::Type;
    using cuDataType2 = typename CudaEquivalentType<T2>::Type;
    overflow_check(n, incx);

    queue.submit([&](sycl::handler& cgh) {
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto res_acc = result.template get_access<sycl::access::mode::write>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            // By default the pointer mode is the CUBLAS_POINTER_MODE_HOST
            // when the data is on buffer, it must be set to
            // CUBLAS_POINTER_MODE_DEVICE mode otherwise it causes the segmentation
            // fault. When it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
            auto x_ = sc.get_mem<cuDataType1*>(x_acc);
            auto res_ = sc.get_mem<cuDataType2*>(res_acc);
            cublasStatus_t err;
            // ASUM does not support negative index
            cublas_native_named_func(func_name, func, err, handle, n, x_, std::abs(incx), res_);
            // Higher level BLAS functions expect CUBLAS_POINTER_MODE_HOST
            // to be set, therfore we need to reset this to the default value
            // in order to avoid CUDA_ERROR_ILLEGAL_ADRESS errors
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        });
    });
}

#define ASUM_LAUNCHER(TYPE1, TYPE2, CUBLAS_ROUTINE)                                         \
    void asum(sycl::queue& queue, int64_t n, sycl::buffer<TYPE1, 1>& x, const int64_t incx, \
              sycl::buffer<TYPE2, 1>& result) {                                             \
        asum(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, result);                   \
    }
ASUM_LAUNCHER(float, float, cublasSasum)
ASUM_LAUNCHER(double, double, cublasDasum)
ASUM_LAUNCHER(std::complex<float>, float, cublasScasum)
ASUM_LAUNCHER(std::complex<double>, double, cublasDzasum)
#undef ASUM_LAUNCHER

template <typename Func, typename T1, typename T2>
inline void scal(const char* func_name, Func func, sycl::queue& queue, int64_t n, T1 a,
                 sycl::buffer<T2, 1>& x, int64_t incx) {
    using cuDataType1 = typename CudaEquivalentType<T1>::Type;
    using cuDataType2 = typename CudaEquivalentType<T2>::Type;
    overflow_check(n, incx);
    queue.submit([&](sycl::handler& cgh) {
        auto x_acc = x.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto x_ = sc.get_mem<cuDataType2*>(x_acc);
            cublasStatus_t err;
            // SCAL does not support negative incx
            cublas_native_named_func(func_name, func, err, handle, n, (cuDataType1*)&a, x_,
                                     std::abs(incx));
        });
    });
}

#define SCAL_LAUNCHER(TYPE1, TYPE2, CUBLAS_ROUTINE)                                              \
    void scal(sycl::queue& queue, int64_t n, TYPE1 a, sycl::buffer<TYPE2, 1>& x, int64_t incx) { \
        scal(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, a, x, incx);                             \
    }
SCAL_LAUNCHER(float, float, cublasSscal)
SCAL_LAUNCHER(double, double, cublasDscal)
SCAL_LAUNCHER(std::complex<float>, std::complex<float>, cublasCscal)
SCAL_LAUNCHER(std::complex<double>, std::complex<double>, cublasZscal)
SCAL_LAUNCHER(float, std::complex<float>, cublasCsscal)
SCAL_LAUNCHER(double, std::complex<double>, cublasZdscal)
#undef SCAL_LAUNCHER

template <typename Func, typename T>
inline void axpy(const char* func_name, Func func, sycl::queue& queue, int64_t n, T alpha,
                 sycl::buffer<T, 1>& x, int64_t incx, sycl::buffer<T, 1>& y, int64_t incy) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    queue.submit([&](sycl::handler& cgh) {
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            auto y_ = sc.get_mem<cuDataType*>(y_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle, n, (cuDataType*)&alpha, x_, incx,
                                     y_, incy);
        });
    });
}

#define AXPY_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                      \
    void axpy(sycl::queue& queue, int64_t n, TYPE alpha, sycl::buffer<TYPE, 1>& x, int64_t incx, \
              sycl::buffer<TYPE, 1>& y, int64_t incy) {                                          \
        axpy(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, alpha, x, incx, y, incy);                \
    }

AXPY_LAUNCHER(float, cublasSaxpy)
AXPY_LAUNCHER(double, cublasDaxpy)
AXPY_LAUNCHER(std::complex<float>, cublasCaxpy)
AXPY_LAUNCHER(std::complex<double>, cublasZaxpy)
#undef AXPY_LAUNCHER

void axpby(sycl::queue& queue, int64_t n, float alpha, sycl::buffer<float, 1>& x, int64_t incx,
           float beta, sycl::buffer<float, 1>& y, int64_t incy) {
    throw unimplemented("blas", "axpby", "for column_major layout");
}

void axpby(sycl::queue& queue, int64_t n, double alpha, sycl::buffer<double, 1>& x, int64_t incx,
           double beta, sycl::buffer<double, 1>& y, int64_t incy) {
    throw unimplemented("blas", "axpby", "for column_major layout");
}

void axpby(sycl::queue& queue, int64_t n, std::complex<float> alpha,
           sycl::buffer<std::complex<float>, 1>& x, int64_t incx, std::complex<float> beta,
           sycl::buffer<std::complex<float>, 1>& y, int64_t incy) {
    throw unimplemented("blas", "axpby", "for column_major layout");
}

void axpby(sycl::queue& queue, int64_t n, std::complex<double> alpha,
           sycl::buffer<std::complex<double>, 1>& x, int64_t incx, std::complex<double> beta,
           sycl::buffer<std::complex<double>, 1>& y, int64_t incy) {
    throw unimplemented("blas", "axpby", "for column_major layout");
}

template <typename Func, typename T1, typename T2>
inline void rotg(const char* func_name, Func func, sycl::queue& queue, sycl::buffer<T1, 1>& a,
                 sycl::buffer<T1, 1>& b, sycl::buffer<T2, 1>& c, sycl::buffer<T1, 1>& s) {
    using cuDataType1 = typename CudaEquivalentType<T1>::Type;
    using cuDataType2 = typename CudaEquivalentType<T2>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read_write>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        auto s_acc = s.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            // By default the pointer mode is the CUBLAS_POINTER_MODE_HOST
            // when the data is on buffer, it must be set to
            // CUBLAS_POINTER_MODE_DEVICE mode otherwise it causes the segmentation
            // fault. When it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
            auto a_ = sc.get_mem<cuDataType1*>(a_acc);
            auto b_ = sc.get_mem<cuDataType1*>(b_acc);
            auto c_ = sc.get_mem<cuDataType2*>(c_acc);
            auto s_ = sc.get_mem<cuDataType1*>(s_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle, a_, b_, c_, s_);
            // Higher level BLAS functions expect CUBLAS_POINTER_MODE_HOST
            // to be set, therfore we need to reset this to the default value
            // in order to avoid CUDA_ERROR_ILLEGAL_ADRESS errors
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        });
    });
}

#define ROTG_LAUNCHER(TYPE1, TYPE2, CUBLAS_ROUTINE)                                     \
    void rotg(sycl::queue& queue, sycl::buffer<TYPE1, 1>& a, sycl::buffer<TYPE1, 1>& b, \
              sycl::buffer<TYPE2, 1>& c, sycl::buffer<TYPE1, 1>& s) {                   \
        rotg(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, a, b, c, s);                       \
    }

ROTG_LAUNCHER(float, float, cublasSrotg)
ROTG_LAUNCHER(double, double, cublasDrotg)
ROTG_LAUNCHER(std::complex<float>, float, cublasCrotg)
ROTG_LAUNCHER(std::complex<double>, double, cublasZrotg)
#undef ROTG_LAUNCHER

template <typename Func, typename T>
inline void rotm(const char* func_name, Func func, sycl::queue& queue, int64_t n,
                 sycl::buffer<T, 1>& x, int64_t incx, sycl::buffer<T, 1>& y, int64_t incy,
                 sycl::buffer<T, 1>& param) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    queue.submit([&](sycl::handler& cgh) {
        auto x_acc = x.template get_access<sycl::access::mode::read_write>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        auto param_acc = param.template get_access<sycl::access::mode::read>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            // By default the pointer mode is the CUBLAS_POINTER_MODE_HOST
            // when the data is on buffer, it must be set to
            // CUBLAS_POINTER_MODE_DEVICE mode otherwise it causes the segmentation
            // fault. When it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            auto y_ = sc.get_mem<cuDataType*>(y_acc);
            auto param_ = sc.get_mem<cuDataType*>(param_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle, n, x_, incx, y_, incy, param_);
            // Higher level BLAS functions expect CUBLAS_POINTER_MODE_HOST
            // to be set, therfore we need to reset this to the default value
            // in order to avoid CUDA_ERROR_ILLEGAL_ADRESS errors
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        });
    });
}

#define ROTM_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                           \
    void rotm(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, int64_t incx,  \
              sycl::buffer<TYPE, 1>& y, int64_t incy, sycl::buffer<TYPE, 1>& param) { \
        rotm(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, y, incy, param);     \
    }

ROTM_LAUNCHER(float, cublasSrotm)
ROTM_LAUNCHER(double, cublasDrotm)
#undef ROTM_LAUNCHER

template <typename Func, typename T>
inline void copy(const char* func_name, Func func, sycl::queue& queue, int64_t n,
                 sycl::buffer<T, 1>& x, int64_t incx, sycl::buffer<T, 1>& y, int64_t incy) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    queue.submit([&](sycl::handler& cgh) {
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            auto y_ = sc.get_mem<cuDataType*>(y_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle, n, x_, incx, y_, incy);
        });
    });
}

#define COPY_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                          \
    void copy(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, int64_t incx, \
              sycl::buffer<TYPE, 1>& y, int64_t incy) {                              \
        copy(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, y, incy);           \
    }

COPY_LAUNCHER(float, cublasScopy)
COPY_LAUNCHER(double, cublasDcopy)
COPY_LAUNCHER(std::complex<float>, cublasCcopy)
COPY_LAUNCHER(std::complex<double>, cublasZcopy)
#undef COPY_LAUNCHER

template <typename Func, typename T>
inline void dot(const char* func_name, Func func, sycl::queue& queue, int64_t n,
                sycl::buffer<T, 1>& x, const int64_t incx, sycl::buffer<T, 1>& y, int64_t incy,
                sycl::buffer<T, 1>& result) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    queue.submit([&](sycl::handler& cgh) {
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read>(cgh);
        auto res_acc = result.template get_access<sycl::access::mode::write>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            // By default the pointer mode is the CUBLAS_POINTER_MODE_HOST
            // when the data is on buffer, it must be set to
            // CUBLAS_POINTER_MODE_DEVICE mode otherwise it causes the segmentation
            // fault. When it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            auto y_ = sc.get_mem<cuDataType*>(y_acc);
            auto res_ = sc.get_mem<cuDataType*>(res_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle, n, x_, incx, y_, incy, res_);
            // Higher level BLAS functions expect CUBLAS_POINTER_MODE_HOST
            // to be set, therfore we need to reset this to the default value
            // in order to avoid CUDA_ERROR_ILLEGAL_ADRESS errors
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        });
    });
}

#define DOT_LAUNCHER(EXT, TYPE, CUBLAS_ROUTINE)                                                  \
    void dot##EXT(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, const int64_t incx,   \
                  sycl::buffer<TYPE, 1>& y, const int64_t incy, sycl::buffer<TYPE, 1>& result) { \
        dot(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, y, incy, result);                \
    }
DOT_LAUNCHER(, float, cublasSdot)
DOT_LAUNCHER(, double, cublasDdot)
DOT_LAUNCHER(c, std::complex<float>, cublasCdotc)
DOT_LAUNCHER(c, std::complex<double>, cublasZdotc)
DOT_LAUNCHER(u, std::complex<float>, cublasCdotu)
DOT_LAUNCHER(u, std::complex<double>, cublasZdotu)
#undef DOT_LAUNCHER

template <typename Func, typename T1, typename T2, typename T3>
inline void rot(const char* func_name, Func func, sycl::queue& queue, int64_t n,
                sycl::buffer<T1, 1>& x, const int64_t incx, sycl::buffer<T1, 1>& y, int64_t incy,
                T2 c, T3 s) {
    using cuDataType1 = typename CudaEquivalentType<T1>::Type;
    using cuDataType2 = typename CudaEquivalentType<T2>::Type;
    using cuDataType3 = typename CudaEquivalentType<T3>::Type;
    overflow_check(n, incx, incy);
    queue.submit([&](sycl::handler& cgh) {
        auto x_acc = x.template get_access<sycl::access::mode::read_write>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            // By default the pointer mode is the CUBLAS_POINTER_MODE_HOST
            // when the data is on buffer, it must be set to
            // CUBLAS_POINTER_MODE_DEVICE mode otherwise it causes the segmentation
            // fault. When it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            // cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
            auto x_ = sc.get_mem<cuDataType1*>(x_acc);
            auto y_ = sc.get_mem<cuDataType1*>(y_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle, n, x_, incx, y_, incy,
                                     (cuDataType2*)&c, (cuDataType3*)&s);
        });
    });
}

#define ROT_LAUNCHER(TYPE1, TYPE2, TYPE3, CUBLAS_ROUTINE)                                  \
    void rot(sycl::queue& queue, int64_t n, sycl::buffer<TYPE1, 1>& x, const int64_t incx, \
             sycl::buffer<TYPE1, 1>& y, int64_t incy, TYPE2 c, TYPE3 s) {                  \
        rot(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, y, incy, c, s);            \
    }

ROT_LAUNCHER(float, float, float, cublasSrot)
ROT_LAUNCHER(double, double, double, cublasDrot)
ROT_LAUNCHER(std::complex<float>, float, float, cublasCsrot)
ROT_LAUNCHER(std::complex<double>, double, double, cublasZdrot)
#undef ROT_LAUNCHER

void sdsdot(sycl::queue& queue, int64_t n, float sb, sycl::buffer<float, 1>& x, int64_t incx,
            sycl::buffer<float, 1>& y, int64_t incy, sycl::buffer<float, 1>& result) {
    overflow_check(n, incx, incy);
    // cuBLAS does not support sdot so we need to mimic sdot.
    queue.submit([&](sycl::handler& cgh) {
        auto x_acc = x.get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.get_access<sycl::access::mode::read>(cgh);
        auto res_acc = result.get_access<sycl::access::mode::write>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            // By default the pointer mode is the CUBLAS_POINTER_MODE_HOST
            // when the data is on buffer, it must be set to
            // CUBLAS_POINTER_MODE_DEVICE mode otherwise it causes the segmentation
            // fault. When it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
            auto x_ = sc.get_mem<float*>(x_acc);
            auto y_ = sc.get_mem<float*>(y_acc);
            auto res_ = sc.get_mem<float*>(res_acc);
            cublasStatus_t err;
            cublas_native_func(cublasSdot, err, handle, n, x_, incx, y_, incy, res_);
            // Higher level BLAS functions expect CUBLAS_POINTER_MODE_HOST
            // to be set, therfore we need to reset this to the default value
            // in order to avoid CUDA_ERROR_ILLEGAL_ADRESS errors
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        });
    });
    // Since SB is a host pointer we need to bring the result back to the host and
    // add sb to it.
    result.get_host_access(sycl::read_write)[0] += sb;
}

void dot(sycl::queue& queue, int64_t n, sycl::buffer<float, 1>& x, int64_t incx,
         sycl::buffer<float, 1>& y, int64_t incy, sycl::buffer<double, 1>& result) {
    throw unimplemented("blas", "dot", "for column_major layout");
}

template <typename Func, typename T>
inline void rotmg(const char* func_name, Func func, sycl::queue& queue, sycl::buffer<T, 1>& d1,
                  sycl::buffer<T, 1>& d2, sycl::buffer<T, 1>& x1, T y1, sycl::buffer<T, 1>& param) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    sycl::buffer<T, 1> y1_buff(&y1, sycl::range<1>(1));
    queue.submit([&](sycl::handler& cgh) {
        auto d1_acc = d1.template get_access<sycl::access::mode::read_write>(cgh);
        auto d2_acc = d2.template get_access<sycl::access::mode::read_write>(cgh);
        auto x1_acc = x1.template get_access<sycl::access::mode::read_write>(cgh);
        auto y1_acc = y1_buff.template get_access<sycl::access::mode::read>(cgh);
        auto param_acc = param.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            // By default the pointer mode is the CUBLAS_POINTER_MODE_HOST
            // when the data is on buffer, it must be set to
            // CUBLAS_POINTER_MODE_DEVICE mode otherwise it causes the segmentation
            // fault. When it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
            auto d1_ = sc.get_mem<cuDataType*>(d1_acc);
            auto d2_ = sc.get_mem<cuDataType*>(d2_acc);
            auto x1_ = sc.get_mem<cuDataType*>(x1_acc);
            auto y1_ = sc.get_mem<cuDataType*>(y1_acc);
            auto param_ = sc.get_mem<cuDataType*>(param_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle, d1_, d2_, x1_, y1_, param_);
            // Higher level BLAS functions expect CUBLAS_POINTER_MODE_HOST
            // to be set, therfore we need to reset this to the default value
            // in order to avoid CUDA_ERROR_ILLEGAL_ADRESS errors
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        });
    });
}

#define ROTMG_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                             \
    void rotmg(sycl::queue& queue, sycl::buffer<TYPE, 1>& d1, sycl::buffer<TYPE, 1>& d2, \
               sycl::buffer<TYPE, 1>& x1, TYPE y1, sycl::buffer<TYPE, 1>& param) {       \
        rotmg(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, d1, d2, x1, y1, param);            \
    }

ROTMG_LAUNCHER(float, cublasSrotmg)
ROTMG_LAUNCHER(double, cublasDrotmg)
#undef ROTMG_LAUNCHER

template <typename Func, typename T>
inline void iamax(const char* func_name, Func func, sycl::queue& queue, int64_t n,
                  sycl::buffer<T, 1>& x, const int64_t incx, sycl::buffer<int64_t, 1>& result) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx);
    // cuBLAS does not support int64_t as return type for the data. So we need to
    // mimic iamax. We are converting the result to be the int and then we convert
    // it back to the actual data on the host.
    // This change may cause failure as the result of integer overflow
    // based on the size. Alternatively either we need to write a sycl kernel
    // to elementwise copy the data between two buffer, or allow reinterpret cast
    // to convert to different type with different typesize size.
    sycl::buffer<int, 1> int_res_buff{ sycl::range<1>(1) };
    queue.submit([&](sycl::handler& cgh) {
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto int_res_acc = int_res_buff.template get_access<sycl::access::mode::write>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            // By default the pointer mode is the CUBLAS_POINTER_MODE_HOST
            // when the data is on buffer, it must be set to
            // CUBLAS_POINTER_MODE_DEVICE mode otherwise it causes the segmentation
            // fault. When it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            auto int_res_ = sc.get_mem<int*>(int_res_acc);
            cublasStatus_t err;
            // For negative incx, iamax returns 0. This behaviour is similar to that of
            // reference netlib BLAS.
            cublas_native_named_func(func_name, func, err, handle, n, x_, incx, int_res_);
            // Higher level BLAS functions expect CUBLAS_POINTER_MODE_HOST
            // to be set, therfore we need to reset this to the default value
            // in order to avoid CUDA_ERROR_ILLEGAL_ADRESS errors
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        });
    });

    queue.submit([&](sycl::handler& cgh) {
        auto int_res_acc = int_res_buff.template get_access<sycl::access::mode::read>(cgh);
        auto result_acc = result.template get_access<sycl::access::mode::write>(cgh);
        cgh.single_task(
            [=]() { result_acc[0] = std::max((int64_t)int_res_acc[0] - 1, (int64_t)0); });
    });
}

#define IAMAX_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                \
    void iamax(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, const int64_t incx, \
               sycl::buffer<int64_t, 1>& result) {                                          \
        iamax(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, result);                  \
    }
IAMAX_LAUNCHER(float, cublasIsamax)
IAMAX_LAUNCHER(double, cublasIdamax)
IAMAX_LAUNCHER(std::complex<float>, cublasIcamax)
IAMAX_LAUNCHER(std::complex<double>, cublasIzamax)
#undef IAMAX_LAUNCHER

template <typename Func, typename T>
inline void swap(const char* func_name, Func func, sycl::queue& queue, int64_t n,
                 sycl::buffer<T, 1>& x, int64_t incx, sycl::buffer<T, 1>& y, int64_t incy) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    queue.submit([&](sycl::handler& cgh) {
        auto x_acc = x.template get_access<sycl::access::mode::read_write>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            auto y_ = sc.get_mem<cuDataType*>(y_acc);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle, n, x_, incx, y_, incy);
        });
    });
}

#define SWAP_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                          \
    void swap(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, int64_t incx, \
              sycl::buffer<TYPE, 1>& y, int64_t incy) {                              \
        swap(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, y, incy);           \
    }

SWAP_LAUNCHER(float, cublasSswap)
SWAP_LAUNCHER(double, cublasDswap)
SWAP_LAUNCHER(std::complex<float>, cublasCswap)
SWAP_LAUNCHER(std::complex<double>, cublasZswap)
#undef SWAP_LAUNCHER

template <typename Func, typename T>
inline void iamin(const char* func_name, Func func, sycl::queue& queue, int64_t n,
                  sycl::buffer<T, 1>& x, const int64_t incx, sycl::buffer<int64_t, 1>& result) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx);
    // cuBLAS does not support int64_t as return type for the data. So we need to
    // mimic iamin we are converting the result to be the int and then we convert
    // it back to the actual data on the host.
    // This change may cause failure as the result of integer overflow
    // based on the size. Alternatively, either we need to write a sycl kernel
    // to elementwise copy the data between two buffer, or allow reinterpret cast
    // to convert to different type with different typesize size.
    sycl::buffer<int, 1> int_res_buff{ sycl::range<1>(1) };
    queue.submit([&](sycl::handler& cgh) {
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto int_res_acc = int_res_buff.template get_access<sycl::access::mode::write>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            // By default the pointer mode is the CUBLAS_POINTER_MODE_HOST
            // when the data is on buffer, it must be set to
            // CUBLAS_POINTER_MODE_DEVICE mode otherwise it causes the segmentation
            // fault. When it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
            auto x_ = sc.get_mem<cuDataType*>(x_acc);
            auto int_res_ = sc.get_mem<int*>(int_res_acc);
            cublasStatus_t err;
            // For negative incx, iamin returns 0. This behaviour is similar to that of
            // implemented as a reference IAMIN.
            cublas_native_named_func(func_name, func, err, handle, n, x_, incx, int_res_);
            // Higher level BLAS functions expect CUBLAS_POINTER_MODE_HOST
            // to be set, therfore we need to reset this to the default value
            // in order to avoid CUDA_ERROR_ILLEGAL_ADRESS errors
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        });
    });

    queue.submit([&](sycl::handler& cgh) {
        auto int_res_acc = int_res_buff.template get_access<sycl::access::mode::read>(cgh);
        auto result_acc = result.template get_access<sycl::access::mode::write>(cgh);
        cgh.single_task(
            [=]() { result_acc[0] = std::max((int64_t)int_res_acc[0] - 1, (int64_t)0); });
    });
}

#define IAMIN_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                \
    void iamin(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, const int64_t incx, \
               sycl::buffer<int64_t, 1>& result) {                                          \
        iamin(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, result);                  \
    }
IAMIN_LAUNCHER(float, cublasIsamin)
IAMIN_LAUNCHER(double, cublasIdamin)
IAMIN_LAUNCHER(std::complex<float>, cublasIcamin)
IAMIN_LAUNCHER(std::complex<double>, cublasIzamin)
#undef IAMIN_LAUNCHER

template <typename Func, typename T1, typename T2>
inline void nrm2(const char* func_name, Func func, sycl::queue& queue, int64_t n,
                 sycl::buffer<T1, 1>& x, const int64_t incx, sycl::buffer<T2, 1>& result) {
    using cuDataType1 = typename CudaEquivalentType<T1>::Type;
    using cuDataType2 = typename CudaEquivalentType<T2>::Type;
    overflow_check(n, incx);

    queue.submit([&](sycl::handler& cgh) {
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto res_acc = result.template get_access<sycl::access::mode::write>(cgh);
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            // By default the pointer mode is the CUBLAS_POINTER_MODE_HOST
            // when the data is on buffer, it must be set to
            // CUBLAS_POINTER_MODE_DEVICE mode otherwise it causes the segmentation
            // fault. When it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
            auto x_ = sc.get_mem<cuDataType1*>(x_acc);
            auto res_ = sc.get_mem<cuDataType2*>(res_acc);
            cublasStatus_t err;
            // NRM2 does not support negative index
            cublas_native_named_func(func_name, func, err, handle, n, x_, std::abs(incx), res_);
            // Higher level BLAS functions expect CUBLAS_POINTER_MODE_HOST
            // to be set, therfore we need to reset this to the default value
            // in order to avoid CUDA_ERROR_ILLEGAL_ADRESS errors
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        });
    });
}

#define NRM2_LAUNCHER(TYPE1, TYPE2, CUBLAS_ROUTINE)                                         \
    void nrm2(sycl::queue& queue, int64_t n, sycl::buffer<TYPE1, 1>& x, const int64_t incx, \
              sycl::buffer<TYPE2, 1>& result) {                                             \
        nrm2(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, result);                   \
    }
NRM2_LAUNCHER(float, float, cublasSnrm2)
NRM2_LAUNCHER(double, double, cublasDnrm2)
NRM2_LAUNCHER(std::complex<float>, float, cublasScnrm2)
NRM2_LAUNCHER(std::complex<double>, double, cublasDznrm2)
#undef NRM2_LAUNCHER

// USM APIs

// Level 1
template <typename Func, typename T1, typename T2>
inline sycl::event asum(const char* func_name, Func func, sycl::queue& queue, int64_t n,
                        const T1* x, const int64_t incx, T2* result,
                        const std::vector<sycl::event>& dependencies) {
    using cuDataType1 = typename CudaEquivalentType<T1>::Type;
    using cuDataType2 = typename CudaEquivalentType<T2>::Type;
    overflow_check(n, incx);

    bool result_on_device =
        sycl::get_pointer_type(result, queue.get_context()) == sycl::usm::alloc::device;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto x_ = reinterpret_cast<const cuDataType1*>(x);
            auto res_ = reinterpret_cast<cuDataType2*>(result);
            if (result_on_device) {
                cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
            }
            cublasStatus_t err;
            // ASUM does not support negative index
            cublas_native_named_func(func_name, func, err, handle, n, x_, std::abs(incx), res_);
            if (result_on_device) {
                cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
            }
        });
    });
    return done;
}

#define ASUM_LAUNCHER_USM(TYPE1, TYPE2, CUBLAS_ROUTINE)                                        \
    sycl::event asum(sycl::queue& queue, int64_t n, const TYPE1* x, const int64_t incx,        \
                     TYPE2* result, const std::vector<sycl::event>& dependencies) {            \
        return asum(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, result, dependencies); \
    }
ASUM_LAUNCHER_USM(float, float, cublasSasum)
ASUM_LAUNCHER_USM(double, double, cublasDasum)
ASUM_LAUNCHER_USM(std::complex<float>, float, cublasScasum)
ASUM_LAUNCHER_USM(std::complex<double>, double, cublasDzasum)
#undef ASUM_LAUNCHER_USM

template <typename Func, typename T1, typename T2>
inline sycl::event scal(const char* func_name, Func func, sycl::queue& queue, int64_t n, T1 a,
                        T2* x, int64_t incx, const std::vector<sycl::event>& dependencies) {
    using cuDataType1 = typename CudaEquivalentType<T1>::Type;
    using cuDataType2 = typename CudaEquivalentType<T2>::Type;
    overflow_check(n, incx);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto x_ = reinterpret_cast<cuDataType2*>(x);
            cublasStatus_t err;
            // SCAL does not support negative incx
            cublas_native_named_func(func_name, func, err, handle, n, (cuDataType1*)&a, x_,
                                     std::abs(incx));
        });
    });
    return done;
}

#define SCAL_LAUNCHER_USM(TYPE1, TYPE2, CUBLAS_ROUTINE)                                   \
    sycl::event scal(sycl::queue& queue, int64_t n, TYPE1 a, TYPE2* x, int64_t incx,      \
                     const std::vector<sycl::event>& dependencies) {                      \
        return scal(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, a, x, incx, dependencies); \
    }
SCAL_LAUNCHER_USM(float, float, cublasSscal)
SCAL_LAUNCHER_USM(double, double, cublasDscal)
SCAL_LAUNCHER_USM(std::complex<float>, std::complex<float>, cublasCscal)
SCAL_LAUNCHER_USM(std::complex<double>, std::complex<double>, cublasZscal)
SCAL_LAUNCHER_USM(float, std::complex<float>, cublasCsscal)
SCAL_LAUNCHER_USM(double, std::complex<double>, cublasZdscal)
#undef SCAL_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event axpy(const char* func_name, Func func, sycl::queue& queue, int64_t n, T alpha,
                        const T* x, int64_t incx, T* y, int64_t incy,
                        const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto x_ = reinterpret_cast<const cuDataType*>(x);
            auto y_ = reinterpret_cast<cuDataType*>(y);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle, n, (cuDataType*)&alpha, x_, incx,
                                     y_, incy);
        });
    });
    return done;
}

#define AXPY_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                              \
    sycl::event axpy(sycl::queue& queue, int64_t n, TYPE alpha, const TYPE* x, int64_t incx, \
                     TYPE* y, int64_t incy, const std::vector<sycl::event>& dependencies) {  \
        return axpy(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, alpha, x, incx, y, incy,      \
                    dependencies);                                                           \
    }

AXPY_LAUNCHER_USM(float, cublasSaxpy)
AXPY_LAUNCHER_USM(double, cublasDaxpy)
AXPY_LAUNCHER_USM(std::complex<float>, cublasCaxpy)
AXPY_LAUNCHER_USM(std::complex<double>, cublasZaxpy)
#undef AXPY_LAUNCHER_USM

sycl::event axpby(sycl::queue& queue, int64_t n, float alpha, const float* x, int64_t incx,
                  float beta, float* y, int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpby", "for column_major layout");
}
sycl::event axpby(sycl::queue& queue, int64_t n, double alpha, const double* x, int64_t incx,
                  double beta, double* y, int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpby", "for column_major layout");
}
sycl::event axpby(sycl::queue& queue, int64_t n, std::complex<float> alpha,
                  const std::complex<float>* x, int64_t incx, std::complex<float> beta,
                  std::complex<float>* y, int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpby", "for column_major layout");
}
sycl::event axpby(sycl::queue& queue, int64_t n, std::complex<double> alpha,
                  const std::complex<double>* x, int64_t incx, std::complex<double> beta,
                  std::complex<double>* y, int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpby", "for column_major layout");
}

template <typename Func, typename T1, typename T2>
inline sycl::event rotg(const char* func_name, Func func, sycl::queue& queue, T1* a, T1* b, T2* c,
                        T1* s, const std::vector<sycl::event>& dependencies) {
    using cuDataType1 = typename CudaEquivalentType<T1>::Type;
    using cuDataType2 = typename CudaEquivalentType<T2>::Type;
    auto ctx = queue.get_context();
    bool results_on_device = (sycl::get_pointer_type(a, ctx) == sycl::usm::alloc::device ||
                              sycl::get_pointer_type(b, ctx) == sycl::usm::alloc::device ||
                              sycl::get_pointer_type(c, ctx) == sycl::usm::alloc::device ||
                              sycl::get_pointer_type(s, ctx) == sycl::usm::alloc::device);
    if (results_on_device) {
        if (sycl::get_pointer_type(a, ctx) == sycl::usm::alloc::unknown ||
            sycl::get_pointer_type(b, ctx) == sycl::usm::alloc::unknown ||
            sycl::get_pointer_type(c, ctx) == sycl::usm::alloc::unknown ||
            sycl::get_pointer_type(s, ctx) == sycl::usm::alloc::unknown) {
            throw oneapi::math::exception(
                "blas", "rotg",
                "If any pointer is only device accessible, all must be device accessible");
        }
    }
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType1*>(a);
            auto b_ = reinterpret_cast<cuDataType1*>(b);
            auto c_ = reinterpret_cast<cuDataType2*>(c);
            auto s_ = reinterpret_cast<cuDataType1*>(s);
            if (results_on_device) {
                cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
            }
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle, a_, b_, c_, s_);
            if (results_on_device) {
                cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
            }
        });
    });
    return done;
}

#define ROTG_LAUNCHER_USM(TYPE1, TYPE2, CUBLAS_ROUTINE)                                \
    sycl::event rotg(sycl::queue& queue, TYPE1* a, TYPE1* b, TYPE2* c, TYPE1* s,       \
                     const std::vector<sycl::event>& dependencies) {                   \
        return rotg(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, a, b, c, s, dependencies); \
    }

ROTG_LAUNCHER_USM(float, float, cublasSrotg)
ROTG_LAUNCHER_USM(double, double, cublasDrotg)
ROTG_LAUNCHER_USM(std::complex<float>, float, cublasCrotg)
ROTG_LAUNCHER_USM(std::complex<double>, double, cublasZrotg)
#undef ROTG_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event rotm(const char* func_name, Func func, sycl::queue& queue, int64_t n, T* x,
                        int64_t incx, T* y, int64_t incy, T* param,
                        const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto x_ = reinterpret_cast<cuDataType*>(x);
            auto y_ = reinterpret_cast<cuDataType*>(y);
            auto param_ = reinterpret_cast<cuDataType*>(param);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle, n, x_, incx, y_, incy, param_);
        });
    });
    return done;
}

#define ROTM_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                   \
    sycl::event rotm(sycl::queue& queue, int64_t n, TYPE* x, int64_t incx, TYPE* y, int64_t incy, \
                     TYPE* param, const std::vector<sycl::event>& dependencies) {                 \
        return rotm(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, y, incy, param,           \
                    dependencies);                                                                \
    }

ROTM_LAUNCHER_USM(float, cublasSrotm)
ROTM_LAUNCHER_USM(double, cublasDrotm)
#undef ROTM_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event copy(const char* func_name, Func func, sycl::queue& queue, int64_t n, const T* x,
                        int64_t incx, T* y, int64_t incy,
                        const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto x_ = reinterpret_cast<const cuDataType*>(x);
            auto y_ = reinterpret_cast<cuDataType*>(y);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle, n, x_, incx, y_, incy);
        });
    });
    return done;
}

#define COPY_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                 \
    sycl::event copy(sycl::queue& queue, int64_t n, const TYPE* x, int64_t incx, TYPE* y,       \
                     int64_t incy, const std::vector<sycl::event>& dependencies) {              \
        return copy(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, y, incy, dependencies); \
    }

COPY_LAUNCHER_USM(float, cublasScopy)
COPY_LAUNCHER_USM(double, cublasDcopy)
COPY_LAUNCHER_USM(std::complex<float>, cublasCcopy)
COPY_LAUNCHER_USM(std::complex<double>, cublasZcopy)
#undef COPY_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event dot(const char* func_name, Func func, sycl::queue& queue, int64_t n, const T* x,
                       const int64_t incx, const T* y, int64_t incy, T* result,
                       const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    bool result_on_device =
        sycl::get_pointer_type(result, queue.get_context()) == sycl::usm::alloc::device;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto x_ = reinterpret_cast<const cuDataType*>(x);
            auto y_ = reinterpret_cast<const cuDataType*>(y);
            auto res_ = reinterpret_cast<cuDataType*>(result);
            if (result_on_device) {
                cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
            }
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle, n, x_, incx, y_, incy, res_);
            if (result_on_device) {
                cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
            }
        });
    });
    return done;
}

#define DOT_LAUNCHER_USM(EXT, TYPE, CUBLAS_ROUTINE)                                        \
    sycl::event dot##EXT(sycl::queue& queue, int64_t n, const TYPE* x, const int64_t incx, \
                         const TYPE* y, const int64_t incy, TYPE* result,                  \
                         const std::vector<sycl::event>& dependencies) {                   \
        return dot(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, y, incy, result,    \
                   dependencies);                                                          \
    }
DOT_LAUNCHER_USM(, float, cublasSdot)
DOT_LAUNCHER_USM(, double, cublasDdot)
DOT_LAUNCHER_USM(c, std::complex<float>, cublasCdotc)
DOT_LAUNCHER_USM(c, std::complex<double>, cublasZdotc)
DOT_LAUNCHER_USM(u, std::complex<float>, cublasCdotu)
DOT_LAUNCHER_USM(u, std::complex<double>, cublasZdotu)
#undef DOT_LAUNCHER_USM

template <typename Func, typename T1, typename T2, typename T3>
inline sycl::event rot(const char* func_name, Func func, sycl::queue& queue, int64_t n, T1* x,
                       const int64_t incx, T1* y, int64_t incy, T2 c, T3 s,
                       const std::vector<sycl::event>& dependencies) {
    using cuDataType1 = typename CudaEquivalentType<T1>::Type;
    using cuDataType2 = typename CudaEquivalentType<T2>::Type;
    using cuDataType3 = typename CudaEquivalentType<T3>::Type;
    overflow_check(n, incx, incy);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto x_ = reinterpret_cast<cuDataType1*>(x);
            auto y_ = reinterpret_cast<cuDataType1*>(y);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle, n, x_, incx, y_, incy,
                                     (cuDataType2*)&c, (cuDataType3*)&s);
        });
    });
    return done;
}

#define ROT_LAUNCHER_USM(TYPE1, TYPE2, TYPE3, CUBLAS_ROUTINE)                              \
    sycl::event rot(sycl::queue& queue, int64_t n, TYPE1* x, const int64_t incx, TYPE1* y, \
                    int64_t incy, TYPE2 c, TYPE3 s,                                        \
                    const std::vector<sycl::event>& dependencies) {                        \
        return rot(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, y, incy, c, s,      \
                   dependencies);                                                          \
    }

ROT_LAUNCHER_USM(float, float, float, cublasSrot)
ROT_LAUNCHER_USM(double, double, double, cublasDrot)
ROT_LAUNCHER_USM(std::complex<float>, float, float, cublasCsrot)
ROT_LAUNCHER_USM(std::complex<double>, double, double, cublasZdrot)
#undef ROT_LAUNCHER_USM

sycl::event sdsdot(sycl::queue& queue, int64_t n, float sb, const float* x, int64_t incx,
                   const float* y, int64_t incy, float* result,
                   const std::vector<sycl::event>& dependencies) {
    overflow_check(n, incx, incy);
    bool result_on_device =
        sycl::get_pointer_type(result, queue.get_context()) == sycl::usm::alloc::device;
    // cuBLAS does not support sdsdot so we need to mimic sdot.
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto x_ = reinterpret_cast<const float*>(x);
            auto y_ = reinterpret_cast<const float*>(y);
            auto res_ = reinterpret_cast<float*>(result);
            if (result_on_device) {
                cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
            }
            cublasStatus_t err;
            cublas_native_func(cublasSdot, err, handle, n, x_, incx, y_, incy, res_);
            if (result_on_device) {
                cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
            }
        });
    });
    done.wait();
    if (result_on_device) {
        // The following does copy device to host and then host to device
        // just to adjust with sb constant. This is pretty inefficient, and
        // should maybe be replaced with a sycl GPU kernel, but it duplicated what
        // is done in the buffer API
        float host_result;
        queue.memcpy(&host_result, result, sizeof(float)).wait();
        host_result += sb;
        auto last_ev = queue.memcpy(result, &host_result, sizeof(float));
        return last_ev;
    }
    else {
        result[0] = result[0] + sb;
        return done;
    }
}

sycl::event dot(sycl::queue& queue, int64_t n, const float* x, int64_t incx, const float* y,
                int64_t incy, double* result, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dot", "for column_major layout");
}

template <typename Func, typename T>
inline sycl::event rotmg(const char* func_name, Func func, sycl::queue& queue, T* d1, T* d2, T* x1,
                         T y1, T* param, const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    auto ctx = queue.get_context();
    bool results_on_device = (sycl::get_pointer_type(d1, ctx) == sycl::usm::alloc::device ||
                              sycl::get_pointer_type(d2, ctx) == sycl::usm::alloc::device ||
                              sycl::get_pointer_type(x1, ctx) == sycl::usm::alloc::device);
    if (results_on_device) {
        if (sycl::get_pointer_type(d1, ctx) == sycl::usm::alloc::unknown ||
            sycl::get_pointer_type(d2, ctx) == sycl::usm::alloc::unknown ||
            sycl::get_pointer_type(x1, ctx) == sycl::usm::alloc::unknown) {
            throw oneapi::math::exception(
                "blas", "rotmg",
                "If any pointer is only device accessible, all must be device accessible");
        }
    }
    cuDataType* y1_;
    if (results_on_device) {
        y1_ = sycl::malloc_device<cuDataType>(1, queue);
        queue.memcpy(y1_, &y1, sizeof(cuDataType)).wait();
    }
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto d1_ = reinterpret_cast<cuDataType*>(d1);
            auto d2_ = reinterpret_cast<cuDataType*>(d2);
            auto x1_ = reinterpret_cast<cuDataType*>(x1);
            auto param_ = reinterpret_cast<cuDataType*>(param);
            cublasStatus_t err;
            if (results_on_device) {
                cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
                cublas_native_named_func(func_name, func, err, handle, d1_, d2_, x1_, y1_, param_);
                cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
            }
            else {
                auto y1_c = reinterpret_cast<const cuDataType*>(&y1);
                cublas_native_named_func(func_name, func, err, handle, d1_, d2_, x1_, y1_c, param_);
            }
        });
    });
    if (results_on_device) {
        done.wait();
        queue.memcpy(&y1, y1_, sizeof(cuDataType)).wait();
        sycl::free(y1_, queue);
    }
    return done;
}

#define ROTMG_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                   \
    sycl::event rotmg(sycl::queue& queue, TYPE* d1, TYPE* d2, TYPE* x1, TYPE y1, TYPE* param,      \
                      const std::vector<sycl::event>& dependencies) {                              \
        return rotmg(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, d1, d2, x1, y1, param, dependencies); \
    }

ROTMG_LAUNCHER_USM(float, cublasSrotmg)
ROTMG_LAUNCHER_USM(double, cublasDrotmg)
#undef ROTMG_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event iamax(const char* func_name, Func func, sycl::queue& queue, int64_t n,
                         const T* x, const int64_t incx, int64_t* result,
                         const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx);
    // cuBLAS does not support int64_t as return type for the data. So we need to
    // mimic iamax. We are converting the result to be the int and then we convert
    // it back to the actual data on the host.
    // This change may cause failure as the result of integer overflow
    // based on the size.
    int int_res = 0;
    int* int_res_p = nullptr;
    bool result_on_device =
        sycl::get_pointer_type(result, queue.get_context()) == sycl::usm::alloc::device;
    if (result_on_device) {
        int_res_p = sycl::malloc_device<int>(1, queue);
    }
    else {
        int_res_p = &int_res;
    }
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto x_ = reinterpret_cast<const cuDataType*>(x);
            if (result_on_device) {
                cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
            }
            cublasStatus_t err;
            // For negative incx, iamax returns 0. This behaviour is similar to that of
            // reference iamax.
            cublas_native_named_func(func_name, func, err, handle, n, x_, incx, int_res_p);
            if (result_on_device) {
                cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
            }
        });
    });
    done.wait();
    if (result_on_device) {
        auto last_ev = queue.submit([&](sycl::handler& cgh) {
            cgh.single_task([=]() { *result = std::max((int64_t)*int_res_p - 1, (int64_t)0); });
        });
        last_ev.wait();
        sycl::free(int_res_p, queue);
        return last_ev;
    }
    else {
        result[0] = std::max((int64_t)(*int_res_p - 1), int64_t{ 0 });
        return done;
    }
}

#define IAMAX_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                \
    sycl::event iamax(sycl::queue& queue, int64_t n, const TYPE* x, const int64_t incx,         \
                      int64_t* result, const std::vector<sycl::event>& dependencies) {          \
        return iamax(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, result, dependencies); \
    }
IAMAX_LAUNCHER_USM(float, cublasIsamax)
IAMAX_LAUNCHER_USM(double, cublasIdamax)
IAMAX_LAUNCHER_USM(std::complex<float>, cublasIcamax)
IAMAX_LAUNCHER_USM(std::complex<double>, cublasIzamax)
#undef IAMAX_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event swap(const char* func_name, Func func, sycl::queue& queue, int64_t n, T* x,
                        int64_t incx, T* y, int64_t incy,
                        const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx, incy);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto x_ = reinterpret_cast<cuDataType*>(x);
            auto y_ = reinterpret_cast<cuDataType*>(y);
            cublasStatus_t err;
            cublas_native_named_func(func_name, func, err, handle, n, x_, incx, y_, incy);
        });
    });
    return done;
}

#define SWAP_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                   \
    sycl::event swap(sycl::queue& queue, int64_t n, TYPE* x, int64_t incx, TYPE* y, int64_t incy, \
                     const std::vector<sycl::event>& dependencies) {                              \
        return swap(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, y, incy, dependencies);   \
    }

SWAP_LAUNCHER_USM(float, cublasSswap)
SWAP_LAUNCHER_USM(double, cublasDswap)
SWAP_LAUNCHER_USM(std::complex<float>, cublasCswap)
SWAP_LAUNCHER_USM(std::complex<double>, cublasZswap)
#undef SWAP_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event iamin(const char* func_name, Func func, sycl::queue& queue, int64_t n,
                         const T* x, const int64_t incx, int64_t* result,
                         const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, incx);
    // cuBLAS does not support int64_t as return type for the data. So we need to
    // mimic iamin. We are converting the result to be the int and then we convert
    // it back to the actual data on the host.
    // This change may cause failure as the result of integer overflow
    // based on the size.
    int int_res = 0;
    int* int_res_p = nullptr;
    bool result_on_device =
        sycl::get_pointer_type(result, queue.get_context()) == sycl::usm::alloc::device;
    if (result_on_device) {
        int_res_p = sycl::malloc_device<int>(1, queue);
    }
    else {
        int_res_p = &int_res;
    }
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto x_ = reinterpret_cast<const cuDataType*>(x);
            if (result_on_device) {
                cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
            }
            cublasStatus_t err;
            // For negative incx, iamin returns 0. This behaviour is similar to that of
            // implemented iamin.
            cublas_native_named_func(func_name, func, err, handle, n, x_, incx, int_res_p);
            if (result_on_device) {
                cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
            }
        });
    });
    done.wait();
    if (result_on_device) {
        auto last_ev = queue.submit([&](sycl::handler& cgh) {
            cgh.single_task([=]() { *result = std::max((int64_t)*int_res_p - 1, (int64_t)0); });
        });
        last_ev.wait();
        sycl::free(int_res_p, queue);
        return last_ev;
    }
    else {
        result[0] = std::max((int64_t)(*int_res_p - 1), int64_t{ 0 });
        return done;
    }
}

#define IAMIN_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                \
    sycl::event iamin(sycl::queue& queue, int64_t n, const TYPE* x, const int64_t incx,         \
                      int64_t* result, const std::vector<sycl::event>& dependencies) {          \
        return iamin(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, result, dependencies); \
    }
IAMIN_LAUNCHER_USM(float, cublasIsamin)
IAMIN_LAUNCHER_USM(double, cublasIdamin)
IAMIN_LAUNCHER_USM(std::complex<float>, cublasIcamin)
IAMIN_LAUNCHER_USM(std::complex<double>, cublasIzamin)
#undef IAMIN_LAUNCHER_USM

template <typename Func, typename T1, typename T2>
inline sycl::event nrm2(const char* func_name, Func func, sycl::queue& queue, int64_t n,
                        const T1* x, const int64_t incx, T2* result,
                        const std::vector<sycl::event>& dependencies) {
    using cuDataType1 = typename CudaEquivalentType<T1>::Type;
    using cuDataType2 = typename CudaEquivalentType<T2>::Type;
    overflow_check(n, incx);

    bool result_on_device =
        sycl::get_pointer_type(result, queue.get_context()) == sycl::usm::alloc::device;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto x_ = reinterpret_cast<const cuDataType1*>(x);
            auto res_ = reinterpret_cast<cuDataType2*>(result);
            if (result_on_device) {
                cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
            }
            cublasStatus_t err;
            // NRM2 does not support negative index
            cublas_native_named_func(func_name, func, err, handle, n, x_, std::abs(incx), res_);
            if (result_on_device) {
                cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
            }
        });
    });
    return done;
}

#define NRM2_LAUNCHER_USM(TYPE1, TYPE2, CUBLAS_ROUTINE)                                        \
    sycl::event nrm2(sycl::queue& queue, int64_t n, const TYPE1* x, const int64_t incx,        \
                     TYPE2* result, const std::vector<sycl::event>& dependencies) {            \
        return nrm2(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, result, dependencies); \
    }
NRM2_LAUNCHER_USM(float, float, cublasSnrm2)
NRM2_LAUNCHER_USM(double, double, cublasDnrm2)
NRM2_LAUNCHER_USM(std::complex<float>, float, cublasScnrm2)
NRM2_LAUNCHER_USM(std::complex<double>, double, cublasDznrm2)
#undef NRM2_LAUNCHER_USM

} // namespace column_major
namespace row_major {

// Buffer APIs

// Level 1
template <typename Func, typename T1, typename T2>
inline void asum(const char* func_name, Func func, sycl::queue& queue, int64_t n,
                 sycl::buffer<T1, 1>& x, const int64_t incx, sycl::buffer<T2, 1>& result) {
    throw unimplemented("blas", "asum", "for row_major layout");
}

#define ASUM_LAUNCHER(TYPE1, TYPE2, CUBLAS_ROUTINE)                                         \
    void asum(sycl::queue& queue, int64_t n, sycl::buffer<TYPE1, 1>& x, const int64_t incx, \
              sycl::buffer<TYPE2, 1>& result) {                                             \
        asum(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, result);                   \
    }
ASUM_LAUNCHER(float, float, cublasSasum)
ASUM_LAUNCHER(double, double, cublasDasum)
ASUM_LAUNCHER(std::complex<float>, float, cublasScasum)
ASUM_LAUNCHER(std::complex<double>, double, cublasDzasum)
#undef ASUM_LAUNCHER

template <typename Func, typename T1, typename T2>
inline void scal(const char* func_name, Func func, sycl::queue& queue, int64_t n, T1 a,
                 sycl::buffer<T2, 1>& x, int64_t incx) {
    throw unimplemented("blas", "scal", "for row_major layout");
}

#define SCAL_LAUNCHER(TYPE1, TYPE2, CUBLAS_ROUTINE)                                              \
    void scal(sycl::queue& queue, int64_t n, TYPE1 a, sycl::buffer<TYPE2, 1>& x, int64_t incx) { \
        scal(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, a, x, incx);                             \
    }
SCAL_LAUNCHER(float, float, cublasSscal)
SCAL_LAUNCHER(double, double, cublasDscal)
SCAL_LAUNCHER(std::complex<float>, std::complex<float>, cublasCscal)
SCAL_LAUNCHER(std::complex<double>, std::complex<double>, cublasZscal)
SCAL_LAUNCHER(float, std::complex<float>, cublasCsscal)
SCAL_LAUNCHER(double, std::complex<double>, cublasZdscal)
#undef SCAL_LAUNCHER

template <typename Func, typename T>
inline void axpy(const char* func_name, Func func, sycl::queue& queue, int64_t n, T alpha,
                 sycl::buffer<T, 1>& x, int64_t incx, sycl::buffer<T, 1>& y, int64_t incy) {
    throw unimplemented("blas", "axpy", "for row_major layout");
}

#define AXPY_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                      \
    void axpy(sycl::queue& queue, int64_t n, TYPE alpha, sycl::buffer<TYPE, 1>& x, int64_t incx, \
              sycl::buffer<TYPE, 1>& y, int64_t incy) {                                          \
        axpy(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, alpha, x, incx, y, incy);                \
    }

AXPY_LAUNCHER(float, cublasSaxpy)
AXPY_LAUNCHER(double, cublasDaxpy)
AXPY_LAUNCHER(std::complex<float>, cublasCaxpy)
AXPY_LAUNCHER(std::complex<double>, cublasZaxpy)
#undef AXPY_LAUNCHER

void axpby(sycl::queue& queue, int64_t n, float alpha, sycl::buffer<float, 1>& x, int64_t incx,
           float beta, sycl::buffer<float, 1>& y, int64_t incy) {
    throw unimplemented("blas", "axpby", "for row_major layout");
}

void axpby(sycl::queue& queue, int64_t n, double alpha, sycl::buffer<double, 1>& x, int64_t incx,
           double beta, sycl::buffer<double, 1>& y, int64_t incy) {
    throw unimplemented("blas", "axpby", "for row_major layout");
}

void axpby(sycl::queue& queue, int64_t n, std::complex<float> alpha,
           sycl::buffer<std::complex<float>, 1>& x, int64_t incx, std::complex<float> beta,
           sycl::buffer<std::complex<float>, 1>& y, int64_t incy) {
    throw unimplemented("blas", "axpby", "for row_major layout");
}

void axpby(sycl::queue& queue, int64_t n, std::complex<double> alpha,
           sycl::buffer<std::complex<double>, 1>& x, int64_t incx, std::complex<double> beta,
           sycl::buffer<std::complex<double>, 1>& y, int64_t incy) {
    throw unimplemented("blas", "axpby", "for row_major layout");
}

template <typename Func, typename T1, typename T2>
inline void rotg(const char* func_name, Func func, sycl::queue& queue, sycl::buffer<T1, 1>& a,
                 sycl::buffer<T1, 1>& b, sycl::buffer<T2, 1>& c, sycl::buffer<T1, 1>& s) {
    throw unimplemented("blas", "rotg", "for row_major layout");
}

#define ROTG_LAUNCHER(TYPE1, TYPE2, CUBLAS_ROUTINE)                                     \
    void rotg(sycl::queue& queue, sycl::buffer<TYPE1, 1>& a, sycl::buffer<TYPE1, 1>& b, \
              sycl::buffer<TYPE2, 1>& c, sycl::buffer<TYPE1, 1>& s) {                   \
        rotg(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, a, b, c, s);                       \
    }

ROTG_LAUNCHER(float, float, cublasSrotg)
ROTG_LAUNCHER(double, double, cublasDrotg)
ROTG_LAUNCHER(std::complex<float>, float, cublasCrotg)
ROTG_LAUNCHER(std::complex<double>, double, cublasZrotg)
#undef ROTG_LAUNCHER

template <typename Func, typename T>
inline void rotm(const char* func_name, Func func, sycl::queue& queue, int64_t n,
                 sycl::buffer<T, 1>& x, int64_t incx, sycl::buffer<T, 1>& y, int64_t incy,
                 sycl::buffer<T, 1>& param) {
    throw unimplemented("blas", "rotm", "for row_major layout");
}

#define ROTM_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                           \
    void rotm(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, int64_t incx,  \
              sycl::buffer<TYPE, 1>& y, int64_t incy, sycl::buffer<TYPE, 1>& param) { \
        rotm(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, y, incy, param);     \
    }

ROTM_LAUNCHER(float, cublasSrotm)
ROTM_LAUNCHER(double, cublasDrotm)
#undef ROTM_LAUNCHER

template <typename Func, typename T>
inline void copy(const char* func_name, Func func, sycl::queue& queue, int64_t n,
                 sycl::buffer<T, 1>& x, int64_t incx, sycl::buffer<T, 1>& y, int64_t incy) {
    throw unimplemented("blas", "copy", "for row_major layout");
}

#define COPY_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                          \
    void copy(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, int64_t incx, \
              sycl::buffer<TYPE, 1>& y, int64_t incy) {                              \
        copy(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, y, incy);           \
    }

COPY_LAUNCHER(float, cublasScopy)
COPY_LAUNCHER(double, cublasDcopy)
COPY_LAUNCHER(std::complex<float>, cublasCcopy)
COPY_LAUNCHER(std::complex<double>, cublasZcopy)
#undef COPY_LAUNCHER

template <typename Func, typename T>
inline void dot(const char* func_name, Func func, sycl::queue& queue, int64_t n,
                sycl::buffer<T, 1>& x, const int64_t incx, sycl::buffer<T, 1>& y, int64_t incy,
                sycl::buffer<T, 1>& result) {
    throw unimplemented("blas", "dot", "for row_major layout");
}

#define DOT_LAUNCHER(EXT, TYPE, CUBLAS_ROUTINE)                                                  \
    void dot##EXT(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, const int64_t incx,   \
                  sycl::buffer<TYPE, 1>& y, const int64_t incy, sycl::buffer<TYPE, 1>& result) { \
        dot(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, y, incy, result);                \
    }
DOT_LAUNCHER(, float, cublasSdot)
DOT_LAUNCHER(, double, cublasDdot)
DOT_LAUNCHER(c, std::complex<float>, cublasCdotc)
DOT_LAUNCHER(c, std::complex<double>, cublasZdotc)
DOT_LAUNCHER(u, std::complex<float>, cublasCdotu)
DOT_LAUNCHER(u, std::complex<double>, cublasZdotu)
#undef DOT_LAUNCHER

template <typename Func, typename T1, typename T2, typename T3>
inline void rot(const char* func_name, Func func, sycl::queue& queue, int64_t n,
                sycl::buffer<T1, 1>& x, const int64_t incx, sycl::buffer<T1, 1>& y, int64_t incy,
                T2 c, T3 s) {
    throw unimplemented("blas", "rot", "for row_major layout");
}

#define ROT_LAUNCHER(TYPE1, TYPE2, TYPE3, CUBLAS_ROUTINE)                                  \
    void rot(sycl::queue& queue, int64_t n, sycl::buffer<TYPE1, 1>& x, const int64_t incx, \
             sycl::buffer<TYPE1, 1>& y, int64_t incy, TYPE2 c, TYPE3 s) {                  \
        rot(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, y, incy, c, s);            \
    }

ROT_LAUNCHER(float, float, float, cublasSrot)
ROT_LAUNCHER(double, double, double, cublasDrot)
ROT_LAUNCHER(std::complex<float>, float, float, cublasCsrot)
ROT_LAUNCHER(std::complex<double>, double, double, cublasZdrot)
#undef ROT_LAUNCHER

void sdsdot(sycl::queue& queue, int64_t n, float sb, sycl::buffer<float, 1>& x, int64_t incx,
            sycl::buffer<float, 1>& y, int64_t incy, sycl::buffer<float, 1>& result) {
    throw unimplemented("blas", "sdsdot", "for row_major layout");
}

void dot(sycl::queue& queue, int64_t n, sycl::buffer<float, 1>& x, int64_t incx,
         sycl::buffer<float, 1>& y, int64_t incy, sycl::buffer<double, 1>& result) {
    throw unimplemented("blas", "dot", "for row_major layout");
}

template <typename Func, typename T>
inline void rotmg(const char* func_name, Func func, sycl::queue& queue, sycl::buffer<T, 1>& d1,
                  sycl::buffer<T, 1>& d2, sycl::buffer<T, 1>& x1, T y1, sycl::buffer<T, 1>& param) {
    throw unimplemented("blas", "rotmg", "for row_major layout");
}

#define ROTMG_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                             \
    void rotmg(sycl::queue& queue, sycl::buffer<TYPE, 1>& d1, sycl::buffer<TYPE, 1>& d2, \
               sycl::buffer<TYPE, 1>& x1, TYPE y1, sycl::buffer<TYPE, 1>& param) {       \
        rotmg(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, d1, d2, x1, y1, param);            \
    }

ROTMG_LAUNCHER(float, cublasSrotmg)
ROTMG_LAUNCHER(double, cublasDrotmg)
#undef ROTMG_LAUNCHER

template <typename Func, typename T>
inline void iamax(const char* func_name, Func func, sycl::queue& queue, int64_t n,
                  sycl::buffer<T, 1>& x, const int64_t incx, sycl::buffer<int64_t, 1>& result) {
    throw unimplemented("blas", "iamax", "for row_major layout");
}

#define IAMAX_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                \
    void iamax(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, const int64_t incx, \
               sycl::buffer<int64_t, 1>& result) {                                          \
        iamax(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, result);                  \
    }
IAMAX_LAUNCHER(float, cublasIsamax)
IAMAX_LAUNCHER(double, cublasIdamax)
IAMAX_LAUNCHER(std::complex<float>, cublasIcamax)
IAMAX_LAUNCHER(std::complex<double>, cublasIzamax)
#undef IAMAX_LAUNCHER

template <typename Func, typename T>
inline void swap(const char* func_name, Func func, sycl::queue& queue, int64_t n,
                 sycl::buffer<T, 1>& x, int64_t incx, sycl::buffer<T, 1>& y, int64_t incy) {
    throw unimplemented("blas", "swap", "for row_major layout");
}

#define SWAP_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                          \
    void swap(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, int64_t incx, \
              sycl::buffer<TYPE, 1>& y, int64_t incy) {                              \
        swap(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, y, incy);           \
    }

SWAP_LAUNCHER(float, cublasSswap)
SWAP_LAUNCHER(double, cublasDswap)
SWAP_LAUNCHER(std::complex<float>, cublasCswap)
SWAP_LAUNCHER(std::complex<double>, cublasZswap)
#undef SWAP_LAUNCHER

template <typename Func, typename T>
inline void iamin(const char* func_name, Func func, sycl::queue& queue, int64_t n,
                  sycl::buffer<T, 1>& x, const int64_t incx, sycl::buffer<int64_t, 1>& result) {
    throw unimplemented("blas", "iamin", "for row_major layout");
}

#define IAMIN_LAUNCHER(TYPE, CUBLAS_ROUTINE)                                                \
    void iamin(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, const int64_t incx, \
               sycl::buffer<int64_t, 1>& result) {                                          \
        iamin(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, result);                  \
    }
IAMIN_LAUNCHER(float, cublasIsamin)
IAMIN_LAUNCHER(double, cublasIdamin)
IAMIN_LAUNCHER(std::complex<float>, cublasIcamin)
IAMIN_LAUNCHER(std::complex<double>, cublasIzamin)
#undef IAMIN_LAUNCHER

template <typename Func, typename T1, typename T2>
inline void nrm2(const char* func_name, Func func, sycl::queue& queue, int64_t n,
                 sycl::buffer<T1, 1>& x, const int64_t incx, sycl::buffer<T2, 1>& result) {
    throw unimplemented("blas", "nrm2", "for row_major layout");
}

#define NRM2_LAUNCHER(TYPE1, TYPE2, CUBLAS_ROUTINE)                                         \
    void nrm2(sycl::queue& queue, int64_t n, sycl::buffer<TYPE1, 1>& x, const int64_t incx, \
              sycl::buffer<TYPE2, 1>& result) {                                             \
        nrm2(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, result);                   \
    }
NRM2_LAUNCHER(float, float, cublasSnrm2)
NRM2_LAUNCHER(double, double, cublasDnrm2)
NRM2_LAUNCHER(std::complex<float>, float, cublasScnrm2)
NRM2_LAUNCHER(std::complex<double>, double, cublasDznrm2)
#undef NRM2_LAUNCHER

// USM APIs

// Level 1
template <typename Func, typename T1, typename T2>
inline sycl::event asum(const char* func_name, Func func, sycl::queue& queue, int64_t n,
                        const T1* x, const int64_t incx, T2* result,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "asum", "for row_major layout");
}

#define ASUM_LAUNCHER_USM(TYPE1, TYPE2, CUBLAS_ROUTINE)                                        \
    sycl::event asum(sycl::queue& queue, int64_t n, const TYPE1* x, const int64_t incx,        \
                     TYPE2* result, const std::vector<sycl::event>& dependencies) {            \
        return asum(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, result, dependencies); \
    }
ASUM_LAUNCHER_USM(float, float, cublasSasum)
ASUM_LAUNCHER_USM(double, double, cublasDasum)
ASUM_LAUNCHER_USM(std::complex<float>, float, cublasScasum)
ASUM_LAUNCHER_USM(std::complex<double>, double, cublasDzasum)
#undef ASUM_LAUNCHER_USM

template <typename Func, typename T1, typename T2>
inline sycl::event scal(const char* func_name, Func func, sycl::queue& queue, int64_t n, T1 a,
                        T2* x, int64_t incx, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "scal", "for row_major layout");
}

#define SCAL_LAUNCHER_USM(TYPE1, TYPE2, CUBLAS_ROUTINE)                                   \
    sycl::event scal(sycl::queue& queue, int64_t n, TYPE1 a, TYPE2* x, int64_t incx,      \
                     const std::vector<sycl::event>& dependencies) {                      \
        return scal(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, a, x, incx, dependencies); \
    }
SCAL_LAUNCHER_USM(float, float, cublasSscal)
SCAL_LAUNCHER_USM(double, double, cublasDscal)
SCAL_LAUNCHER_USM(std::complex<float>, std::complex<float>, cublasCscal)
SCAL_LAUNCHER_USM(std::complex<double>, std::complex<double>, cublasZscal)
SCAL_LAUNCHER_USM(float, std::complex<float>, cublasCsscal)
SCAL_LAUNCHER_USM(double, std::complex<double>, cublasZdscal)
#undef SCAL_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event axpy(const char* func_name, Func func, sycl::queue& queue, int64_t n, T alpha,
                        const T* x, int64_t incx, T* y, int64_t incy,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpy", "for row_major layout");
}

#define AXPY_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                              \
    sycl::event axpy(sycl::queue& queue, int64_t n, TYPE alpha, const TYPE* x, int64_t incx, \
                     TYPE* y, int64_t incy, const std::vector<sycl::event>& dependencies) {  \
        return axpy(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, alpha, x, incx, y, incy,      \
                    dependencies);                                                           \
    }

AXPY_LAUNCHER_USM(float, cublasSaxpy)
AXPY_LAUNCHER_USM(double, cublasDaxpy)
AXPY_LAUNCHER_USM(std::complex<float>, cublasCaxpy)
AXPY_LAUNCHER_USM(std::complex<double>, cublasZaxpy)
#undef AXPY_LAUNCHER_USM

sycl::event axpby(sycl::queue& queue, int64_t n, float alpha, const float* x, int64_t incx,
                  float beta, float* y, int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpby", "for row_major layout");
}
sycl::event axpby(sycl::queue& queue, int64_t n, double alpha, const double* x, int64_t incx,
                  double beta, double* y, int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpby", "for row_major layout");
}
sycl::event axpby(sycl::queue& queue, int64_t n, std::complex<float> alpha,
                  const std::complex<float>* x, int64_t incx, std::complex<float> beta,
                  std::complex<float>* y, int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpby", "for row_major layout");
}
sycl::event axpby(sycl::queue& queue, int64_t n, std::complex<double> alpha,
                  const std::complex<double>* x, int64_t incx, std::complex<double> beta,
                  std::complex<double>* y, int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpby", "for row_major layout");
}

template <typename Func, typename T1, typename T2>
inline sycl::event rotg(const char* func_name, Func func, sycl::queue& queue, T1* a, T1* b, T2* c,
                        T1* s, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "rotg", "for row_major layout");
}

#define ROTG_LAUNCHER_USM(TYPE1, TYPE2, CUBLAS_ROUTINE)                                \
    sycl::event rotg(sycl::queue& queue, TYPE1* a, TYPE1* b, TYPE2* c, TYPE1* s,       \
                     const std::vector<sycl::event>& dependencies) {                   \
        return rotg(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, a, b, c, s, dependencies); \
    }

ROTG_LAUNCHER_USM(float, float, cublasSrotg)
ROTG_LAUNCHER_USM(double, double, cublasDrotg)
ROTG_LAUNCHER_USM(std::complex<float>, float, cublasCrotg)
ROTG_LAUNCHER_USM(std::complex<double>, double, cublasZrotg)
#undef ROTG_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event rotm(const char* func_name, Func func, sycl::queue& queue, int64_t n, T* x,
                        int64_t incx, T* y, int64_t incy, T* param,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "rotm", "for row_major layout");
}

#define ROTM_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                   \
    sycl::event rotm(sycl::queue& queue, int64_t n, TYPE* x, int64_t incx, TYPE* y, int64_t incy, \
                     TYPE* param, const std::vector<sycl::event>& dependencies) {                 \
        return rotm(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, y, incy, param,           \
                    dependencies);                                                                \
    }

ROTM_LAUNCHER_USM(float, cublasSrotm)
ROTM_LAUNCHER_USM(double, cublasDrotm)
#undef ROTM_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event copy(const char* func_name, Func func, sycl::queue& queue, int64_t n, const T* x,
                        int64_t incx, T* y, int64_t incy,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "copy", "for row_major layout");
}

#define COPY_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                 \
    sycl::event copy(sycl::queue& queue, int64_t n, const TYPE* x, int64_t incx, TYPE* y,       \
                     int64_t incy, const std::vector<sycl::event>& dependencies) {              \
        return copy(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, y, incy, dependencies); \
    }

COPY_LAUNCHER_USM(float, cublasScopy)
COPY_LAUNCHER_USM(double, cublasDcopy)
COPY_LAUNCHER_USM(std::complex<float>, cublasCcopy)
COPY_LAUNCHER_USM(std::complex<double>, cublasZcopy)
#undef COPY_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event dot(const char* func_name, Func func, sycl::queue& queue, int64_t n, const T* x,
                       const int64_t incx, const T* y, int64_t incy, T* result,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dot", "for row_major layout");
}

#define DOT_LAUNCHER_USM(EXT, TYPE, CUBLAS_ROUTINE)                                        \
    sycl::event dot##EXT(sycl::queue& queue, int64_t n, const TYPE* x, const int64_t incx, \
                         const TYPE* y, const int64_t incy, TYPE* result,                  \
                         const std::vector<sycl::event>& dependencies) {                   \
        return dot(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, y, incy, result,    \
                   dependencies);                                                          \
    }
DOT_LAUNCHER_USM(, float, cublasSdot)
DOT_LAUNCHER_USM(, double, cublasDdot)
DOT_LAUNCHER_USM(c, std::complex<float>, cublasCdotc)
DOT_LAUNCHER_USM(c, std::complex<double>, cublasZdotc)
DOT_LAUNCHER_USM(u, std::complex<float>, cublasCdotu)
DOT_LAUNCHER_USM(u, std::complex<double>, cublasZdotu)
#undef DOT_LAUNCHER_USM

template <typename Func, typename T1, typename T2, typename T3>
inline sycl::event rot(const char* func_name, Func func, sycl::queue& queue, int64_t n, T1* x,
                       const int64_t incx, T1* y, int64_t incy, T2 c, T3 s,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "rot", "for row_major layout");
}

#define ROT_LAUNCHER_USM(TYPE1, TYPE2, TYPE3, CUBLAS_ROUTINE)                              \
    sycl::event rot(sycl::queue& queue, int64_t n, TYPE1* x, const int64_t incx, TYPE1* y, \
                    int64_t incy, TYPE2 c, TYPE3 s,                                        \
                    const std::vector<sycl::event>& dependencies) {                        \
        return rot(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, y, incy, c, s,      \
                   dependencies);                                                          \
    }

ROT_LAUNCHER_USM(float, float, float, cublasSrot)
ROT_LAUNCHER_USM(double, double, double, cublasDrot)
ROT_LAUNCHER_USM(std::complex<float>, float, float, cublasCsrot)
ROT_LAUNCHER_USM(std::complex<double>, double, double, cublasZdrot)
#undef ROT_LAUNCHER_USM

sycl::event sdsdot(sycl::queue& queue, int64_t n, float sb, const float* x, int64_t incx,
                   const float* y, int64_t incy, float* result,
                   const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "sdsdot", "for row_major layout");
}

sycl::event dot(sycl::queue& queue, int64_t n, const float* x, int64_t incx, const float* y,
                int64_t incy, double* result, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dot", "for row_major layout");
}

template <typename Func, typename T>
inline sycl::event rotmg(const char* func_name, Func func, sycl::queue& queue, T* d1, T* d2, T* x1,
                         T y1, T* param, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "rotmg", "for row_major layout");
}

#define ROTMG_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                   \
    sycl::event rotmg(sycl::queue& queue, TYPE* d1, TYPE* d2, TYPE* x1, TYPE y1, TYPE* param,      \
                      const std::vector<sycl::event>& dependencies) {                              \
        return rotmg(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, d1, d2, x1, y1, param, dependencies); \
    }

ROTMG_LAUNCHER_USM(float, cublasSrotmg)
ROTMG_LAUNCHER_USM(double, cublasDrotmg)
#undef ROTMG_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event iamax(const char* func_name, Func func, sycl::queue& queue, int64_t n,
                         const T* x, const int64_t incx, int64_t* result,
                         const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "iamax", "for row_major layout");
}

#define IAMAX_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                \
    sycl::event iamax(sycl::queue& queue, int64_t n, const TYPE* x, const int64_t incx,         \
                      int64_t* result, const std::vector<sycl::event>& dependencies) {          \
        return iamax(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, result, dependencies); \
    }
IAMAX_LAUNCHER_USM(float, cublasIsamax)
IAMAX_LAUNCHER_USM(double, cublasIdamax)
IAMAX_LAUNCHER_USM(std::complex<float>, cublasIcamax)
IAMAX_LAUNCHER_USM(std::complex<double>, cublasIzamax)
#undef IAMAX_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event swap(const char* func_name, Func func, sycl::queue& queue, int64_t n, T* x,
                        int64_t incx, T* y, int64_t incy,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "swap", "for row_major layout");
}

#define SWAP_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                   \
    sycl::event swap(sycl::queue& queue, int64_t n, TYPE* x, int64_t incx, TYPE* y, int64_t incy, \
                     const std::vector<sycl::event>& dependencies) {                              \
        return swap(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, y, incy, dependencies);   \
    }

SWAP_LAUNCHER_USM(float, cublasSswap)
SWAP_LAUNCHER_USM(double, cublasDswap)
SWAP_LAUNCHER_USM(std::complex<float>, cublasCswap)
SWAP_LAUNCHER_USM(std::complex<double>, cublasZswap)
#undef SWAP_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event iamin(const char* func_name, Func func, sycl::queue& queue, int64_t n,
                         const T* x, const int64_t incx, int64_t* result,
                         const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "iamin", "for row_major layout");
}

#define IAMIN_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                                \
    sycl::event iamin(sycl::queue& queue, int64_t n, const TYPE* x, const int64_t incx,         \
                      int64_t* result, const std::vector<sycl::event>& dependencies) {          \
        return iamin(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, result, dependencies); \
    }
IAMIN_LAUNCHER_USM(float, cublasIsamin)
IAMIN_LAUNCHER_USM(double, cublasIdamin)
IAMIN_LAUNCHER_USM(std::complex<float>, cublasIcamin)
IAMIN_LAUNCHER_USM(std::complex<double>, cublasIzamin)
#undef IAMIN_LAUNCHER_USM

template <typename Func, typename T1, typename T2>
inline sycl::event nrm2(const char* func_name, Func func, sycl::queue& queue, int64_t n,
                        const T1* x, const int64_t incx, T2* result,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "nrm2", "for row_major layout");
}

#define NRM2_LAUNCHER_USM(TYPE1, TYPE2, CUBLAS_ROUTINE)                                        \
    sycl::event nrm2(sycl::queue& queue, int64_t n, const TYPE1* x, const int64_t incx,        \
                     TYPE2* result, const std::vector<sycl::event>& dependencies) {            \
        return nrm2(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, n, x, incx, result, dependencies); \
    }
NRM2_LAUNCHER_USM(float, float, cublasSnrm2)
NRM2_LAUNCHER_USM(double, double, cublasDnrm2)
NRM2_LAUNCHER_USM(std::complex<float>, float, cublasScnrm2)
NRM2_LAUNCHER_USM(std::complex<double>, double, cublasDznrm2)
#undef NRM2_LAUNCHER_USM

} // namespace row_major
} // namespace cublas
} // namespace blas
} // namespace math
} // namespace oneapi
