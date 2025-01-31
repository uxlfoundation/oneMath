/*******************************************************************************
* Copyright 2025 SiPearl
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

// Buffer APIs
template <typename T, typename CBLAS_FUNC>
void copy_batch(sycl::queue& queue, int64_t n, sycl::buffer<T, 1>& x, int64_t incx, int64_t stridex,
                sycl::buffer<T, 1>& y, int64_t incy, int64_t stridey, int64_t batch_size,
                CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_copy_batch_strided>(cgh, [=]() {
            for (size_t i = 0; i < batch_size; ++i) {
                cblas_func(n, accessor_x.GET_MULTI_PTR + i * stridex, incx,
                           accessor_y.GET_MULTI_PTR + i * stridey, incy);
            }
        });
    });
}

#define COPY_BATCH_LAUNCHER(TYPE, ROUTINE)                                                    \
    void copy_batch(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, int64_t incx,    \
                    int64_t stridex, sycl::buffer<TYPE, 1>& y, int64_t incy, int64_t stridey, \
                    int64_t batch_size) {                                                     \
        copy_batch(queue, n, x, incx, stridex, y, incy, stridey, batch_size, ROUTINE);        \
    }

COPY_BATCH_LAUNCHER(float, ::cblas_scopy)
COPY_BATCH_LAUNCHER(double, ::cblas_dcopy)
COPY_BATCH_LAUNCHER(std::complex<float>, ::cblas_ccopy)
COPY_BATCH_LAUNCHER(std::complex<double>, ::cblas_zcopy)

template <typename T, typename CBLAS_FUNC>
void axpy_batch(sycl::queue& queue, int64_t n, T alpha, sycl::buffer<T, 1>& x, int64_t incx,
                int64_t stridex, sycl::buffer<T, 1>& y, int64_t incy, int64_t stridey,
                int64_t batch_size, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_axpy_batch_strided>(cgh, [=]() {
            for (size_t i = 0; i < batch_size; ++i) {
                cblas_func(n, cast_to_void_if_complex(alpha),
                           accessor_x.GET_MULTI_PTR + i * stridex, incx,
                           accessor_y.GET_MULTI_PTR + i * stridey, incy);
            }
        });
    });
}

#define AXPY_BATCH_LAUNCHER(TYPE, ROUTINE)                                                    \
    void axpy_batch(sycl::queue& queue, int64_t n, TYPE alpha, sycl::buffer<TYPE, 1>& x,      \
                    int64_t incx, int64_t stridex, sycl::buffer<TYPE, 1>& y, int64_t incy,    \
                    int64_t stridey, int64_t batch_size) {                                    \
        axpy_batch(queue, n, alpha, x, incx, stridex, y, incy, stridey, batch_size, ROUTINE); \
    }

AXPY_BATCH_LAUNCHER(float, ::cblas_saxpy)
AXPY_BATCH_LAUNCHER(double, ::cblas_daxpy)
AXPY_BATCH_LAUNCHER(std::complex<float>, ::cblas_caxpy)
AXPY_BATCH_LAUNCHER(std::complex<double>, ::cblas_zaxpy)

template <typename T, typename CBLAS_FUNC>
void gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n, T alpha,
                sycl::buffer<T, 1>& a, int64_t lda, int64_t stride_a, sycl::buffer<T, 1>& x,
                int64_t incx, int64_t stride_x, T beta, sycl::buffer<T, 1>& y, int64_t incy,
                int64_t stride_y, int64_t batch_size, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read_write>(cgh);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        host_task<class armpl_kernel_gemv_batch_strided>(cgh, [=]() {
            for (size_t i = 0; i < batch_size; ++i) {
                cblas_func(MAJOR, transa_, m, n, cast_to_void_if_complex(alpha),
                           accessor_a.GET_MULTI_PTR + i * stride_a, lda,
                           accessor_x.GET_MULTI_PTR + i * stride_x, incx,
                           cast_to_void_if_complex(beta), accessor_y.GET_MULTI_PTR + i * stride_y,
                           incy);
            }
        });
    });
}

#define GEMV_BATCH_LAUNCHER(TYPE, ROUTINE)                                                         \
    void gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n, TYPE alpha,        \
                    sycl::buffer<TYPE, 1>& a, int64_t lda, int64_t stride_a,                       \
                    sycl::buffer<TYPE, 1>& x, int64_t incx, int64_t stride_x, TYPE beta,           \
                    sycl::buffer<TYPE, 1>& y, int64_t incy, int64_t stride_y,                      \
                    int64_t batch_size) {                                                          \
        gemv_batch(queue, transa, m, n, alpha, a, lda, stride_a, x, incx, stride_x, beta, y, incy, \
                   stride_y, batch_size, ROUTINE);                                                 \
    }

GEMV_BATCH_LAUNCHER(float, ::cblas_sgemv)
GEMV_BATCH_LAUNCHER(double, ::cblas_dgemv)
GEMV_BATCH_LAUNCHER(std::complex<float>, ::cblas_cgemv)
GEMV_BATCH_LAUNCHER(std::complex<double>, ::cblas_zgemv)

template <typename T, typename CBLAS_FUNC>
void dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n, sycl::buffer<T, 1>& a,
                int64_t lda, int64_t stride_a, sycl::buffer<T, 1>& x, int64_t incx,
                int64_t stride_x, sycl::buffer<T, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read_write>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_dgmm_batch_strided>(cgh, [=]() {
            int64_t ldx = (left_right == side::left) ? m : n;
            int64_t totalsize_x = ldx * ldx;
            int64_t absincx = std::abs(incx);

            auto x_array_val = new T[totalsize_x * batch_size]{};
            if (!x_array_val) {
                std::cerr << "Error: cannot allocate input array\n";
                delete[] x_array_val;
                throw oneapi::math::host_bad_alloc("blas", __func__);
            }

            for (size_t i = 0; i < batch_size; ++i) {
                T* x_vector = x_acc.GET_MULTI_PTR + i * stride_x;

                if (incx > 0) {
                    for (size_t j = 0; j < ldx; ++j) {
                        x_array_val[i * totalsize_x + j * ldx + j] = x_vector[j * incx];
                    }
                }
                else {
                    for (size_t j = 0; j < ldx; ++j) {
                        x_array_val[i * totalsize_x + j * ldx + j] =
                            x_vector[(ldx - j - 1) * absincx];
                    }
                }
            }

            CBLAS_TRANSPOSE transa_ = CblasNoTrans;
            CBLAS_TRANSPOSE transb_ = CblasNoTrans;

            int64_t k = ldx;

            T alpha = 1.0f;
            T beta = 0.0f;

            armpl_int_t one = 1;

            T** a_array = new T*[batch_size]();
            T** x_array = new T*[batch_size]();
            T** c_array = new T*[batch_size]();
            if (!a_array || !x_array || !c_array) {
                std::cerr << "Error: cannot allocate input arrays\n";
                delete[] a_array;
                delete[] x_array;
                delete[] c_array;
                throw oneapi::math::host_bad_alloc("blas", __func__);
            }
            for (size_t i = 0; i < batch_size; ++i) {
                a_array[i] = a_acc.GET_MULTI_PTR + i * stride_a;
                x_array[i] = x_array_val + i * totalsize_x;
                c_array[i] = c_acc.GET_MULTI_PTR + i * stride_c;
            }
            if constexpr (is_complex<T>) {
                if (left_right == side::left) {
                    cblas_func(MAJOR, &transa_, &transb_, (const armpl_int_t*)&m,
                               (const armpl_int_t*)&n, (const armpl_int_t*)&k, &alpha,
                               (const void* const*)x_array, (const armpl_int_t*)&ldx,
                               (const void* const*)a_array, (const armpl_int_t*)&lda, &beta,
                               (void**)c_array, (const armpl_int_t*)&ldc, one,
                               (const armpl_int_t*)&batch_size);
                }
                else {
                    cblas_func(MAJOR, &transa_, &transb_, (const armpl_int_t*)&m,
                               (const armpl_int_t*)&n, (const armpl_int_t*)&k, &alpha,
                               (const void* const*)a_array, (const armpl_int_t*)&lda,
                               (const void* const*)x_array, (const armpl_int_t*)&ldx, &beta,
                               (void**)c_array, (const armpl_int_t*)&ldc, one,
                               (const armpl_int_t*)&batch_size);
                }
            }
            else {
                if (left_right == side::left) {
                    cblas_func(MAJOR, &transa_, &transb_, (const armpl_int_t*)&m,
                               (const armpl_int_t*)&n, (const armpl_int_t*)&k, &alpha, x_array,
                               (const armpl_int_t*)&ldx, a_array, (const armpl_int_t*)&lda, &beta,
                               (T**)c_array, (const armpl_int_t*)&ldc, one,
                               (const armpl_int_t*)&batch_size);
                }
                else {
                    cblas_func(MAJOR, &transa_, &transb_, (const armpl_int_t*)&m,
                               (const armpl_int_t*)&n, (const armpl_int_t*)&k, &alpha,
                               cast_to_void_if_complex(a_array), (const armpl_int_t*)&lda,
                               cast_to_void_if_complex(x_array), (const armpl_int_t*)&ldx, &beta,
                               (T**)c_array, (const armpl_int_t*)&ldc, one,
                               (const armpl_int_t*)&batch_size);
                }
            }

            delete[] x_array_val;
            delete[] a_array;
            delete[] x_array;
            delete[] c_array;
        });
    });
}

#define DGMM_BATCH_LAUNCHER(TYPE, ROUTINE)                                                         \
    void dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,                     \
                    sycl::buffer<TYPE, 1>& a, int64_t lda, int64_t stride_a,                       \
                    sycl::buffer<TYPE, 1>& x, int64_t incx, int64_t stride_x,                      \
                    sycl::buffer<TYPE, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size) { \
        dgmm_batch(queue, left_right, m, n, a, lda, stride_a, x, incx, stride_x, c, ldc, stride_c, \
                   batch_size, ROUTINE);                                                           \
    }

DGMM_BATCH_LAUNCHER(float, ::cblas_sgemm_batch)
DGMM_BATCH_LAUNCHER(double, ::cblas_dgemm_batch)
DGMM_BATCH_LAUNCHER(std::complex<float>, ::cblas_cgemm_batch)
DGMM_BATCH_LAUNCHER(std::complex<double>, ::cblas_zgemm_batch)

template <typename T, typename CBLAS_FUNC>
void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, T alpha, sycl::buffer<T, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<T, 1>& b, int64_t ldb, int64_t stride_b, T beta, sycl::buffer<T, 1>& c,
                int64_t ldc, int64_t stride_c, int64_t batch_size, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read_write>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        armpl_int_t one = 1;
        host_task<class armpl_kernel_gemm_batch_strided>(cgh, [=]() {
            T** a_array = new T*[batch_size]();
            T** b_array = new T*[batch_size]();
            T** c_array = new T*[batch_size]();
            if (!a_array || !b_array || !c_array) {
                std::cerr << "Error cannot allocate input arrays\n";
                delete[] a_array;
                delete[] b_array;
                delete[] c_array;
                return;
            }
            for (size_t i = 0; i < batch_size; ++i) {
                a_array[i] = a_acc.GET_MULTI_PTR + i * stride_a;
                b_array[i] = b_acc.GET_MULTI_PTR + i * stride_b;
                c_array[i] = c_acc.GET_MULTI_PTR + i * stride_c;
            }
            if constexpr (is_complex<T>)
                cblas_func(MAJOR, &transa_, &transb_, (const armpl_int_t*)&m,
                           (const armpl_int_t*)&n, (const armpl_int_t*)&k, &alpha,
                           (const void* const*)a_array, (const armpl_int_t*)&lda,
                           (const void* const*)b_array, (const armpl_int_t*)&ldb, &beta,
                           (void**)c_array, (const armpl_int_t*)&ldc, one,
                           (const armpl_int_t*)&batch_size);
            else
                cblas_func(MAJOR, &transa_, &transb_, (const armpl_int_t*)&m,
                           (const armpl_int_t*)&n, (const armpl_int_t*)&k, &alpha,
                           (const T**)a_array, (const armpl_int_t*)&lda, (const T**)b_array,
                           (const armpl_int_t*)&ldb, &beta, (T**)c_array, (const armpl_int_t*)&ldc,
                           one, (const armpl_int_t*)&batch_size);

            delete[] a_array;
            delete[] b_array;
            delete[] c_array;
        });
    });
}

#define GEMM_BATCH_LAUNCHER(TYPE, ROUTINE)                                                        \
    void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n, \
                    int64_t k, TYPE alpha, sycl::buffer<TYPE, 1>& a, int64_t lda,                 \
                    int64_t stride_a, sycl::buffer<TYPE, 1>& b, int64_t ldb, int64_t stride_b,    \
                    TYPE beta, sycl::buffer<TYPE, 1>& c, int64_t ldc, int64_t stride_c,           \
                    int64_t batch_size) {                                                         \
        gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,     \
                   beta, c, ldc, stride_c, batch_size, ROUTINE);                                  \
    }

GEMM_BATCH_LAUNCHER(float, ::cblas_sgemm_batch)
GEMM_BATCH_LAUNCHER(double, ::cblas_dgemm_batch)
GEMM_BATCH_LAUNCHER(std::complex<float>, ::cblas_cgemm_batch)
GEMM_BATCH_LAUNCHER(std::complex<double>, ::cblas_zgemm_batch)

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, sycl::half alpha, sycl::buffer<sycl::half, 1>& a, int64_t lda,
                int64_t stride_a, sycl::buffer<sycl::half, 1>& b, int64_t ldb, int64_t stride_b,
                sycl::half beta, sycl::buffer<sycl::half, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    queue.submit([&](sycl::handler& cgh) {
#ifndef __HIPSYCL__ //hip reports aspect as not supported even if it works
        if (!verify_support<sycl::half, sycl::half>(queue, sycl::aspect::fp16)) {
            throw oneapi::math::unimplemented(
                "blas", "sycl::half", "half is not supported by the device or the sycl compiler");
        }
#endif
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        armpl_int_t one = 1;
        host_task<class armpl_kernel_hgemm_batch_strided>(cgh, [=]() {
            int64_t totalsize_a, totalsize_b, totalsize_c;
            int64_t size_a, size_b, size_c;
#ifdef COLUMN_MAJOR
            size_a = (transa == transpose::N) ? lda * k : lda * m;
            size_b = (transb == transpose::N) ? ldb * n : ldb * k;
            size_c = ldc * n;
#endif
#ifdef ROW_MAJOR
            size_a = (transa == transpose::N) ? lda * m : lda * k;
            size_b = (transb == transpose::N) ? ldb * k : ldb * n;
            size_c = ldc * m;
#endif
            totalsize_a = (batch_size - 1) * stride_a + size_a;
            totalsize_b = (batch_size - 1) * stride_b + size_b;
            totalsize_c = (batch_size - 1) * stride_c + size_c;

            float* f32_a = new float[totalsize_a]();
            float* f32_b = new float[totalsize_b]();
            float* f32_c = new float[totalsize_c]();
            if (!f32_a || !f32_b || !f32_c) {
                std::cerr << "Error cannot allocate input arrays\n";
                delete[] f32_a;
                delete[] f32_b;
                delete[] f32_c;
                return;
            }
            float** a_array = new float*[batch_size]();
            float** b_array = new float*[batch_size]();
            float** c_array = new float*[batch_size]();
            if (!a_array || !b_array || !c_array) {
                std::cerr << "Error cannot allocate input arrays\n";
                delete[] a_array;
                delete[] b_array;
                delete[] c_array;
                return;
            }
            // copy A, B and C to float
            copy_mat(a_acc, CblasColMajor, transpose::N, totalsize_a, 1, totalsize_a, 0.0f, f32_a);
            copy_mat(b_acc, CblasColMajor, transpose::N, totalsize_b, 1, totalsize_b, 0.0f, f32_b);
            copy_mat(c_acc, CblasColMajor, transpose::N, totalsize_c, 1, totalsize_c, 0.0f, f32_c);
            float alphaf = (float)alpha;
            float betaf = (float)beta;
            for (size_t i = 0; i < batch_size; ++i) {
                a_array[i] = f32_a + i * stride_a;
                b_array[i] = f32_b + i * stride_b;
                c_array[i] = f32_c + i * stride_c;
            }
            ::cblas_sgemm_batch(
                MAJOR, &transa_, &transb_, (const armpl_int_t*)&m, (const armpl_int_t*)&n,
                (const armpl_int_t*)&k, &alphaf, (const float**)a_array, (const armpl_int_t*)&lda,
                (const float**)b_array, (const armpl_int_t*)&ldb, &betaf, (float**)c_array,
                (const armpl_int_t*)&ldc, one, (const armpl_int_t*)&batch_size);
            // copy C back to half
            sycl::half co = 0.0f;
            copy_mat(f32_c, CblasColMajor, totalsize_c, 1, totalsize_c, offset::F, &co,
                     (sycl::half*)c_acc.GET_MULTI_PTR);
            delete[] a_array;
            delete[] b_array;
            delete[] c_array;
            delete[] f32_a;
            delete[] f32_b;
            delete[] f32_c;
        });
    });
}

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, float alpha, sycl::buffer<sycl::half, 1>& a, int64_t lda,
                int64_t stride_a, sycl::buffer<sycl::half, 1>& b, int64_t ldb, int64_t stride_b,
                float beta, sycl::buffer<float, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    queue.submit([&](sycl::handler& cgh) {
#ifndef __HIPSYCL__ //hip reports aspect as not supported even if it works
        if (!verify_support<sycl::half, sycl::half>(queue, sycl::aspect::fp16)) {
            throw oneapi::math::unimplemented(
                "blas", "sycl::half", "half is not supported by the device or the sycl compiler");
        }
#endif
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        armpl_int_t one = 1;
        host_task<class armpl_kernel_hgemm_batch_strided>(cgh, [=]() {
            int64_t totalsize_a, totalsize_b, totalsize_c;
            int64_t size_a, size_b, size_c;
#ifdef COLUMN_MAJOR
            size_a = (transa == transpose::N) ? lda * k : lda * m;
            size_b = (transb == transpose::N) ? ldb * n : ldb * k;
            size_c = ldc * n;
#endif
#ifdef ROW_MAJOR
            size_a = (transa == transpose::N) ? lda * m : lda * k;
            size_b = (transb == transpose::N) ? ldb * k : ldb * n;
            size_c = ldc * m;
#endif
            totalsize_a = (batch_size - 1) * stride_a + size_a;
            totalsize_b = (batch_size - 1) * stride_b + size_b;
            totalsize_c = (batch_size - 1) * stride_c + size_c;

            float* f32_a = new float[totalsize_a]();
            float* f32_b = new float[totalsize_b]();
            if (!f32_a || !f32_b) {
                std::cerr << "Error cannot allocate input arrays\n";
                delete[] f32_a;
                delete[] f32_b;
                return;
            }
            float** a_array = new float*[batch_size]();
            float** b_array = new float*[batch_size]();
            float** c_array = new float*[batch_size]();
            if (!a_array || !b_array) {
                std::cerr << "Error cannot allocate input arrays\n";
                delete[] a_array;
                delete[] b_array;
                delete[] c_array;
                return;
            }
            // copy A, B to float
            copy_mat(a_acc, CblasColMajor, transpose::N, totalsize_a, 1, totalsize_a, 0.0f, f32_a);
            copy_mat(b_acc, CblasColMajor, transpose::N, totalsize_b, 1, totalsize_b, 0.0f, f32_b);
            float alphaf = (float)alpha;
            float betaf = (float)beta;
            for (size_t i = 0; i < batch_size; ++i) {
                a_array[i] = f32_a + i * stride_a;
                b_array[i] = f32_b + i * stride_b;
                c_array[i] = c_acc.GET_MULTI_PTR + i * stride_c;
            }
            ::cblas_sgemm_batch(
                MAJOR, &transa_, &transb_, (const armpl_int_t*)&m, (const armpl_int_t*)&n,
                (const armpl_int_t*)&k, &alphaf, (const float**)a_array, (const armpl_int_t*)&lda,
                (const float**)b_array, (const armpl_int_t*)&ldb, &betaf, (float**)c_array,
                (const armpl_int_t*)&ldc, one, (const armpl_int_t*)&batch_size);
            // copy C back to half
            delete[] a_array;
            delete[] b_array;
            delete[] c_array;
            delete[] f32_a;
            delete[] f32_b;
        });
    });
}

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, float alpha, sycl::buffer<int8_t, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<int8_t, 1>& b, int64_t ldb, int64_t stride_b, float beta,
                sycl::buffer<float, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, float alpha, sycl::buffer<int8_t, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<int8_t, 1>& b, int64_t ldb, int64_t stride_b, float beta,
                sycl::buffer<int32_t, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

template <typename T, typename CBLAS_FUNC>
void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, T alpha, sycl::buffer<T, 1>& a, int64_t lda,
                int64_t stride_a, sycl::buffer<T, 1>& b, int64_t ldb, int64_t stride_b,
                int64_t batch_size, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read_write>(cgh);
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(trans);
        CBLAS_DIAG diag_ = cblas_convert(unit_diag);
        host_task<class armpl_kernel_trsm_batch>(cgh, [=]() {
            for (size_t i = 0; i < batch_size; ++i) {
                T* a_array = a_acc.GET_MULTI_PTR + i * stride_a;
                T* b_array = b_acc.GET_MULTI_PTR + i * stride_b;
                cblas_func(MAJOR, left_right_, upper_lower_, transa_, diag_, m, n,
                           cast_to_void_if_complex(alpha), a_array, lda, b_array, ldb);
            }
        });
    });
}

#define TRSM_BATCH_LAUNCHER(TYPE, ROUTINE)                                                      \
    void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,     \
                    diag unit_diag, int64_t m, int64_t n, TYPE alpha, sycl::buffer<TYPE, 1>& a, \
                    int64_t lda, int64_t stride_a, sycl::buffer<TYPE, 1>& b, int64_t ldb,       \
                    int64_t stride_b, int64_t batch_size) {                                     \
        trsm_batch(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,       \
                   stride_a, b, ldb, stride_b, batch_size, ROUTINE);                            \
    }

TRSM_BATCH_LAUNCHER(float, ::cblas_strsm)
TRSM_BATCH_LAUNCHER(double, ::cblas_dtrsm)
TRSM_BATCH_LAUNCHER(std::complex<float>, ::cblas_ctrsm)
TRSM_BATCH_LAUNCHER(std::complex<double>, ::cblas_ztrsm)

template <typename T, typename CBLAS_FUNC>
void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                T alpha, sycl::buffer<T, 1>& a, int64_t lda, int64_t stride_a, T beta,
                sycl::buffer<T, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(trans);
        host_task<class armpl_kernel_syrk_batch>(cgh, [=]() {
            for (size_t i = 0; i < batch_size; ++i) {
                T* a_array = a_acc.GET_MULTI_PTR + i * stride_a;
                T* c_array = c_acc.GET_MULTI_PTR + i * stride_c;
                cblas_func(MAJOR, upper_lower_, transa_, n, k, cast_to_void_if_complex(alpha),
                           a_array, lda, cast_to_void_if_complex(beta), c_array, ldc);
            }
        });
    });
}

#define SYRK_BATCH_LAUNCHER(TYPE, ROUTINE)                                                       \
    void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k, \
                    TYPE alpha, sycl::buffer<TYPE, 1>& a, int64_t lda, int64_t stride_a,         \
                    TYPE beta, sycl::buffer<TYPE, 1>& c, int64_t ldc, int64_t stride_c,          \
                    int64_t batch_size) {                                                        \
        syrk_batch(queue, upper_lower, trans, n, k, alpha, a, lda, stride_a, beta, c, ldc,       \
                   stride_c, batch_size, ROUTINE);                                               \
    }

SYRK_BATCH_LAUNCHER(float, ::cblas_ssyrk)
SYRK_BATCH_LAUNCHER(double, ::cblas_dsyrk)
SYRK_BATCH_LAUNCHER(std::complex<float>, ::cblas_csyrk)
SYRK_BATCH_LAUNCHER(std::complex<double>, ::cblas_zsyrk)

template <typename T, typename CBLAS_FUNC>
void omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, T alpha,
                    sycl::buffer<T, 1>& a, int64_t lda, int64_t stride_a, sycl::buffer<T, 1>& b,
                    int64_t ldb, int64_t stride_b, int64_t batch_size, CBLAS_FUNC cblas_func) {
    using ArmPLDataType = typename ArmPLEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read_write>(cgh);
        auto trans_ = fortran_char(trans);
#ifdef COLUMN_MAJOR
        char order = 'C';
#endif
#ifdef ROW_MAJOR
        check_armpl_version(
            24, 10, 0,
            "omatcopy row major is not functional"); // Row major omatcopy is broken (reported) in armpl <=24.04
        char order = 'R';
#endif
        host_task<class armpl_kernel_omatcopy_batch>(cgh, [=]() {
            for (size_t i = 0; i < batch_size; ++i) {
                T* a_array = a_acc.GET_MULTI_PTR + i * stride_a;
                T* b_array = b_acc.GET_MULTI_PTR + i * stride_b;
                if constexpr (is_complex<T>) {
                    ArmPLDataType alpha_ = { alpha.real(), alpha.imag() };
                    cblas_func(order, trans_, m, n, alpha_,
                               reinterpret_cast<const ArmPLDataType*>(a_array), lda,
                               reinterpret_cast<ArmPLDataType*>(b_array), ldb);
                }
                else {
                    cblas_func(order, trans_, m, n, cast_to_void_if_complex(alpha), a_array, lda,
                               b_array, ldb);
                }
            }
        });
    });
}

#define OMATCOPY_BATCH_LAUNCHER(TYPE, ROUTINE)                                                    \
    void omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, TYPE alpha,    \
                        sycl::buffer<TYPE, 1>& a, int64_t lda, int64_t stride_a,                  \
                        sycl::buffer<TYPE, 1>& b, int64_t ldb, int64_t stride_b,                  \
                        int64_t batch_size) {                                                     \
        omatcopy_batch(queue, trans, m, n, alpha, a, lda, stride_a, b, ldb, stride_b, batch_size, \
                       ROUTINE);                                                                  \
    }

OMATCOPY_BATCH_LAUNCHER(float, ::somatcopy)
OMATCOPY_BATCH_LAUNCHER(double, ::domatcopy)
OMATCOPY_BATCH_LAUNCHER(std::complex<float>, ::comatcopy)
OMATCOPY_BATCH_LAUNCHER(std::complex<double>, ::zomatcopy)

template <typename T, typename CBLAS_FUNC>
void imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, T alpha,
                    sycl::buffer<T, 1>& ab, int64_t lda, int64_t ldb, int64_t stride,
                    int64_t batch_size, CBLAS_FUNC cblas_func) {
    using ArmPLDataType = typename ArmPLEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto ab_acc = ab.template get_access<sycl::access::mode::read_write>(cgh);
        auto trans_ = fortran_char(trans);
#ifdef COLUMN_MAJOR
        char order = 'C';
#endif
#ifdef ROW_MAJOR
        check_armpl_version(
            24, 10, 0,
            "imatcopy row major is not functional"); // Row major imatcopy is broken (reported) in armpl <=24.04
        char order = 'R';
#endif

        host_task<class armpl_kernel_imatcopy_batch>(cgh, [=]() {
            for (size_t i = 0; i < batch_size; ++i) {
                T* ab_array = ab_acc.GET_MULTI_PTR + i * stride;
                if constexpr (is_complex<T>) {
                    ArmPLDataType alpha_ = { alpha.real(), alpha.imag() };
                    cblas_func(order, trans_, m, n, alpha_,
                               reinterpret_cast<ArmPLDataType*>(ab_array), lda, ldb);
                }
                else {
                    cblas_func(order, trans_, m, n, cast_to_void_if_complex(alpha), ab_array, lda,
                               ldb);
                }
            }
        });
    });
}

#define IMATCOPY_BATCH_LAUNCHER(TYPE, ROUTINE)                                                 \
    void imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, TYPE alpha, \
                        sycl::buffer<TYPE, 1>& ab, int64_t lda, int64_t ldb, int64_t stride,   \
                        int64_t batch_size) {                                                  \
        imatcopy_batch(queue, trans, m, n, alpha, ab, lda, ldb, stride, batch_size, ROUTINE);  \
    }

IMATCOPY_BATCH_LAUNCHER(float, ::simatcopy)
IMATCOPY_BATCH_LAUNCHER(double, ::dimatcopy)
IMATCOPY_BATCH_LAUNCHER(std::complex<float>, ::cimatcopy)
IMATCOPY_BATCH_LAUNCHER(std::complex<double>, ::zimatcopy)

#define OMATADD_BATCH_LAUNCHER(TYPE)                                                              \
    void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m,         \
                       int64_t n, TYPE alpha, sycl::buffer<TYPE, 1>& a, int64_t lda,              \
                       int64_t stride_a, TYPE beta, sycl::buffer<TYPE, 1>& b, int64_t ldb,        \
                       int64_t stride_b, sycl::buffer<TYPE, 1>& c, int64_t ldc, int64_t stride_c, \
                       int64_t batch_size) {                                                      \
        throw unimplemented("blas", "omatadd_batch", MAJOR_MINOR_TEXT);                           \
    }

OMATADD_BATCH_LAUNCHER(float)
OMATADD_BATCH_LAUNCHER(double)
OMATADD_BATCH_LAUNCHER(std::complex<float>)
OMATADD_BATCH_LAUNCHER(std::complex<double>)

// USM APIs

template <typename T, typename CBLAS_FUNC>
sycl::event copy_batch(sycl::queue& queue, int64_t* n, const T** x, int64_t* incx, T** y,
                       int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (size_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_copy_batch_group_usm>(cgh, [=]() {
            int64_t offset = 0;
            for (size_t i = 0; i < group_count; ++i) {
                for (size_t j = 0; j < group_size[i]; ++j) {
                    cblas_func(n[i], x[offset], incx[i], y[offset], incy[i]);
                    offset++;
                }
            }
        });
    });
    return done;
}

#define COPY_BATCH_USM_LAUNCHER(TYPE, ROUTINE)                                                \
    sycl::event copy_batch(sycl::queue& queue, int64_t* n, const TYPE** x, int64_t* incx,     \
                           TYPE** y, int64_t* incy, int64_t group_count, int64_t* group_size, \
                           const std::vector<sycl::event>& dependencies) {                    \
        return copy_batch(queue, n, x, incx, y, incy, group_count, group_size, dependencies,  \
                          ROUTINE);                                                           \
    }

COPY_BATCH_USM_LAUNCHER(float, ::cblas_scopy)
COPY_BATCH_USM_LAUNCHER(double, ::cblas_dcopy)
COPY_BATCH_USM_LAUNCHER(std::complex<float>, ::cblas_ccopy)
COPY_BATCH_USM_LAUNCHER(std::complex<double>, ::cblas_zcopy)

template <typename T, typename CBLAS_FUNC>
sycl::event copy_batch(sycl::queue& queue, int64_t n, const T* x, int64_t incx, int64_t stridex,
                       T* y, int64_t incy, int64_t stridey, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (size_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_copy_batch_strided_usm>(cgh, [=]() {
            for (size_t i = 0; i < batch_size; ++i) {
                cblas_func(n, x + i * stridex, incx, y + i * stridey, incy);
            }
        });
    });
    return done;
}

#define COPY_BATCH_STRIDED_USM_LAUNCHER(TYPE, ROUTINE)                                            \
    sycl::event copy_batch(sycl::queue& queue, int64_t n, const TYPE* x, int64_t incx,            \
                           int64_t stridex, TYPE* y, int64_t incy, int64_t stridey,               \
                           int64_t batch_size, const std::vector<sycl::event>& dependencies) {    \
        return copy_batch(queue, n, x, incx, stridex, y, incy, stridey, batch_size, dependencies, \
                          ROUTINE);                                                               \
    }

COPY_BATCH_STRIDED_USM_LAUNCHER(float, ::cblas_scopy)
COPY_BATCH_STRIDED_USM_LAUNCHER(double, ::cblas_dcopy)
COPY_BATCH_STRIDED_USM_LAUNCHER(std::complex<float>, ::cblas_ccopy)
COPY_BATCH_STRIDED_USM_LAUNCHER(std::complex<double>, ::cblas_zcopy)

template <typename T, typename CBLAS_FUNC>
sycl::event axpy_batch(sycl::queue& queue, int64_t* n, T* alpha, const T** x, int64_t* incx, T** y,
                       int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (size_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_axpy_batch_group_usm>(cgh, [=]() {
            int64_t offset = 0;
            for (size_t i = 0; i < group_count; ++i) {
                for (size_t j = 0; j < group_size[i]; ++j) {
                    cblas_func(n[i], cast_to_void_if_complex(alpha[i]), x[offset], incx[i],
                               y[offset], incy[i]);
                    offset++;
                }
            }
        });
    });
    return done;
}

#define AXPY_BATCH_USM_LAUNCHER(TYPE, ROUTINE)                                                  \
    sycl::event axpy_batch(sycl::queue& queue, int64_t* n, TYPE* alpha, const TYPE** x,         \
                           int64_t* incx, TYPE** y, int64_t* incy, int64_t group_count,         \
                           int64_t* group_size, const std::vector<sycl::event>& dependencies) { \
        return axpy_batch(queue, n, alpha, x, incx, y, incy, group_count, group_size,           \
                          dependencies, ROUTINE);                                               \
    }

AXPY_BATCH_USM_LAUNCHER(float, ::cblas_saxpy)
AXPY_BATCH_USM_LAUNCHER(double, ::cblas_daxpy)
AXPY_BATCH_USM_LAUNCHER(std::complex<float>, ::cblas_caxpy)
AXPY_BATCH_USM_LAUNCHER(std::complex<double>, ::cblas_zaxpy)

template <typename T, typename CBLAS_FUNC>
sycl::event axpy_batch(sycl::queue& queue, int64_t n, T alpha, const T* x, int64_t incx,
                       int64_t stridex, T* y, int64_t incy, int64_t stridey, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (size_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_axpy_batch_strided_usm>(cgh, [=]() {
            for (size_t i = 0; i < batch_size; ++i) {
                cblas_func(n, cast_to_void_if_complex(alpha), x + i * stridex, incx,
                           y + i * stridey, incy);
            }
        });
    });
    return done;
}

#define AXPY_BATCH_STRIDED_USM_LAUNCHER(TYPE, ROUTINE)                                             \
    sycl::event axpy_batch(sycl::queue& queue, int64_t n, TYPE alpha, const TYPE* x, int64_t incx, \
                           int64_t stridex, TYPE* y, int64_t incy, int64_t stridey,                \
                           int64_t batch_size, const std::vector<sycl::event>& dependencies) {     \
        return axpy_batch(queue, n, alpha, x, incx, stridex, y, incy, stridey, batch_size,         \
                          dependencies, ROUTINE);                                                  \
    }

AXPY_BATCH_STRIDED_USM_LAUNCHER(float, ::cblas_saxpy)
AXPY_BATCH_STRIDED_USM_LAUNCHER(double, ::cblas_daxpy)
AXPY_BATCH_STRIDED_USM_LAUNCHER(std::complex<float>, ::cblas_caxpy)
AXPY_BATCH_STRIDED_USM_LAUNCHER(std::complex<double>, ::cblas_zaxpy)

template <typename T, typename CBLAS_FUNC>
sycl::event gemv_batch(sycl::queue& queue, transpose* transa, int64_t* m, int64_t* n, T* alpha,
                       const T** a, int64_t* lda, const T** x, int64_t* incx, T* beta, T** y,
                       int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (size_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_gemv_batch_group_usm>(cgh, [=]() {
            int64_t offset = 0;
            for (size_t i = 0; i < group_count; ++i) {
                CBLAS_TRANSPOSE transa_ = cblas_convert(transa[i]);
                for (size_t j = 0; j < group_size[i]; ++j) {
                    cblas_func(MAJOR, transa_, m[i], n[i], cast_to_void_if_complex(alpha[i]),
                               a[offset], lda[i], x[offset], incx[i],
                               cast_to_void_if_complex(beta[i]), y[offset], incy[i]);
                    offset++;
                }
            }
        });
    });
    return done;
}

#define GEMV_BATCH_USM_LAUNCHER(TYPE, ROUTINE)                                                     \
    sycl::event gemv_batch(sycl::queue& queue, transpose* transa, int64_t* m, int64_t* n,          \
                           TYPE* alpha, const TYPE** a, int64_t* lda, const TYPE** x,              \
                           int64_t* incx, TYPE* beta, TYPE** y, int64_t* incy,                     \
                           int64_t group_count, int64_t* group_size,                               \
                           const std::vector<sycl::event>& dependencies) {                         \
        return gemv_batch(queue, transa, m, n, alpha, a, lda, x, incx, beta, y, incy, group_count, \
                          group_size, dependencies, ROUTINE);                                      \
    }

GEMV_BATCH_USM_LAUNCHER(float, ::cblas_sgemv)
GEMV_BATCH_USM_LAUNCHER(double, ::cblas_dgemv)
GEMV_BATCH_USM_LAUNCHER(std::complex<float>, ::cblas_cgemv)
GEMV_BATCH_USM_LAUNCHER(std::complex<double>, ::cblas_zgemv)

template <typename T, typename CBLAS_FUNC>
sycl::event gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n, T alpha,
                       const T* a, int64_t lda, int64_t stride_a, const T* x, int64_t incx,
                       int64_t stride_x, T beta, T* y, int64_t incy, int64_t stride_y,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies,
                       CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (size_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_gemv_batch>(cgh, [=]() {
            CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
            for (size_t i = 0; i < batch_size; ++i) {
                cblas_func(MAJOR, transa_, m, n, cast_to_void_if_complex(alpha), a + i * stride_a,
                           lda, x + i * stride_x, incx, cast_to_void_if_complex(beta),
                           y + i * stride_y, incy);
            }
        });
    });
    return done;
}

#define GEMV_BATCH_STRIDED_USM_LAUNCHER(TYPE, ROUTINE)                                             \
    sycl::event gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n, TYPE alpha, \
                           const TYPE* a, int64_t lda, int64_t stride_a, const TYPE* x,            \
                           int64_t incx, int64_t stride_x, TYPE beta, TYPE* y, int64_t incy,       \
                           int64_t stride_y, int64_t batch_size,                                   \
                           const std::vector<sycl::event>& dependencies) {                         \
        return gemv_batch(queue, transa, m, n, alpha, a, lda, stride_a, x, incx, stride_x, beta,   \
                          y, incy, stride_y, batch_size, dependencies, ROUTINE);                   \
    }

GEMV_BATCH_STRIDED_USM_LAUNCHER(float, ::cblas_sgemv)
GEMV_BATCH_STRIDED_USM_LAUNCHER(double, ::cblas_dgemv)
GEMV_BATCH_STRIDED_USM_LAUNCHER(std::complex<float>, ::cblas_cgemv)
GEMV_BATCH_STRIDED_USM_LAUNCHER(std::complex<double>, ::cblas_zgemv)

template <typename T, typename CBLAS_FUNC>
sycl::event dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n, const T* a,
                       int64_t lda, int64_t stride_a, const T* x, int64_t incx, int64_t stride_x,
                       T* c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (size_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_dgmm_batch_strided_usm>(cgh, [=]() {
            int64_t ldx = (left_right == side::left) ? m : n;
            int64_t totalsize_x = ldx * ldx;
            int64_t absincx = std::abs(incx);

            T* x_array_val = new T[totalsize_x * batch_size]{};
            if (!x_array_val) {
                std::cerr << "Error: cannot allocate input array\n";
                delete[] x_array_val;
                throw oneapi::math::host_bad_alloc("blas", __func__);
            }

            for (size_t i = 0; i < batch_size; ++i) {
                T* x_vector = (T*)x + i * stride_x;
                if (incx > 0) {
                    for (size_t j = 0; j < ldx; ++j) {
                        x_array_val[i * totalsize_x + j * ldx + j] = x_vector[j * incx];
                    }
                }
                else {
                    for (size_t j = 0; j < ldx; ++j) {
                        x_array_val[i * totalsize_x + j * ldx + j] =
                            x_vector[(ldx - j - 1) * absincx];
                    }
                }
            }

            CBLAS_TRANSPOSE transa_ = CblasNoTrans;
            CBLAS_TRANSPOSE transb_ = CblasNoTrans;

            int64_t k = ldx;

            T alpha = 1.0f;
            T beta = 0.0f;

            armpl_int_t one = 1;

            T** a_array = new T*[batch_size]();
            T** x_array = new T*[batch_size]();
            T** c_array = new T*[batch_size]();
            if (!a_array || !x_array || !c_array) {
                std::cerr << "Error: cannot allocate input arrays\n";
                delete[] a_array;
                delete[] x_array;
                delete[] c_array;
                throw oneapi::math::host_bad_alloc("blas", __func__);
            }
            for (size_t i = 0; i < batch_size; ++i) {
                a_array[i] = (T*)a + i * stride_a;
                x_array[i] = x_array_val + i * totalsize_x;
                c_array[i] = (T*)c + i * stride_c;
            }
            if constexpr (is_complex<T>) {
                if (left_right == side::left) {
                    cblas_func(MAJOR, &transa_, &transb_, (const armpl_int_t*)&m,
                               (const armpl_int_t*)&n, (const armpl_int_t*)&k, &alpha,
                               (const void* const*)x_array, (const armpl_int_t*)&ldx,
                               (const void* const*)a_array, (const armpl_int_t*)&lda, &beta,
                               (void**)c_array, (const armpl_int_t*)&ldc, one,
                               (const armpl_int_t*)&batch_size);
                }
                else {
                    cblas_func(MAJOR, &transa_, &transb_, (const armpl_int_t*)&m,
                               (const armpl_int_t*)&n, (const armpl_int_t*)&k, &alpha,
                               (const void* const*)a_array, (const armpl_int_t*)&lda,
                               (const void* const*)x_array, (const armpl_int_t*)&ldx, &beta,
                               (void**)c_array, (const armpl_int_t*)&ldc, one,
                               (const armpl_int_t*)&batch_size);
                }
            }
            else {
                if (left_right == side::left) {
                    cblas_func(MAJOR, &transa_, &transb_, (const armpl_int_t*)&m,
                               (const armpl_int_t*)&n, (const armpl_int_t*)&k, &alpha,
                               (const T**)x_array, (const armpl_int_t*)&ldx, (const T**)a_array,
                               (const armpl_int_t*)&lda, &beta, (T**)c_array,
                               (const armpl_int_t*)&ldc, one, (const armpl_int_t*)&batch_size);
                }
                else {
                    cblas_func(MAJOR, &transa_, &transb_, (const armpl_int_t*)&m,
                               (const armpl_int_t*)&n, (const armpl_int_t*)&k, &alpha,
                               (const T**)a_array, (const armpl_int_t*)&lda, (const T**)x_array,
                               (const armpl_int_t*)&ldx, &beta, (T**)c_array,
                               (const armpl_int_t*)&ldc, one, (const armpl_int_t*)&batch_size);
                }
            }

            delete[] x_array_val;
            delete[] a_array;
            delete[] x_array;
            delete[] c_array;
        });
    });
    return done;
}

#define DGMM_BATCH_STRIDED_USM_LAUNCHER(TYPE, ROUTINE)                                             \
    sycl::event dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,              \
                           const TYPE* a, int64_t lda, int64_t stride_a, const TYPE* x,            \
                           int64_t incx, int64_t stride_x, TYPE* c, int64_t ldc, int64_t stride_c, \
                           int64_t batch_size, const std::vector<sycl::event>& dependencies) {     \
        return dgmm_batch(queue, left_right, m, n, a, lda, stride_a, x, incx, stride_x, c, ldc,    \
                          stride_c, batch_size, dependencies, ROUTINE);                            \
    }

DGMM_BATCH_STRIDED_USM_LAUNCHER(float, ::cblas_sgemm_batch)
DGMM_BATCH_STRIDED_USM_LAUNCHER(double, ::cblas_dgemm_batch)
DGMM_BATCH_STRIDED_USM_LAUNCHER(std::complex<float>, ::cblas_cgemm_batch)
DGMM_BATCH_STRIDED_USM_LAUNCHER(std::complex<double>, ::cblas_zgemm_batch)

template <typename T, typename CBLAS_FUNC>
sycl::event dgmm_batch(sycl::queue& queue, side* left_right, int64_t* m, int64_t* n, const T** a,
                       int64_t* lda, const T** x, int64_t* incx, T** c, int64_t* ldc,
                       int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (size_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_sdgmm_batch_group_usm>(cgh, [=]() {
            int64_t offset = 0;
            for (size_t i = 0; i < group_count; ++i) {
                // left = m, right = n
                int64_t ldx = (left_right[i] == side::left) ? m[i] : n[i];
                int64_t totalsize_x = ldx * ldx;
                int64_t absincx = std::abs(incx[i]);

                T* x_array_val = new T[totalsize_x * group_size[i]]{};
                if (!x_array_val) {
                    std::cerr << "Error: cannot allocate input array\n";
                    delete[] x_array_val;
                    throw oneapi::math::host_bad_alloc("blas", __func__);
                }

                // Fill X arrays
                for (size_t j = 0; j < group_size[i]; ++j) {
                    T* x_vector = (T*)x[offset + j];

                    if (incx[i] > 0) { // positive stride
                        for (int64_t k = 0; k < ldx; ++k) {
                            x_array_val[j * totalsize_x + k * ldx + k] = x_vector[k * incx[i]];
                        }
                    }
                    else { // negative stride
                        for (int64_t k = 0; k < ldx; ++k) {
                            x_array_val[j * totalsize_x + k * ldx + k] =
                                x_vector[(ldx - k - 1) * absincx];
                        }
                    }
                }

                // Usefull variable to call cblas_sgemm_batch
                CBLAS_TRANSPOSE transa_ = CblasNoTrans;
                CBLAS_TRANSPOSE transb_ = CblasNoTrans;

                int64_t k = ldx;

                T alpha = 1.0f;
                T beta = 0.0f;

                armpl_int_t one = 1;

                // Allocate memory
                T** x_array = new T*[group_size[i]]();
                if (!x_array) {
                    std::cerr << "Error: cannot allocate input array\n";
                    delete[] x_array;
                    throw oneapi::math::host_bad_alloc("blas", __func__);
                }
                for (size_t j = 0; j < group_size[i]; ++j) {
                    x_array[j] = x_array_val + j * totalsize_x;
                }

                // Apply SGEMM
                // C = alpha * diag(X) * A + beta * C
                // or
                // C = alpha * A * diag(X) + beta * C
                if constexpr (is_complex<T>)
                    if (left_right[i] == side::left) { // left
                        // c_array = one * x_array * a_array + zero * c_array
                        cblas_func(MAJOR, &transa_, &transb_, (const armpl_int_t*)&m[i],
                                   (const armpl_int_t*)&n[i], (const armpl_int_t*)&k, &alpha,
                                   (const void* const*)x_array, (const armpl_int_t*)&ldx,
                                   (const void* const*)&a[offset], (const armpl_int_t*)&lda[i],
                                   &beta, (void**)&c[offset], (const armpl_int_t*)&ldc[i], one,
                                   (const armpl_int_t*)&group_size[i]);
                    }
                    else { // right
                        // c_array = one * a_array * x_array + zero * c_array
                        cblas_func(MAJOR, &transa_, &transb_, (const armpl_int_t*)&m[i],
                                   (const armpl_int_t*)&n[i], (const armpl_int_t*)&k, &alpha,
                                   (const void* const*)&a[offset], (const armpl_int_t*)&lda[i],
                                   (const void* const*)x_array, (const armpl_int_t*)&ldx, &beta,
                                   (void**)&c[offset], (const armpl_int_t*)&ldc[i], one,
                                   (const armpl_int_t*)&group_size[i]);
                    }
                else

                    if (left_right[i] == side::left) { // left
                    // c_array = one * x_array * a_array + zero * c_array
                    cblas_func(MAJOR, &transa_, &transb_, (const armpl_int_t*)&m[i],
                               (const armpl_int_t*)&n[i], (const armpl_int_t*)&k, &alpha,
                               (const T**)x_array, (const armpl_int_t*)&ldx, (const T**)&a[offset],
                               (const armpl_int_t*)&lda[i], &beta, (T**)&c[offset],
                               (const armpl_int_t*)&ldc[i], one,
                               (const armpl_int_t*)&group_size[i]);
                }
                else { // right
                    // c_array = one * a_array * x_array + zero * c_array
                    cblas_func(MAJOR, &transa_, &transb_, (const armpl_int_t*)&m[i],
                               (const armpl_int_t*)&n[i], (const armpl_int_t*)&k, &alpha,
                               (const T**)&a[offset], (const armpl_int_t*)&lda[i],
                               (const T**)x_array, (const armpl_int_t*)&ldx, &beta, (T**)&c[offset],
                               (const armpl_int_t*)&ldc[i], one,
                               (const armpl_int_t*)&group_size[i]);
                }

                // Release memory
                delete[] x_array_val;
                delete[] x_array;

                // Increment offset
                offset += group_size[i];
            }
        });
    });
    return done;
}

#define DGMM_BATCH_USM_LAUNCHER(TYPE, ROUTINE)                                                    \
    sycl::event dgmm_batch(sycl::queue& queue, side* left_right, int64_t* m, int64_t* n,          \
                           const TYPE** a, int64_t* lda, const TYPE** x, int64_t* incx, TYPE** c, \
                           int64_t* ldc, int64_t group_count, int64_t* group_size,                \
                           const std::vector<sycl::event>& dependencies) {                        \
        return dgmm_batch(queue, left_right, m, n, a, lda, x, incx, c, ldc, group_count,          \
                          group_size, dependencies, ROUTINE);                                     \
    }

DGMM_BATCH_USM_LAUNCHER(float, ::cblas_sgemm_batch)
DGMM_BATCH_USM_LAUNCHER(double, ::cblas_dgemm_batch)
DGMM_BATCH_USM_LAUNCHER(std::complex<float>, ::cblas_cgemm_batch)
DGMM_BATCH_USM_LAUNCHER(std::complex<double>, ::cblas_zgemm_batch)

template <typename T, typename CBLAS_FUNC>
sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, T* alpha, const T** a, int64_t* lda, const T** b,
                       int64_t* ldb, T* beta, T** c, int64_t* ldc, int64_t group_count,
                       int64_t* group_size, const std::vector<sycl::event>& dependencies,
                       CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (size_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_gemm_batch_group_usm>(cgh, [=]() {
            CBLAS_TRANSPOSE* transa_ = new CBLAS_TRANSPOSE[group_count]();
            CBLAS_TRANSPOSE* transb_ = new CBLAS_TRANSPOSE[group_count]();
            if (!transa_ || !transb_) {
                std::cerr << "Error cannot allocate trans arrays\n";
                delete[] transa_;
                delete[] transb_;
                return;
            }
            for (size_t i = 0; i < group_count; ++i) {
                transa_[i] = cblas_convert(transa[i]);
                transb_[i] = cblas_convert(transb[i]);
            }
            if constexpr (is_complex<T>)
                cblas_func(MAJOR, transa_, transb_, (const armpl_int_t*)m, (const armpl_int_t*)n,
                           (const armpl_int_t*)k, alpha, (const void**)a, (const armpl_int_t*)lda,
                           (const void* const*)b, (const armpl_int_t*)ldb, beta, (void**)c,
                           (const armpl_int_t*)ldc, group_count, (const armpl_int_t*)group_size);
            else
                cblas_func(MAJOR, transa_, transb_, (const armpl_int_t*)m, (const armpl_int_t*)n,
                           (const armpl_int_t*)k, alpha, a, (const armpl_int_t*)lda, b,
                           (const armpl_int_t*)ldb, beta, c, (const armpl_int_t*)ldc, group_count,
                           (const armpl_int_t*)group_size);
            delete[] transa_;
            delete[] transb_;
        });
    });
    return done;
}

#define GEMM_BATCH_USM_LAUNCHER(TYPE, ROUTINE)                                                   \
    sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m, \
                           int64_t* n, int64_t* k, TYPE* alpha, const TYPE** a, int64_t* lda,    \
                           const TYPE** b, int64_t* ldb, TYPE* beta, TYPE** c, int64_t* ldc,     \
                           int64_t group_count, int64_t* group_size,                             \
                           const std::vector<sycl::event>& dependencies) {                       \
        return gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,   \
                          group_count, group_size, dependencies, ROUTINE);                       \
    }

GEMM_BATCH_USM_LAUNCHER(float, ::cblas_sgemm_batch)
GEMM_BATCH_USM_LAUNCHER(double, ::cblas_dgemm_batch)
GEMM_BATCH_USM_LAUNCHER(std::complex<float>, ::cblas_cgemm_batch)
GEMM_BATCH_USM_LAUNCHER(std::complex<double>, ::cblas_zgemm_batch)

sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, sycl::half* alpha, const sycl::half** a,
                       int64_t* lda, const sycl::half** b, int64_t* ldb, sycl::half* beta,
                       sycl::half** c, int64_t* ldc, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
#ifndef __HIPSYCL__ //hip reports aspect as not supported even if it works
        if (!verify_support<sycl::half, sycl::half>(queue, sycl::aspect::fp16)) {
            throw oneapi::math::unimplemented(
                "blas", "sycl::half", "half is not supported by the device or the sycl compiler");
        }
#endif
        int64_t num_events = dependencies.size();
        for (size_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_hgemm_batch_group_usm>(cgh, [=]() {
            CBLAS_TRANSPOSE* transa_ = new CBLAS_TRANSPOSE[group_count]();
            CBLAS_TRANSPOSE* transb_ = new CBLAS_TRANSPOSE[group_count]();
            float* alphaf = new float[group_count]();
            float* betaf = new float[group_count]();
            if (!transa_ || !transb_ || !alphaf || !betaf) {
                std::cerr << "Error cannot allocate input arrays\n";
                delete[] transa_;
                delete[] transb_;
                delete[] alphaf;
                delete[] betaf;
                return;
            }
            int64_t totalbatch_size = 0;
            for (size_t i = 0; i < group_count; ++i) {
                transa_[i] = cblas_convert(transa[i]);
                transb_[i] = cblas_convert(transb[i]);
                alphaf[i] = (float)alpha[i];
                betaf[i] = (float)beta[i];
                totalbatch_size += group_size[i];
            }
            float** a_array = new float*[totalbatch_size]();
            float** b_array = new float*[totalbatch_size]();
            float** c_array = new float*[totalbatch_size]();
            if (!a_array || !b_array || !c_array) {
                std::cerr << "Error cannot allocate input arrays\n";
                delete[] a_array;
                delete[] b_array;
                delete[] c_array;
                return;
            }
            int64_t size_a, size_b, size_c, idx;
            sycl::half co = 0.0f;
            for (size_t i = 0, idx = 0; i < group_count; ++i) {
#ifdef COLUMN_MAJOR
                size_a = (transa[i] == transpose::N) ? lda[i] * k[i] : lda[i] * m[i];
                size_b = (transb[i] == transpose::N) ? ldb[i] * n[i] : ldb[i] * k[i];
                size_c = ldc[i] * n[i];
#endif
#ifdef ROW_MAJOR
                size_a = (transa[i] == transpose::N) ? lda[i] * m[i] : lda[i] * k[i];
                size_b = (transb[i] == transpose::N) ? ldb[i] * k[i] : ldb[i] * n[i];
                size_c = ldc[i] * m[i];
#endif
                for (size_t j = 0; j < group_size[i]; ++j, ++idx) {
                    a_array[idx] = new float[size_a]();
                    b_array[idx] = new float[size_b]();
                    c_array[idx] = new float[size_c]();
                    copy_mat(a[idx], MAJOR, transa[i], m[i], k[i], lda[i], 0.0f, a_array[idx]);
                    copy_mat(b[idx], MAJOR, transb[i], k[i], n[i], ldb[i], 0.0f, b_array[idx]);
                    copy_mat(c[idx], MAJOR, transpose::N, m[i], n[i], ldc[i], 0.0f, c_array[idx]);
                }
            }
            ::cblas_sgemm_batch(
                MAJOR, transa_, transb_, (const armpl_int_t*)m, (const armpl_int_t*)n,
                (const armpl_int_t*)k, alphaf, (const float**)a_array, (const armpl_int_t*)lda,
                (const float**)b_array, (const armpl_int_t*)ldb, betaf, (float**)c_array,
                (const armpl_int_t*)ldc, group_count, (const armpl_int_t*)group_size);
            for (size_t i = 0, idx = 0; i < group_count; ++i) {
                size_c = ldc[i] * n[i];
                for (size_t j = 0; j < group_size[i]; ++j, ++idx) {
                    copy_mat(c_array[idx], MAJOR, m[i], n[i], ldc[i], offset::F, &co, c[idx]);
                    delete[] a_array[idx];
                    delete[] b_array[idx];
                    delete[] c_array[idx];
                }
            }
            delete[] alphaf;
            delete[] betaf;
            delete[] a_array;
            delete[] b_array;
            delete[] c_array;
            delete[] transa_;
            delete[] transb_;
        });
    });
    return done;
}

sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, float* alpha, const sycl::half** a, int64_t* lda,
                       const sycl::half** b, int64_t* ldb, float* beta, float** c, int64_t* ldc,
                       int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
#ifndef __HIPSYCL__ //hip reports aspect as not supported even if it works
        if (!verify_support<sycl::half, sycl::half>(queue, sycl::aspect::fp16)) {
            throw oneapi::math::unimplemented(
                "blas", "sycl::half", "half is not supported by the device or the sycl compiler");
        }
#endif
        int64_t num_events = dependencies.size();
        for (size_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_hgemm_batch_group_usm>(cgh, [=]() {
            CBLAS_TRANSPOSE* transa_ = new CBLAS_TRANSPOSE[group_count]();
            CBLAS_TRANSPOSE* transb_ = new CBLAS_TRANSPOSE[group_count]();
            float* alphaf = new float[group_count]();
            float* betaf = new float[group_count]();
            if (!transa_ || !transb_ || !alphaf || !betaf) {
                std::cerr << "Error cannot allocate input arrays\n";
                delete[] transa_;
                delete[] transb_;
                delete[] alphaf;
                delete[] betaf;
                return;
            }
            int64_t totalbatch_size = 0;
            for (size_t i = 0; i < group_count; ++i) {
                transa_[i] = cblas_convert(transa[i]);
                transb_[i] = cblas_convert(transb[i]);
                alphaf[i] = (float)alpha[i];
                betaf[i] = (float)beta[i];
                totalbatch_size += group_size[i];
            }
            float** a_array = new float*[totalbatch_size]();
            float** b_array = new float*[totalbatch_size]();
            if (!a_array || !b_array) {
                std::cerr << "Error cannot allocate input arrays\n";
                delete[] a_array;
                delete[] b_array;
                return;
            }
            int64_t size_a, size_b, idx;
            sycl::half co = 0.0f;
            for (size_t i = 0, idx = 0; i < group_count; ++i) {
#ifdef COLUMN_MAJOR
                size_a = (transa[i] == transpose::N) ? lda[i] * k[i] : lda[i] * m[i];
                size_b = (transb[i] == transpose::N) ? ldb[i] * n[i] : ldb[i] * k[i];
#endif
#ifdef ROW_MAJOR
                size_a = (transa[i] == transpose::N) ? lda[i] * m[i] : lda[i] * k[i];
                size_b = (transb[i] == transpose::N) ? ldb[i] * k[i] : ldb[i] * n[i];
#endif
                for (size_t j = 0; j < group_size[i]; ++j, ++idx) {
                    a_array[idx] = new float[size_a]();
                    b_array[idx] = new float[size_b]();
                    copy_mat(a[idx], MAJOR, transa[i], m[i], k[i], lda[i], 0.0f, a_array[idx]);
                    copy_mat(b[idx], MAJOR, transb[i], k[i], n[i], ldb[i], 0.0f, b_array[idx]);
                }
            }
            ::cblas_sgemm_batch(
                MAJOR, transa_, transb_, (const armpl_int_t*)m, (const armpl_int_t*)n,
                (const armpl_int_t*)k, alphaf, (const float**)a_array, (const armpl_int_t*)lda,
                (const float**)b_array, (const armpl_int_t*)ldb, betaf, (float**)c,
                (const armpl_int_t*)ldc, group_count, (const armpl_int_t*)group_size);
            for (size_t i = 0, idx = 0; i < group_count; ++i) {
                for (size_t j = 0; j < group_size[i]; ++j, ++idx) {
                    delete[] a_array[idx];
                    delete[] b_array[idx];
                }
            }
            delete[] alphaf;
            delete[] betaf;
            delete[] a_array;
            delete[] b_array;
            delete[] transa_;
            delete[] transb_;
        });
    });
    return done;
}

sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, float* alpha, const int8_t** a, int64_t* lda,
                       const int8_t** b, int64_t* ldb, float* beta, float** c, int64_t* ldc,
                       int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, float* alpha, const int8_t** a, int64_t* lda,
                       const int8_t** b, int64_t* ldb, float* beta, int32_t** c, int64_t* ldc,
                       int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

template <typename T, typename CBLAS_FUNC>
sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, T alpha, const T* a, int64_t lda, int64_t stride_a, const T* b,
                       int64_t ldb, int64_t stride_b, T beta, T* c, int64_t ldc, int64_t stride_c,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies,
                       CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (size_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        armpl_int_t one = 1;
        host_task<class armpl_kernel_gemm_batch_strided_usm>(cgh, [=]() {
            T** a_array = new T*[batch_size]();
            T** b_array = new T*[batch_size]();
            T** c_array = new T*[batch_size]();
            if (!a_array || !b_array || !c_array) {
                std::cerr << "Error cannot allocate input arrays\n";
                delete[] a_array;
                delete[] b_array;
                delete[] c_array;
                return;
            }
            for (size_t i = 0; i < batch_size; ++i) {
                a_array[i] = (T*)a + i * stride_a;
                b_array[i] = (T*)b + i * stride_b;
                c_array[i] = (T*)c + i * stride_c;
            }
            if constexpr (is_complex<T>)
                cblas_func(MAJOR, &transa_, &transb_, (const armpl_int_t*)&m,
                           (const armpl_int_t*)&n, (const armpl_int_t*)&k, &alpha,
                           (const void**)a_array, (const armpl_int_t*)&lda, (const void**)b_array,
                           (const armpl_int_t*)&ldb, &beta, (void**)c_array,
                           (const armpl_int_t*)&ldc, one, (const armpl_int_t*)&batch_size);
            else
                cblas_func(MAJOR, &transa_, &transb_, (const armpl_int_t*)&m,
                           (const armpl_int_t*)&n, (const armpl_int_t*)&k, &alpha, a_array,
                           (const armpl_int_t*)&lda, b_array, (const armpl_int_t*)&ldb, &beta,
                           c_array, (const armpl_int_t*)&ldc, one, (const armpl_int_t*)&batch_size);
            delete[] a_array;
            delete[] b_array;
            delete[] c_array;
        });
    });
    return done;
}

#define GEMM_BATCH_STRIDED_USM_LAUNCHER(TYPE, ROUTINE)                                            \
    sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m,     \
                           int64_t n, int64_t k, TYPE alpha, const TYPE* a, int64_t lda,          \
                           int64_t stride_a, const TYPE* b, int64_t ldb, int64_t stride_b,        \
                           TYPE beta, TYPE* c, int64_t ldc, int64_t stride_c, int64_t batch_size, \
                           const std::vector<sycl::event>& dependencies) {                        \
        return gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,        \
                          stride_b, beta, c, ldc, stride_c, batch_size, dependencies, ROUTINE);   \
    }

GEMM_BATCH_STRIDED_USM_LAUNCHER(float, ::cblas_sgemm_batch)
GEMM_BATCH_STRIDED_USM_LAUNCHER(double, ::cblas_dgemm_batch)
GEMM_BATCH_STRIDED_USM_LAUNCHER(std::complex<float>, ::cblas_cgemm_batch)
GEMM_BATCH_STRIDED_USM_LAUNCHER(std::complex<double>, ::cblas_zgemm_batch)

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, sycl::half alpha, const sycl::half* a, int64_t lda,
                       int64_t stride_a, const sycl::half* b, int64_t ldb, int64_t stride_b,
                       sycl::half beta, sycl::half* c, int64_t ldc, int64_t stride_c,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
#ifndef __HIPSYCL__ //hip reports aspect as not supported even if it works
        if (!verify_support<sycl::half, sycl::half>(queue, sycl::aspect::fp16)) {
            throw oneapi::math::unimplemented(
                "blas", "sycl::half", "half is not supported by the device or the sycl compiler");
        }
#endif
        int64_t num_events = dependencies.size();
        for (size_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        armpl_int_t one = 1;
        host_task<class armpl_kernel_hgemm_batch_strided_usm>(cgh, [=]() {
            int64_t totalsize_a, totalsize_b, totalsize_c;
            int64_t size_a, size_b, size_c;
#ifdef COLUMN_MAJOR
            size_a = (transa == transpose::N) ? lda * k : lda * m;
            size_b = (transb == transpose::N) ? ldb * n : ldb * k;
            size_c = ldc * n;
#endif
#ifdef ROW_MAJOR
            size_a = (transa == transpose::N) ? lda * m : lda * k;
            size_b = (transb == transpose::N) ? ldb * k : ldb * n;
            size_c = ldc * m;
#endif
            totalsize_a = (batch_size - 1) * stride_a + size_a;
            totalsize_b = (batch_size - 1) * stride_b + size_b;
            totalsize_c = (batch_size - 1) * stride_c + size_c;

            // copy A, B and C to float
            float* f32_a = new float[totalsize_a]();
            float* f32_b = new float[totalsize_b]();
            float* f32_c = new float[totalsize_c]();
            if (!f32_a || !f32_b || !f32_c) {
                std::cerr << "Error cannot allocate input arrays\n";
                delete[] f32_a;
                delete[] f32_b;
                delete[] f32_c;
                return;
            }
            copy_mat(a, CblasColMajor, transpose::N, totalsize_a, 1, totalsize_a, 0.0f, f32_a);
            copy_mat(b, CblasColMajor, transpose::N, totalsize_b, 1, totalsize_b, 0.0f, f32_b);
            copy_mat(c, CblasColMajor, transpose::N, totalsize_c, 1, totalsize_c, 0.0f, f32_c);

            float alphaf = (float)alpha;
            float betaf = (float)beta;
            float** a_array = new float*[batch_size]();
            float** b_array = new float*[batch_size]();
            float** c_array = new float*[batch_size]();
            if (!a_array || !b_array || !c_array) {
                std::cerr << "Error cannot allocate input arrays\n";
                delete[] a_array;
                delete[] b_array;
                delete[] c_array;
                return;
            }
            for (size_t i = 0; i < batch_size; ++i) {
                a_array[i] = (float*)f32_a + i * stride_a;
                b_array[i] = (float*)f32_b + i * stride_b;
                c_array[i] = (float*)f32_c + i * stride_c;
            }
            ::cblas_sgemm_batch(
                MAJOR, &transa_, &transb_, (const armpl_int_t*)&m, (const armpl_int_t*)&n,
                (const armpl_int_t*)&k, &alphaf, (const float**)a_array, (const armpl_int_t*)&lda,
                (const float**)b_array, (const armpl_int_t*)&ldb, &betaf, (float**)c_array,
                (const armpl_int_t*)&ldc, one, (const armpl_int_t*)&batch_size);

            sycl::half co = 0.0f;
            copy_mat(f32_c, CblasColMajor, totalsize_c, 1, totalsize_c, offset::F, &co, c);
            delete[] a_array;
            delete[] b_array;
            delete[] c_array;
            delete[] f32_a;
            delete[] f32_b;
            delete[] f32_c;
        });
    });
    return done;
}

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, float alpha, const sycl::half* a, int64_t lda, int64_t stride_a,
                       const sycl::half* b, int64_t ldb, int64_t stride_b, float beta, float* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
#ifndef __HIPSYCL__ //hip reports aspect as not supported even if it works
        if (!verify_support<sycl::half, sycl::half>(queue, sycl::aspect::fp16)) {
            throw oneapi::math::unimplemented(
                "blas", "sycl::half", "half is not supported by the device or the sycl compiler");
        }
#endif
        int64_t num_events = dependencies.size();
        for (size_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        armpl_int_t one = 1;
        host_task<class armpl_kernel_hgemm_batch_strided_usm>(cgh, [=]() {
            int64_t totalsize_a, totalsize_b;
            int64_t size_a, size_b;
#ifdef COLUMN_MAJOR
            size_a = (transa == transpose::N) ? lda * k : lda * m;
            size_b = (transb == transpose::N) ? ldb * n : ldb * k;
#endif
#ifdef ROW_MAJOR
            size_a = (transa == transpose::N) ? lda * m : lda * k;
            size_b = (transb == transpose::N) ? ldb * k : ldb * n;
#endif
            totalsize_a = (batch_size - 1) * stride_a + size_a;
            totalsize_b = (batch_size - 1) * stride_b + size_b;

            // copy A and B to float
            float* f32_a = new float[totalsize_a]();
            float* f32_b = new float[totalsize_b]();
            if (!f32_a || !f32_b) {
                std::cerr << "Error cannot allocate input arrays\n";
                delete[] f32_a;
                delete[] f32_b;
                return;
            }
            copy_mat(a, CblasColMajor, transpose::N, totalsize_a, 1, totalsize_a, 0.0f, f32_a);
            copy_mat(b, CblasColMajor, transpose::N, totalsize_b, 1, totalsize_b, 0.0f, f32_b);

            float alphaf = (float)alpha;
            float betaf = (float)beta;
            float** a_array = new float*[batch_size]();
            float** b_array = new float*[batch_size]();
            float** c_array = new float*[batch_size]();
            if (!a_array || !b_array || !c_array) {
                std::cerr << "Error cannot allocate input arrays\n";
                delete[] a_array;
                delete[] b_array;
                delete[] c_array;
                return;
            }
            for (size_t i = 0; i < batch_size; ++i) {
                a_array[i] = (float*)f32_a + i * stride_a;
                b_array[i] = (float*)f32_b + i * stride_b;
                c_array[i] = c + i * stride_c;
            }
            ::cblas_sgemm_batch(
                MAJOR, &transa_, &transb_, (const armpl_int_t*)&m, (const armpl_int_t*)&n,
                (const armpl_int_t*)&k, &alphaf, (const float**)a_array, (const armpl_int_t*)&lda,
                (const float**)b_array, (const armpl_int_t*)&ldb, &betaf, (float**)c_array,
                (const armpl_int_t*)&ldc, one, (const armpl_int_t*)&batch_size);

            delete[] a_array;
            delete[] b_array;
            delete[] c_array;
            delete[] f32_a;
            delete[] f32_b;
        });
    });
    return done;
}

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, float alpha, const int8_t* a, int64_t lda, int64_t stride_a,
                       const int8_t* b, int64_t ldb, int64_t stride_b, float beta, float* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, float alpha, const int8_t* a, int64_t lda, int64_t stride_a,
                       const int8_t* b, int64_t ldb, int64_t stride_b, float beta, int32_t* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

template <typename T, typename CBLAS_FUNC>
sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, T alpha, const T* a, int64_t lda,
                       int64_t stride_a, T* b, int64_t ldb, int64_t stride_b, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (size_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(trans);
        CBLAS_DIAG diag_ = cblas_convert(unit_diag);
        host_task<class armpl_kernel_trsm_batch>(cgh, [=]() {
            for (size_t i = 0; i < batch_size; ++i) {
                cblas_func(MAJOR, left_right_, upper_lower_, transa_, diag_, m, n,
                           cast_to_void_if_complex(alpha), a + i * stride_a, lda, b + i * stride_b,
                           ldb);
            }
        });
    });
    return done;
}

#define TRSM_BATCH_STRIDED_USM_LAUNCHER(TYPE, ROUTINE)                                             \
    sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans, \
                           diag unit_diag, int64_t m, int64_t n, TYPE alpha, const TYPE* a,        \
                           int64_t lda, int64_t stride_a, TYPE* b, int64_t ldb, int64_t stride_b,  \
                           int64_t batch_size, const std::vector<sycl::event>& dependencies) {     \
        return trsm_batch(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,   \
                          stride_a, b, ldb, stride_b, batch_size, dependencies, ROUTINE);          \
    }

TRSM_BATCH_STRIDED_USM_LAUNCHER(float, ::cblas_strsm)
TRSM_BATCH_STRIDED_USM_LAUNCHER(double, ::cblas_dtrsm)
TRSM_BATCH_STRIDED_USM_LAUNCHER(std::complex<float>, ::cblas_ctrsm)
TRSM_BATCH_STRIDED_USM_LAUNCHER(std::complex<double>, ::cblas_ztrsm)

template <typename T, typename CBLAS_FUNC>
sycl::event trsm_batch(sycl::queue& queue, side* left_right, uplo* upper_lower, transpose* trans,
                       diag* unit_diag, int64_t* m, int64_t* n, T* alpha, const T** a, int64_t* lda,
                       T** b, int64_t* ldb, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (size_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_trsm_batch_group_usm>(cgh, [=]() {
            int64_t offset = 0;
            for (size_t i = 0; i < group_count; ++i) {
                for (size_t j = 0; j < group_size[i]; ++j) {
                    CBLAS_SIDE left_right_ = cblas_convert(left_right[i]);
                    CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower[i]);
                    CBLAS_TRANSPOSE transa_ = cblas_convert(trans[i]);
                    CBLAS_DIAG diag_ = cblas_convert(unit_diag[i]);
                    cblas_func(MAJOR, left_right_, upper_lower_, transa_, diag_, m[i], n[i],
                               cast_to_void_if_complex(alpha[i]), a[offset], lda[i], b[offset],
                               ldb[i]);
                    offset++;
                }
            }
        });
    });
    return done;
}

#define TRSM_BATCH_USM_LAUNCHER(TYPE, ROUTINE)                                                     \
    sycl::event trsm_batch(sycl::queue& queue, side* left_right, uplo* upper_lower,                \
                           transpose* trans, diag* unit_diag, int64_t* m, int64_t* n, TYPE* alpha, \
                           const TYPE** a, int64_t* lda, TYPE** b, int64_t* ldb,                   \
                           int64_t group_count, int64_t* group_size,                               \
                           const std::vector<sycl::event>& dependencies) {                         \
        return trsm_batch(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,   \
                          b, ldb, group_count, group_size, dependencies, ROUTINE);                 \
    }

TRSM_BATCH_USM_LAUNCHER(float, ::cblas_strsm)
TRSM_BATCH_USM_LAUNCHER(double, ::cblas_dtrsm)
TRSM_BATCH_USM_LAUNCHER(std::complex<float>, ::cblas_ctrsm)
TRSM_BATCH_USM_LAUNCHER(std::complex<double>, ::cblas_ztrsm)

template <typename T, typename CBLAS_FUNC>
sycl::event syrk_batch(sycl::queue& queue, uplo* upper_lower, transpose* trans, int64_t* n,
                       int64_t* k, T* alpha, const T** a, int64_t* lda, T* beta, T** c,
                       int64_t* ldc, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (size_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_syrk_batch_group_usm>(cgh, [=]() {
            int64_t offset = 0;
            for (size_t i = 0; i < group_count; ++i) {
                for (size_t j = 0; j < group_size[i]; ++j) {
                    CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower[i]);
                    CBLAS_TRANSPOSE transa_ = cblas_convert(trans[i]);
                    cblas_func(MAJOR, upper_lower_, transa_, n[i], k[i],
                               cast_to_void_if_complex(alpha[i]), a[offset], lda[i],
                               cast_to_void_if_complex(beta[i]), c[offset], ldc[i]);
                    offset++;
                }
            }
        });
    });
    return done;
}

#define SYRK_BATCH_USM_LAUNCHER(TYPE, ROUTINE)                                                  \
    sycl::event syrk_batch(sycl::queue& queue, uplo* upper_lower, transpose* trans, int64_t* n, \
                           int64_t* k, TYPE* alpha, const TYPE** a, int64_t* lda, TYPE* beta,   \
                           TYPE** c, int64_t* ldc, int64_t group_count, int64_t* group_size,    \
                           const std::vector<sycl::event>& dependencies) {                      \
        return syrk_batch(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc,         \
                          group_count, group_size, dependencies, ROUTINE);                      \
    }

SYRK_BATCH_USM_LAUNCHER(float, ::cblas_ssyrk)
SYRK_BATCH_USM_LAUNCHER(double, ::cblas_dsyrk)
SYRK_BATCH_USM_LAUNCHER(std::complex<float>, ::cblas_csyrk)
SYRK_BATCH_USM_LAUNCHER(std::complex<double>, ::cblas_zsyrk)

template <typename T, typename CBLAS_FUNC>
sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       T alpha, const T* a, int64_t lda, int64_t stride_a, T beta, T* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (size_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(trans);
        host_task<class armpl_kernel_syrk_batch>(cgh, [=]() {
            for (size_t i = 0; i < batch_size; ++i) {
                const T* a_array = a + i * stride_a;
                T* c_array = c + i * stride_c;
                cblas_func(MAJOR, upper_lower_, transa_, n, k, cast_to_void_if_complex(alpha),
                           a_array, lda, cast_to_void_if_complex(beta), c_array, ldc);
            }
        });
    });
    return done;
}

#define SYRK_BATCH_STRIDED_USM_LAUNCHER(TYPE, ROUTINE)                                            \
    sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n,      \
                           int64_t k, TYPE alpha, const TYPE* a, int64_t lda, int64_t stride_a,   \
                           TYPE beta, TYPE* c, int64_t ldc, int64_t stride_c, int64_t batch_size, \
                           const std::vector<sycl::event>& dependencies) {                        \
        return syrk_batch(queue, upper_lower, trans, n, k, alpha, a, lda, stride_a, beta, c, ldc, \
                          stride_c, batch_size, dependencies, ROUTINE);                           \
    }

SYRK_BATCH_STRIDED_USM_LAUNCHER(float, ::cblas_ssyrk)
SYRK_BATCH_STRIDED_USM_LAUNCHER(double, ::cblas_dsyrk)
SYRK_BATCH_STRIDED_USM_LAUNCHER(std::complex<float>, ::cblas_csyrk)
SYRK_BATCH_STRIDED_USM_LAUNCHER(std::complex<double>, ::cblas_zsyrk)

template <typename T, typename CBLAS_FUNC>
sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, T alpha,
                           const T* a, int64_t lda, int64_t stride_a, T* b, int64_t ldb,
                           int64_t stride_b, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    using ArmPLDataType = typename ArmPLEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (size_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        auto trans_ = fortran_char(trans);
#ifdef COLUMN_MAJOR
        char order = 'C';
#endif
#ifdef ROW_MAJOR
        check_armpl_version(
            24, 10, 0,
            "omatcopy row major is not functional"); // Row major omatcopy is broken (reported) in armpl <=24.04
        char order = 'R';
#endif
        host_task<class armpl_kernel_omatcopy_batch>(cgh, [=]() {
            for (size_t i = 0; i < batch_size; ++i) {
                const T* a_array = a + i * stride_a;
                T* b_array = b + i * stride_b;
                if constexpr (is_complex<T>) {
                    ArmPLDataType alpha_ = { alpha.real(), alpha.imag() };
                    cblas_func(order, trans_, m, n, alpha_,
                               reinterpret_cast<const ArmPLDataType*>(a_array), lda,
                               reinterpret_cast<ArmPLDataType*>(b_array), ldb);
                }
                else {
                    cblas_func(order, trans_, m, n, cast_to_void_if_complex(alpha), a_array, lda,
                               b_array, ldb);
                }
            }
        });
    });
    return done;
}

#define OMATCOPY_BATCH_STRIDED_USM_LAUNCHER(TYPE, ROUTINE)                                        \
    sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,         \
                               TYPE alpha, const TYPE* a, int64_t lda, int64_t stride_a, TYPE* b, \
                               int64_t ldb, int64_t stride_b, int64_t batch_size,                 \
                               const std::vector<sycl::event>& dependencies) {                    \
        return omatcopy_batch(queue, trans, m, n, alpha, a, lda, stride_a, b, ldb, stride_b,      \
                              batch_size, dependencies, ROUTINE);                                 \
    }

OMATCOPY_BATCH_STRIDED_USM_LAUNCHER(float, ::somatcopy)
OMATCOPY_BATCH_STRIDED_USM_LAUNCHER(double, ::domatcopy)
OMATCOPY_BATCH_STRIDED_USM_LAUNCHER(std::complex<float>, ::comatcopy)
OMATCOPY_BATCH_STRIDED_USM_LAUNCHER(std::complex<double>, ::zomatcopy)

template <typename T, typename CBLAS_FUNC>
sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, T alpha,
                           T* ab, int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    using ArmPLDataType = typename ArmPLEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (size_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        auto trans_ = fortran_char(trans);
#ifdef COLUMN_MAJOR
        char order = 'C';
#endif
#ifdef ROW_MAJOR
        check_armpl_version(
            24, 10, 0,
            "imatcopy row major is not functional"); // Row major imatcopy is broken (reported) in armpl <=24.04
        char order = 'R';
#endif
        host_task<class armpl_kernel_imatcopy_batch>(cgh, [=]() {
            for (size_t i = 0; i < batch_size; ++i) {
                T* ab_array = ab + i * stride;
                if constexpr (is_complex<T>) {
                    ArmPLDataType alpha_ = { alpha.real(), alpha.imag() };
                    cblas_func(order, trans_, m, n, alpha_,
                               reinterpret_cast<ArmPLDataType*>(ab_array), lda, ldb);
                }
                else {
                    cblas_func(order, trans_, m, n, cast_to_void_if_complex(alpha), ab_array, lda,
                               ldb);
                }
            }
        });
    });
    return done;
}

#define IMATCOPY_BATCH_STRIDED_USM_LAUNCHER(TYPE, ROUTINE)                                         \
    sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,          \
                               TYPE alpha, TYPE* ab, int64_t lda, int64_t ldb, int64_t stride,     \
                               int64_t batch_size, const std::vector<sycl::event>& dependencies) { \
        return imatcopy_batch(queue, trans, m, n, alpha, ab, lda, ldb, stride, batch_size,         \
                              dependencies, ROUTINE);                                              \
    }

IMATCOPY_BATCH_STRIDED_USM_LAUNCHER(float, ::simatcopy)
IMATCOPY_BATCH_STRIDED_USM_LAUNCHER(double, ::dimatcopy)
IMATCOPY_BATCH_STRIDED_USM_LAUNCHER(std::complex<float>, ::cimatcopy)
IMATCOPY_BATCH_STRIDED_USM_LAUNCHER(std::complex<double>, ::zimatcopy)

#ifdef COLUMN_MAJOR
#define OMATADD_BATCH_STRIDED_USM_LAUNCHER(TYPE)                                                   \
    sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m,   \
                              int64_t n, TYPE alpha, const TYPE* a, int64_t lda, int64_t stride_a, \
                              TYPE beta, const TYPE* b, int64_t ldb, int64_t stride_b, TYPE* c,    \
                              int64_t ldc, int64_t stride_c, int64_t batch_size,                   \
                              const std::vector<sycl::event>& dependencies) {                      \
        throw unimplemented("blas", "omatadd_batch", "for column_major layout for ArmPL");         \
    }
#endif
#ifdef ROW_MAJOR
#define OMATADD_BATCH_STRIDED_USM_LAUNCHER(TYPE)                                                   \
    sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m,   \
                              int64_t n, TYPE alpha, const TYPE* a, int64_t lda, int64_t stride_a, \
                              TYPE beta, const TYPE* b, int64_t ldb, int64_t stride_b, TYPE* c,    \
                              int64_t ldc, int64_t stride_c, int64_t batch_size,                   \
                              const std::vector<sycl::event>& dependencies) {                      \
        throw unimplemented("blas", "omatadd_batch", "for column_major layout for ArmPL");         \
    }
#endif

OMATADD_BATCH_STRIDED_USM_LAUNCHER(float)
OMATADD_BATCH_STRIDED_USM_LAUNCHER(double)
OMATADD_BATCH_STRIDED_USM_LAUNCHER(std::complex<float>)
OMATADD_BATCH_STRIDED_USM_LAUNCHER(std::complex<double>)

template <typename T, typename CBLAS_FUNC>
sycl::event omatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n, T* alpha,
                           const T** a, int64_t* lda, T** b, int64_t* ldb, int64_t group_count,
                           int64_t* group_size, const std::vector<sycl::event>& dependencies,
                           CBLAS_FUNC cblas_func) {
    using ArmPLDataType = typename ArmPLEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (size_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
#ifdef COLUMN_MAJOR
        char order = 'C';
#endif
#ifdef ROW_MAJOR
        check_armpl_version(
            24, 10, 0,
            "omatcopy row major is not functional"); // Row major omatcopy is broken (reported) in armpl <=24.04
        char order = 'R';
#endif
        host_task<class armpl_kernel_omatcopy_batch_group_usm>(cgh, [=]() {
            int64_t offset = 0;
            for (size_t i = 0; i < group_count; ++i) {
                for (size_t j = 0; j < group_size[i]; ++j) {
                    auto trans_ = fortran_char(trans[i]);
                    if constexpr (is_complex<T>) {
                        ArmPLDataType alpha_ = { alpha[i].real(), alpha[i].imag() };
                        cblas_func(order, trans_, m[i], n[i], alpha_,
                                   reinterpret_cast<const ArmPLDataType*>(a[offset]), lda[i],
                                   reinterpret_cast<ArmPLDataType*>(b[offset]), ldb[i]);
                    }
                    else {
                        cblas_func(order, trans_, m[i], n[i], cast_to_void_if_complex(alpha[i]),
                                   a[offset], lda[i], b[offset], ldb[i]);
                    }
                    offset++;
                }
            }
        });
    });
    return done;
}

#define OMATCOPY_BATCH_USM_LAUNCHER(TYPE, ROUTINE)                                                \
    sycl::event omatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,      \
                               TYPE* alpha, const TYPE** a, int64_t* lda, TYPE** b, int64_t* ldb, \
                               int64_t group_count, int64_t* group_size,                          \
                               const std::vector<sycl::event>& dependencies) {                    \
        return omatcopy_batch(queue, trans, m, n, alpha, a, lda, b, ldb, group_count, group_size, \
                              dependencies, ROUTINE);                                             \
    }

OMATCOPY_BATCH_USM_LAUNCHER(float, ::somatcopy)
OMATCOPY_BATCH_USM_LAUNCHER(double, ::domatcopy)
OMATCOPY_BATCH_USM_LAUNCHER(std::complex<float>, ::comatcopy)
OMATCOPY_BATCH_USM_LAUNCHER(std::complex<double>, ::zomatcopy)

template <typename T, typename CBLAS_FUNC>
sycl::event imatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n, T* alpha,
                           T** ab, int64_t* lda, int64_t* ldb, int64_t group_count,
                           int64_t* group_size, const std::vector<sycl::event>& dependencies,
                           CBLAS_FUNC cblas_func) {
    using ArmPLDataType = typename ArmPLEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (size_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
#ifdef COLUMN_MAJOR
        char order = 'C';
#endif
#ifdef ROW_MAJOR
        check_armpl_version(
            24, 10, 0,
            "imatcopy row major is not functional"); // Row major imatcopy is broken (reported) in armpl <=24.04
        char order = 'R';
#endif
        host_task<class armpl_kernel_omatcopy_batch_group_usm>(cgh, [=]() {
            int64_t offset = 0;
            for (size_t i = 0; i < group_count; ++i) {
                for (size_t j = 0; j < group_size[i]; ++j) {
                    auto trans_ = fortran_char(trans[i]);
                    if constexpr (is_complex<T>) {
                        ArmPLDataType alpha_ = { alpha[i].real(), alpha[i].imag() };
                        cblas_func(order, trans_, m[i], n[i], alpha_,
                                   reinterpret_cast<ArmPLDataType*>(ab[offset]), lda[i], ldb[i]);
                    }
                    else {
                        cblas_func(order, trans_, m[i], n[i], cast_to_void_if_complex(alpha[i]),
                                   ab[offset], lda[i], ldb[i]);
                    }
                    offset++;
                }
            }
        });
    });
    return done;
}

#define IMATCOPY_BATCH_USM_LAUNCHER(TYPE, ROUTINE)                                              \
    sycl::event imatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,    \
                               TYPE* alpha, TYPE** ab, int64_t* lda, int64_t* ldb,              \
                               int64_t group_count, int64_t* group_size,                        \
                               const std::vector<sycl::event>& dependencies) {                  \
        return imatcopy_batch(queue, trans, m, n, alpha, ab, lda, ldb, group_count, group_size, \
                              dependencies, ROUTINE);                                           \
    }

IMATCOPY_BATCH_USM_LAUNCHER(float, ::simatcopy)
IMATCOPY_BATCH_USM_LAUNCHER(double, ::dimatcopy)
IMATCOPY_BATCH_USM_LAUNCHER(std::complex<float>, ::cimatcopy)
IMATCOPY_BATCH_USM_LAUNCHER(std::complex<double>, ::zimatcopy)
