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

template <typename Ta, typename Tb>
void gemm_bias_fallback(sycl::queue& queue, CBLAS_LAYOUT layout, transpose transa, transpose transb,
                        offset offsetc, int64_t m, int64_t n, int64_t k, float alpha,
                        sycl::buffer<Ta, 1>& a, int64_t lda, Ta ao, sycl::buffer<Tb, 1>& b,
                        int64_t ldb, Tb bo, float beta, sycl::buffer<int32_t, 1>& c, int64_t ldc,
                        sycl::buffer<int32_t, 1>& co) {
    queue.submit([&](sycl::handler& cgh) {
        auto transa_ = cblas_convert(transa);
        auto transb_ = cblas_convert(transb);
        auto offsetc_ = cblas_convert(offsetc);
        int64_t sizea, sizeb, sizec;
#ifdef COLUMN_MAJOR
        sizea = (transa == transpose::nontrans) ? lda * k : lda * m;
        sizeb = (transb == transpose::nontrans) ? ldb * n : ldb * k;
        sizec = ldc * n;
#endif
#ifdef ROW_MAJOR
        sizea = (transa == transpose::nontrans) ? lda * m : lda * k;
        sizeb = (transb == transpose::nontrans) ? ldb * k : ldb * n;
        sizec = ldc * m;
#endif
        auto accessor_a = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_b = b.template get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_c = c.template get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_co = co.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_gemm_bias_fallback>(cgh, [=]() {
            auto a_mat = (Ta*)static_cast<void*>(accessor_a.GET_MULTI_PTR);
            auto b_mat = (Tb*)static_cast<void*>(accessor_b.GET_MULTI_PTR);
            auto c_mat = (int32_t*)static_cast<void*>(accessor_c.GET_MULTI_PTR);
            auto co_vec = (int32_t*)static_cast<void*>(accessor_co.GET_MULTI_PTR);
            auto ad = new double[sizea]();
            auto bd = new double[sizeb]();
            auto cd = new double[sizec]();
            double alphad = alpha, betad = beta;
            double aod = ao, bod = bo;
            copy_mat(a_mat, layout, transa, m, k, lda, aod, ad);
            copy_mat(b_mat, layout, transb, k, n, ldb, bod, bd);
            copy_mat(c_mat, layout, transpose::nontrans, m, n, ldc, 0.0, cd);
            ::cblas_dgemm(MAJOR, transa_, transb_, m, n, k, alphad, ad, lda, bd, ldb, betad, cd,
                          ldc);
            copy_mat(cd, layout, m, n, ldc, offsetc_, co_vec, c_mat);
            delete[] ad;
            delete[] bd;
            delete[] cd;
        });
    });
}

#define GEMM_BIAS_LAUNCHER(TYPEA, TYPEB)                                                         \
    void gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc,       \
                   int64_t m, int64_t n, int64_t k, float alpha, sycl::buffer<TYPEA, 1>& a,      \
                   int64_t lda, TYPEA ao, sycl::buffer<TYPEB, 1>& b, int64_t ldb, TYPEB bo,      \
                   float beta, sycl::buffer<int32_t, 1>& c, int64_t ldc,                         \
                   sycl::buffer<int32_t, 1>& co) {                                               \
        gemm_bias_fallback(queue, MAJOR, transa, transb, offsetc, m, n, k, alpha, a, lda, ao, b, \
                           ldb, bo, beta, c, ldc, co);                                           \
    }

GEMM_BIAS_LAUNCHER(int8_t, int8_t)
GEMM_BIAS_LAUNCHER(int8_t, uint8_t)
GEMM_BIAS_LAUNCHER(uint8_t, int8_t)
GEMM_BIAS_LAUNCHER(uint8_t, uint8_t)

template <typename T, typename CBLAS_FUNC>
void gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, T alpha, sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& b,
           int64_t ldb, T beta, sycl::buffer<T, 1>& c, int64_t ldc, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        auto accessor_a = a.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_b = b.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_c = c.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_gemmt>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, transa_, transb_, n, k, cast_to_void_if_complex(alpha),
                       accessor_a.GET_MULTI_PTR, lda, accessor_b.GET_MULTI_PTR, ldb,
                       cast_to_void_if_complex(beta), accessor_c.GET_MULTI_PTR, ldc);
        });
    });
}

#define GEMMT_LAUNCHER(TYPE, ROUTINE)                                                        \
    void gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb,     \
               int64_t n, int64_t k, TYPE alpha, sycl::buffer<TYPE, 1>& a, int64_t lda,      \
               sycl::buffer<TYPE, 1>& b, int64_t ldb, TYPE beta, sycl::buffer<TYPE, 1>& c,   \
               int64_t ldc) {                                                                \
        gemmt(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc, \
              ROUTINE);                                                                      \
    }

GEMMT_LAUNCHER(float, ::cblas_sgemmt)
GEMMT_LAUNCHER(double, ::cblas_dgemmt)
GEMMT_LAUNCHER(std::complex<float>, ::cblas_cgemmt)
GEMMT_LAUNCHER(std::complex<double>, ::cblas_zgemmt)

template <typename T, typename CBLAS_FUNC>
void omatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, T alpha,
              sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& b, int64_t ldb,
              CBLAS_FUNC cblas_func) {
    using ArmPLDataType = typename ArmPLEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto trans_ = fortran_char(trans);
#ifdef COLUMN_MAJOR
        char order = 'C';
#endif
#ifdef ROW_MAJOR
        check_armpl_version(
            24, 10, 0,
            "omatcopy row major is not fucntional"); // Row major omatcopy is broken (reported) in armpl <=24.04
        char order = 'R';
#endif
        auto accessor_a = a.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_b = b.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_omatcopy>(cgh, [=]() {
            if constexpr (is_complex<T>) {
                ArmPLDataType alpha_ = { alpha.real(), alpha.imag() };
                cblas_func(order, trans_, m, n, alpha_,
                           reinterpret_cast<const ArmPLDataType*>(accessor_a.GET_MULTI_PTR), lda,
                           reinterpret_cast<ArmPLDataType*>(accessor_b.GET_MULTI_PTR), ldb);
            }
            else {
                cblas_func(order, trans_, m, n, alpha, accessor_a.GET_MULTI_PTR, lda,
                           accessor_b.GET_MULTI_PTR, ldb);
            }
        });
    });
}

#define OMATCOPY_LAUNCHER(TYPE, ROUTINE)                                                          \
    void omatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, TYPE alpha,          \
                  sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& b, int64_t ldb) { \
        omatcopy(queue, trans, m, n, alpha, a, lda, b, ldb, ROUTINE);                             \
    }

OMATCOPY_LAUNCHER(float, ::somatcopy)
OMATCOPY_LAUNCHER(double, ::domatcopy)
OMATCOPY_LAUNCHER(std::complex<float>, ::comatcopy)
OMATCOPY_LAUNCHER(std::complex<double>, ::zomatcopy)

#define OMATCOPY2_LAUNCHER(TYPE)                                                          \
    void omatcopy2(sycl::queue& queue, transpose trans, int64_t m, int64_t n, TYPE alpha, \
                   sycl::buffer<TYPE, 1>& a, int64_t lda, std::int64_t stridea,           \
                   sycl::buffer<TYPE, 1>& b, int64_t ldb, std::int64_t strideb) {         \
        throw unimplemented("blas", "omatcopy2", MAJOR_MINOR_TEXT);                       \
    }

OMATCOPY2_LAUNCHER(float)
OMATCOPY2_LAUNCHER(double)
OMATCOPY2_LAUNCHER(std::complex<float>)
OMATCOPY2_LAUNCHER(std::complex<double>)

template <typename T, typename CBLAS_FUNC>
void imatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, T alpha,
              sycl::buffer<T, 1>& ab, int64_t lda, int64_t ldb, CBLAS_FUNC cblas_func) {
    using ArmPLDataType = typename ArmPLEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
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
        auto accessor_ab = ab.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_imatcopy>(cgh, [=]() {
            if constexpr (is_complex<T>) {
                ArmPLDataType alpha_ = { alpha.real(), alpha.imag() };
                cblas_func(order, trans_, m, n, alpha_,
                           reinterpret_cast<ArmPLDataType*>(accessor_ab.GET_MULTI_PTR), lda, ldb);
            }
            else {
                cblas_func(order, trans_, m, n, alpha, accessor_ab.GET_MULTI_PTR, lda, ldb);
            }
        });
    });
}

#define IMATCOPY_LAUNCHER(TYPE, ROUTINE)                                                 \
    void imatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, TYPE alpha, \
                  sycl::buffer<TYPE, 1>& ab, int64_t lda, int64_t ldb) {                 \
        imatcopy(queue, trans, m, n, alpha, ab, lda, ldb, ROUTINE);                      \
    }

IMATCOPY_LAUNCHER(float, ::simatcopy)
IMATCOPY_LAUNCHER(double, ::dimatcopy)
IMATCOPY_LAUNCHER(std::complex<float>, ::cimatcopy)
IMATCOPY_LAUNCHER(std::complex<double>, ::zimatcopy)

#define OMATADD_LAUNCHER(TYPE)                                                                   \
    void omatadd(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,   \
                 TYPE alpha, sycl::buffer<TYPE, 1>& a, int64_t lda, TYPE beta,                   \
                 sycl::buffer<TYPE, 1>& b, int64_t ldb, sycl::buffer<TYPE, 1>& c, int64_t ldc) { \
        throw unimplemented("blas", "omatadd", MAJOR_MINOR_TEXT);                                \
    }

OMATADD_LAUNCHER(float)
OMATADD_LAUNCHER(double)
OMATADD_LAUNCHER(std::complex<float>)
OMATADD_LAUNCHER(std::complex<double>)

// USM APIs

template <typename Ta, typename Tb>
sycl::event gemm_bias_fallback(sycl::queue& queue, CBLAS_LAYOUT layout, transpose transa,
                               transpose transb, offset offsetc, int64_t m, int64_t n, int64_t k,
                               float alpha, const Ta* a, int64_t lda, Ta ao, const Tb* b,
                               int64_t ldb, Tb bo, float beta, int32_t* c, int64_t ldc,
                               const int32_t* co, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        auto transa_ = cblas_convert(transa);
        auto transb_ = cblas_convert(transb);
        auto offsetc_ = cblas_convert(offsetc);
        int64_t sizea, sizeb, sizec;
#ifdef COLUMN_MAJOR
        sizea = (transa == transpose::nontrans) ? lda * k : lda * m;
        sizeb = (transb == transpose::nontrans) ? ldb * n : ldb * k;
        sizec = ldc * n;
#endif
#ifdef ROW_MAJOR
        sizea = (transa == transpose::nontrans) ? lda * m : lda * k;
        sizeb = (transb == transpose::nontrans) ? ldb * k : ldb * n;
        sizec = ldc * m;
#endif
        host_task<class armpl_kernel_gemm_bias_fallback_usm>(cgh, [=]() {
            double* ad = new double[sizea]();
            double* bd = new double[sizeb]();
            double* cd = new double[sizec]();
            double alphad = alpha, betad = beta;
            double aod = ao, bod = bo;
            copy_mat(a, layout, transa, m, k, lda, aod, ad);
            copy_mat(b, layout, transb, k, n, ldb, bod, bd);
            copy_mat(c, layout, transpose::nontrans, m, n, ldc, 0.0, cd);
            ::cblas_dgemm(MAJOR, transa_, transb_, m, n, k, alphad, ad, lda, bd, ldb, betad, cd,
                          ldc);
            copy_mat(cd, layout, m, n, ldc, offsetc_, co, c);
            delete[] ad;
            delete[] bd;
            delete[] cd;
        });
    });
    return done;
}

#define GEMM_BIAS_USM_LAUNCHER(TYPEA, TYPEB)                                                      \
    sycl::event gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc, \
                          int64_t m, int64_t n, int64_t k, float alpha, const TYPEA* a,           \
                          int64_t lda, TYPEA ao, const TYPEB* b, int64_t ldb, TYPEB bo,           \
                          float beta, int32_t* c, int64_t ldc, const int32_t* co,                 \
                          const std::vector<sycl::event>& dependencies) {                         \
        return gemm_bias_fallback(queue, MAJOR, transa, transb, offsetc, m, n, k, alpha, a, lda,  \
                                  ao, b, ldb, bo, beta, c, ldc, co, dependencies);                \
    }

GEMM_BIAS_USM_LAUNCHER(int8_t, int8_t)
GEMM_BIAS_USM_LAUNCHER(int8_t, uint8_t)
GEMM_BIAS_USM_LAUNCHER(uint8_t, int8_t)
GEMM_BIAS_USM_LAUNCHER(uint8_t, uint8_t)

template <typename T, typename CBLAS_FUNC>
sycl::event gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb,
                  int64_t n, int64_t k, T alpha, const T* a, int64_t lda, const T* b, int64_t ldb,
                  T beta, T* c, int64_t ldc, const std::vector<sycl::event>& dependencies,
                  CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        host_task<class armpl_kernel_sgemmt_usm>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, transa_, transb_, n, k, cast_to_void_if_complex(alpha),
                       a, lda, b, ldb, cast_to_void_if_complex(beta), c, ldc);
        });
    });
    return done;
}

#define GEMMT_USM_LAUNCHER(TYPE, ROUTINE)                                                          \
    sycl::event gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb,    \
                      int64_t n, int64_t k, TYPE alpha, const TYPE* a, int64_t lda, const TYPE* b, \
                      int64_t ldb, TYPE beta, TYPE* c, int64_t ldc,                                \
                      const std::vector<sycl::event>& dependencies) {                              \
        return gemmt(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,     \
                     ldc, dependencies, ROUTINE);                                                  \
    }

GEMMT_USM_LAUNCHER(float, ::cblas_sgemmt)
GEMMT_USM_LAUNCHER(double, ::cblas_dgemmt)
GEMMT_USM_LAUNCHER(std::complex<float>, ::cblas_cgemmt)
GEMMT_USM_LAUNCHER(std::complex<double>, ::cblas_zgemmt)

template <typename T, typename CBLAS_FUNC>
sycl::event omatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, T alpha, const T* a,
                     int64_t lda, T* b, int64_t ldb, const std::vector<sycl::event>& dependencies,
                     CBLAS_FUNC cblas_func) {
    using ArmPLDataType = typename ArmPLEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
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
        host_task<class armpl_kernel_omatcopy>(cgh, [=]() {
            if constexpr (is_complex<T>) {
                ArmPLDataType alpha_ = { alpha.real(), alpha.imag() };
                cblas_func(order, trans_, m, n, alpha_, reinterpret_cast<const ArmPLDataType*>(a),
                           lda, reinterpret_cast<ArmPLDataType*>(b), ldb);
            }
            else {
                cblas_func(order, trans_, m, n, alpha, a, lda, b, ldb);
            }
        });
    });
    return done;
}

#define OMATCOPY_USM_LAUNCHER(TYPE, ROUTINE)                                                    \
    sycl::event omatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, TYPE alpha, \
                         const TYPE* a, int64_t lda, TYPE* b, int64_t ldb,                      \
                         const std::vector<sycl::event>& dependencies) {                        \
        return omatcopy(queue, trans, m, n, alpha, a, lda, b, ldb, dependencies, ROUTINE);      \
    }

OMATCOPY_USM_LAUNCHER(float, ::somatcopy)
OMATCOPY_USM_LAUNCHER(double, ::domatcopy)
OMATCOPY_USM_LAUNCHER(std::complex<float>, ::comatcopy)
OMATCOPY_USM_LAUNCHER(std::complex<double>, ::zomatcopy)

#define OMATCOPY2_USM_LAUNCHER(TYPE)                                                              \
    sycl::event omatcopy2(sycl::queue& queue, transpose trans, int64_t m, int64_t n, TYPE alpha,  \
                          const TYPE* a, int64_t lda, std::int64_t stridea, TYPE* b, int64_t ldb, \
                          std::int64_t strideb, const std::vector<sycl::event>& dependencies) {   \
        throw unimplemented("blas", "omatcopy2", MAJOR_MINOR_TEXT);                               \
    }

OMATCOPY2_USM_LAUNCHER(float)
OMATCOPY2_USM_LAUNCHER(double)
OMATCOPY2_USM_LAUNCHER(std::complex<float>)
OMATCOPY2_USM_LAUNCHER(std::complex<double>)

template <typename T, typename CBLAS_FUNC>
sycl::event imatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, T alpha, T* ab,
                     int64_t lda, int64_t ldb, const std::vector<sycl::event>& dependencies,
                     CBLAS_FUNC cblas_func) {
    using ArmPLDataType = typename ArmPLEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
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
        host_task<class armpl_kernel_imatcopy>(cgh, [=]() {
            if constexpr (is_complex<T>) {
                ArmPLDataType alpha_ = { alpha.real(), alpha.imag() };
                cblas_func(order, trans_, m, n, alpha_, reinterpret_cast<ArmPLDataType*>(ab), lda,
                           ldb);
            }
            else {
                cblas_func(order, trans_, m, n, alpha, ab, lda, ldb);
            }
        });
    });
    return done;
}

#define IMATCOPY_USM_LAUNCHER(TYPE, ROUTINE)                                                    \
    sycl::event imatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, TYPE alpha, \
                         TYPE* ab, int64_t lda, int64_t ldb,                                    \
                         const std::vector<sycl::event>& dependencies) {                        \
        return imatcopy(queue, trans, m, n, alpha, ab, lda, ldb, dependencies, ROUTINE);        \
    }

IMATCOPY_USM_LAUNCHER(float, ::simatcopy)
IMATCOPY_USM_LAUNCHER(double, ::dimatcopy)
IMATCOPY_USM_LAUNCHER(std::complex<float>, ::cimatcopy)
IMATCOPY_USM_LAUNCHER(std::complex<double>, ::zimatcopy)

#define OMATADD_USM_LAUNCHER(TYPE)                                                         \
    sycl::event omatadd(sycl::queue& queue, transpose transa, transpose transb, int64_t m, \
                        int64_t n, TYPE alpha, const TYPE* a, int64_t lda, TYPE beta,      \
                        const TYPE* b, int64_t ldb, TYPE* c, int64_t ldc,                  \
                        const std::vector<sycl::event>& dependencies) {                    \
        throw unimplemented("blas", "omatadd", MAJOR_MINOR_TEXT);                          \
    }

OMATADD_USM_LAUNCHER(float)
OMATADD_USM_LAUNCHER(double)
OMATADD_USM_LAUNCHER(std::complex<float>)
OMATADD_USM_LAUNCHER(std::complex<double>)
