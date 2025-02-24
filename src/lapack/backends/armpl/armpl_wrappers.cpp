/***************************************************************************
*  Copyright 2025 SiPearl
*
*  Adapted from cusolver backend.
*
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

// BUFFER APIs

template <typename Func, typename T_A, typename T_B>
inline void gebrd(Func func, const char* func_name, sycl::queue& queue, std::int64_t m,
                  std::int64_t n, sycl::buffer<T_A>& a, std::int64_t lda, sycl::buffer<T_B>& d,
                  sycl::buffer<T_B>& e, sycl::buffer<T_A>& tauq, sycl::buffer<T_A>& taup,
                  sycl::buffer<T_A>& s, std::int64_t scratchpad_size) {
    using ArmDataType = typename ArmEquivalentType<T_A>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto d_acc = d.template get_access<sycl::access::mode::write>(cgh);
        auto e_acc = e.template get_access<sycl::access::mode::write>(cgh);
        auto s_acc = s.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto tauq_acc =
            tauq.template reinterpret<ArmDataType>().template get_access<sycl::access::mode::write>(
                cgh);
        auto taup_acc =
            taup.template reinterpret<ArmDataType>().template get_access<sycl::access::mode::write>(
                cgh);
        host_task<class armpl_kernel_gebrd>(cgh, [=]() {
            std::int64_t err =
                func(LAPACK_COL_MAJOR, m, n, a_acc.GET_MULTI_PTR, lda, d_acc.GET_MULTI_PTR,
                     e_acc.GET_MULTI_PTR, tauq_acc.GET_MULTI_PTR, taup_acc.GET_MULTI_PTR,
                     s_acc.GET_MULTI_PTR, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define GEBRD_LAUNCHER(TYPE_A, TYPE_B, ROUTINE)                                             \
    void gebrd(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<TYPE_A>& a, \
               std::int64_t lda, sycl::buffer<TYPE_B>& d, sycl::buffer<TYPE_B>& e,          \
               sycl::buffer<TYPE_A>& tauq, sycl::buffer<TYPE_A>& taup,                      \
               sycl::buffer<TYPE_A>& scratchpad, std::int64_t scratchpad_size) {            \
        gebrd(ROUTINE, #ROUTINE, queue, m, n, a, lda, d, e, tauq, taup, scratchpad,         \
              scratchpad_size);                                                             \
    }

GEBRD_LAUNCHER(float, float, LAPACKE_sgebrd_work)
GEBRD_LAUNCHER(double, double, LAPACKE_dgebrd_work)
GEBRD_LAUNCHER(std::complex<float>, float, LAPACKE_cgebrd_work)
GEBRD_LAUNCHER(std::complex<double>, double, LAPACKE_zgebrd_work)

#undef GEBRD_LAUNCHER

template <typename Func, typename T>
inline void gerqf(Func func, const char* func_name, sycl::queue& queue, std::int64_t m,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& tau,
                  sycl::buffer<T>& s, std::int64_t scratchpad_size) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc =
            tau.template reinterpret<ArmDataType>().template get_access<sycl::access::mode::write>(
                cgh);
        auto s_acc = s.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_gerqf>(cgh, [=]() {
            std::int64_t err = func(LAPACK_COL_MAJOR, m, n, a_acc.GET_MULTI_PTR, lda,
                                    tau_acc.GET_MULTI_PTR, s_acc.GET_MULTI_PTR, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define GERQF_LAUNCHER(TYPE, ROUTINE)                                                     \
    void gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<TYPE>& a, \
               std::int64_t lda, sycl::buffer<TYPE>& tau, sycl::buffer<TYPE>& scratchpad, \
               std::int64_t scratchpad_size) {                                            \
        gerqf(ROUTINE, #ROUTINE, queue, m, n, a, lda, tau, scratchpad, scratchpad_size);  \
    }

GERQF_LAUNCHER(float, LAPACKE_sgerqf_work)
GERQF_LAUNCHER(double, LAPACKE_dgerqf_work)
GERQF_LAUNCHER(std::complex<float>, LAPACKE_cgerqf_work)
GERQF_LAUNCHER(std::complex<double>, LAPACKE_zgerqf_work)

#undef GERQF_LAUNCHER

template <typename Func, typename T>
inline void geqrf(Func func, const char* func_name, sycl::queue& queue, std::int64_t m,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& tau,
                  sycl::buffer<T>& s, std::int64_t scratchpad_size) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc =
            tau.template reinterpret<ArmDataType>().template get_access<sycl::access::mode::write>(
                cgh);
        auto s_acc = s.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);

        host_task<class armpl_kernel_geqrf>(cgh, [=]() {
            std::int64_t err = func(LAPACK_COL_MAJOR, m, n, a_acc.GET_MULTI_PTR, lda,
                                    tau_acc.GET_MULTI_PTR, s_acc.GET_MULTI_PTR, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define GEQRF_LAUNCHER(TYPE, ROUTINE)                                                     \
    void geqrf(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<TYPE>& a, \
               std::int64_t lda, sycl::buffer<TYPE>& tau, sycl::buffer<TYPE>& scratchpad, \
               std::int64_t scratchpad_size) {                                            \
        geqrf(ROUTINE, #ROUTINE, queue, m, n, a, lda, tau, scratchpad, scratchpad_size);  \
    }

GEQRF_LAUNCHER(float, LAPACKE_sgeqrf_work)
GEQRF_LAUNCHER(double, LAPACKE_dgeqrf_work)
GEQRF_LAUNCHER(std::complex<float>, LAPACKE_cgeqrf_work)
GEQRF_LAUNCHER(std::complex<double>, LAPACKE_zgeqrf_work)

#undef GEQRF_LAUNCHER

template <typename Func, typename T>
void getrf(Func func, const char* func_name, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto ipiv_acc = ipiv.template get_access<sycl::access::mode::write>(cgh);
        host_task<class armpl_kernel_getrf>(cgh, [=]() {
            std::int64_t err =
                func(LAPACK_COL_MAJOR, m, n, a_acc.GET_MULTI_PTR, lda, ipiv_acc.GET_MULTI_PTR);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });

#define GETRF_LAUNCHER(TYPE, ROUTINE)                                                              \
    void getrf(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<TYPE>& a,          \
               std::int64_t lda, sycl::buffer<std::int64_t>& ipiv, sycl::buffer<TYPE>& scratchpad, \
               std::int64_t scratchpad_size) {                                                     \
        getrf(ROUTINE, #ROUTINE, queue, m, n, a, lda, ipiv);                                       \
    }
}

GETRF_LAUNCHER(float, LAPACKE_sgetrf_work)
GETRF_LAUNCHER(double, LAPACKE_dgetrf_work)
GETRF_LAUNCHER(std::complex<float>, LAPACKE_cgetrf_work)
GETRF_LAUNCHER(std::complex<double>, LAPACKE_zgetrf_work)

#undef GETRF_LAUNCHER

template <typename Func, typename T>
void getri(Func func, const char* func_name, sycl::queue& queue, std::int64_t n, sycl::buffer<T>& a,
           std::int64_t lda, sycl::buffer<std::int64_t>& ipiv, sycl::buffer<T>& s,
           std::int64_t scratchpad_size) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto ipiv_acc = ipiv.template get_access<sycl::access::mode::write>(cgh);
        auto s_acc = s.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);

        host_task<class armpl_kernel_getri>(cgh, [=]() {
            std::int64_t err = func(LAPACK_COL_MAJOR, n, a_acc.GET_MULTI_PTR, lda,
                                    ipiv_acc.GET_MULTI_PTR, s_acc.GET_MULTI_PTR, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });

#define GETRI_LAUNCHER(TYPE, ROUTINE)                                                       \
    void getri(sycl::queue& queue, std::int64_t n, sycl::buffer<TYPE>& a, std::int64_t lda, \
               sycl::buffer<std::int64_t>& ipiv, sycl::buffer<TYPE>& scratchpad,            \
               std::int64_t scratchpad_size) {                                              \
        getri(ROUTINE, #ROUTINE, queue, n, a, lda, ipiv, scratchpad, scratchpad_size);      \
    }
}

GETRI_LAUNCHER(float, LAPACKE_sgetri_work)
GETRI_LAUNCHER(double, LAPACKE_dgetri_work)
GETRI_LAUNCHER(std::complex<float>, LAPACKE_cgetri_work)
GETRI_LAUNCHER(std::complex<double>, LAPACKE_zgetri_work)

template <typename Func, typename T>
inline void getrs(Func func, const char* func_name, sycl::queue& queue,
                  oneapi::math::transpose trans, std::int64_t n, std::int64_t nrhs,
                  sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
                  sycl::buffer<T>& b, std::int64_t ldb) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc =
            a.template reinterpret<ArmDataType>().template get_access<sycl::access::mode::read>(
                cgh);
        auto ipiv_acc = ipiv.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc =
            b.template reinterpret<ArmDataType>().template get_access<sycl::access::mode::write>(
                cgh);

        host_task<class armpl_kernel_getrs>(cgh, [=]() {
            std::int64_t err =
                func(LAPACK_COL_MAJOR, get_operation(trans), n, nrhs, a_acc.GET_MULTI_PTR, lda,
                     ipiv_acc.GET_MULTI_PTR, b_acc.GET_MULTI_PTR, ldb);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define GETRS_LAUNCHER(TYPE, ROUTINE)                                                     \
    void getrs(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,         \
               std::int64_t nrhs, sycl::buffer<TYPE>& a, std::int64_t lda,                \
               sycl::buffer<std::int64_t>& ipiv, sycl::buffer<TYPE>& b, std::int64_t ldb, \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {            \
        getrs(ROUTINE, #ROUTINE, queue, trans, n, nrhs, a, lda, ipiv, b, ldb);            \
    }

GETRS_LAUNCHER(float, LAPACKE_sgetrs_work)
GETRS_LAUNCHER(double, LAPACKE_dgetrs_work)
GETRS_LAUNCHER(std::complex<float>, LAPACKE_cgetrs_work)
GETRS_LAUNCHER(std::complex<double>, LAPACKE_zgetrs_work)

#undef GETRS_LAUNCHER

template <typename Func, typename T_A, typename T_B>
inline void gesvd(Func func, const char* func_name, sycl::queue& queue, oneapi::math::jobsvd jobu,
                  oneapi::math::jobsvd jobvt, std::int64_t m, std::int64_t n, sycl::buffer<T_A>& a,
                  std::int64_t lda, sycl::buffer<T_B>& s, sycl::buffer<T_A>& u, std::int64_t ldu,
                  sycl::buffer<T_A>& vt, std::int64_t ldvt, sycl::buffer<T_A>& scratchpad,
                  std::int64_t scratchpad_size) {
    using ArmDataType = typename ArmEquivalentType<T_A>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto s_acc = s.template get_access<sycl::access::mode::write>(cgh);
        auto u_acc =
            u.template reinterpret<ArmDataType>().template get_access<sycl::access::mode::write>(
                cgh);
        auto vt_acc =
            vt.template reinterpret<ArmDataType>().template get_access<sycl::access::mode::write>(
                cgh);
        auto scratch_acc = scratchpad.template reinterpret<ArmDataType>()
                               .template get_access<sycl::access::mode::read_write>(cgh);

        host_task<class armpl_kernel_gesvd>(cgh, [=]() {
            std::int64_t err = 0;
            if constexpr (is_complex<T_A>) {
                T_B* rwork = new T_B[5 * std::min(m, n)];
                err = func(LAPACK_COL_MAJOR, get_jobsvd(jobu), get_jobsvd(jobvt), m, n,
                           a_acc.GET_MULTI_PTR, lda, s_acc.GET_MULTI_PTR, u_acc.GET_MULTI_PTR, ldu,
                           vt_acc.GET_MULTI_PTR, ldvt, scratch_acc.GET_MULTI_PTR, scratchpad_size,
                           rwork);
                delete[] rwork;
            }
            else {
                err = func(LAPACK_COL_MAJOR, get_jobsvd(jobu), get_jobsvd(jobvt), m, n,
                           a_acc.GET_MULTI_PTR, lda, s_acc.GET_MULTI_PTR, u_acc.GET_MULTI_PTR, ldu,
                           vt_acc.GET_MULTI_PTR, ldvt, scratch_acc.GET_MULTI_PTR, scratchpad_size);
            }
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define GESVD_LAUNCHER(TYPE_A, TYPE_B, ROUTINE)                                               \
    void gesvd(sycl::queue& queue, oneapi::math::jobsvd jobu, oneapi::math::jobsvd jobvt,     \
               std::int64_t m, std::int64_t n, sycl::buffer<TYPE_A>& a, std::int64_t lda,     \
               sycl::buffer<TYPE_B>& s, sycl::buffer<TYPE_A>& u, std::int64_t ldu,            \
               sycl::buffer<TYPE_A>& vt, std::int64_t ldvt, sycl::buffer<TYPE_A>& scratchpad, \
               std::int64_t scratchpad_size) {                                                \
        gesvd(ROUTINE, #ROUTINE, queue, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,       \
              scratchpad, scratchpad_size);                                                   \
    }

GESVD_LAUNCHER(float, float, LAPACKE_sgesvd_work)
GESVD_LAUNCHER(double, double, LAPACKE_dgesvd_work)
GESVD_LAUNCHER(std::complex<float>, float, LAPACKE_cgesvd_work)
GESVD_LAUNCHER(std::complex<double>, double, LAPACKE_zgesvd_work)

#undef GESVD_LAUNCHER
template <typename Func, typename T_A, typename T_B>
inline void heevd(Func func, const char* func_name, sycl::queue& queue, oneapi::math::job jobz,
                  oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<T_A>& a, std::int64_t lda,
                  sycl::buffer<T_B>& w) {
    using ArmDataType = typename ArmEquivalentType<T_A>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto w_acc = w.template get_access<sycl::access::mode::write>(cgh);
        host_task<class armpl_kernel_heevd>(cgh, [=]() {
            std::int64_t err = func(LAPACK_COL_MAJOR, get_job(jobz), get_fill_mode(uplo), n,
                                    a_acc.GET_MULTI_PTR, lda, w_acc.GET_MULTI_PTR);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define HEEVD_LAUNCHER(TYPE_A, TYPE_B, ROUTINE)                                                    \
    void heevd(sycl::queue& queue, oneapi::math::job jobz, oneapi::math::uplo uplo,                \
               std::int64_t n, sycl::buffer<TYPE_A>& a, std::int64_t lda, sycl::buffer<TYPE_B>& w, \
               sycl::buffer<TYPE_A>& scratchpad, std::int64_t scratchpad_size) {                   \
        heevd(ROUTINE, #ROUTINE, queue, jobz, uplo, n, a, lda, w);                                 \
    }

HEEVD_LAUNCHER(std::complex<float>, float, LAPACKE_cheevd)
HEEVD_LAUNCHER(std::complex<double>, double, LAPACKE_zheevd)

#undef HEEVD_LAUNCHER

template <typename Func, typename T_A, typename T_B>
inline void hegvd(Func func, const char* func_name, sycl::queue& queue, std::int64_t itype,
                  oneapi::math::job jobz, oneapi::math::uplo uplo, std::int64_t n,
                  sycl::buffer<T_A>& a, std::int64_t lda, sycl::buffer<T_A>& b, std::int64_t ldb,
                  sycl::buffer<T_B>& w) {
    using ArmDataType = typename ArmEquivalentType<T_A>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto b_acc = b.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto w_acc = w.template get_access<sycl::access::mode::write>(cgh);
        host_task<class armpl_kernel_hegvd>(cgh, [=]() {
            std::int64_t err =
                func(LAPACK_COL_MAJOR, itype, get_job(jobz), get_fill_mode(uplo), n,
                     a_acc.GET_MULTI_PTR, lda, b_acc.GET_MULTI_PTR, ldb, w_acc.GET_MULTI_PTR);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define HEGVD_LAUNCHER(TYPE_A, TYPE_B, ROUTINE)                                                    \
    void hegvd(sycl::queue& queue, std::int64_t itype, oneapi::math::job jobz,                     \
               oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<TYPE_A>& a, std::int64_t lda, \
               sycl::buffer<TYPE_A>& b, std::int64_t ldb, sycl::buffer<TYPE_B>& w,                 \
               sycl::buffer<TYPE_A>& scratchpad, std::int64_t scratchpad_size) {                   \
        hegvd(ROUTINE, #ROUTINE, queue, itype, jobz, uplo, n, a, lda, b, ldb, w);                  \
    }

HEGVD_LAUNCHER(std::complex<float>, float, LAPACKE_chegvd)
HEGVD_LAUNCHER(std::complex<double>, double, LAPACKE_zhegvd)

#undef HEGVD_LAUNCHER

template <typename Func, typename T_A, typename T_B>
inline void hetrd(Func func, const char* func_name, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, sycl::buffer<T_A>& a, std::int64_t lda, sycl::buffer<T_B>& d,
                  sycl::buffer<T_B>& e, sycl::buffer<T_A>& tau, sycl::buffer<T_A>& s,
                  std::int64_t scratchpad_size) {
    using ArmDataType = typename ArmEquivalentType<T_A>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto d_acc = d.template get_access<sycl::access::mode::write>(cgh);
        auto e_acc = e.template get_access<sycl::access::mode::write>(cgh);
        auto tau_acc =
            tau.template reinterpret<ArmDataType>().template get_access<sycl::access::mode::write>(
                cgh);

        auto s_acc = s.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);

        host_task<class armpl_kernel_hetrd>(cgh, [=]() {
            std::int64_t err = func(LAPACK_COL_MAJOR, get_fill_mode(uplo), n, a_acc.GET_MULTI_PTR,
                                    lda, d_acc.GET_MULTI_PTR, e_acc.GET_MULTI_PTR,
                                    tau_acc.GET_MULTI_PTR, s_acc.GET_MULTI_PTR, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define HETRD_LAUNCHER(TYPE_A, TYPE_B, ROUTINE)                                                   \
    void hetrd(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,                       \
               sycl::buffer<TYPE_A>& a, std::int64_t lda, sycl::buffer<TYPE_B>& d,                \
               sycl::buffer<TYPE_B>& e, sycl::buffer<TYPE_A>& tau,                                \
               sycl::buffer<TYPE_A>& scratchpad, std::int64_t scratchpad_size) {                  \
        hetrd(ROUTINE, #ROUTINE, queue, uplo, n, a, lda, d, e, tau, scratchpad, scratchpad_size); \
    }

HETRD_LAUNCHER(std::complex<float>, float, LAPACKE_chetrd_work)
HETRD_LAUNCHER(std::complex<double>, double, LAPACKE_zhetrd_work)

#undef HETRD_LAUNCHER

template <typename Func, typename T>
inline void hetrf(Func func, const char* func_name, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda,
                  sycl::buffer<std::int64_t>& ipiv, sycl::buffer<T>& s,
                  std::int64_t scratchpad_size) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto ipiv_acc = ipiv.template get_access<sycl::access::mode::write>(cgh);

        auto s_acc = s.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);

        host_task<class armpl_kernel_hetrf>(cgh, [=]() {
            std::int64_t err =
                func(LAPACK_COL_MAJOR, get_fill_mode(uplo), n, a_acc.GET_MULTI_PTR, lda,
                     ipiv_acc.GET_MULTI_PTR, s_acc.GET_MULTI_PTR, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define HETRF_LAUNCHER(TYPE_A, ROUTINE)                                                      \
    void hetrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,                  \
               sycl::buffer<TYPE_A>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,  \
               sycl::buffer<TYPE_A>& scratchpad, std::int64_t scratchpad_size) {             \
        hetrf(ROUTINE, #ROUTINE, queue, uplo, n, a, lda, ipiv, scratchpad, scratchpad_size); \
    }

HETRF_LAUNCHER(std::complex<float>, LAPACKE_chetrf_work)
HETRF_LAUNCHER(std::complex<double>, LAPACKE_zhetrf_work)

#undef HETRF_LAUNCHER

template <typename Func, typename T>
inline void orgbr(Func func, const char* func_name, sycl::queue& queue, oneapi::math::generate vec,
                  std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<T>& a,
                  std::int64_t lda, sycl::buffer<T>& tau, sycl::buffer<T>& s,
                  std::int64_t scratchpad_size) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc =
            tau.template reinterpret<ArmDataType>().template get_access<sycl::access::mode::read>(
                cgh);
        auto s_acc = s.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);

        host_task<class armpl_kernel_orgbr>(cgh, [=]() {
            std::int64_t err =
                func(LAPACK_COL_MAJOR, get_generate(vec), m, n, k, a_acc.GET_MULTI_PTR, lda,
                     tau_acc.GET_MULTI_PTR, s_acc.GET_MULTI_PTR, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define ORGBR_LAUNCHER(TYPE, ROUTINE)                                                            \
    void orgbr(sycl::queue& queue, oneapi::math::generate vec, std::int64_t m, std::int64_t n,   \
               std::int64_t k, sycl::buffer<TYPE>& a, std::int64_t lda, sycl::buffer<TYPE>& tau, \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {                   \
        orgbr(ROUTINE, #ROUTINE, queue, vec, m, n, k, a, lda, tau, scratchpad, scratchpad_size); \
    }

ORGBR_LAUNCHER(float, LAPACKE_sorgbr_work)
ORGBR_LAUNCHER(double, LAPACKE_dorgbr_work)

#undef ORGBR_LAUNCHER

template <typename Func, typename T>
inline void orgqr(Func func, const char* func_name, sycl::queue& queue, std::int64_t m,
                  std::int64_t n, std::int64_t k, sycl::buffer<T>& a, std::int64_t lda,
                  sycl::buffer<T>& tau, sycl::buffer<T>& s, std::int64_t scratchpad_size) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc =
            tau.template reinterpret<ArmDataType>().template get_access<sycl::access::mode::read>(
                cgh);
        auto s_acc = s.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_orgqr>(cgh, [=]() {
            std::int64_t err = func(LAPACK_COL_MAJOR, m, n, k, a_acc.GET_MULTI_PTR, lda,
                                    tau_acc.GET_MULTI_PTR, s_acc.GET_MULTI_PTR, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define ORGQR_LAUNCHER(TYPE, ROUTINE)                                                       \
    void orgqr(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,          \
               sycl::buffer<TYPE>& a, std::int64_t lda, sycl::buffer<TYPE>& tau,            \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {              \
        orgqr(ROUTINE, #ROUTINE, queue, m, n, k, a, lda, tau, scratchpad, scratchpad_size); \
    }

ORGQR_LAUNCHER(float, LAPACKE_sorgqr_work)
ORGQR_LAUNCHER(double, LAPACKE_dorgqr_work)

#undef ORGQR_LAUNCHER

template <typename Func, typename T>
inline void orgtr(Func func, const char* func_name, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& tau,
                  sycl::buffer<T>& s, std::int64_t scratchpad_size) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc =
            tau.template reinterpret<ArmDataType>().template get_access<sycl::access::mode::read>(
                cgh);
        auto s_acc = s.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_orgtr>(cgh, [=]() {
            std::int64_t err =
                func(LAPACK_COL_MAJOR, get_fill_mode(uplo), n, a_acc.GET_MULTI_PTR, lda,
                     tau_acc.GET_MULTI_PTR, s_acc.GET_MULTI_PTR, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define ORGTR_LAUNCHER(TYPE, ROUTINE)                                                              \
    void orgtr(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<TYPE>& a, \
               std::int64_t lda, sycl::buffer<TYPE>& tau, sycl::buffer<TYPE>& scratchpad,          \
               std::int64_t scratchpad_size) {                                                     \
        orgtr(ROUTINE, #ROUTINE, queue, uplo, n, a, lda, tau, scratchpad, scratchpad_size);        \
    }

ORGTR_LAUNCHER(float, LAPACKE_sorgtr_work)
ORGTR_LAUNCHER(double, LAPACKE_dorgtr_work)

#undef ORGTR_LAUNCHER

template <typename Func, typename T>
inline void ormtr(Func func, const char* func_name, sycl::queue& queue, oneapi::math::side side,
                  oneapi::math::uplo uplo, oneapi::math::transpose trans, std::int64_t m,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& tau,
                  sycl::buffer<T>& c, std::int64_t ldc, sycl::buffer<T>& s,
                  std::int64_t scratchpad_size) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc = tau.template reinterpret<ArmDataType>()
                           .template get_access<sycl::access::mode::read_write>(cgh);
        auto c_acc = c.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto s_acc = s.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);

        host_task<class armpl_kernel_ormtr>(cgh, [=]() {
            std::int64_t err =
                func(LAPACK_COL_MAJOR, get_side_mode(side), get_fill_mode(uplo),
                     get_operation(trans), m, n, a_acc.GET_MULTI_PTR, lda, tau_acc.GET_MULTI_PTR,
                     c_acc.GET_MULTI_PTR, ldc, s_acc.GET_MULTI_PTR, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define ORMTR_LAUNCHER(TYPE, ROUTINE)                                                             \
    void ormtr(sycl::queue& queue, oneapi::math::side side, oneapi::math::uplo uplo,              \
               oneapi::math::transpose trans, std::int64_t m, std::int64_t n,                     \
               sycl::buffer<TYPE>& a, std::int64_t lda, sycl::buffer<TYPE>& tau,                  \
               sycl::buffer<TYPE>& c, std::int64_t ldc, sycl::buffer<TYPE>& scratchpad,           \
               std::int64_t scratchpad_size) {                                                    \
        ormtr(ROUTINE, #ROUTINE, queue, side, uplo, trans, m, n, a, lda, tau, c, ldc, scratchpad, \
              scratchpad_size);                                                                   \
    }

ORMTR_LAUNCHER(float, LAPACKE_sormtr_work)
ORMTR_LAUNCHER(double, LAPACKE_dormtr_work)

#undef ORMTR_LAUNCHER

template <typename Func, typename T>
inline void ormrq(Func func, const char* func_name, sycl::queue& queue, oneapi::math::side side,
                  oneapi::math::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
                  sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& tau, sycl::buffer<T>& c,
                  std::int64_t ldc, sycl::buffer<T>& s, std::int64_t scratchpad_size) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc = tau.template reinterpret<ArmDataType>()
                           .template get_access<sycl::access::mode::read_write>(cgh);
        auto c_acc = c.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto s_acc = s.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);

        host_task<class armpl_kernel_ormrq>(cgh, [=]() {
            std::int64_t err = func(LAPACK_COL_MAJOR, get_side_mode(side), get_operation(trans), m,
                                    n, k, a_acc.GET_MULTI_PTR, lda, tau_acc.GET_MULTI_PTR,
                                    c_acc.GET_MULTI_PTR, ldc, s_acc.GET_MULTI_PTR, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define ORMRQ_LAUNCHER(TYPE, ROUTINE)                                                              \
    void ormrq(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,         \
               std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<TYPE>& a,              \
               std::int64_t lda, sycl::buffer<TYPE>& tau, sycl::buffer<TYPE>& c, std::int64_t ldc, \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {                     \
        ormrq(ROUTINE, #ROUTINE, queue, side, trans, m, n, k, a, lda, tau, c, ldc, scratchpad,     \
              scratchpad_size);                                                                    \
    }

ORMRQ_LAUNCHER(float, LAPACKE_sormrq_work)
ORMRQ_LAUNCHER(double, LAPACKE_dormrq_work)

#undef ORMRQ_LAUNCHER

template <typename Func, typename T>
inline void ormqr(Func func, const char* func_name, sycl::queue& queue, oneapi::math::side side,
                  oneapi::math::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
                  sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& tau, sycl::buffer<T>& c,
                  std::int64_t ldc, sycl::buffer<T>& s, std::int64_t scratchpad_size) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc =
            a.template reinterpret<ArmDataType>().template get_access<sycl::access::mode::read>(
                cgh);
        auto tau_acc =
            tau.template reinterpret<ArmDataType>().template get_access<sycl::access::mode::read>(
                cgh);
        auto c_acc = c.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);

        auto s_acc = s.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);

        host_task<class armpl_kernel_ormqr>(cgh, [=]() {
            std::int64_t err = func(LAPACK_COL_MAJOR, get_side_mode(side), get_operation(trans), m,
                                    n, k, a_acc.GET_MULTI_PTR, lda, tau_acc.GET_MULTI_PTR,
                                    c_acc.GET_MULTI_PTR, ldc, s_acc.GET_MULTI_PTR, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define ORMQR_LAUNCHER(TYPE, ROUTINE)                                                              \
    void ormqr(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,         \
               std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<TYPE>& a,              \
               std::int64_t lda, sycl::buffer<TYPE>& tau, sycl::buffer<TYPE>& c, std::int64_t ldc, \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {                     \
        ormqr(ROUTINE, #ROUTINE, queue, side, trans, m, n, k, a, lda, tau, c, ldc, scratchpad,     \
              scratchpad_size);                                                                    \
    }

ORMQR_LAUNCHER(float, LAPACKE_sormqr_work)
ORMQR_LAUNCHER(double, LAPACKE_dormqr_work)

#undef ORMQR_LAUNCHER
template <typename Func, typename T>
inline void potrf(Func func, const char* func_name, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_potrf>(cgh, [=]() {
            std::int64_t err =
                func(LAPACK_COL_MAJOR, get_fill_mode(uplo), n, a_acc.GET_MULTI_PTR, lda);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define POTRF_LAUNCHER(TYPE, ROUTINE)                                                              \
    void potrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<TYPE>& a, \
               std::int64_t lda, sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {   \
        potrf(ROUTINE, #ROUTINE, queue, uplo, n, a, lda);                                          \
    }

POTRF_LAUNCHER(float, LAPACKE_spotrf_work)
POTRF_LAUNCHER(double, LAPACKE_dpotrf_work)
POTRF_LAUNCHER(std::complex<float>, LAPACKE_cpotrf_work)
POTRF_LAUNCHER(std::complex<double>, LAPACKE_zpotrf_work)

#undef POTRF_LAUNCHER

template <typename Func, typename T>
inline void potri(Func func, const char* func_name, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_potri>(cgh, [=]() {
            std::int64_t err =
                func(LAPACK_COL_MAJOR, get_fill_mode(uplo), n, a_acc.GET_MULTI_PTR, lda);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define POTRI_LAUNCHER(TYPE, ROUTINE)                                                              \
    void potri(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<TYPE>& a, \
               std::int64_t lda, sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {   \
        potri(ROUTINE, #ROUTINE, queue, uplo, n, a, lda);                                          \
    }

POTRI_LAUNCHER(float, LAPACKE_spotri_work)
POTRI_LAUNCHER(double, LAPACKE_dpotri_work)
POTRI_LAUNCHER(std::complex<float>, LAPACKE_cpotri_work)
POTRI_LAUNCHER(std::complex<double>, LAPACKE_zpotri_work)

#undef POTRI_LAUNCHER

template <typename Func, typename T>
inline void potrs(Func func, const char* func_name, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, std::int64_t nrhs, sycl::buffer<T>& a, std::int64_t lda,
                  sycl::buffer<T>& b, std::int64_t ldb) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc =
            a.template reinterpret<ArmDataType>().template get_access<sycl::access::mode::read>(
                cgh);
        auto b_acc = b.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);

        host_task<class armpl_kernel_potrs>(cgh, [=]() {
            std::int64_t err = func(LAPACK_COL_MAJOR, get_fill_mode(uplo), n, nrhs,
                                    a_acc.GET_MULTI_PTR, lda, b_acc.GET_MULTI_PTR, ldb);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define POTRS_LAUNCHER(TYPE, ROUTINE)                                                            \
    void potrs(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,   \
               sycl::buffer<TYPE>& a, std::int64_t lda, sycl::buffer<TYPE>& b, std::int64_t ldb, \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {                   \
        potrs(ROUTINE, #ROUTINE, queue, uplo, n, nrhs, a, lda, b, ldb);                          \
    }

POTRS_LAUNCHER(float, LAPACKE_spotrs_work)
POTRS_LAUNCHER(double, LAPACKE_dpotrs_work)
POTRS_LAUNCHER(std::complex<float>, LAPACKE_cpotrs_work)
POTRS_LAUNCHER(std::complex<double>, LAPACKE_zpotrs_work)

#undef POTRS_LAUNCHER

template <typename Func, typename T>
inline void syevd(Func func, const char* func_name, sycl::queue& queue, oneapi::math::job jobz,
                  oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<T>& a, std::int64_t lda,
                  sycl::buffer<T>& w) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto w_acc =
            w.template reinterpret<ArmDataType>().template get_access<sycl::access::mode::write>(
                cgh);
        host_task<class armpl_kernel_syevd>(cgh, [=]() {
            std::int64_t err = func(LAPACK_COL_MAJOR, get_job(jobz), get_fill_mode(uplo), n,
                                    a_acc.GET_MULTI_PTR, lda, w_acc.GET_MULTI_PTR);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define SYEVD_LAUNCHER(TYPE, ROUTINE)                                                          \
    void syevd(sycl::queue& queue, oneapi::math::job jobz, oneapi::math::uplo uplo,            \
               std::int64_t n, sycl::buffer<TYPE>& a, std::int64_t lda, sycl::buffer<TYPE>& w, \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {                 \
        syevd(ROUTINE, #ROUTINE, queue, jobz, uplo, n, a, lda, w);                             \
    }

SYEVD_LAUNCHER(float, LAPACKE_ssyevd)
SYEVD_LAUNCHER(double, LAPACKE_dsyevd)

#undef SYEVD_LAUNCHER

template <typename Func, typename T>
inline void sygvd(Func func, const char* func_name, sycl::queue& queue, std::int64_t itype,
                  oneapi::math::job jobz, oneapi::math::uplo uplo, std::int64_t n,
                  sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& b, std::int64_t ldb,
                  sycl::buffer<T>& w) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto b_acc = b.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto w_acc =
            w.template reinterpret<ArmDataType>().template get_access<sycl::access::mode::write>(
                cgh);
        host_task<class armpl_kernel_sygvd>(cgh, [=]() {
            std::int64_t err =
                func(LAPACK_COL_MAJOR, itype, get_job(jobz), get_fill_mode(uplo), n,
                     a_acc.GET_MULTI_PTR, lda, b_acc.GET_MULTI_PTR, ldb, w_acc.GET_MULTI_PTR);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define SYGVD_LAUNCHER(TYPE, ROUTINE)                                                            \
    void sygvd(sycl::queue& queue, std::int64_t itype, oneapi::math::job jobz,                   \
               oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<TYPE>& a, std::int64_t lda, \
               sycl::buffer<TYPE>& b, std::int64_t ldb, sycl::buffer<TYPE>& w,                   \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {                   \
        sygvd(ROUTINE, #ROUTINE, queue, itype, jobz, uplo, n, a, lda, b, ldb, w);                \
    }

SYGVD_LAUNCHER(float, LAPACKE_ssygvd)
SYGVD_LAUNCHER(double, LAPACKE_dsygvd)

#undef SYGVD_LAUNCH

template <typename Func, typename T>
inline void sytrd(Func func, const char* func_name, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& d,
                  sycl::buffer<T>& e, sycl::buffer<T>& tau, sycl::buffer<T>& s,
                  std::int64_t scratchpad_size) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto d_acc =
            d.template reinterpret<ArmDataType>().template get_access<sycl::access::mode::write>(
                cgh);
        auto e_acc =
            e.template reinterpret<ArmDataType>().template get_access<sycl::access::mode::write>(
                cgh);
        auto tau_acc =
            tau.template reinterpret<ArmDataType>().template get_access<sycl::access::mode::write>(
                cgh);
        auto s_acc = s.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);

        host_task<class armpl_kernel_sytrd>(cgh, [=]() {
            std::int64_t err = func(LAPACK_COL_MAJOR, get_fill_mode(uplo), n, a_acc.GET_MULTI_PTR,
                                    lda, d_acc.GET_MULTI_PTR, e_acc.GET_MULTI_PTR,
                                    tau_acc.GET_MULTI_PTR, s_acc.GET_MULTI_PTR, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define SYTRD_LAUNCHER(TYPE, ROUTINE)                                                              \
    void sytrd(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<TYPE>& a, \
               std::int64_t lda, sycl::buffer<TYPE>& d, sycl::buffer<TYPE>& e,                     \
               sycl::buffer<TYPE>& tau, sycl::buffer<TYPE>& scratchpad,                            \
               std::int64_t scratchpad_size) {                                                     \
        sytrd(ROUTINE, #ROUTINE, queue, uplo, n, a, lda, d, e, tau, scratchpad, scratchpad_size);  \
    }

SYTRD_LAUNCHER(float, LAPACKE_ssytrd_work)
SYTRD_LAUNCHER(double, LAPACKE_dsytrd_work)

#undef SYTRD_LAUNCHER

template <typename Func, typename T>
inline void sytrf(Func func, const char* func_name, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda,
                  sycl::buffer<std::int64_t>& ipiv, sycl::buffer<T>& s,
                  std::int64_t scratchpad_size) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto ipiv_acc = ipiv.template get_access<sycl::access::mode::write>(cgh);
        auto s_acc = s.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);

        host_task<class armpl_kernel_sytrf>(cgh, [=]() {
            std::int64_t err =
                func(LAPACK_COL_MAJOR, get_fill_mode(uplo), n, a_acc.GET_MULTI_PTR, lda,
                     ipiv_acc.GET_MULTI_PTR, s_acc.GET_MULTI_PTR, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define SYTRF_LAUNCHER(TYPE, ROUTINE)                                                              \
    void sytrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<TYPE>& a, \
               std::int64_t lda, sycl::buffer<std::int64_t>& ipiv, sycl::buffer<TYPE>& scratchpad, \
               std::int64_t scratchpad_size) {                                                     \
        sytrf(ROUTINE, #ROUTINE, queue, uplo, n, a, lda, ipiv, scratchpad, scratchpad_size);       \
    }

SYTRF_LAUNCHER(float, LAPACKE_ssytrf_work)
SYTRF_LAUNCHER(double, LAPACKE_dsytrf_work)
SYTRF_LAUNCHER(std::complex<float>, LAPACKE_csytrf_work)
SYTRF_LAUNCHER(std::complex<double>, LAPACKE_zsytrf_work)

#undef SYTRF_LAUNCHER

template <typename Func, typename T>
inline void trtrs(Func func, const char* func_name, sycl::queue& queue, oneapi::math::uplo uplo,
                  oneapi::math::transpose trans, oneapi::math::diag diag, std::int64_t n,
                  std::int64_t nrhs, sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& b,
                  std::int64_t ldb) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto b_acc =
            b.template reinterpret<ArmDataType>().template get_access<sycl::access::mode::write>(
                cgh);
        host_task<class armpl_kernel_trtrs>(cgh, [=]() {
            std::int64_t err =
                func(LAPACK_COL_MAJOR, get_fill_mode(uplo), get_operation(trans), get_diag(diag), n,
                     nrhs, a_acc.GET_MULTI_PTR, lda, b_acc.GET_MULTI_PTR, ldb);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define TRTRS_LAUNCHER(TYPE, ROUTINE)                                                             \
    void trtrs(sycl::queue& queue, oneapi::math::uplo uplo, oneapi::math::transpose trans,        \
               oneapi::math::diag diag, std::int64_t n, std::int64_t nrhs, sycl::buffer<TYPE>& a, \
               std::int64_t lda, sycl::buffer<TYPE>& b, std::int64_t ldb,                         \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {                    \
        trtrs(ROUTINE, #ROUTINE, queue, uplo, trans, diag, n, nrhs, a, lda, b, ldb);              \
    }

TRTRS_LAUNCHER(float, LAPACKE_strtrs)
TRTRS_LAUNCHER(double, LAPACKE_dtrtrs)
TRTRS_LAUNCHER(std::complex<float>, LAPACKE_ctrtrs)
TRTRS_LAUNCHER(std::complex<double>, LAPACKE_ztrtrs)

#undef TRTRS_LAUNCHER

template <typename Func, typename T>
inline void ungbr(Func func, const char* func_name, sycl::queue& queue, oneapi::math::generate vec,
                  std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<T>& a,
                  std::int64_t lda, sycl::buffer<T>& tau, sycl::buffer<T>& s,
                  std::int64_t scratchpad_size) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc =
            tau.template reinterpret<ArmDataType>().template get_access<sycl::access::mode::write>(
                cgh);
        auto s_acc = s.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);

        host_task<class armpl_kernel_ungbr>(cgh, [=]() {
            std::int64_t err =
                func(LAPACK_COL_MAJOR, get_generate(vec), m, n, k, a_acc.GET_MULTI_PTR, lda,
                     tau_acc.GET_MULTI_PTR, s_acc.GET_MULTI_PTR, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define UNGBR_LAUNCHER(TYPE, ROUTINE)                                                            \
    void ungbr(sycl::queue& queue, oneapi::math::generate vec, std::int64_t m, std::int64_t n,   \
               std::int64_t k, sycl::buffer<TYPE>& a, std::int64_t lda, sycl::buffer<TYPE>& tau, \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {                   \
        ungbr(ROUTINE, #ROUTINE, queue, vec, m, n, k, a, lda, tau, scratchpad, scratchpad_size); \
    }

UNGBR_LAUNCHER(std::complex<float>, LAPACKE_cungbr_work)
UNGBR_LAUNCHER(std::complex<double>, LAPACKE_zungbr_work)

#undef UNGBR_LAUNCHER

template <typename Func, typename T>
inline void ungqr(Func func, const char* func_name, sycl::queue& queue, std::int64_t m,
                  std::int64_t n, std::int64_t k, sycl::buffer<T>& a, std::int64_t lda,
                  sycl::buffer<T>& tau, sycl::buffer<T>& s, std::int64_t scratchpad_size) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc =
            tau.template reinterpret<ArmDataType>().template get_access<sycl::access::mode::write>(
                cgh);
        auto s_acc = s.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);

        host_task<class armpl_kernel_ungqr>(cgh, [=]() {
            std::int64_t err = func(LAPACK_COL_MAJOR, m, n, k, a_acc.GET_MULTI_PTR, lda,
                                    tau_acc.GET_MULTI_PTR, s_acc.GET_MULTI_PTR, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define UNGQR_LAUNCHER(TYPE, ROUTINE)                                                       \
    void ungqr(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,          \
               sycl::buffer<TYPE>& a, std::int64_t lda, sycl::buffer<TYPE>& tau,            \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {              \
        ungqr(ROUTINE, #ROUTINE, queue, m, n, k, a, lda, tau, scratchpad, scratchpad_size); \
    }

UNGQR_LAUNCHER(std::complex<float>, LAPACKE_cungqr_work)
UNGQR_LAUNCHER(std::complex<double>, LAPACKE_zungqr_work)

#undef UNGQR_LAUNCHER

template <typename Func, typename T>
inline void ungtr(Func func, const char* func_name, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& tau,
                  sycl::buffer<T>& s, std::int64_t scratchpad_size) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc =
            tau.template reinterpret<ArmDataType>().template get_access<sycl::access::mode::write>(
                cgh);
        auto s_acc = s.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);

        host_task<class armpl_kernel_ungtr>(cgh, [=]() {
            std::int64_t err =
                func(LAPACK_COL_MAJOR, get_fill_mode(uplo), n, a_acc.GET_MULTI_PTR, lda,
                     tau_acc.GET_MULTI_PTR, s_acc.GET_MULTI_PTR, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define UNGTR_LAUNCHER(TYPE, ROUTINE)                                                              \
    void ungtr(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<TYPE>& a, \
               std::int64_t lda, sycl::buffer<TYPE>& tau, sycl::buffer<TYPE>& scratchpad,          \
               std::int64_t scratchpad_size) {                                                     \
        ungtr(ROUTINE, #ROUTINE, queue, uplo, n, a, lda, tau, scratchpad, scratchpad_size);        \
    }

UNGTR_LAUNCHER(std::complex<float>, LAPACKE_cungtr_work)
UNGTR_LAUNCHER(std::complex<double>, LAPACKE_zungtr_work)

#undef UNGTR_LAUNCHER

template <typename Func, typename T>
inline void unmrq(Func func, const char* func_name, sycl::queue& queue, oneapi::math::side side,
                  oneapi::math::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
                  sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& tau, sycl::buffer<T>& c,
                  std::int64_t ldc, sycl::buffer<T>& s, std::int64_t scratchpad_size) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto c_acc = c.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc =
            tau.template reinterpret<ArmDataType>().template get_access<sycl::access::mode::write>(
                cgh);
        auto s_acc = s.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);

        host_task<class armpl_kernel_unmrq>(cgh, [=]() {
            std::int64_t err = func(LAPACK_COL_MAJOR, get_side_mode(side), get_operation(trans), m,
                                    n, k, a_acc.GET_MULTI_PTR, lda, tau_acc.GET_MULTI_PTR,
                                    c_acc.GET_MULTI_PTR, ldc, s_acc.GET_MULTI_PTR, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define UNMRQ_LAUNCHER(TYPE, ROUTINE)                                                              \
    void unmrq(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,         \
               std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<TYPE>& a,              \
               std::int64_t lda, sycl::buffer<TYPE>& tau, sycl::buffer<TYPE>& c, std::int64_t ldc, \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {                     \
        unmrq(ROUTINE, #ROUTINE, queue, side, trans, m, n, k, a, lda, tau, c, ldc, scratchpad,     \
              scratchpad_size);                                                                    \
    }

UNMRQ_LAUNCHER(std::complex<float>, LAPACKE_cunmrq_work)
UNMRQ_LAUNCHER(std::complex<double>, LAPACKE_zunmrq_work)

#undef UNMRQ_LAUNCHER

template <typename Func, typename T>
inline void unmqr(Func func, const char* func_name, sycl::queue& queue, oneapi::math::side side,
                  oneapi::math::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
                  sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& tau, sycl::buffer<T>& c,
                  std::int64_t ldc, sycl::buffer<T>& s, std::int64_t scratchpad_size) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc =
            tau.template reinterpret<ArmDataType>().template get_access<sycl::access::mode::write>(
                cgh);
        auto c_acc = c.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto s_acc = s.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);

        host_task<class armpl_kernel_unmqr>(cgh, [=]() {
            std::int64_t err = func(LAPACK_COL_MAJOR, get_side_mode(side), get_operation(trans), m,
                                    n, k, a_acc.GET_MULTI_PTR, lda, tau_acc.GET_MULTI_PTR,
                                    c_acc.GET_MULTI_PTR, ldc, s_acc.GET_MULTI_PTR, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define UNMQR_LAUNCHER(TYPE, ROUTINE)                                                              \
    void unmqr(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,         \
               std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<TYPE>& a,              \
               std::int64_t lda, sycl::buffer<TYPE>& tau, sycl::buffer<TYPE>& c, std::int64_t ldc, \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {                     \
        unmqr(ROUTINE, #ROUTINE, queue, side, trans, m, n, k, a, lda, tau, c, ldc, scratchpad,     \
              scratchpad_size);                                                                    \
    }

UNMQR_LAUNCHER(std::complex<float>, LAPACKE_cunmqr_work)
UNMQR_LAUNCHER(std::complex<double>, LAPACKE_zunmqr_work)

#undef UNMQR_LAUNCHER

template <typename Func, typename T>
inline void unmtr(Func func, const char* func_name, sycl::queue& queue, oneapi::math::side side,
                  oneapi::math::uplo uplo, oneapi::math::transpose trans, std::int64_t m,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& tau,
                  sycl::buffer<T>& c, std::int64_t ldc, sycl::buffer<T>& s,
                  std::int64_t scratchpad_size) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc =
            tau.template reinterpret<ArmDataType>().template get_access<sycl::access::mode::write>(
                cgh);
        auto c_acc = c.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);
        auto s_acc = s.template reinterpret<ArmDataType>()
                         .template get_access<sycl::access::mode::read_write>(cgh);

        host_task<class armpl_kernel_unmtr>(cgh, [=]() {
            std::int64_t err =
                func(LAPACK_COL_MAJOR, get_side_mode(side), get_fill_mode(uplo),
                     get_operation(trans), m, n, a_acc.GET_MULTI_PTR, lda, tau_acc.GET_MULTI_PTR,
                     c_acc.GET_MULTI_PTR, ldc, s_acc.GET_MULTI_PTR, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
}

#define UNMTR_LAUNCHER(TYPE, ROUTINE)                                                             \
    void unmtr(sycl::queue& queue, oneapi::math::side side, oneapi::math::uplo uplo,              \
               oneapi::math::transpose trans, std::int64_t m, std::int64_t n,                     \
               sycl::buffer<TYPE>& a, std::int64_t lda, sycl::buffer<TYPE>& tau,                  \
               sycl::buffer<TYPE>& c, std::int64_t ldc, sycl::buffer<TYPE>& scratchpad,           \
               std::int64_t scratchpad_size) {                                                    \
        unmtr(ROUTINE, #ROUTINE, queue, side, uplo, trans, m, n, a, lda, tau, c, ldc, scratchpad, \
              scratchpad_size);                                                                   \
    }

UNMTR_LAUNCHER(std::complex<float>, LAPACKE_cunmtr_work)
UNMTR_LAUNCHER(std::complex<double>, LAPACKE_zunmtr_work)

#undef UNMTR_LAUNCHER

// USM APIs

template <typename Func, typename T_A, typename T_B>
inline sycl::event gebrd(Func func, const char* func_name, sycl::queue& queue, std::int64_t m,
                         std::int64_t n, T_A* a, std::int64_t lda, T_B* d, T_B* e, T_A* tauq,
                         T_A* taup, T_A* s, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T_A>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_gebrd>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto d_ = reinterpret_cast<T_B*>(d);
            auto e_ = reinterpret_cast<T_B*>(e);
            auto tauq_ = reinterpret_cast<ArmDataType*>(tauq);
            auto taup_ = reinterpret_cast<ArmDataType*>(taup);
            auto s_ = reinterpret_cast<ArmDataType*>(s);
            std::int64_t err =
                func(LAPACK_COL_MAJOR, m, n, a_, lda, d_, e_, tauq_, taup_, s_, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
    return done;
}

#define GEBRD_LAUNCHER_USM(TYPE_A, TYPE_B, ROUTINE)                                        \
    sycl::event gebrd(sycl::queue& queue, std::int64_t m, std::int64_t n, TYPE_A* a,       \
                      std::int64_t lda, TYPE_B* d, TYPE_B* e, TYPE_A* tauq, TYPE_A* taup,  \
                      TYPE_A* scratchpad, std::int64_t scratchpad_size,                    \
                      const std::vector<sycl::event>& dependencies) {                      \
        return gebrd(ROUTINE, #ROUTINE, queue, m, n, a, lda, d, e, tauq, taup, scratchpad, \
                     scratchpad_size, dependencies);                                       \
    }

GEBRD_LAUNCHER_USM(float, float, LAPACKE_sgebrd_work)
GEBRD_LAUNCHER_USM(double, double, LAPACKE_dgebrd_work)
GEBRD_LAUNCHER_USM(std::complex<float>, float, LAPACKE_cgebrd_work)
GEBRD_LAUNCHER_USM(std::complex<double>, double, LAPACKE_zgebrd_work)

#undef GEBRD_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event gerqf(Func func, const char* func_name, sycl::queue& queue, std::int64_t m,
                         std::int64_t n, T* a, std::int64_t lda, T* tau, T* s,
                         std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_gerqf>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto tau_ = reinterpret_cast<ArmDataType*>(tau);
            auto s_ = reinterpret_cast<ArmDataType*>(s);
            std::int64_t err = func(LAPACK_COL_MAJOR, m, n, a_, lda, tau_, s_, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
    return done;
}

#define GERQF_LAUNCHER_USM(TYPE, ROUTINE)                                                          \
    sycl::event gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n, TYPE* a,                 \
                      std::int64_t lda, TYPE* tau, TYPE* scratchpad, std::int64_t scratchpad_size, \
                      const std::vector<sycl::event>& dependencies) {                              \
        return gerqf(ROUTINE, #ROUTINE, queue, m, n, a, lda, tau, scratchpad, scratchpad_size,     \
                     dependencies);                                                                \
    }

GERQF_LAUNCHER_USM(float, LAPACKE_sgerqf_work)
GERQF_LAUNCHER_USM(double, LAPACKE_dgerqf_work)
GERQF_LAUNCHER_USM(std::complex<float>, LAPACKE_cgerqf_work)
GERQF_LAUNCHER_USM(std::complex<double>, LAPACKE_zgerqf_work)

#undef GERQF_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event geqrf(Func func, const char* func_name, sycl::queue& queue, std::int64_t m,
                         std::int64_t n, T* a, std::int64_t lda, T* tau, T* s,
                         std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_geqrf>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto tau_ = reinterpret_cast<ArmDataType*>(tau);
            auto s_ = reinterpret_cast<ArmDataType*>(s);

            std::int64_t err = func(LAPACK_COL_MAJOR, m, n, a_, lda, tau_, s_, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
    return done;
}

#define GEQRF_LAUNCHER_USM(TYPE, ROUTINE)                                                          \
    sycl::event geqrf(sycl::queue& queue, std::int64_t m, std::int64_t n, TYPE* a,                 \
                      std::int64_t lda, TYPE* tau, TYPE* scratchpad, std::int64_t scratchpad_size, \
                      const std::vector<sycl::event>& dependencies) {                              \
        return geqrf(ROUTINE, #ROUTINE, queue, m, n, a, lda, tau, scratchpad, scratchpad_size,     \
                     dependencies);                                                                \
    }

GEQRF_LAUNCHER_USM(float, LAPACKE_sgeqrf_work)
GEQRF_LAUNCHER_USM(double, LAPACKE_dgeqrf_work)
GEQRF_LAUNCHER_USM(std::complex<float>, LAPACKE_cgeqrf_work)
GEQRF_LAUNCHER_USM(std::complex<double>, LAPACKE_zgeqrf_work)

#undef GEQRF_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event getrf(Func func, const char* func_name, sycl::queue& queue, std::int64_t m,
                         std::int64_t n, T* a, std::int64_t lda, std::int64_t* ipiv, T* s,
                         std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_getrf>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto ipiv_ = reinterpret_cast<int64_t*>(ipiv);
            auto s_ = reinterpret_cast<ArmDataType*>(s);

            std::int64_t err = func(LAPACK_COL_MAJOR, m, n, a_, lda, ipiv_);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });

    return done;
}

#define GETRF_LAUNCHER_USM(TYPE, ROUTINE)                                                       \
    sycl::event getrf(sycl::queue& queue, std::int64_t m, std::int64_t n, TYPE* a,              \
                      std::int64_t lda, std::int64_t* ipiv, TYPE* scratchpad,                   \
                      std::int64_t scratchpad_size,                                             \
                      const std::vector<sycl::event>& dependencies) {                           \
        return getrf(ROUTINE, #ROUTINE, queue, m, n, a, lda, ipiv, scratchpad, scratchpad_size, \
                     dependencies);                                                             \
    }

GETRF_LAUNCHER_USM(float, LAPACKE_sgetrf_work)
GETRF_LAUNCHER_USM(double, LAPACKE_dgetrf_work)
GETRF_LAUNCHER_USM(std::complex<float>, LAPACKE_cgetrf_work)
GETRF_LAUNCHER_USM(std::complex<double>, LAPACKE_zgetrf_work)

#undef GETRF_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event getri(Func func, const char* func_name, sycl::queue& queue, std::int64_t n, T* a,
                         std::int64_t lda, std::int64_t* ipiv, T* s, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_getri>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto ipiv_ = reinterpret_cast<int64_t*>(ipiv);
            auto s_ = reinterpret_cast<ArmDataType*>(s);

            std::int64_t err = func(LAPACK_COL_MAJOR, n, a_, lda, ipiv_, s_, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });

    return done;
}

#define GETRI_LAUNCHER_USM(TYPE, ROUTINE)                                                    \
    sycl::event getri(sycl::queue& queue, std::int64_t n, TYPE* a, std::int64_t lda,         \
                      std::int64_t* ipiv, TYPE* scratchpad, std::int64_t scratchpad_size,    \
                      const std::vector<sycl::event>& dependencies) {                        \
        return getri(ROUTINE, #ROUTINE, queue, n, a, lda, ipiv, scratchpad, scratchpad_size, \
                     dependencies);                                                          \
    }

GETRI_LAUNCHER_USM(float, LAPACKE_sgetri_work)
GETRI_LAUNCHER_USM(double, LAPACKE_dgetri_work)
GETRI_LAUNCHER_USM(std::complex<float>, LAPACKE_cgetri_work)
GETRI_LAUNCHER_USM(std::complex<double>, LAPACKE_zgetri_work)

#undef GETRI_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event getrs(Func func, const char* func_name, sycl::queue& queue,
                         oneapi::math::transpose trans, std::int64_t n, std::int64_t nrhs, T* a,
                         std::int64_t lda, std::int64_t* ipiv, T* b, std::int64_t ldb, T* s,
                         std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_getrs>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto ipiv_ = reinterpret_cast<int64_t*>(ipiv);
            auto b_ = reinterpret_cast<ArmDataType*>(b);

            std::int64_t err =
                func(LAPACK_COL_MAJOR, get_operation(trans), n, nrhs, a_, lda, ipiv_, b_, ldb);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });

    return done;
}

#define GETRS_LAUNCHER_USM(TYPE, ROUTINE)                                                        \
    sycl::event getrs(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,         \
                      std::int64_t nrhs, TYPE* a, std::int64_t lda, std::int64_t* ipiv, TYPE* b, \
                      std::int64_t ldb, TYPE* scratchpad, std::int64_t scratchpad_size,          \
                      const std::vector<sycl::event>& dependencies) {                            \
        return getrs(ROUTINE, #ROUTINE, queue, trans, n, nrhs, a, lda, ipiv, b, ldb, scratchpad, \
                     scratchpad_size, dependencies);                                             \
    }

GETRS_LAUNCHER_USM(float, LAPACKE_sgetrs_work)
GETRS_LAUNCHER_USM(double, LAPACKE_dgetrs_work)
GETRS_LAUNCHER_USM(std::complex<float>, LAPACKE_cgetrs_work)
GETRS_LAUNCHER_USM(std::complex<double>, LAPACKE_zgetrs_work)

#undef GETRS_LAUNCHER_USM

template <typename Func, typename T_A, typename T_B>
inline sycl::event gesvd(Func func, const char* func_name, sycl::queue& queue,
                         oneapi::math::jobsvd jobu, oneapi::math::jobsvd jobvt, std::int64_t m,
                         std::int64_t n, T_A* a, std::int64_t lda, T_B* s, T_A* u, std::int64_t ldu,
                         T_A* vt, std::int64_t ldvt, T_A* scratch, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T_A>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_gesvd>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto s_ = reinterpret_cast<T_B*>(s);
            auto u_ = reinterpret_cast<ArmDataType*>(u);
            auto vt_ = reinterpret_cast<ArmDataType*>(vt);
            auto scratch_ = reinterpret_cast<ArmDataType*>(scratch);
            std::int64_t err = 0;
            if constexpr (is_complex<T_A>) {
                T_B* rwork = new T_B[5 * std::min(m, n)];
                err = func(LAPACK_COL_MAJOR, get_jobsvd(jobu), get_jobsvd(jobvt), m, n, a_, lda, s_,
                           u_, ldu, vt_, ldvt, scratch_, scratchpad_size, rwork);
                delete[] rwork;
            }
            else {
                err = func(LAPACK_COL_MAJOR, get_jobsvd(jobu), get_jobsvd(jobvt), m, n, a_, lda, s_,
                           u_, ldu, vt_, ldvt, scratch_, scratchpad_size);
            }
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
    return done;
}

#define GESVD_LAUNCHER_USM(TYPE_A, TYPE_B, ROUTINE)                                              \
    sycl::event gesvd(sycl::queue& queue, oneapi::math::jobsvd jobu, oneapi::math::jobsvd jobvt, \
                      std::int64_t m, std::int64_t n, TYPE_A* a, std::int64_t lda, TYPE_B* s,    \
                      TYPE_A* u, std::int64_t ldu, TYPE_A* vt, std::int64_t ldvt,                \
                      TYPE_A* scratchpad, std::int64_t scratchpad_size,                          \
                      const std::vector<sycl::event>& dependencies) {                            \
        return gesvd(ROUTINE, #ROUTINE, queue, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,   \
                     scratchpad, scratchpad_size, dependencies);                                 \
    }

GESVD_LAUNCHER_USM(float, float, LAPACKE_sgesvd_work)
GESVD_LAUNCHER_USM(double, double, LAPACKE_dgesvd_work)
GESVD_LAUNCHER_USM(std::complex<float>, float, LAPACKE_cgesvd_work)
GESVD_LAUNCHER_USM(std::complex<double>, double, LAPACKE_zgesvd_work)

#undef GESVD_LAUNCHER_USM

template <typename Func, typename T_A, typename T_B>
inline sycl::event heevd(Func func, const char* func_name, sycl::queue& queue,
                         oneapi::math::job jobz, oneapi::math::uplo uplo, std::int64_t n, T_A*& a,
                         std::int64_t lda, T_B*& w, const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T_A>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_heevd>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto w_ = reinterpret_cast<T_B*>(w);
            std::int64_t err =
                func(LAPACK_COL_MAJOR, get_job(jobz), get_fill_mode(uplo), n, a_, lda, w_);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
    return done;
}

#define HEEVD_LAUNCHER_USM(TYPE_A, TYPE_B, ROUTINE)                                               \
    sycl::event heevd(sycl::queue& queue, oneapi::math::job jobz, oneapi::math::uplo uplo,        \
                      std::int64_t n, TYPE_A* a, std::int64_t lda, TYPE_B* w, TYPE_A* scratchpad, \
                      std::int64_t scratchpad_size,                                               \
                      const std::vector<sycl::event>& dependencies) {                             \
        return heevd(ROUTINE, #ROUTINE, queue, jobz, uplo, n, a, lda, w, dependencies);           \
    }

HEEVD_LAUNCHER_USM(std::complex<float>, float, LAPACKE_cheevd)
HEEVD_LAUNCHER_USM(std::complex<double>, double, LAPACKE_zheevd)

#undef HEEVD_LAUNCHER_USM

template <typename Func, typename T_A, typename T_B>
inline sycl::event hegvd(Func func, const char* func_name, sycl::queue& queue, std::int64_t itype,
                         oneapi::math::job jobz, oneapi::math::uplo uplo, std::int64_t n, T_A*& a,
                         std::int64_t lda, T_A*& b, std::int64_t ldb, T_B*& w,
                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T_A>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_hegvd>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto b_ = reinterpret_cast<ArmDataType*>(b);
            auto w_ = reinterpret_cast<T_B*>(w);
            std::int64_t err = func(LAPACK_COL_MAJOR, itype, get_job(jobz), get_fill_mode(uplo), n,
                                    a_, lda, b_, ldb, w_);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
    return done;
}

#define HEGVD_LAUNCHER_USM(TYPE_A, TYPE_B, ROUTINE)                                         \
    sycl::event hegvd(sycl::queue& queue, std::int64_t itype, oneapi::math::job jobz,       \
                      oneapi::math::uplo uplo, std::int64_t n, TYPE_A* a, std::int64_t lda, \
                      TYPE_A* b, std::int64_t ldb, TYPE_B* w, TYPE_A* scratchpad,           \
                      std::int64_t scratchpad_size,                                         \
                      const std::vector<sycl::event>& dependencies) {                       \
        return hegvd(ROUTINE, #ROUTINE, queue, itype, jobz, uplo, n, a, lda, b, ldb, w,     \
                     dependencies);                                                         \
    }

HEGVD_LAUNCHER_USM(std::complex<float>, float, LAPACKE_chegvd)
HEGVD_LAUNCHER_USM(std::complex<double>, double, LAPACKE_zhegvd)

#undef HEGVD_LAUNCHER_USM

template <typename Func, typename T_A, typename T_B>
inline sycl::event hetrd(Func func, const char* func_name, sycl::queue& queue,
                         oneapi::math::uplo uplo, std::int64_t n, T_A* a, std::int64_t lda, T_B* d,
                         T_B* e, T_A* tau, T_A* s, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T_A>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_hetrd>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto d_ = reinterpret_cast<T_B*>(d);
            auto e_ = reinterpret_cast<T_B*>(e);
            auto tau_ = reinterpret_cast<ArmDataType*>(tau);
            auto s_ = reinterpret_cast<ArmDataType*>(s);
            std::int64_t err = func(LAPACK_COL_MAJOR, get_fill_mode(uplo), n, a_, lda, d_, e_, tau_,
                                    s_, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
    return done;
}

#define HETRD_LAUNCHER_USM(TYPE_A, TYPE_B, ROUTINE)                                            \
    sycl::event hetrd(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, TYPE_A* a,  \
                      std::int64_t lda, TYPE_B* d, TYPE_B* e, TYPE_A* tau, TYPE_A* scratchpad, \
                      std::int64_t scratchpad_size,                                            \
                      const std::vector<sycl::event>& dependencies) {                          \
        return hetrd(ROUTINE, #ROUTINE, queue, uplo, n, a, lda, d, e, tau, scratchpad,         \
                     scratchpad_size, dependencies);                                           \
    }

HETRD_LAUNCHER_USM(std::complex<float>, float, LAPACKE_chetrd_work)
HETRD_LAUNCHER_USM(std::complex<double>, double, LAPACKE_zhetrd_work)

#undef HETRD_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event hetrf(Func func, const char* func_name, sycl::queue& queue,
                         oneapi::math::uplo uplo, std::int64_t n, T* a, std::int64_t lda,
                         std::int64_t* ipiv, T* s, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_hetrf>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto ipiv_ = reinterpret_cast<std::int64_t*>(ipiv);
            auto s_ = reinterpret_cast<ArmDataType*>(s);
            std::int64_t err =
                func(LAPACK_COL_MAJOR, get_fill_mode(uplo), n, a_, lda, ipiv_, s_, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
    return done;
}

#define HETRF_LAUNCHER_USM(TYPE, ROUTINE)                                                          \
    sycl::event hetrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, TYPE* a,        \
                      std::int64_t lda, std::int64_t* ipiv, TYPE* scratchpad,                      \
                      std::int64_t scratchpad_size,                                                \
                      const std::vector<sycl::event>& dependencies) {                              \
        return hetrf(ROUTINE, #ROUTINE, queue, uplo, n, a, lda, ipiv, scratchpad, scratchpad_size, \
                     dependencies);                                                                \
    }

HETRF_LAUNCHER_USM(std::complex<float>, LAPACKE_chetrf_work)
HETRF_LAUNCHER_USM(std::complex<double>, LAPACKE_zhetrf_work)

#undef HETRF_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event orgbr(Func func, const char* func_name, sycl::queue& queue,
                         oneapi::math::generate vec, std::int64_t m, std::int64_t n, std::int64_t k,
                         T* a, std::int64_t lda, T* tau, T* s, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_orgbr>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto tau_ = reinterpret_cast<ArmDataType*>(tau);
            auto s_ = reinterpret_cast<ArmDataType*>(s);
            std::int64_t err = func(LAPACK_COL_MAJOR, get_generate(vec), m, n, k, a_, lda, tau_, s_,
                                    scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
    return done;
}

#define ORGBR_LAUNCHER_USM(TYPE, ROUTINE)                                                   \
    sycl::event orgbr(sycl::queue& queue, oneapi::math::generate vec, std::int64_t m,       \
                      std::int64_t n, std::int64_t k, TYPE* a, std::int64_t lda, TYPE* tau, \
                      TYPE* scratchpad, std::int64_t scratchpad_size,                       \
                      const std::vector<sycl::event>& dependencies) {                       \
        return orgbr(ROUTINE, #ROUTINE, queue, vec, m, n, k, a, lda, tau, scratchpad,       \
                     scratchpad_size, dependencies);                                        \
    }

ORGBR_LAUNCHER_USM(float, LAPACKE_sorgbr_work)
ORGBR_LAUNCHER_USM(double, LAPACKE_dorgbr_work)

#undef ORGBR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event orgqr(Func func, const char* func_name, sycl::queue& queue, std::int64_t m,
                         std::int64_t n, std::int64_t k, T* a, std::int64_t lda, T* tau, T* s,
                         std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_orgqr>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto tau_ = reinterpret_cast<ArmDataType*>(tau);
            auto s_ = reinterpret_cast<ArmDataType*>(s);
            std::int64_t err = func(LAPACK_COL_MAJOR, m, n, k, a_, lda, tau_, s_, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
    return done;
}

#define ORGQR_LAUNCHER_USM(TYPE, ROUTINE)                                                          \
    sycl::event orgqr(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k, TYPE* a, \
                      std::int64_t lda, TYPE* tau, TYPE* scratchpad, std::int64_t scratchpad_size, \
                      const std::vector<sycl::event>& dependencies) {                              \
        return orgqr(ROUTINE, #ROUTINE, queue, m, n, k, a, lda, tau, scratchpad, scratchpad_size,  \
                     dependencies);                                                                \
    }

ORGQR_LAUNCHER_USM(float, LAPACKE_sorgqr_work)
ORGQR_LAUNCHER_USM(double, LAPACKE_dorgqr_work)

#undef ORGQR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event orgtr(Func func, const char* func_name, sycl::queue& queue,
                         oneapi::math::uplo uplo, std::int64_t n, T* a, std::int64_t lda, T* tau,
                         T* s, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_orgtr>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto tau_ = reinterpret_cast<ArmDataType*>(tau);
            auto s_ = reinterpret_cast<ArmDataType*>(s);
            std::int64_t err =
                func(LAPACK_COL_MAJOR, get_fill_mode(uplo), n, a_, lda, tau_, s_, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
    return done;
}

#define ORGTR_LAUNCHER_USM(TYPE, ROUTINE)                                                          \
    sycl::event orgtr(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, TYPE* a,        \
                      std::int64_t lda, TYPE* tau, TYPE* scratchpad, std::int64_t scratchpad_size, \
                      const std::vector<sycl::event>& dependencies) {                              \
        return orgtr(ROUTINE, #ROUTINE, queue, uplo, n, a, lda, tau, scratchpad, scratchpad_size,  \
                     dependencies);                                                                \
    }

ORGTR_LAUNCHER_USM(float, LAPACKE_sorgtr_work)
ORGTR_LAUNCHER_USM(double, LAPACKE_dorgtr_work)

#undef ORGTR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event ormtr(Func func, const char* func_name, sycl::queue& queue,
                         oneapi::math::side side, oneapi::math::uplo uplo,
                         oneapi::math::transpose trans, std::int64_t m, std::int64_t n, T* a,
                         std::int64_t lda, T* tau, T* c, std::int64_t ldc, T* s,
                         std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_ormtr>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto tau_ = reinterpret_cast<ArmDataType*>(tau);
            auto c_ = reinterpret_cast<ArmDataType*>(c);
            auto s_ = reinterpret_cast<ArmDataType*>(s);
            std::int64_t err =
                func(LAPACK_COL_MAJOR, get_side_mode(side), get_fill_mode(uplo),
                     get_operation(trans), m, n, a_, lda, tau_, c_, ldc, s_, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
    return done;
}

#define ORMTR_LAUNCHER_USM(TYPE, ROUTINE)                                                       \
    sycl::event ormtr(sycl::queue& queue, oneapi::math::side side, oneapi::math::uplo uplo,     \
                      oneapi::math::transpose trans, std::int64_t m, std::int64_t n, TYPE* a,   \
                      std::int64_t lda, TYPE* tau, TYPE* c, std::int64_t ldc, TYPE* scratchpad, \
                      std::int64_t scratchpad_size,                                             \
                      const std::vector<sycl::event>& dependencies) {                           \
        return ormtr(ROUTINE, #ROUTINE, queue, side, uplo, trans, m, n, a, lda, tau, c, ldc,    \
                     scratchpad, scratchpad_size, dependencies);                                \
    }

ORMTR_LAUNCHER_USM(float, LAPACKE_sormtr_work)
ORMTR_LAUNCHER_USM(double, LAPACKE_dormtr_work)

#undef ORMTR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event ormrq(Func func, const char* func_name, sycl::queue& queue,
                         oneapi::math::side side, oneapi::math::transpose trans, std::int64_t m,
                         std::int64_t n, std::int64_t k, T* a, std::int64_t lda, T* tau, T* c,
                         std::int64_t ldc, T* s, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_ormrq>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto tau_ = reinterpret_cast<ArmDataType*>(tau);
            auto c_ = reinterpret_cast<ArmDataType*>(c);
            auto s_ = reinterpret_cast<ArmDataType*>(s);
            std::int64_t err = func(LAPACK_COL_MAJOR, get_side_mode(side), get_operation(trans), m,
                                    n, k, a_, lda, tau_, c_, ldc, s_, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
    return done;
}

#define ORMRQ_LAUNCHER_USM(TYPE, ROUTINE)                                                         \
    sycl::event ormrq(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans, \
                      std::int64_t m, std::int64_t n, std::int64_t k, TYPE* a, std::int64_t lda,  \
                      TYPE* tau, TYPE* c, std::int64_t ldc, TYPE* scratchpad,                     \
                      std::int64_t scratchpad_size,                                               \
                      const std::vector<sycl::event>& dependencies) {                             \
        return ormrq(ROUTINE, #ROUTINE, queue, side, trans, m, n, k, a, lda, tau, c, ldc,         \
                     scratchpad, scratchpad_size, dependencies);                                  \
    }

ORMRQ_LAUNCHER_USM(float, LAPACKE_sormrq_work)
ORMRQ_LAUNCHER_USM(double, LAPACKE_dormrq_work)

#undef ORMRQ_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event ormqr(Func func, const char* func_name, sycl::queue& queue,
                         oneapi::math::side side, oneapi::math::transpose trans, std::int64_t m,
                         std::int64_t n, std::int64_t k, T* a, std::int64_t lda, T* tau, T* c,
                         std::int64_t ldc, T* s, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_ormqr>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto tau_ = reinterpret_cast<ArmDataType*>(tau);
            auto c_ = reinterpret_cast<ArmDataType*>(c);
            auto s_ = reinterpret_cast<ArmDataType*>(s);
            std::int64_t err = func(LAPACK_COL_MAJOR, get_side_mode(side), get_operation(trans), m,
                                    n, k, a_, lda, tau_, c_, ldc, s_, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
    return done;
}

#define ORMQR_LAUNCHER_USM(TYPE, ROUTINE)                                                         \
    sycl::event ormqr(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans, \
                      std::int64_t m, std::int64_t n, std::int64_t k, TYPE* a, std::int64_t lda,  \
                      TYPE* tau, TYPE* c, std::int64_t ldc, TYPE* scratchpad,                     \
                      std::int64_t scratchpad_size,                                               \
                      const std::vector<sycl::event>& dependencies) {                             \
        return ormqr(ROUTINE, #ROUTINE, queue, side, trans, m, n, k, a, lda, tau, c, ldc,         \
                     scratchpad, scratchpad_size, dependencies);                                  \
    }

ORMQR_LAUNCHER_USM(float, LAPACKE_sormqr_work)
ORMQR_LAUNCHER_USM(double, LAPACKE_dormqr_work)

#undef ORMQR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event potrf(Func func, const char* func_name, sycl::queue& queue,
                         oneapi::math::uplo uplo, std::int64_t n, T* a, std::int64_t lda,
                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_potrf>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            std::int64_t err = func(LAPACK_COL_MAJOR, get_fill_mode(uplo), n, a_, lda);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
    return done;
}

#define POTRF_LAUNCHER_USM(TYPE, ROUTINE)                                                   \
    sycl::event potrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, TYPE* a, \
                      std::int64_t lda, TYPE* scratchpad, std::int64_t scratchpad_size,     \
                      const std::vector<sycl::event>& dependencies) {                       \
        return potrf(ROUTINE, #ROUTINE, queue, uplo, n, a, lda, dependencies);              \
    }

POTRF_LAUNCHER_USM(float, LAPACKE_spotrf_work)
POTRF_LAUNCHER_USM(double, LAPACKE_dpotrf_work)
POTRF_LAUNCHER_USM(std::complex<float>, LAPACKE_cpotrf_work)
POTRF_LAUNCHER_USM(std::complex<double>, LAPACKE_zpotrf_work)

#undef POTRF_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event potri(Func func, const char* func_name, sycl::queue& queue,
                         oneapi::math::uplo uplo, std::int64_t n, T* a, std::int64_t lda,
                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_potri>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);

            std::int64_t err = func(LAPACK_COL_MAJOR, get_fill_mode(uplo), n, a_, lda);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
    return done;
}

#define POTRI_LAUNCHER_USM(TYPE, ROUTINE)                                                   \
    sycl::event potri(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, TYPE* a, \
                      std::int64_t lda, TYPE* scratchpad, std::int64_t scratchpad_size,     \
                      const std::vector<sycl::event>& dependencies) {                       \
        return potri(ROUTINE, #ROUTINE, queue, uplo, n, a, lda, dependencies);              \
    }

POTRI_LAUNCHER_USM(float, LAPACKE_spotri_work)
POTRI_LAUNCHER_USM(double, LAPACKE_dpotri_work)
POTRI_LAUNCHER_USM(std::complex<float>, LAPACKE_cpotri_work)
POTRI_LAUNCHER_USM(std::complex<double>, LAPACKE_zpotri_work)

#undef POTRI_LAUNCHER_USM

// cusolverDnXpotrs does not use scratchpad memory
template <typename Func, typename T>
inline sycl::event potrs(Func func, const char* func_name, sycl::queue& queue,
                         oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs, T* a,
                         std::int64_t lda, T* b, std::int64_t ldb,
                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_potrs>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto b_ = reinterpret_cast<ArmDataType*>(b);
            std::int64_t err =
                func(LAPACK_COL_MAJOR, get_fill_mode(uplo), n, nrhs, a_, lda, b_, ldb);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
    return done;
}

#define POTRS_LAUNCHER_USM(TYPE, ROUTINE)                                                      \
    sycl::event potrs(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,             \
                      std::int64_t nrhs, TYPE* a, std::int64_t lda, TYPE* b, std::int64_t ldb, \
                      TYPE* scratchpad, std::int64_t scratchpad_size,                          \
                      const std::vector<sycl::event>& dependencies) {                          \
        return potrs(ROUTINE, #ROUTINE, queue, uplo, n, nrhs, a, lda, b, ldb, dependencies);   \
    }

POTRS_LAUNCHER_USM(float, LAPACKE_spotrs_work)
POTRS_LAUNCHER_USM(double, LAPACKE_dpotrs_work)
POTRS_LAUNCHER_USM(std::complex<float>, LAPACKE_cpotrs_work)
POTRS_LAUNCHER_USM(std::complex<double>, LAPACKE_zpotrs_work)

#undef POTRS_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event syevd(Func func, const char* func_name, sycl::queue& queue,
                         oneapi::math::job jobz, oneapi::math::uplo uplo, std::int64_t n, T* a,
                         std::int64_t lda, T* w, const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_syevd>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto w_ = reinterpret_cast<ArmDataType*>(w);
            std::int64_t err =
                func(LAPACK_COL_MAJOR, get_job(jobz), get_fill_mode(uplo), n, a_, lda, w_);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
    return done;
}

#define SYEVD_LAUNCHER_USM(TYPE, ROUTINE)                                                   \
    sycl::event syevd(sycl::queue& queue, oneapi::math::job jobz, oneapi::math::uplo uplo,  \
                      std::int64_t n, TYPE* a, std::int64_t lda, TYPE* w, TYPE* scratchpad, \
                      std::int64_t scratchpad_size,                                         \
                      const std::vector<sycl::event>& dependencies) {                       \
        return syevd(ROUTINE, #ROUTINE, queue, jobz, uplo, n, a, lda, w, dependencies);     \
    }

SYEVD_LAUNCHER_USM(float, LAPACKE_ssyevd)
SYEVD_LAUNCHER_USM(double, LAPACKE_dsyevd)

#undef SYEVD_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event sygvd(Func func, const char* func_name, sycl::queue& queue, std::int64_t itype,
                         oneapi::math::job jobz, oneapi::math::uplo uplo, std::int64_t n, T* a,
                         std::int64_t lda, T* b, std::int64_t ldb, T* w,
                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_sygvd>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto b_ = reinterpret_cast<ArmDataType*>(b);
            auto w_ = reinterpret_cast<ArmDataType*>(w);
            std::int64_t err = func(LAPACK_COL_MAJOR, itype, get_job(jobz), get_fill_mode(uplo), n,
                                    a_, lda, b_, ldb, w_);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
    return done;
}

#define SYGVD_LAUNCHER_USM(TYPE, ROUTINE)                                                          \
    sycl::event sygvd(sycl::queue& queue, std::int64_t itype, oneapi::math::job jobz,              \
                      oneapi::math::uplo uplo, std::int64_t n, TYPE* a, std::int64_t lda, TYPE* b, \
                      std::int64_t ldb, TYPE* w, TYPE* scratchpad, std::int64_t scratchpad_size,   \
                      const std::vector<sycl::event>& dependencies) {                              \
        return sygvd(ROUTINE, #ROUTINE, queue, itype, jobz, uplo, n, a, lda, b, ldb, w,            \
                     dependencies);                                                                \
    }

SYGVD_LAUNCHER_USM(float, LAPACKE_ssygvd)
SYGVD_LAUNCHER_USM(double, LAPACKE_dsygvd)

#undef SYGVD_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event sytrd(Func func, const char* func_name, sycl::queue& queue,
                         oneapi::math::uplo uplo, std::int64_t n, T* a, std::int64_t lda, T* d,
                         T* e, T* tau, T* s, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_sytrd>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto d_ = reinterpret_cast<ArmDataType*>(d);
            auto e_ = reinterpret_cast<ArmDataType*>(e);
            auto tau_ = reinterpret_cast<ArmDataType*>(tau);
            auto s_ = reinterpret_cast<ArmDataType*>(s);

            std::int64_t err = func(LAPACK_COL_MAJOR, get_fill_mode(uplo), n, a_, lda, d_, e_, tau_,
                                    s_, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
    return done;
}

#define SYTRD_LAUNCHER_USM(TYPE, ROUTINE)                                                   \
    sycl::event sytrd(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, TYPE* a, \
                      std::int64_t lda, TYPE* d, TYPE* e, TYPE* tau, TYPE* scratchpad,      \
                      std::int64_t scratchpad_size,                                         \
                      const std::vector<sycl::event>& dependencies) {                       \
        return sytrd(ROUTINE, #ROUTINE, queue, uplo, n, a, lda, d, e, tau, scratchpad,      \
                     scratchpad_size, dependencies);                                        \
    }

SYTRD_LAUNCHER_USM(float, LAPACKE_ssytrd_work)
SYTRD_LAUNCHER_USM(double, LAPACKE_dsytrd_work)

#undef SYTRD_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event sytrf(Func func, const char* func_name, sycl::queue& queue,
                         oneapi::math::uplo uplo, std::int64_t n, T* a, std::int64_t lda,
                         std::int64_t* ipiv, T* s, std::int64_t scratchpad_size,

                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_sytrf>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto ipiv_ = reinterpret_cast<int64_t*>(ipiv);
            auto s_ = reinterpret_cast<ArmDataType*>(s);

            std::int64_t err =
                func(LAPACK_COL_MAJOR, get_fill_mode(uplo), n, a_, lda, ipiv_, s_, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });

    return done;
}

#define SYTRF_LAUNCHER_USM(TYPE, ROUTINE)                                                          \
    sycl::event sytrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, TYPE* a,        \
                      std::int64_t lda, std::int64_t* ipiv, TYPE* scratchpad,                      \
                      std::int64_t scratchpad_size,                                                \
                      const std::vector<sycl::event>& dependencies) {                              \
        return sytrf(ROUTINE, #ROUTINE, queue, uplo, n, a, lda, ipiv, scratchpad, scratchpad_size, \
                     dependencies);                                                                \
    }

SYTRF_LAUNCHER_USM(float, LAPACKE_ssytrf_work)
SYTRF_LAUNCHER_USM(double, LAPACKE_dsytrf_work)
SYTRF_LAUNCHER_USM(std::complex<float>, LAPACKE_csytrf_work)
SYTRF_LAUNCHER_USM(std::complex<double>, LAPACKE_zsytrf_work)

#undef SYTRF_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event trtrs(Func func, const char* func_name, sycl::queue& queue,
                         oneapi::math::uplo uplo, oneapi::math::transpose trans,
                         oneapi::math::diag diag, std::int64_t n, std::int64_t nrhs, T* a,
                         std::int64_t lda, T* b, std::int64_t ldb,
                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_trtrs>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto b_ = reinterpret_cast<ArmDataType*>(b);
            std::int64_t err = func(LAPACK_COL_MAJOR, get_fill_mode(uplo), get_operation(trans),
                                    get_diag(diag), n, nrhs, a_, lda, b_, ldb);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });

    return done;
}

#define TRTRS_LAUNCHER_USM(TYPE, ROUTINE)                                                         \
    sycl::event trtrs(sycl::queue& queue, oneapi::math::uplo uplo, oneapi::math::transpose trans, \
                      oneapi::math::diag diag, std::int64_t n, std::int64_t nrhs, TYPE* a,        \
                      std::int64_t lda, TYPE* b, std::int64_t ldb, TYPE* scratchpad,              \
                      std::int64_t scratchpad_size,                                               \
                      const std::vector<sycl::event>& dependencies) {                             \
        return trtrs(ROUTINE, #ROUTINE, queue, uplo, trans, diag, n, nrhs, a, lda, b, ldb,        \
                     dependencies);                                                               \
    }

TRTRS_LAUNCHER_USM(float, LAPACKE_strtrs)
TRTRS_LAUNCHER_USM(double, LAPACKE_dtrtrs)
TRTRS_LAUNCHER_USM(std::complex<float>, LAPACKE_ctrtrs)
TRTRS_LAUNCHER_USM(std::complex<double>, LAPACKE_ztrtrs)

#undef TRTRS_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event ungbr(Func func, const char* func_name, sycl::queue& queue,
                         oneapi::math::generate vec, std::int64_t m, std::int64_t n, std::int64_t k,
                         T* a, std::int64_t lda, T* tau, T* s, std::int64_t scratchpad_size,

                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_ungbr>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto tau_ = reinterpret_cast<ArmDataType*>(tau);
            auto s_ = reinterpret_cast<ArmDataType*>(s);

            std::int64_t err = func(LAPACK_COL_MAJOR, get_generate(vec), m, n, k, a_, lda, tau_, s_,
                                    scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
    return done;
}

#define UNGBR_LAUNCHER_USM(TYPE, ROUTINE)                                                   \
    sycl::event ungbr(sycl::queue& queue, oneapi::math::generate vec, std::int64_t m,       \
                      std::int64_t n, std::int64_t k, TYPE* a, std::int64_t lda, TYPE* tau, \
                      TYPE* scratchpad, std::int64_t scratchpad_size,                       \
                      const std::vector<sycl::event>& dependencies) {                       \
        return ungbr(ROUTINE, #ROUTINE, queue, vec, m, n, k, a, lda, tau, scratchpad,       \
                     scratchpad_size, dependencies);                                        \
    }

UNGBR_LAUNCHER_USM(std::complex<float>, LAPACKE_cungbr_work)
UNGBR_LAUNCHER_USM(std::complex<double>, LAPACKE_zungbr_work)

#undef UNGBR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event ungqr(Func func, const char* func_name, sycl::queue& queue, std::int64_t m,
                         std::int64_t n, std::int64_t k, T* a, std::int64_t lda, T* tau, T* s,
                         std::int64_t scratchpad_size,

                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_ungqr>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto tau_ = reinterpret_cast<ArmDataType*>(tau);
            auto s_ = reinterpret_cast<ArmDataType*>(s);

            std::int64_t err = func(LAPACK_COL_MAJOR, m, n, k, a_, lda, tau_, s_, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
    return done;
}

#define UNGQR_LAUNCHER_USM(TYPE, ROUTINE)                                                          \
    sycl::event ungqr(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k, TYPE* a, \
                      std::int64_t lda, TYPE* tau, TYPE* scratchpad, std::int64_t scratchpad_size, \
                      const std::vector<sycl::event>& dependencies) {                              \
        return ungqr(ROUTINE, #ROUTINE, queue, m, n, k, a, lda, tau, scratchpad, scratchpad_size,  \
                     dependencies);                                                                \
    }

UNGQR_LAUNCHER_USM(std::complex<float>, LAPACKE_cungqr_work)
UNGQR_LAUNCHER_USM(std::complex<double>, LAPACKE_zungqr_work)

#undef UNGQR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event ungtr(Func func, const char* func_name, sycl::queue& queue,
                         oneapi::math::uplo uplo, std::int64_t n, T* a, std::int64_t lda, T* tau,
                         T* s, std::int64_t scratchpad_size,

                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_ungtr>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto tau_ = reinterpret_cast<ArmDataType*>(tau);
            auto s_ = reinterpret_cast<ArmDataType*>(s);

            std::int64_t err =
                func(LAPACK_COL_MAJOR, get_fill_mode(uplo), n, a_, lda, tau_, s_, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
    return done;
}

#define UNGTR_LAUNCHER_USM(TYPE, ROUTINE)                                                          \
    sycl::event ungtr(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, TYPE* a,        \
                      std::int64_t lda, TYPE* tau, TYPE* scratchpad, std::int64_t scratchpad_size, \
                      const std::vector<sycl::event>& dependencies) {                              \
        return ungtr(ROUTINE, #ROUTINE, queue, uplo, n, a, lda, tau, scratchpad, scratchpad_size,  \
                     dependencies);                                                                \
    }

UNGTR_LAUNCHER_USM(std::complex<float>, LAPACKE_cungtr_work)
UNGTR_LAUNCHER_USM(std::complex<double>, LAPACKE_zungtr_work)

#undef UNGTR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event unmrq(Func func, const char* func_name, sycl::queue& queue,
                         oneapi::math::side side, oneapi::math::transpose trans, std::int64_t m,
                         std::int64_t n, std::int64_t k, T* a, std::int64_t lda, T* tau, T* c,
                         std::int64_t ldc, T* s, std::int64_t scratchpad_size,

                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_unmrq>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto tau_ = reinterpret_cast<ArmDataType*>(tau);
            auto c_ = reinterpret_cast<ArmDataType*>(c);
            auto s_ = reinterpret_cast<ArmDataType*>(s);

            std::int64_t err = func(LAPACK_COL_MAJOR, get_side_mode(side), get_operation(trans), m,
                                    n, k, a_, lda, tau_, c_, ldc, s_, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
    return done;
}

#define UNMRQ_LAUNCHER_USM(TYPE, ROUTINE)                                                         \
    sycl::event unmrq(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans, \
                      std::int64_t m, std::int64_t n, std::int64_t k, TYPE* a, std::int64_t lda,  \
                      TYPE* tau, TYPE* c, std::int64_t ldc, TYPE* scratchpad,                     \
                      std::int64_t scratchpad_size,                                               \
                      const std::vector<sycl::event>& dependencies) {                             \
        return unmrq(ROUTINE, #ROUTINE, queue, side, trans, m, n, k, a, lda, tau, c, ldc,         \
                     scratchpad, scratchpad_size, dependencies);                                  \
    }

UNMRQ_LAUNCHER_USM(std::complex<float>, LAPACKE_cunmrq_work)
UNMRQ_LAUNCHER_USM(std::complex<double>, LAPACKE_zunmrq_work)

#undef UNMRQ_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event unmqr(Func func, const char* func_name, sycl::queue& queue,
                         oneapi::math::side side, oneapi::math::transpose trans, std::int64_t m,
                         std::int64_t n, std::int64_t k, T* a, std::int64_t lda, T* tau, T* c,
                         std::int64_t ldc, T* s, std::int64_t scratchpad_size,

                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_unmqr>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto tau_ = reinterpret_cast<ArmDataType*>(tau);
            auto c_ = reinterpret_cast<ArmDataType*>(c);
            auto s_ = reinterpret_cast<ArmDataType*>(s);

            std::int64_t err = func(LAPACK_COL_MAJOR, get_side_mode(side), get_operation(trans), m,
                                    n, k, a_, lda, tau_, c_, ldc, s_, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
    return done;
}

#define UNMQR_LAUNCHER_USM(TYPE, ROUTINE)                                                         \
    sycl::event unmqr(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans, \
                      std::int64_t m, std::int64_t n, std::int64_t k, TYPE* a, std::int64_t lda,  \
                      TYPE* tau, TYPE* c, std::int64_t ldc, TYPE* scratchpad,                     \
                      std::int64_t scratchpad_size,                                               \
                      const std::vector<sycl::event>& dependencies) {                             \
        return unmqr(ROUTINE, #ROUTINE, queue, side, trans, m, n, k, a, lda, tau, c, ldc,         \
                     scratchpad, scratchpad_size, dependencies);                                  \
    }

UNMQR_LAUNCHER_USM(std::complex<float>, LAPACKE_cunmqr_work)
UNMQR_LAUNCHER_USM(std::complex<double>, LAPACKE_zunmqr_work)

#undef UNMQR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event unmtr(Func func, const char* func_name, sycl::queue& queue,
                         oneapi::math::side side, oneapi::math::uplo uplo,
                         oneapi::math::transpose trans, std::int64_t m, std::int64_t n, T* a,
                         std::int64_t lda, T* tau, T* c, std::int64_t ldc, T* s,
                         std::int64_t scratchpad_size,

                         const std::vector<sycl::event>& dependencies) {
    using ArmDataType = typename ArmEquivalentType<T>::Type;
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_unmtr>(cgh, [=]() {
            auto a_ = reinterpret_cast<ArmDataType*>(a);
            auto tau_ = reinterpret_cast<ArmDataType*>(tau);
            auto c_ = reinterpret_cast<ArmDataType*>(c);
            auto s_ = reinterpret_cast<ArmDataType*>(s);

            std::int64_t err =
                func(LAPACK_COL_MAJOR, get_side_mode(side), get_fill_mode(uplo),
                     get_operation(trans), m, n, a_, lda, tau_, c_, ldc, s_, scratchpad_size);
            if (err != 0) {
                throw armpl_lapacke_error(func_name, err);
            }
        });
    });
    return done;
}

#define UNMTR_LAUNCHER_USM(TYPE, ROUTINE)                                                       \
    sycl::event unmtr(sycl::queue& queue, oneapi::math::side side, oneapi::math::uplo uplo,     \
                      oneapi::math::transpose trans, std::int64_t m, std::int64_t n, TYPE* a,   \
                      std::int64_t lda, TYPE* tau, TYPE* c, std::int64_t ldc, TYPE* scratchpad, \
                      std::int64_t scratchpad_size,                                             \
                      const std::vector<sycl::event>& dependencies) {                           \
        return unmtr(ROUTINE, #ROUTINE, queue, side, uplo, trans, m, n, a, lda, tau, c, ldc,    \
                     scratchpad, scratchpad_size, dependencies);                                \
    }

UNMTR_LAUNCHER_USM(std::complex<float>, LAPACKE_cunmtr_work)
UNMTR_LAUNCHER_USM(std::complex<double>, LAPACKE_zunmtr_work)

#undef UNMTR_LAUNCHER_USM

// SCRATCHPAD APIs
template <typename Func, typename TYPE>
inline void gebrd_scratchpad_size(Func func, const char* func_name, sycl::queue& queue,
                                  std::int64_t m, std::int64_t n, std::int64_t lda,
                                  TYPE* work_query) {
    using ArmDataType = typename ArmEquivalentType<TYPE>::Type;
    queue
        .submit(
            [&](sycl::handler&
                    cgh) { //auto w_acc = work_query.template get_access<sycl::access::mode::read_write>(cgh);
                host_task<class armpl_kernel_gebrd_scratchpad_size>(cgh, [=]() {
                    std::int64_t err = func(LAPACK_COL_MAJOR, m, n, nullptr, lda, nullptr, nullptr,
                                            nullptr, nullptr, work_query, -1);
                    if (err != 0) {
                        throw armpl_lapacke_error(func_name, err);
                    }
                });
            })
        .wait();
}

#define GEBRD_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                                     \
    template <>                                                                                   \
    std::int64_t gebrd_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t m, std::int64_t n, \
                                             std::int64_t lda) {                                  \
        using ArmDataType = typename ArmEquivalentType<TYPE>::Type;                               \
        ArmDataType work_query;                                                                   \
        gebrd_scratchpad_size(ROUTINE, #ROUTINE, queue, m, n, lda, &work_query);                  \
        return cast_to_int_if_complex(work_query);                                                \
    }

GEBRD_LAUNCHER_SCRATCH(float, LAPACKE_sgebrd_work)
GEBRD_LAUNCHER_SCRATCH(double, LAPACKE_dgebrd_work)
GEBRD_LAUNCHER_SCRATCH(std::complex<float>, LAPACKE_cgebrd_work)
GEBRD_LAUNCHER_SCRATCH(std::complex<double>, LAPACKE_zgebrd_work)

#undef GEBRD_LAUNCHER_SCRATCH

template <typename Func, typename TYPE>
inline void gerqf_scratchpad_size(Func func, const char* func_name, sycl::queue& queue,
                                  std::int64_t m, std::int64_t n, std::int64_t lda,
                                  TYPE* work_query) {
    using ArmDataType = typename ArmEquivalentType<TYPE>::Type;
    queue
        .submit([&](sycl::handler& cgh) {
            host_task<class armpl_kernel_gerqf_scratchpad_size>(cgh, [=]() {
                std::int64_t err =
                    func(LAPACK_COL_MAJOR, m, n, nullptr, lda, nullptr, work_query, -1);
                if (err != 0) {
                    throw armpl_lapacke_error(func_name, err);
                }
            });
        })
        .wait();
}

#define GERQF_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                                     \
    template <>                                                                                   \
    std::int64_t gerqf_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t m, std::int64_t n, \
                                             std::int64_t lda) {                                  \
        using ArmDataType = typename ArmEquivalentType<TYPE>::Type;                               \
        ArmDataType work_query;                                                                   \
        gerqf_scratchpad_size(ROUTINE, #ROUTINE, queue, m, n, lda, &work_query);                  \
        return cast_to_int_if_complex(work_query);                                                \
    }

GERQF_LAUNCHER_SCRATCH(float, LAPACKE_sgerqf_work)
GERQF_LAUNCHER_SCRATCH(double, LAPACKE_dgerqf_work)
GERQF_LAUNCHER_SCRATCH(std::complex<float>, LAPACKE_cgerqf_work)
GERQF_LAUNCHER_SCRATCH(std::complex<double>, LAPACKE_zgerqf_work)

template <typename Func, typename TYPE>
inline void geqrf_scratchpad_size(Func func, const char* func_name, sycl::queue& queue,
                                  std::int64_t m, std::int64_t n, std::int64_t lda,
                                  TYPE* work_query) {
    using ArmDataType = typename ArmEquivalentType<TYPE>::Type;
    queue
        .submit([&](sycl::handler& cgh) {
            host_task<class armpl_kernel_geqrf_scratchpad_size>(cgh, [=]() {
                std::int64_t err =
                    func(LAPACK_COL_MAJOR, m, n, nullptr, lda, nullptr, work_query, -1);
                if (err != 0) {
                    throw armpl_lapacke_error(func_name, err);
                }
            });
        })
        .wait();
}

#define GEQRF_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                                     \
    template <>                                                                                   \
    std::int64_t geqrf_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t m, std::int64_t n, \
                                             std::int64_t lda) {                                  \
        using ArmDataType = typename ArmEquivalentType<TYPE>::Type;                               \
        ArmDataType work_query;                                                                   \
        geqrf_scratchpad_size(ROUTINE, #ROUTINE, queue, m, n, lda, &work_query);                  \
        return cast_to_int_if_complex(work_query);                                                \
    }

GEQRF_LAUNCHER_SCRATCH(float, LAPACKE_sgeqrf_work)
GEQRF_LAUNCHER_SCRATCH(double, LAPACKE_dgeqrf_work)
GEQRF_LAUNCHER_SCRATCH(std::complex<float>, LAPACKE_cgeqrf_work)
GEQRF_LAUNCHER_SCRATCH(std::complex<double>, LAPACKE_zgeqrf_work)

#undef GEQRF_LAUNCHER_SCRATCH

template <typename Func, typename TYPE>
inline void gesvd_scratchpad_size(Func func, const char* func_name, sycl::queue& queue,
                                  oneapi::math::jobsvd jobu, oneapi::math::jobsvd jobvt,
                                  std::int64_t m, std::int64_t n, std::int64_t lda,
                                  std::int64_t ldu, std::int64_t ldvt, TYPE* work_query) {
    using ArmDataType = typename ArmEquivalentType<TYPE>::Type;
    queue
        .submit([&](sycl::handler& cgh) {
            host_task<class armpl_kernel_gesvd_scratchpad_size>(cgh, [=]() {
                std::int64_t err = 0;
                if constexpr (is_complex<TYPE>) {
                    err = func(LAPACK_COL_MAJOR, get_jobsvd(jobu), get_jobsvd(jobvt), m, n, nullptr,
                               lda, nullptr, nullptr, ldu, nullptr, ldvt, work_query, -1, nullptr);
                }
                else {
                    err = func(LAPACK_COL_MAJOR, get_jobsvd(jobu), get_jobsvd(jobvt), m, n, nullptr,
                               lda, nullptr, nullptr, ldu, nullptr, ldvt, work_query, -1);
                }
                if (err != 0) {
                    throw armpl_lapacke_error(func_name, err);
                }
            });
        })
        .wait();
}

#define GESVD_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                                    \
    template <>                                                                                  \
    std::int64_t gesvd_scratchpad_size<TYPE>(                                                    \
        sycl::queue & queue, oneapi::math::jobsvd jobu, oneapi::math::jobsvd jobvt,              \
        std::int64_t m, std::int64_t n, std::int64_t lda, std::int64_t ldu, std::int64_t ldvt) { \
        using ArmDataType = typename ArmEquivalentType<TYPE>::Type;                              \
        ArmDataType work_query;                                                                  \
        gesvd_scratchpad_size(ROUTINE, #ROUTINE, queue, jobu, jobvt, m, n, lda, ldu, ldvt,       \
                              &work_query);                                                      \
        return cast_to_int_if_complex(work_query);                                               \
    }

GESVD_LAUNCHER_SCRATCH(float, LAPACKE_sgesvd_work)
GESVD_LAUNCHER_SCRATCH(double, LAPACKE_dgesvd_work)
GESVD_LAUNCHER_SCRATCH(std::complex<float>, LAPACKE_cgesvd_work)
GESVD_LAUNCHER_SCRATCH(std::complex<double>, LAPACKE_zgesvd_work)

#undef GESVD_LAUNCHER_SCRATCH

#define GETRF_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                                     \
    template <>                                                                                   \
    std::int64_t getrf_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t m, std::int64_t n, \
                                             std::int64_t lda) {                                  \
        return 0;                                                                                 \
    }

GETRF_LAUNCHER_SCRATCH(float, LAPACKE_sgetrf_work)
GETRF_LAUNCHER_SCRATCH(double, LAPACKE_dgetrf_work)
GETRF_LAUNCHER_SCRATCH(std::complex<float>, LAPACKE_cgetrf_work)
GETRF_LAUNCHER_SCRATCH(std::complex<double>, LAPACKE_zgetrf_work)

#undef GETRF_LAUNCHER_SCRATCH

template <typename Func, typename TYPE>
inline void getri_scratchpad_size(Func func, const char* func_name, sycl::queue& queue,
                                  std::int64_t n, std::int64_t lda, TYPE* work_query) {
    using ArmDataType = typename ArmEquivalentType<TYPE>::Type;
    queue
        .submit([&](sycl::handler& cgh) {
            host_task<class armpl_kernel_getri_scratchpad_size>(cgh, [=]() {
                std::int64_t err = func(LAPACK_COL_MAJOR, n, nullptr, lda, nullptr, work_query, -1);
                if (err != 0) {
                    throw armpl_lapacke_error(func_name, err);
                }
            });
        })
        .wait();
}

#define GETRI_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                     \
    template <>                                                                   \
    std::int64_t getri_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t n, \
                                             std::int64_t lda) {                  \
        using ArmDataType = typename ArmEquivalentType<TYPE>::Type;               \
        ArmDataType work_query;                                                   \
        getri_scratchpad_size(ROUTINE, #ROUTINE, queue, n, lda, &work_query);     \
        return cast_to_int_if_complex(work_query);                                \
    }

GETRI_LAUNCHER_SCRATCH(float, LAPACKE_sgetri_work)
GETRI_LAUNCHER_SCRATCH(double, LAPACKE_dgetri_work)
GETRI_LAUNCHER_SCRATCH(std::complex<float>, LAPACKE_cgetri_work)
GETRI_LAUNCHER_SCRATCH(std::complex<double>, LAPACKE_zgetri_work)

#undef GETRI_LAUNCHER_SCRATCH

#define GETRS_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                                     \
    template <>                                                                                   \
    std::int64_t getrs_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::transpose trans,  \
                                             std::int64_t n, std::int64_t nrhs, std::int64_t lda, \
                                             std::int64_t ldb) {                                  \
        return 0;                                                                                 \
    }

GETRS_LAUNCHER_SCRATCH(float, LAPACKE_sgetrs_work)
GETRS_LAUNCHER_SCRATCH(double, LAPACKE_dgetrs_work)
GETRS_LAUNCHER_SCRATCH(std::complex<float>, LAPACKE_cgetrs_work)
GETRS_LAUNCHER_SCRATCH(std::complex<double>, LAPACKE_zgetrs_work)

#undef GETRS_LAUNCHER_SCRATCH

//These calls uses three separate work array, querying a single value is hard, stick to self contained call
#define HEEVD_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                             \
    template <>                                                                           \
    std::int64_t heevd_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::job jobz, \
                                             oneapi::math::uplo uplo, std::int64_t n,     \
                                             std::int64_t lda) {                          \
        return 0;                                                                         \
    }

HEEVD_LAUNCHER_SCRATCH(std::complex<float>, LAPACKE_cheevd)
HEEVD_LAUNCHER_SCRATCH(std::complex<double>, LAPACKE_zheevd)

#undef HEEVD_LAUNCHER_SCRATCH

#define HEGVD_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                                      \
    template <>                                                                                    \
    std::int64_t hegvd_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t itype,              \
                                             oneapi::math::job jobz, oneapi::math::uplo uplo,      \
                                             std::int64_t n, std::int64_t lda, std::int64_t ldb) { \
        return 0;                                                                                  \
    }

HEGVD_LAUNCHER_SCRATCH(std::complex<float>, LAPACKE_chegvd)
HEGVD_LAUNCHER_SCRATCH(std::complex<double>, LAPACKE_zhegvd)

#undef HEGVD_LAUNCHER_SCRATCH

template <typename Func, typename TYPE>
inline void hetrd_scratchpad_size(Func func, const char* func_name, sycl::queue& queue,
                                  oneapi::math::uplo uplo, std::int64_t n, std::int64_t lda,
                                  TYPE* work_query) {
    using ArmDataType = typename ArmEquivalentType<TYPE>::Type;
    queue
        .submit([&](sycl::handler& cgh) {
            host_task<class armpl_kernel_hetrd_scratchpad_size>(cgh, [=]() {
                std::int64_t err = func(LAPACK_COL_MAJOR, get_fill_mode(uplo), n, nullptr, lda,
                                        nullptr, nullptr, nullptr, work_query, -1);
                if (err != 0) {
                    throw armpl_lapacke_error(func_name, err);
                }
            });
        })
        .wait();
}

#define HETRD_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                              \
    template <>                                                                            \
    std::int64_t hetrd_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::uplo uplo, \
                                             std::int64_t n, std::int64_t lda) {           \
        using ArmDataType = typename ArmEquivalentType<TYPE>::Type;                        \
        ArmDataType work_query;                                                            \
        hetrd_scratchpad_size(ROUTINE, #ROUTINE, queue, uplo, n, lda, &work_query);        \
        return cast_to_int_if_complex(work_query);                                         \
    }

HETRD_LAUNCHER_SCRATCH(std::complex<float>, LAPACKE_chetrd_work)
HETRD_LAUNCHER_SCRATCH(std::complex<double>, LAPACKE_zhetrd_work)

#undef HETRD_LAUNCHER_SCRATCH

template <typename Func, typename TYPE>
inline void hetrf_scratchpad_size(Func func, const char* func_name, sycl::queue& queue,
                                  oneapi::math::uplo uplo, std::int64_t n, std::int64_t lda,
                                  TYPE* work_query) {
    using ArmDataType = typename ArmEquivalentType<TYPE>::Type;
    queue
        .submit([&](sycl::handler& cgh) {
            host_task<class armpl_kernel_hetrf_scratchpad_size>(cgh, [=]() {
                std::int64_t err = func(LAPACK_COL_MAJOR, get_fill_mode(uplo), n, nullptr, lda,
                                        nullptr, work_query, -1);
                if (err != 0) {
                    throw armpl_lapacke_error(func_name, err);
                }
            });
        })
        .wait();
}

#define HETRF_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                              \
    template <>                                                                            \
    std::int64_t hetrf_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::uplo uplo, \
                                             std::int64_t n, std::int64_t lda) {           \
        using ArmDataType = typename ArmEquivalentType<TYPE>::Type;                        \
        ArmDataType work_query;                                                            \
        hetrf_scratchpad_size(ROUTINE, #ROUTINE, queue, uplo, n, lda, &work_query);        \
        return cast_to_int_if_complex(work_query);                                         \
    }

HETRF_LAUNCHER_SCRATCH(std::complex<float>, LAPACKE_chetrf_work)
HETRF_LAUNCHER_SCRATCH(std::complex<double>, LAPACKE_zhetrf_work)
#undef HETRF_LAUNCHER_SCRATCH

template <typename Func, typename TYPE>
inline void orgbr_scratchpad_size(Func func, const char* func_name, sycl::queue& queue,
                                  oneapi::math::generate vec, std::int64_t m, std::int64_t n,
                                  std::int64_t k, std::int64_t lda, TYPE* work_query) {
    using ArmDataType = typename ArmEquivalentType<TYPE>::Type;
    queue
        .submit([&](sycl::handler& cgh) {
            host_task<class armpl_kernel_orgbr_scratchpad_size>(cgh, [=]() {
                std::int64_t err = func(LAPACK_COL_MAJOR, get_generate(vec), m, n, k, nullptr, lda,
                                        nullptr, work_query, -1);
                if (err != 0) {
                    throw armpl_lapacke_error(func_name, err);
                }
            });
        })
        .wait();
}

#define ORGBR_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                                 \
    template <>                                                                               \
    std::int64_t orgbr_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::generate vec, \
                                             std::int64_t m, std::int64_t n, std::int64_t k,  \
                                             std::int64_t lda) {                              \
        using ArmDataType = typename ArmEquivalentType<TYPE>::Type;                           \
        ArmDataType work_query;                                                               \
        orgbr_scratchpad_size(ROUTINE, #ROUTINE, queue, vec, m, n, k, lda, &work_query);      \
        return cast_to_int_if_complex(work_query);                                            \
    }

ORGBR_LAUNCHER_SCRATCH(float, LAPACKE_sorgbr_work)
ORGBR_LAUNCHER_SCRATCH(double, LAPACKE_dorgbr_work)

#undef ORGBR_LAUNCHER_SCRATCH

template <typename Func, typename TYPE>
inline void orgtr_scratchpad_size(Func func, const char* func_name, sycl::queue& queue,
                                  oneapi::math::uplo uplo, std::int64_t n, std::int64_t lda,
                                  TYPE* work_query) {
    using ArmDataType = typename ArmEquivalentType<TYPE>::Type;
    queue
        .submit([&](sycl::handler& cgh) {
            host_task<class armpl_kernel_orgtr_scratchpad_size>(cgh, [=]() {
                std::int64_t err = func(LAPACK_COL_MAJOR, get_fill_mode(uplo), n, nullptr, lda,
                                        nullptr, work_query, -1);
                if (err != 0) {
                    throw armpl_lapacke_error(func_name, err);
                }
            });
        })
        .wait();
}

#define ORGTR_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                              \
    template <>                                                                            \
    std::int64_t orgtr_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::uplo uplo, \
                                             std::int64_t n, std::int64_t lda) {           \
        using ArmDataType = typename ArmEquivalentType<TYPE>::Type;                        \
        ArmDataType work_query;                                                            \
        orgtr_scratchpad_size(ROUTINE, #ROUTINE, queue, uplo, n, lda, &work_query);        \
        return cast_to_int_if_complex(work_query);                                         \
    }

ORGTR_LAUNCHER_SCRATCH(float, LAPACKE_sorgtr_work)
ORGTR_LAUNCHER_SCRATCH(double, LAPACKE_dorgtr_work)

#undef ORGTR_LAUNCHER_SCRATCH

template <typename Func, typename TYPE>
inline void orgqr_scratchpad_size(Func func, const char* func_name, sycl::queue& queue,
                                  std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda,
                                  TYPE* work_query) {
    using ArmDataType = typename ArmEquivalentType<TYPE>::Type;
    queue
        .submit([&](sycl::handler& cgh) {
            host_task<class armpl_kernel_orgqr_scratchpad_size>(cgh, [=]() {
                std::int64_t err =
                    func(LAPACK_COL_MAJOR, m, n, k, nullptr, lda, nullptr, work_query, -1);
                if (err != 0) {
                    throw armpl_lapacke_error(func_name, err);
                }
            });
        })
        .wait();
}

#define ORGQR_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                                     \
    template <>                                                                                   \
    std::int64_t orgqr_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t m, std::int64_t n, \
                                             std::int64_t k, std::int64_t lda) {                  \
        using ArmDataType = typename ArmEquivalentType<TYPE>::Type;                               \
        ArmDataType work_query;                                                                   \
        orgqr_scratchpad_size(ROUTINE, #ROUTINE, queue, m, n, k, lda, &work_query);               \
        return cast_to_int_if_complex(work_query);                                                \
    }

ORGQR_LAUNCHER_SCRATCH(float, LAPACKE_sorgqr_work)
ORGQR_LAUNCHER_SCRATCH(double, LAPACKE_dorgqr_work)

#undef ORGQR_LAUNCHER_SCRATCH

template <typename Func, typename TYPE>
inline void ormrq_scratchpad_size(Func func, const char* func_name, sycl::queue& queue,
                                  oneapi::math::side side, oneapi::math::transpose trans,
                                  std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda,
                                  std::int64_t ldc, TYPE* work_query) {
    using ArmDataType = typename ArmEquivalentType<TYPE>::Type;
    queue
        .submit([&](sycl::handler& cgh) {
            host_task<class armpl_kernel_ormrq_scratchpad_size>(cgh, [=]() {
                std::int64_t err =
                    func(LAPACK_COL_MAJOR, get_side_mode(side), get_operation(trans), m, n, k,
                         nullptr, lda, nullptr, nullptr, ldc, work_query, -1);
                if (err != 0) {
                    throw armpl_lapacke_error(func_name, err);
                }
            });
        })
        .wait();
}

#define ORMRQ_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                                 \
    template <>                                                                               \
    std::int64_t ormrq_scratchpad_size<TYPE>(                                                 \
        sycl::queue & queue, oneapi::math::side side, oneapi::math::transpose trans,          \
        std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda, std::int64_t ldc) { \
        using ArmDataType = typename ArmEquivalentType<TYPE>::Type;                           \
        ArmDataType work_query;                                                               \
        ormrq_scratchpad_size(ROUTINE, #ROUTINE, queue, side, trans, m, n, k, lda, ldc,       \
                              &work_query);                                                   \
        return cast_to_int_if_complex(work_query);                                            \
    }

ORMRQ_LAUNCHER_SCRATCH(float, LAPACKE_sormrq_work)
ORMRQ_LAUNCHER_SCRATCH(double, LAPACKE_dormrq_work)

#undef ORMRQ_LAUNCHER_SCRATCH

template <typename Func, typename TYPE>
inline void ormqr_scratchpad_size(Func func, const char* func_name, sycl::queue& queue,
                                  oneapi::math::side side, oneapi::math::transpose trans,
                                  std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda,
                                  std::int64_t ldc, TYPE* work_query) {
    using ArmDataType = typename ArmEquivalentType<TYPE>::Type;
    queue
        .submit([&](sycl::handler& cgh) {
            host_task<class armpl_kernel_ormqr_scratchpad_size>(cgh, [=]() {
                std::int64_t err =
                    func(LAPACK_COL_MAJOR, get_side_mode(side), get_operation(trans), m, n, k,
                         nullptr, lda, nullptr, nullptr, ldc, work_query, -1);
                if (err != 0) {
                    throw armpl_lapacke_error(func_name, err);
                }
            });
        })
        .wait();
}

#define ORMQR_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                                 \
    template <>                                                                               \
    std::int64_t ormqr_scratchpad_size<TYPE>(                                                 \
        sycl::queue & queue, oneapi::math::side side, oneapi::math::transpose trans,          \
        std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda, std::int64_t ldc) { \
        using ArmDataType = typename ArmEquivalentType<TYPE>::Type;                           \
        ArmDataType work_query;                                                               \
        ormqr_scratchpad_size(ROUTINE, #ROUTINE, queue, side, trans, m, n, k, lda, ldc,       \
                              &work_query);                                                   \
        return cast_to_int_if_complex(work_query);                                            \
    }

ORMQR_LAUNCHER_SCRATCH(float, LAPACKE_sormqr_work)
ORMQR_LAUNCHER_SCRATCH(double, LAPACKE_dormqr_work)

#undef ORMQR_LAUNCHER_SCRATCH

template <typename Func, typename TYPE>
inline void ormtr_scratchpad_size(Func func, const char* func_name, sycl::queue& queue,
                                  oneapi::math::side side, oneapi::math::uplo uplo,
                                  oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
                                  std::int64_t lda, std::int64_t ldc, TYPE* work_query) {
    using ArmDataType = typename ArmEquivalentType<TYPE>::Type;
    queue
        .submit([&](sycl::handler& cgh) {
            host_task<class armpl_kernel_ormtr_scratchpad_size>(cgh, [=]() {
                std::int64_t err = func(LAPACK_COL_MAJOR, get_side_mode(side), get_fill_mode(uplo),
                                        get_operation(trans), m, n, nullptr, lda, nullptr, nullptr,
                                        ldc, work_query, -1);
                if (err != 0) {
                    throw armpl_lapacke_error(func_name, err);
                }
            });
        })
        .wait();
}

#define ORMTR_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                                      \
    template <>                                                                                    \
    std::int64_t ormtr_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::side side,         \
                                             oneapi::math::uplo uplo,                              \
                                             oneapi::math::transpose trans, std::int64_t m,        \
                                             std::int64_t n, std::int64_t lda, std::int64_t ldc) { \
        using ArmDataType = typename ArmEquivalentType<TYPE>::Type;                                \
        ArmDataType work_query;                                                                    \
        ormtr_scratchpad_size(ROUTINE, #ROUTINE, queue, side, uplo, trans, m, n, lda, ldc,         \
                              &work_query);                                                        \
        return cast_to_int_if_complex(work_query);                                                 \
    }

ORMTR_LAUNCHER_SCRATCH(float, LAPACKE_sormtr_work)
ORMTR_LAUNCHER_SCRATCH(double, LAPACKE_dormtr_work)

#undef ORMTR_LAUNCHER_SCRATCH

#define POTRF_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                              \
    template <>                                                                            \
    std::int64_t potrf_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::uplo uplo, \
                                             std::int64_t n, std::int64_t lda) {           \
        return 0;                                                                          \
    }

POTRF_LAUNCHER_SCRATCH(float, LAPACKE_spotrf_work)
POTRF_LAUNCHER_SCRATCH(double, LAPACKE_dpotrf_work)
POTRF_LAUNCHER_SCRATCH(std::complex<float>, LAPACKE_cpotrf_work)
POTRF_LAUNCHER_SCRATCH(std::complex<double>, LAPACKE_zpotrf_work)

#undef POTRF_LAUNCHER_SCRATCH

#define POTRS_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                                     \
    template <>                                                                                   \
    std::int64_t potrs_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::uplo uplo,        \
                                             std::int64_t n, std::int64_t nrhs, std::int64_t lda, \
                                             std::int64_t ldb) {                                  \
        return 0;                                                                                 \
    }

POTRS_LAUNCHER_SCRATCH(float, LAPACKE_spotrs_work)
POTRS_LAUNCHER_SCRATCH(double, LAPACKE_dpotrs_work)
POTRS_LAUNCHER_SCRATCH(std::complex<float>, LAPACKE_cpotrs_work)
POTRS_LAUNCHER_SCRATCH(std::complex<double>, LAPACKE_zpotrs_work)

#undef POTRS_LAUNCHER_SCRATCH

#define POTRI_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                              \
    template <>                                                                            \
    std::int64_t potri_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::uplo uplo, \
                                             std::int64_t n, std::int64_t lda) {           \
        return 0;                                                                          \
    }

POTRI_LAUNCHER_SCRATCH(float, LAPACKE_spotri_work)
POTRI_LAUNCHER_SCRATCH(double, LAPACKE_dpotri_work)
POTRI_LAUNCHER_SCRATCH(std::complex<float>, LAPACKE_cpotri_work)
POTRI_LAUNCHER_SCRATCH(std::complex<double>, LAPACKE_zpotri_work)

#undef POTRI_LAUNCHER_SCRATCH

template <typename Func, typename TYPE>
inline void sytrf_scratchpad_size(Func func, const char* func_name, sycl::queue& queue,
                                  oneapi::math::uplo uplo, std::int64_t n, std::int64_t lda,
                                  TYPE* work_query) {
    using ArmDataType = typename ArmEquivalentType<TYPE>::Type;
    queue
        .submit([&](sycl::handler& cgh) {
            host_task<class armpl_kernel_sytrf_scratchpad_size>(cgh, [=]() {
                std::int64_t err = func(LAPACK_COL_MAJOR, get_fill_mode(uplo), n, nullptr, lda,
                                        nullptr, work_query, -1);
                if (err != 0) {
                    throw armpl_lapacke_error(func_name, err);
                }
            });
        })
        .wait();
}

#define SYTRF_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                              \
    template <>                                                                            \
    std::int64_t sytrf_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::uplo uplo, \
                                             std::int64_t n, std::int64_t lda) {           \
        using ArmDataType = typename ArmEquivalentType<TYPE>::Type;                        \
        ArmDataType work_query;                                                            \
        sytrf_scratchpad_size(ROUTINE, #ROUTINE, queue, uplo, n, lda, &work_query);        \
        return cast_to_int_if_complex(work_query);                                         \
    }

SYTRF_LAUNCHER_SCRATCH(float, LAPACKE_ssytrf_work)
SYTRF_LAUNCHER_SCRATCH(double, LAPACKE_dsytrf_work)
SYTRF_LAUNCHER_SCRATCH(std::complex<float>, LAPACKE_csytrf_work)
SYTRF_LAUNCHER_SCRATCH(std::complex<double>, LAPACKE_zsytrf_work)

#undef SYTRF_LAUNCHER_SCRATCH

#define SYEVD_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                             \
    template <>                                                                           \
    std::int64_t syevd_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::job jobz, \
                                             oneapi::math::uplo uplo, std::int64_t n,     \
                                             std::int64_t lda) {                          \
        return 0;                                                                         \
    }

SYEVD_LAUNCHER_SCRATCH(float, LAPACKE_ssyevd)
SYEVD_LAUNCHER_SCRATCH(double, LAPACKE_dsyevd)

#undef SYEVD_LAUNCHER_SCRATCH

#define SYGVD_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                                      \
    template <>                                                                                    \
    std::int64_t sygvd_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t itype,              \
                                             oneapi::math::job jobz, oneapi::math::uplo uplo,      \
                                             std::int64_t n, std::int64_t lda, std::int64_t ldb) { \
        return 0;                                                                                  \
    }

SYGVD_LAUNCHER_SCRATCH(float, LAPACKE_ssygvd)
SYGVD_LAUNCHER_SCRATCH(double, LAPACKE_dsygvd)

#undef SYGVD_LAUNCHER_SCRATCH

template <typename Func, typename TYPE>
inline void sytrd_scratchpad_size(Func func, const char* func_name, sycl::queue& queue,
                                  oneapi::math::uplo uplo, std::int64_t n, std::int64_t lda,
                                  TYPE* work_query) {
    using ArmDataType = typename ArmEquivalentType<TYPE>::Type;
    queue
        .submit([&](sycl::handler& cgh) {
            host_task<class armpl_kernel_sytrd_scratchpad_size>(cgh, [=]() {
                std::int64_t err = func(LAPACK_COL_MAJOR, get_fill_mode(uplo), n, nullptr, lda,
                                        nullptr, nullptr, nullptr, work_query, -1);
                if (err != 0) {
                    throw armpl_lapacke_error(func_name, err);
                }
            });
        })
        .wait();
}

#define SYTRD_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                              \
    template <>                                                                            \
    std::int64_t sytrd_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::uplo uplo, \
                                             std::int64_t n, std::int64_t lda) {           \
        using ArmDataType = typename ArmEquivalentType<TYPE>::Type;                        \
        ArmDataType work_query;                                                            \
        sytrd_scratchpad_size(ROUTINE, #ROUTINE, queue, uplo, n, lda, &work_query);        \
        return cast_to_int_if_complex(work_query);                                         \
    }

SYTRD_LAUNCHER_SCRATCH(float, LAPACKE_ssytrd_work)
SYTRD_LAUNCHER_SCRATCH(double, LAPACKE_dsytrd_work)

#undef SYTRD_LAUNCHER_SCRATCH

#define TRTRS_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                         \
    template <>                                                                       \
    std::int64_t trtrs_scratchpad_size<TYPE>(                                         \
        sycl::queue & queue, oneapi::math::uplo uplo, oneapi::math::transpose trans,  \
        oneapi::math::diag diag, std::int64_t n, std::int64_t nrhs, std::int64_t lda, \
        std::int64_t ldb) {                                                           \
        return 0;                                                                     \
    }

TRTRS_LAUNCHER_SCRATCH(float, LAPACKE_strtrs)
TRTRS_LAUNCHER_SCRATCH(double, LAPACKE_dtrtrs)
TRTRS_LAUNCHER_SCRATCH(std::complex<float>, LAPACKE_ctrtrs)
TRTRS_LAUNCHER_SCRATCH(std::complex<double>, LAPACKE_ztrtrs)

#undef TRTRS_LAUNCHER_SCRATCH

template <typename Func, typename TYPE>
inline void ungbr_scratchpad_size(Func func, const char* func_name, sycl::queue& queue,
                                  oneapi::math::generate vec, std::int64_t m, std::int64_t n,
                                  std::int64_t k, std::int64_t lda, TYPE* work_query) {
    using ArmDataType = typename ArmEquivalentType<TYPE>::Type;
    queue
        .submit([&](sycl::handler& cgh) {
            host_task<class armpl_kernel_ungbr_scratchpad_size>(cgh, [=]() {
                std::int64_t err = func(LAPACK_COL_MAJOR, get_generate(vec), m, n, k, nullptr, lda,
                                        nullptr, work_query, -1);
                if (err != 0) {
                    throw armpl_lapacke_error(func_name, err);
                }
            });
        })
        .wait();
}

#define UNGBR_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                                 \
    template <>                                                                               \
    std::int64_t ungbr_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::generate vec, \
                                             std::int64_t m, std::int64_t n, std::int64_t k,  \
                                             std::int64_t lda) {                              \
        using ArmDataType = typename ArmEquivalentType<TYPE>::Type;                           \
        ArmDataType work_query;                                                               \
        ungbr_scratchpad_size(ROUTINE, #ROUTINE, queue, vec, m, n, k, lda, &work_query);      \
        return cast_to_int_if_complex(work_query);                                            \
    }

UNGBR_LAUNCHER_SCRATCH(std::complex<float>, LAPACKE_cungbr_work)
UNGBR_LAUNCHER_SCRATCH(std::complex<double>, LAPACKE_zungbr_work)

#undef UNGBR_LAUNCHER_SCRATCH

template <typename Func, typename TYPE>
inline void ungqr_scratchpad_size(Func func, const char* func_name, sycl::queue& queue,
                                  std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda,
                                  TYPE* work_query) {
    using ArmDataType = typename ArmEquivalentType<TYPE>::Type;
    queue
        .submit([&](sycl::handler& cgh) {
            host_task<class armpl_kernel_ungqr_scratchpad_size>(cgh, [=]() {
                std::int64_t err =
                    func(LAPACK_COL_MAJOR, m, n, k, nullptr, lda, nullptr, work_query, -1);
                if (err != 0) {
                    throw armpl_lapacke_error(func_name, err);
                }
            });
        })
        .wait();
}

#define UNGQR_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                                     \
    template <>                                                                                   \
    std::int64_t ungqr_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t m, std::int64_t n, \
                                             std::int64_t k, std::int64_t lda) {                  \
        using ArmDataType = typename ArmEquivalentType<TYPE>::Type;                               \
        ArmDataType work_query;                                                                   \
        ungqr_scratchpad_size(ROUTINE, #ROUTINE, queue, m, n, k, lda, &work_query);               \
        return cast_to_int_if_complex(work_query);                                                \
    }

UNGQR_LAUNCHER_SCRATCH(std::complex<float>, LAPACKE_cungqr_work)
UNGQR_LAUNCHER_SCRATCH(std::complex<double>, LAPACKE_zungqr_work)

#undef UNGQR_LAUNCHER_SCRATCH
template <typename Func, typename TYPE>
inline void ungtr_scratchpad_size(Func func, const char* func_name, sycl::queue& queue,
                                  oneapi::math::uplo uplo, std::int64_t n, std::int64_t lda,
                                  TYPE* work_query) {
    using ArmDataType = typename ArmEquivalentType<TYPE>::Type;
    queue
        .submit([&](sycl::handler& cgh) {
            host_task<class armpl_kernel_ungtr_scratchpad_size>(cgh, [=]() {
                std::int64_t err = func(LAPACK_COL_MAJOR, get_fill_mode(uplo), n, nullptr, lda,
                                        nullptr, work_query, -1);
                if (err != 0) {
                    throw armpl_lapacke_error(func_name, err);
                }
            });
        })
        .wait();
}

#define UNGTR_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                              \
    template <>                                                                            \
    std::int64_t ungtr_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::uplo uplo, \
                                             std::int64_t n, std::int64_t lda) {           \
        using ArmDataType = typename ArmEquivalentType<TYPE>::Type;                        \
        ArmDataType work_query;                                                            \
        ungtr_scratchpad_size(ROUTINE, #ROUTINE, queue, uplo, n, lda, &work_query);        \
        return cast_to_int_if_complex(work_query);                                         \
    }

UNGTR_LAUNCHER_SCRATCH(std::complex<float>, LAPACKE_cungtr_work)
UNGTR_LAUNCHER_SCRATCH(std::complex<double>, LAPACKE_zungtr_work)

#undef UNGTR_LAUNCHER_SCRATCH

template <typename Func, typename TYPE>
inline void unmrq_scratchpad_size(Func func, const char* func_name, sycl::queue& queue,
                                  oneapi::math::side side, oneapi::math::transpose trans,
                                  std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda,
                                  std::int64_t ldc, TYPE* work_query) {
    using ArmDataType = typename ArmEquivalentType<TYPE>::Type;
    queue
        .submit([&](sycl::handler& cgh) {
            host_task<class armpl_kernel_unmrq_scratchpad_size>(cgh, [=]() {
                std::int64_t err =
                    func(LAPACK_COL_MAJOR, get_side_mode(side), get_operation(trans), m, n, k,
                         nullptr, lda, nullptr, nullptr, ldc, work_query, -1);
                if (err != 0) {
                    throw armpl_lapacke_error(func_name, err);
                }
            });
        })
        .wait();
}

#define UNMRQ_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                                 \
    template <>                                                                               \
    std::int64_t unmrq_scratchpad_size<TYPE>(                                                 \
        sycl::queue & queue, oneapi::math::side side, oneapi::math::transpose trans,          \
        std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda, std::int64_t ldc) { \
        using ArmDataType = typename ArmEquivalentType<TYPE>::Type;                           \
        ArmDataType work_query;                                                               \
        unmrq_scratchpad_size(ROUTINE, #ROUTINE, queue, side, trans, m, n, k, lda, ldc,       \
                              &work_query);                                                   \
        return cast_to_int_if_complex(work_query);                                            \
    }

UNMRQ_LAUNCHER_SCRATCH(std::complex<float>, LAPACKE_cunmrq_work)
UNMRQ_LAUNCHER_SCRATCH(std::complex<double>, LAPACKE_zunmrq_work)

#undef UNMRQ_LAUNCHER_SCRATCH

template <typename Func, typename TYPE>
inline void unmqr_scratchpad_size(Func func, const char* func_name, sycl::queue& queue,
                                  oneapi::math::side side, oneapi::math::transpose trans,
                                  std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda,
                                  std::int64_t ldc, TYPE* work_query) {
    using ArmDataType = typename ArmEquivalentType<TYPE>::Type;
    queue
        .submit([&](sycl::handler& cgh) {
            host_task<class armpl_kernel_unmqr_scratchpad_size>(cgh, [=]() {
                std::int64_t err =
                    func(LAPACK_COL_MAJOR, get_side_mode(side), get_operation(trans), m, n, k,
                         nullptr, lda, nullptr, nullptr, ldc, work_query, -1);
                if (err != 0) {
                    throw armpl_lapacke_error(func_name, err);
                }
            });
        })
        .wait();
}

#define UNMQR_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                                 \
    template <>                                                                               \
    std::int64_t unmqr_scratchpad_size<TYPE>(                                                 \
        sycl::queue & queue, oneapi::math::side side, oneapi::math::transpose trans,          \
        std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda, std::int64_t ldc) { \
        using ArmDataType = typename ArmEquivalentType<TYPE>::Type;                           \
        ArmDataType work_query;                                                               \
        unmqr_scratchpad_size(ROUTINE, #ROUTINE, queue, side, trans, m, n, k, lda, ldc,       \
                              &work_query);                                                   \
        return cast_to_int_if_complex(work_query);                                            \
    }

UNMQR_LAUNCHER_SCRATCH(std::complex<float>, LAPACKE_cunmqr_work)
UNMQR_LAUNCHER_SCRATCH(std::complex<double>, LAPACKE_zunmqr_work)

#undef UNMQR_LAUNCHER_SCRATCH

template <typename Func, typename TYPE>
inline void unmtr_scratchpad_size(Func func, const char* func_name, sycl::queue& queue,
                                  oneapi::math::side side, oneapi::math::uplo uplo,
                                  oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
                                  std::int64_t lda, std::int64_t ldc, TYPE* work_query) {
    using ArmDataType = typename ArmEquivalentType<TYPE>::Type;
    queue
        .submit([&](sycl::handler& cgh) {
            host_task<class armpl_kernel_unmtr_scratchpad_size>(cgh, [=]() {
                std::int64_t err = func(LAPACK_COL_MAJOR, get_side_mode(side), get_fill_mode(uplo),
                                        get_operation(trans), m, n, nullptr, lda, nullptr, nullptr,
                                        ldc, work_query, -1);
                if (err != 0) {
                    throw armpl_lapacke_error(func_name, err);
                }
            });
        })
        .wait();
}

#define UNMTR_LAUNCHER_SCRATCH(TYPE, ROUTINE)                                                      \
    template <>                                                                                    \
    std::int64_t unmtr_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::side side,         \
                                             oneapi::math::uplo uplo,                              \
                                             oneapi::math::transpose trans, std::int64_t m,        \
                                             std::int64_t n, std::int64_t lda, std::int64_t ldc) { \
        using ArmDataType = typename ArmEquivalentType<TYPE>::Type;                                \
        ArmDataType work_query;                                                                    \
        unmtr_scratchpad_size(ROUTINE, #ROUTINE, queue, side, uplo, trans, m, n, lda, ldc,         \
                              &work_query);                                                        \
        return cast_to_int_if_complex(work_query);                                                 \
    }

UNMTR_LAUNCHER_SCRATCH(std::complex<float>, LAPACKE_cunmtr_work)
UNMTR_LAUNCHER_SCRATCH(std::complex<double>, LAPACKE_zunmtr_work)

#undef UNMTR_LAUNCHER_SCRATCH

} // namespace armpl
} // namespace lapack
} // namespace math
} // namespace oneapi
