/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Copyright 2022 Intel Corporation
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
#include "rocsolver_helper.hpp"
#include "rocsolver_task.hpp"

#include "oneapi/math/exceptions.hpp"
#include "oneapi/math/lapack/detail/rocsolver/onemath_lapack_rocsolver.hpp"

namespace oneapi {
namespace math {
namespace lapack {
namespace rocsolver {

// BUFFER APIs

template <typename Func, typename T_A, typename T_B>
inline void gebrd(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                  std::int64_t n, sycl::buffer<T_A>& a, std::int64_t lda, sycl::buffer<T_B>& d,
                  sycl::buffer<T_B>& e, sycl::buffer<T_A>& tauq, sycl::buffer<T_A>& taup,
                  sycl::buffer<T_A>& scratchpad, std::int64_t scratchpad_size) {
    using rocmDataType_A = typename RocmEquivalentType<T_A>::Type;
    using rocmDataType_B = typename RocmEquivalentType<T_B>::Type;
    overflow_check(m, n, lda, scratchpad_size);

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto d_acc = d.template get_access<sycl::access::mode::write>(cgh);
        auto e_acc = e.template get_access<sycl::access::mode::write>(cgh);
        auto tauq_acc = tauq.template get_access<sycl::access::mode::write>(cgh);
        auto taup_acc = taup.template get_access<sycl::access::mode::write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType_A*>(a_acc);
            auto d_ = sc.get_mem<rocmDataType_B*>(d_acc);
            auto e_ = sc.get_mem<rocmDataType_B*>(e_acc);
            auto tauq_ = sc.get_mem<rocmDataType_A*>(tauq_acc);
            auto taup_ = sc.get_mem<rocmDataType_A*>(taup_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, m, n, a_, lda, d_, e_, tauq_,
                                        taup_);
        });
    });
}

#define GEBRD_LAUNCHER(TYPE_A, TYPE_B, ROCSOLVER_ROUTINE)                                   \
    void gebrd(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<TYPE_A>& a, \
               std::int64_t lda, sycl::buffer<TYPE_B>& d, sycl::buffer<TYPE_B>& e,          \
               sycl::buffer<TYPE_A>& tauq, sycl::buffer<TYPE_A>& taup,                      \
               sycl::buffer<TYPE_A>& scratchpad, std::int64_t scratchpad_size) {            \
        gebrd(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, m, n, a, lda, d, e, tauq, taup, \
              scratchpad, scratchpad_size);                                                 \
    }

GEBRD_LAUNCHER(float, float, rocsolver_sgebrd)
GEBRD_LAUNCHER(double, double, rocsolver_dgebrd)
GEBRD_LAUNCHER(std::complex<float>, float, rocsolver_cgebrd)
GEBRD_LAUNCHER(std::complex<double>, double, rocsolver_zgebrd)

#undef GEBRD_LAUNCHER

void gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<float>& a,
           std::int64_t lda, sycl::buffer<float>& tau, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "gerqf");
}
void gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<double>& a,
           std::int64_t lda, sycl::buffer<double>& tau, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "gerqf");
}
void gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<std::complex<float>>& a,
           std::int64_t lda, sycl::buffer<std::complex<float>>& tau,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "gerqf");
}
void gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& tau, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "gerqf");
}

template <typename Func, typename T>
inline void geqrf(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& tau,
                  sycl::buffer<T>& scratchpad, std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(m, n, lda, scratchpad_size);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType*>(a_acc);
            auto tau_ = sc.get_mem<rocmDataType*>(tau_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, m, n, a_, lda, tau_);
        });
    });
}

#define GEQRF_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                            \
    void geqrf(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<TYPE>& a,  \
               std::int64_t lda, sycl::buffer<TYPE>& tau, sycl::buffer<TYPE>& scratchpad,  \
               std::int64_t scratchpad_size) {                                             \
        geqrf(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, m, n, a, lda, tau, scratchpad, \
              scratchpad_size);                                                            \
    }

GEQRF_LAUNCHER(float, rocsolver_sgeqrf)
GEQRF_LAUNCHER(double, rocsolver_dgeqrf)
GEQRF_LAUNCHER(std::complex<float>, rocsolver_cgeqrf)
GEQRF_LAUNCHER(std::complex<double>, rocsolver_zgeqrf)

#undef GEQRF_LAUNCHER

template <typename Func, typename T>
void getrf(const char* func_name, Func func, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<T>& scratchpad, std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(m, n, lda, scratchpad_size);

    // rocsolver legacy api does not accept 64-bit ints.
    // To get around the limitation,
    // create new buffer with 32-bit ints then copy over results
    std::uint64_t ipiv_size = std::min(n, m);
    sycl::buffer<int, 1> ipiv32(sycl::range<1>{ ipiv_size });
    sycl::buffer<int> devInfo{ 1 };

    auto done = queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto ipiv32_acc = ipiv32.template get_access<sycl::access::mode::write>(cgh);
        auto devInfo_acc = devInfo.template get_access<sycl::access::mode::write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType*>(a_acc);
            auto ipiv32_ = sc.get_mem<int*>(ipiv32_acc);
            auto devInfo_ = sc.get_mem<int*>(devInfo_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, m, n, a_, lda, ipiv32_,
                                        devInfo_);
        });
    });

    // Copy from 32-bit buffer to 64-bit
    queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(done);
        auto ipiv32_acc = ipiv32.template get_access<sycl::access::mode::read>(cgh);
        auto ipiv_acc = ipiv.template get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for(sycl::range<1>{ ipiv_size }, [=](sycl::id<1> index) {
            ipiv_acc[index] = static_cast<std::int64_t>(ipiv32_acc[index]);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
}

#define GETRF_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                                    \
    void getrf(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<TYPE>& a,          \
               std::int64_t lda, sycl::buffer<std::int64_t>& ipiv, sycl::buffer<TYPE>& scratchpad, \
               std::int64_t scratchpad_size) {                                                     \
        getrf(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, m, n, a, lda, ipiv, scratchpad,        \
              scratchpad_size);                                                                    \
    }

GETRF_LAUNCHER(float, rocsolver_sgetrf)
GETRF_LAUNCHER(double, rocsolver_dgetrf)
GETRF_LAUNCHER(std::complex<float>, rocsolver_cgetrf)
GETRF_LAUNCHER(std::complex<double>, rocsolver_zgetrf)

#undef GETRF_LAUNCHER

void getri(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>>& a,
           std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getri");
}
void getri(sycl::queue& queue, std::int64_t n, sycl::buffer<double>& a, std::int64_t lda,
           sycl::buffer<std::int64_t>& ipiv, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getri");
}
void getri(sycl::queue& queue, std::int64_t n, sycl::buffer<float>& a, std::int64_t lda,
           sycl::buffer<std::int64_t>& ipiv, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getri");
}
void getri(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>>& a,
           std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getri");
}

template <typename Func, typename T>
inline void getrs(const char* func_name, Func func, sycl::queue& queue,
                  oneapi::math::transpose trans, std::int64_t n, std::int64_t nrhs,
                  sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
                  sycl::buffer<T>& b, std::int64_t ldb, sycl::buffer<T>& scratchpad,
                  std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, nrhs, lda, ldb);

    // rocsolver legacy api does not accept 64-bit ints.
    // To get around the limitation.
    // Create new buffer and convert 64-bit values.
    std::uint64_t ipiv_size = ipiv.size();
    sycl::buffer<int, 1> ipiv32(sycl::range<1>{ ipiv_size });

    queue.submit([&](sycl::handler& cgh) {
        auto ipiv32_acc = ipiv32.template get_access<sycl::access::mode::write>(cgh);
        auto ipiv_acc = ipiv.template get_access<sycl::access::mode::read>(cgh);
        cgh.parallel_for(sycl::range<1>{ ipiv_size }, [=](sycl::id<1> index) {
            ipiv32_acc[index] = static_cast<std::int32_t>(ipiv_acc[index]);
        });
    });

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto ipiv_acc = ipiv32.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType*>(a_acc);
            auto ipiv_ = sc.get_mem<std::int32_t*>(ipiv_acc);
            auto b_ = sc.get_mem<rocmDataType*>(b_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_operation(trans),
                                        n, nrhs, a_, lda, ipiv_, b_, ldb);
        });
    });
}

#define GETRS_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                                   \
    void getrs(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,                 \
               std::int64_t nrhs, sycl::buffer<TYPE>& a, std::int64_t lda,                        \
               sycl::buffer<std::int64_t>& ipiv, sycl::buffer<TYPE>& b, std::int64_t ldb,         \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {                    \
        getrs(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, trans, n, nrhs, a, lda, ipiv, b, ldb, \
              scratchpad, scratchpad_size);                                                       \
    }

GETRS_LAUNCHER(float, rocsolver_sgetrs)
GETRS_LAUNCHER(double, rocsolver_dgetrs)
GETRS_LAUNCHER(std::complex<float>, rocsolver_cgetrs)
GETRS_LAUNCHER(std::complex<double>, rocsolver_zgetrs)

#undef GETRS_LAUNCHER

template <typename Func, typename T_A, typename T_B>
inline void gesvd(const char* func_name, Func func, sycl::queue& queue, oneapi::math::jobsvd jobu,
                  oneapi::math::jobsvd jobvt, std::int64_t m, std::int64_t n, sycl::buffer<T_A>& a,
                  std::int64_t lda, sycl::buffer<T_B>& s, sycl::buffer<T_A>& u, std::int64_t ldu,
                  sycl::buffer<T_A>& vt, std::int64_t ldvt, sycl::buffer<T_A>& scratchpad,
                  std::int64_t scratchpad_size) {
    using rocmDataType_A = typename RocmEquivalentType<T_A>::Type;
    using rocmDataType_B = typename RocmEquivalentType<T_B>::Type;
    overflow_check(n, m, lda, ldu, ldvt, scratchpad_size);
    sycl::buffer<int> devInfo{ 1 };
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto s_acc = s.template get_access<sycl::access::mode::write>(cgh);
        auto u_acc = u.template get_access<sycl::access::mode::write>(cgh);
        auto vt_acc = vt.template get_access<sycl::access::mode::write>(cgh);
        auto devInfo_acc = devInfo.template get_access<sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType_A*>(a_acc);
            auto s_ = sc.get_mem<rocmDataType_B*>(s_acc);
            auto u_ = sc.get_mem<rocmDataType_A*>(u_acc);
            auto vt_ = sc.get_mem<rocmDataType_A*>(vt_acc);
            auto devInfo_ = sc.get_mem<int*>(devInfo_acc);
            auto scratch_ = sc.get_mem<rocmDataType_B*>(scratch_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocsolver_jobsvd(jobu),
                                        get_rocsolver_jobsvd(jobvt), m, n, a_, lda, s_, u_, ldu,
                                        vt_, ldvt, scratch_, rocblas_workmode::rocblas_outofplace,
                                        devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
}

#define GESVD_LAUNCHER(TYPE_A, TYPE_B, ROCSOLVER_ROUTINE)                                         \
    void gesvd(sycl::queue& queue, oneapi::math::jobsvd jobu, oneapi::math::jobsvd jobvt,         \
               std::int64_t m, std::int64_t n, sycl::buffer<TYPE_A>& a, std::int64_t lda,         \
               sycl::buffer<TYPE_B>& s, sycl::buffer<TYPE_A>& u, std::int64_t ldu,                \
               sycl::buffer<TYPE_A>& vt, std::int64_t ldvt, sycl::buffer<TYPE_A>& scratchpad,     \
               std::int64_t scratchpad_size) {                                                    \
        gesvd(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, jobu, jobvt, m, n, a, lda, s, u, ldu, \
              vt, ldvt, scratchpad, scratchpad_size);                                             \
    }

GESVD_LAUNCHER(float, float, rocsolver_sgesvd)
GESVD_LAUNCHER(double, double, rocsolver_dgesvd)
GESVD_LAUNCHER(std::complex<float>, float, rocsolver_cgesvd)
GESVD_LAUNCHER(std::complex<double>, double, rocsolver_zgesvd)

#undef GESVD_LAUNCHER

template <typename Func, typename T_A, typename T_B>
inline void heevd(const char* func_name, Func func, sycl::queue& queue, oneapi::math::job jobz,
                  oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<T_A>& a, std::int64_t lda,
                  sycl::buffer<T_B>& w, sycl::buffer<T_A>& scratchpad,
                  std::int64_t scratchpad_size) {
    using rocmDataType_A = typename RocmEquivalentType<T_A>::Type;
    using rocmDataType_B = typename RocmEquivalentType<T_B>::Type;
    overflow_check(n, lda, scratchpad_size);
    sycl::buffer<int> devInfo{ 1 };
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto w_acc = w.template get_access<sycl::access::mode::write>(cgh);
        auto devInfo_acc = devInfo.template get_access<sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType_A*>(a_acc);
            auto w_ = sc.get_mem<rocmDataType_B*>(w_acc);
            auto devInfo_ = sc.get_mem<int*>(devInfo_acc);
            auto scratch_ = sc.get_mem<rocmDataType_B*>(scratch_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocsolver_job(jobz),
                                        get_rocblas_fill_mode(uplo), n, a_, lda, w_, scratch_,
                                        devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
}

#define HEEVD_LAUNCHER(TYPE_A, TYPE_B, ROCSOLVER_ROUTINE)                                          \
    void heevd(sycl::queue& queue, oneapi::math::job jobz, oneapi::math::uplo uplo,                \
               std::int64_t n, sycl::buffer<TYPE_A>& a, std::int64_t lda, sycl::buffer<TYPE_B>& w, \
               sycl::buffer<TYPE_A>& scratchpad, std::int64_t scratchpad_size) {                   \
        heevd(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, jobz, uplo, n, a, lda, w, scratchpad,  \
              scratchpad_size);                                                                    \
    }

HEEVD_LAUNCHER(std::complex<float>, float, rocsolver_cheevd)
HEEVD_LAUNCHER(std::complex<double>, double, rocsolver_zheevd)

#undef HEEVD_LAUNCHER

template <typename Func, typename T_A, typename T_B>
inline void hegvd(const char* func_name, Func func, sycl::queue& queue, std::int64_t itype,
                  oneapi::math::job jobz, oneapi::math::uplo uplo, std::int64_t n,
                  sycl::buffer<T_A>& a, std::int64_t lda, sycl::buffer<T_A>& b, std::int64_t ldb,
                  sycl::buffer<T_B>& w, sycl::buffer<T_A>& scratchpad,
                  std::int64_t scratchpad_size) {
    using rocmDataType_A = typename RocmEquivalentType<T_A>::Type;
    using rocmDataType_B = typename RocmEquivalentType<T_B>::Type;
    overflow_check(n, lda, ldb, scratchpad_size);
    sycl::buffer<int> devInfo{ 1 };
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read_write>(cgh);
        auto w_acc = w.template get_access<sycl::access::mode::write>(cgh);
        auto devInfo_acc = devInfo.template get_access<sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType_A*>(a_acc);
            auto b_ = sc.get_mem<rocmDataType_A*>(b_acc);
            auto w_ = sc.get_mem<rocmDataType_B*>(w_acc);
            auto devInfo_ = sc.get_mem<int*>(devInfo_acc);
            auto scratch_ = sc.get_mem<rocmDataType_B*>(scratch_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocsolver_itype(itype),
                                        get_rocsolver_job(jobz), get_rocblas_fill_mode(uplo), n, a_,
                                        lda, b_, ldb, w_, scratch_, devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
}

#define HEGVD_LAUNCHER(TYPE_A, TYPE_B, ROCSOLVER_ROUTINE)                                          \
    void hegvd(sycl::queue& queue, std::int64_t itype, oneapi::math::job jobz,                     \
               oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<TYPE_A>& a, std::int64_t lda, \
               sycl::buffer<TYPE_A>& b, std::int64_t ldb, sycl::buffer<TYPE_B>& w,                 \
               sycl::buffer<TYPE_A>& scratchpad, std::int64_t scratchpad_size) {                   \
        hegvd(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, itype, jobz, uplo, n, a, lda, b, ldb,  \
              w, scratchpad, scratchpad_size);                                                     \
    }

HEGVD_LAUNCHER(std::complex<float>, float, rocsolver_chegvd)
HEGVD_LAUNCHER(std::complex<double>, double, rocsolver_zhegvd)

#undef HEGVD_LAUNCHER

template <typename Func, typename T_A, typename T_B>
inline void hetrd(const char* func_name, Func func, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, sycl::buffer<T_A>& a, std::int64_t lda, sycl::buffer<T_B>& d,
                  sycl::buffer<T_B>& e, sycl::buffer<T_A>& tau, sycl::buffer<T_A>& scratchpad,
                  std::int64_t scratchpad_size) {
    using rocmDataType_A = typename RocmEquivalentType<T_A>::Type;
    using rocmDataType_B = typename RocmEquivalentType<T_B>::Type;
    overflow_check(n, lda, scratchpad_size);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto d_acc = d.template get_access<sycl::access::mode::write>(cgh);
        auto e_acc = e.template get_access<sycl::access::mode::write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType_A*>(a_acc);
            auto d_ = sc.get_mem<rocmDataType_B*>(d_acc);
            auto e_ = sc.get_mem<rocmDataType_B*>(e_acc);
            auto tau_ = sc.get_mem<rocmDataType_A*>(tau_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_fill_mode(uplo),
                                        n, a_, lda, d_, e_, tau_);
        });
    });
}

#define HETRD_LAUNCHER(TYPE_A, TYPE_B, ROCSOLVER_ROUTINE)                               \
    void hetrd(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,             \
               sycl::buffer<TYPE_A>& a, std::int64_t lda, sycl::buffer<TYPE_B>& d,      \
               sycl::buffer<TYPE_B>& e, sycl::buffer<TYPE_A>& tau,                      \
               sycl::buffer<TYPE_A>& scratchpad, std::int64_t scratchpad_size) {        \
        hetrd(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, a, lda, d, e, tau, \
              scratchpad, scratchpad_size);                                             \
    }

HETRD_LAUNCHER(std::complex<float>, float, rocsolver_chetrd)
HETRD_LAUNCHER(std::complex<double>, double, rocsolver_zhetrd)

#undef HETRD_LAUNCHER

void hetrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "hetrf");
}
void hetrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::int64_t>& ipiv, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "hetrf");
}

template <typename Func, typename T>
inline void orgbr(const char* func_name, Func func, sycl::queue& queue, oneapi::math::generate vec,
                  std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<T>& a,
                  std::int64_t lda, sycl::buffer<T>& tau, sycl::buffer<T>& scratchpad,
                  std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, scratchpad_size);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::read>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType*>(a_acc);
            auto tau_ = sc.get_mem<rocmDataType*>(tau_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_generate(vec), m,
                                        n, k, a_, lda, tau_);
        });
    });
}

#define ORGBR_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                                    \
    void orgbr(sycl::queue& queue, oneapi::math::generate vec, std::int64_t m, std::int64_t n,     \
               std::int64_t k, sycl::buffer<TYPE>& a, std::int64_t lda, sycl::buffer<TYPE>& tau,   \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {                     \
        orgbr(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, vec, m, n, k, a, lda, tau, scratchpad, \
              scratchpad_size);                                                                    \
    }

ORGBR_LAUNCHER(float, rocsolver_sorgbr)
ORGBR_LAUNCHER(double, rocsolver_dorgbr)

#undef ORGBR_LAUNCHER

template <typename Func, typename T>
inline void orgqr(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                  std::int64_t n, std::int64_t k, sycl::buffer<T>& a, std::int64_t lda,
                  sycl::buffer<T>& tau, sycl::buffer<T>& scratchpad, std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, scratchpad_size);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::read>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType*>(a_acc);
            auto tau_ = sc.get_mem<rocmDataType*>(tau_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, m, n, k, a_, lda, tau_);
        });
    });
}

#define ORGQR_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                               \
    void orgqr(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,            \
               sycl::buffer<TYPE>& a, std::int64_t lda, sycl::buffer<TYPE>& tau,              \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {                \
        orgqr(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, m, n, k, a, lda, tau, scratchpad, \
              scratchpad_size);                                                               \
    }

ORGQR_LAUNCHER(float, rocsolver_sorgqr)
ORGQR_LAUNCHER(double, rocsolver_dorgqr)

#undef ORGQR_LAUNCHER

template <typename Func, typename T>
inline void orgtr(const char* func_name, Func func, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& tau,
                  sycl::buffer<T>& scratchpad, std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::read>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType*>(a_acc);
            auto tau_ = sc.get_mem<rocmDataType*>(tau_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_fill_mode(uplo),
                                        n, a_, lda, tau_);
        });
    });
}

#define ORGTR_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                                    \
    void orgtr(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<TYPE>& a, \
               std::int64_t lda, sycl::buffer<TYPE>& tau, sycl::buffer<TYPE>& scratchpad,          \
               std::int64_t scratchpad_size) {                                                     \
        orgtr(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, a, lda, tau, scratchpad,      \
              scratchpad_size);                                                                    \
    }

ORGTR_LAUNCHER(float, rocsolver_sorgtr)
ORGTR_LAUNCHER(double, rocsolver_dorgtr)

#undef ORGTR_LAUNCHER

template <typename Func, typename T>
inline void ormtr(const char* func_name, Func func, sycl::queue& queue, oneapi::math::side side,
                  oneapi::math::uplo uplo, oneapi::math::transpose trans, std::int64_t m,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& tau,
                  sycl::buffer<T>& c, std::int64_t ldc, sycl::buffer<T>& scratchpad,
                  std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldc, scratchpad_size);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::read_write>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType*>(a_acc);
            auto tau_ = sc.get_mem<rocmDataType*>(tau_acc);
            auto c_ = sc.get_mem<rocmDataType*>(c_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_side_mode(side),
                                        get_rocblas_fill_mode(uplo), get_rocblas_operation(trans),
                                        m, n, a_, lda, tau_, c_, ldc);
        });
    });
}

#define ORMTR_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                                   \
    void ormtr(sycl::queue& queue, oneapi::math::side side, oneapi::math::uplo uplo,              \
               oneapi::math::transpose trans, std::int64_t m, std::int64_t n,                     \
               sycl::buffer<TYPE>& a, std::int64_t lda, sycl::buffer<TYPE>& tau,                  \
               sycl::buffer<TYPE>& c, std::int64_t ldc, sycl::buffer<TYPE>& scratchpad,           \
               std::int64_t scratchpad_size) {                                                    \
        ormtr(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, side, uplo, trans, m, n, a, lda, tau, \
              c, ldc, scratchpad, scratchpad_size);                                               \
    }

ORMTR_LAUNCHER(float, rocsolver_sormtr)
ORMTR_LAUNCHER(double, rocsolver_dormtr)

#undef ORMTR_LAUNCHER

void ormrq(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
           std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<float>& a, std::int64_t lda,
           sycl::buffer<float>& tau, sycl::buffer<float>& c, std::int64_t ldc,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "ormrq");
}
void ormrq(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
           std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<double>& a,
           std::int64_t lda, sycl::buffer<double>& tau, sycl::buffer<double>& c, std::int64_t ldc,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "ormrq");
}

template <typename Func, typename T>
inline void ormqr(const char* func_name, Func func, sycl::queue& queue, oneapi::math::side side,
                  oneapi::math::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
                  sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& tau, sycl::buffer<T>& c,
                  std::int64_t ldc, sycl::buffer<T>& scratchpad, std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, ldc, scratchpad_size);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType*>(a_acc);
            auto tau_ = sc.get_mem<rocmDataType*>(tau_acc);
            auto c_ = sc.get_mem<rocmDataType*>(c_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_side_mode(side),
                                        get_rocblas_operation(trans), m, n, k, a_, lda, tau_, c_,
                                        ldc);
        });
    });
}

#define ORMQR_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                                    \
    void ormqr(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,         \
               std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<TYPE>& a,              \
               std::int64_t lda, sycl::buffer<TYPE>& tau, sycl::buffer<TYPE>& c, std::int64_t ldc, \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {                     \
        ormqr(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, side, trans, m, n, k, a, lda, tau, c,  \
              ldc, scratchpad, scratchpad_size);                                                   \
    }

ORMQR_LAUNCHER(float, rocsolver_sormqr)
ORMQR_LAUNCHER(double, rocsolver_dormqr)

#undef ORMQR_LAUNCHER

template <typename Func, typename T>
inline void potrf(const char* func_name, Func func, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& scratchpad,
                  std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    sycl::buffer<int> devInfo{ 1 };
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto devInfo_acc = devInfo.template get_access<sycl::access::mode::write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType*>(a_acc);
            auto devInfo_ = sc.get_mem<int*>(devInfo_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_fill_mode(uplo),
                                        n, a_, lda, devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
}

#define POTRF_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                                    \
    void potrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<TYPE>& a, \
               std::int64_t lda, sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {   \
        potrf(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, a, lda, scratchpad,           \
              scratchpad_size);                                                                    \
    }

POTRF_LAUNCHER(float, rocsolver_spotrf)
POTRF_LAUNCHER(double, rocsolver_dpotrf)
POTRF_LAUNCHER(std::complex<float>, rocsolver_cpotrf)
POTRF_LAUNCHER(std::complex<double>, rocsolver_zpotrf)

#undef POTRF_LAUNCHER

template <typename Func, typename T>
inline void potri(const char* func_name, Func func, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& scratchpad,
                  std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    sycl::buffer<int> devInfo{ 1 };
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto devInfo_acc = devInfo.template get_access<sycl::access::mode::write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType*>(a_acc);
            auto devInfo_ = sc.get_mem<int*>(devInfo_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_fill_mode(uplo),
                                        n, a_, lda, devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
}

#define POTRI_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                                    \
    void potri(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<TYPE>& a, \
               std::int64_t lda, sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {   \
        potri(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, a, lda, scratchpad,           \
              scratchpad_size);                                                                    \
    }

POTRI_LAUNCHER(float, rocsolver_spotri)
POTRI_LAUNCHER(double, rocsolver_dpotri)
POTRI_LAUNCHER(std::complex<float>, rocsolver_cpotri)
POTRI_LAUNCHER(std::complex<double>, rocsolver_zpotri)

#undef POTRI_LAUNCHER

template <typename Func, typename T>
inline void potrs(const char* func_name, Func func, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, std::int64_t nrhs, sycl::buffer<T>& a, std::int64_t lda,
                  sycl::buffer<T>& b, std::int64_t ldb, sycl::buffer<T>& scratchpad,
                  std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, nrhs, lda, ldb, scratchpad_size);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType*>(a_acc);
            auto b_ = sc.get_mem<rocmDataType*>(b_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_fill_mode(uplo),
                                        n, nrhs, a_, lda, b_, ldb);
        });
    });
}

#define POTRS_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                                  \
    void potrs(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,   \
               sycl::buffer<TYPE>& a, std::int64_t lda, sycl::buffer<TYPE>& b, std::int64_t ldb, \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {                   \
        potrs(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, nrhs, a, lda, b, ldb,       \
              scratchpad, scratchpad_size);                                                      \
    }

POTRS_LAUNCHER(float, rocsolver_spotrs)
POTRS_LAUNCHER(double, rocsolver_dpotrs)
POTRS_LAUNCHER(std::complex<float>, rocsolver_cpotrs)
POTRS_LAUNCHER(std::complex<double>, rocsolver_zpotrs)

#undef POTRS_LAUNCHER

template <typename Func, typename T>
inline void syevd(const char* func_name, Func func, sycl::queue& queue, oneapi::math::job jobz,
                  oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<T>& a, std::int64_t lda,
                  sycl::buffer<T>& w, sycl::buffer<T>& scratchpad, std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    sycl::buffer<int> devInfo{ 1 };
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto w_acc = w.template get_access<sycl::access::mode::write>(cgh);
        auto devInfo_acc = devInfo.template get_access<sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType*>(a_acc);
            auto w_ = sc.get_mem<rocmDataType*>(w_acc);
            auto devInfo_ = sc.get_mem<int*>(devInfo_acc);
            auto scratch_ = sc.get_mem<rocmDataType*>(scratch_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocsolver_job(jobz),
                                        get_rocblas_fill_mode(uplo), n, a_, lda, w_, scratch_,
                                        devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
}

#define SYEVD_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                                   \
    void syevd(sycl::queue& queue, oneapi::math::job jobz, oneapi::math::uplo uplo,               \
               std::int64_t n, sycl::buffer<TYPE>& a, std::int64_t lda, sycl::buffer<TYPE>& w,    \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {                    \
        syevd(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, jobz, uplo, n, a, lda, w, scratchpad, \
              scratchpad_size);                                                                   \
    }

SYEVD_LAUNCHER(float, rocsolver_ssyevd)
SYEVD_LAUNCHER(double, rocsolver_dsyevd)

#undef SYEVD_LAUNCHER

template <typename Func, typename T>
inline void sygvd(const char* func_name, Func func, sycl::queue& queue, std::int64_t itype,
                  oneapi::math::job jobz, oneapi::math::uplo uplo, std::int64_t n,
                  sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& b, std::int64_t ldb,
                  sycl::buffer<T>& w, sycl::buffer<T>& scratchpad, std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, lda, ldb, scratchpad_size);
    sycl::buffer<int> devInfo{ 1 };
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read_write>(cgh);
        auto w_acc = w.template get_access<sycl::access::mode::write>(cgh);
        auto devInfo_acc = devInfo.template get_access<sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType*>(a_acc);
            auto b_ = sc.get_mem<rocmDataType*>(b_acc);
            auto w_ = sc.get_mem<rocmDataType*>(w_acc);
            auto devInfo_ = sc.get_mem<int*>(devInfo_acc);
            auto scratch_ = sc.get_mem<rocmDataType*>(scratch_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocsolver_itype(itype),
                                        get_rocsolver_job(jobz), get_rocblas_fill_mode(uplo), n, a_,
                                        lda, b_, ldb, w_, scratch_, devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
}

#define SYGVD_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                                   \
    void sygvd(sycl::queue& queue, std::int64_t itype, oneapi::math::job jobz,                    \
               oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<TYPE>& a, std::int64_t lda,  \
               sycl::buffer<TYPE>& b, std::int64_t ldb, sycl::buffer<TYPE>& w,                    \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {                    \
        sygvd(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, itype, jobz, uplo, n, a, lda, b, ldb, \
              w, scratchpad, scratchpad_size);                                                    \
    }

SYGVD_LAUNCHER(float, rocsolver_ssygvd)
SYGVD_LAUNCHER(double, rocsolver_dsygvd)

#undef SYGVD_LAUNCH

template <typename Func, typename T>
inline void sytrd(const char* func_name, Func func, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& d,
                  sycl::buffer<T>& e, sycl::buffer<T>& tau, sycl::buffer<T>& scratchpad,
                  std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto d_acc = d.template get_access<sycl::access::mode::write>(cgh);
        auto e_acc = e.template get_access<sycl::access::mode::write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType*>(a_acc);
            auto d_ = sc.get_mem<rocmDataType*>(d_acc);
            auto e_ = sc.get_mem<rocmDataType*>(e_acc);
            auto tau_ = sc.get_mem<rocmDataType*>(tau_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_fill_mode(uplo),
                                        n, a_, lda, d_, e_, tau_);
        });
    });
}

#define SYTRD_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                                    \
    void sytrd(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<TYPE>& a, \
               std::int64_t lda, sycl::buffer<TYPE>& d, sycl::buffer<TYPE>& e,                     \
               sycl::buffer<TYPE>& tau, sycl::buffer<TYPE>& scratchpad,                            \
               std::int64_t scratchpad_size) {                                                     \
        sytrd(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, a, lda, d, e, tau,            \
              scratchpad, scratchpad_size);                                                        \
    }

SYTRD_LAUNCHER(float, rocsolver_ssytrd)
SYTRD_LAUNCHER(double, rocsolver_dsytrd)

#undef SYTRD_LAUNCHER

template <typename Func, typename T>
inline void sytrf(const char* func_name, Func func, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda,
                  sycl::buffer<std::int64_t>& ipiv, sycl::buffer<T>& scratchpad,
                  std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    sycl::buffer<int> devInfo{ 1 };

    // rocsolver legacy api does not accept 64-bit ints.
    // To get around the limitation.
    // Create new buffer with 32-bit ints then copy over results
    std::uint64_t ipiv_size = n;
    sycl::buffer<int, 1> ipiv32(sycl::range<1>{ ipiv_size });

    auto done = queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto ipiv32_acc = ipiv32.template get_access<sycl::access::mode::write>(cgh);
        auto devInfo_acc = devInfo.template get_access<sycl::access::mode::write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType*>(a_acc);
            auto ipiv32_ = sc.get_mem<int*>(ipiv32_acc);
            auto devInfo_ = sc.get_mem<int*>(devInfo_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_fill_mode(uplo),
                                        n, a_, lda, ipiv32_, devInfo_);
        });
    });

    // Copy from 32-bit buffer to 64-bit
    queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(done);
        auto ipiv32_acc = ipiv32.template get_access<sycl::access::mode::read>(cgh);
        auto ipiv_acc = ipiv.template get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for(sycl::range<1>{ ipiv_size }, [=](sycl::id<1> index) {
            ipiv_acc[index] = static_cast<std::int64_t>(ipiv32_acc[index]);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
}

#define SYTRF_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                                    \
    void sytrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<TYPE>& a, \
               std::int64_t lda, sycl::buffer<std::int64_t>& ipiv, sycl::buffer<TYPE>& scratchpad, \
               std::int64_t scratchpad_size) {                                                     \
        sytrf(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, a, lda, ipiv, scratchpad,     \
              scratchpad_size);                                                                    \
    }

SYTRF_LAUNCHER(float, rocsolver_ssytrf)
SYTRF_LAUNCHER(double, rocsolver_dsytrf)
SYTRF_LAUNCHER(std::complex<float>, rocsolver_csytrf)
SYTRF_LAUNCHER(std::complex<double>, rocsolver_zsytrf)

#undef SYTRF_LAUNCHER

void trtrs(sycl::queue& queue, oneapi::math::uplo uplo, oneapi::math::transpose trans,
           oneapi::math::diag diag, std::int64_t n, std::int64_t nrhs,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& b, std::int64_t ldb,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "trtrs");
}
void trtrs(sycl::queue& queue, oneapi::math::uplo uplo, oneapi::math::transpose trans,
           oneapi::math::diag diag, std::int64_t n, std::int64_t nrhs, sycl::buffer<double>& a,
           std::int64_t lda, sycl::buffer<double>& b, std::int64_t ldb,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "trtrs");
}
void trtrs(sycl::queue& queue, oneapi::math::uplo uplo, oneapi::math::transpose trans,
           oneapi::math::diag diag, std::int64_t n, std::int64_t nrhs, sycl::buffer<float>& a,
           std::int64_t lda, sycl::buffer<float>& b, std::int64_t ldb,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "trtrs");
}
void trtrs(sycl::queue& queue, oneapi::math::uplo uplo, oneapi::math::transpose trans,
           oneapi::math::diag diag, std::int64_t n, std::int64_t nrhs,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& b, std::int64_t ldb,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "trtrs");
}

template <typename Func, typename T>
inline void ungbr(const char* func_name, Func func, sycl::queue& queue, oneapi::math::generate vec,
                  std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<T>& a,
                  std::int64_t lda, sycl::buffer<T>& tau, sycl::buffer<T>& scratchpad,
                  std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, scratchpad_size);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType*>(a_acc);
            auto tau_ = sc.get_mem<rocmDataType*>(tau_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_generate(vec), m,
                                        n, k, a_, lda, tau_);
        });
    });
}

#define UNGBR_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                                    \
    void ungbr(sycl::queue& queue, oneapi::math::generate vec, std::int64_t m, std::int64_t n,     \
               std::int64_t k, sycl::buffer<TYPE>& a, std::int64_t lda, sycl::buffer<TYPE>& tau,   \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {                     \
        ungbr(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, vec, m, n, k, a, lda, tau, scratchpad, \
              scratchpad_size);                                                                    \
    }

UNGBR_LAUNCHER(std::complex<float>, rocsolver_cungbr)
UNGBR_LAUNCHER(std::complex<double>, rocsolver_zungbr)

#undef UNGBR_LAUNCHER

template <typename Func, typename T>
inline void ungqr(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                  std::int64_t n, std::int64_t k, sycl::buffer<T>& a, std::int64_t lda,
                  sycl::buffer<T>& tau, sycl::buffer<T>& scratchpad, std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, scratchpad_size);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType*>(a_acc);
            auto tau_ = sc.get_mem<rocmDataType*>(tau_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, m, n, k, a_, lda, tau_);
        });
    });
}

#define UNGQR_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                               \
    void ungqr(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,            \
               sycl::buffer<TYPE>& a, std::int64_t lda, sycl::buffer<TYPE>& tau,              \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {                \
        ungqr(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, m, n, k, a, lda, tau, scratchpad, \
              scratchpad_size);                                                               \
    }

UNGQR_LAUNCHER(std::complex<float>, rocsolver_cungqr)
UNGQR_LAUNCHER(std::complex<double>, rocsolver_zungqr)

#undef UNGQR_LAUNCHER

template <typename Func, typename T>
inline void ungtr(const char* func_name, Func func, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& tau,
                  sycl::buffer<T>& scratchpad, std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType*>(a_acc);
            auto tau_ = sc.get_mem<rocmDataType*>(tau_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_fill_mode(uplo),
                                        n, a_, lda, tau_);
        });
    });
}

#define UNGTR_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                                    \
    void ungtr(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<TYPE>& a, \
               std::int64_t lda, sycl::buffer<TYPE>& tau, sycl::buffer<TYPE>& scratchpad,          \
               std::int64_t scratchpad_size) {                                                     \
        ungtr(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, a, lda, tau, scratchpad,      \
              scratchpad_size);                                                                    \
    }

UNGTR_LAUNCHER(std::complex<float>, rocsolver_cungtr)
UNGTR_LAUNCHER(std::complex<double>, rocsolver_zungtr)

#undef UNGTR_LAUNCHER

void unmrq(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
           std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<std::complex<float>>& a,
           std::int64_t lda, sycl::buffer<std::complex<float>>& tau,
           sycl::buffer<std::complex<float>>& c, std::int64_t ldc,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "unmrq");
}
void unmrq(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
           std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<std::complex<double>>& a,
           std::int64_t lda, sycl::buffer<std::complex<double>>& tau,
           sycl::buffer<std::complex<double>>& c, std::int64_t ldc,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "unmrq");
}

template <typename Func, typename T>
inline void unmqr(const char* func_name, Func func, sycl::queue& queue, oneapi::math::side side,
                  oneapi::math::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
                  sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& tau, sycl::buffer<T>& c,
                  std::int64_t ldc, sycl::buffer<T>& scratchpad, std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::write>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType*>(a_acc);
            auto tau_ = sc.get_mem<rocmDataType*>(tau_acc);
            auto c_ = sc.get_mem<rocmDataType*>(c_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_side_mode(side),
                                        get_rocblas_operation(trans), m, n, k, a_, lda, tau_, c_,
                                        ldc);
        });
    });
}

#define UNMQR_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                                    \
    void unmqr(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,         \
               std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<TYPE>& a,              \
               std::int64_t lda, sycl::buffer<TYPE>& tau, sycl::buffer<TYPE>& c, std::int64_t ldc, \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {                     \
        unmqr(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, side, trans, m, n, k, a, lda, tau, c,  \
              ldc, scratchpad, scratchpad_size);                                                   \
    }

UNMQR_LAUNCHER(std::complex<float>, rocsolver_cunmqr)
UNMQR_LAUNCHER(std::complex<double>, rocsolver_zunmqr)

#undef UNMQR_LAUNCHER

template <typename Func, typename T>
inline void unmtr(const char* func_name, Func func, sycl::queue& queue, oneapi::math::side side,
                  oneapi::math::uplo uplo, oneapi::math::transpose trans, std::int64_t m,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& tau,
                  sycl::buffer<T>& c, std::int64_t ldc, sycl::buffer<T>& scratchpad,
                  std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldc, scratchpad_size);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::write>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType*>(a_acc);
            auto tau_ = sc.get_mem<rocmDataType*>(tau_acc);
            auto c_ = sc.get_mem<rocmDataType*>(c_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_side_mode(side),
                                        get_rocblas_fill_mode(uplo), get_rocblas_operation(trans),
                                        m, n, a_, lda, tau_, c_, ldc);
        });
    });
}

#define UNMTR_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                                   \
    void unmtr(sycl::queue& queue, oneapi::math::side side, oneapi::math::uplo uplo,              \
               oneapi::math::transpose trans, std::int64_t m, std::int64_t n,                     \
               sycl::buffer<TYPE>& a, std::int64_t lda, sycl::buffer<TYPE>& tau,                  \
               sycl::buffer<TYPE>& c, std::int64_t ldc, sycl::buffer<TYPE>& scratchpad,           \
               std::int64_t scratchpad_size) {                                                    \
        unmtr(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, side, uplo, trans, m, n, a, lda, tau, \
              c, ldc, scratchpad, scratchpad_size);                                               \
    }

UNMTR_LAUNCHER(std::complex<float>, rocsolver_cunmtr)
UNMTR_LAUNCHER(std::complex<double>, rocsolver_zunmtr)

#undef UNMTR_LAUNCHER

// USM APIs

template <typename Func, typename T_A, typename T_B>
inline sycl::event gebrd(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                         std::int64_t n, T_A* a, std::int64_t lda, T_B* d, T_B* e, T_A* tauq,
                         T_A* taup, T_A* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using rocmDataType_A = typename RocmEquivalentType<T_A>::Type;
    using rocmDataType_B = typename RocmEquivalentType<T_B>::Type;
    overflow_check(m, n, lda, scratchpad_size);

    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType_A*>(a);
            auto d_ = reinterpret_cast<rocmDataType_B*>(d);
            auto e_ = reinterpret_cast<rocmDataType_B*>(e);
            auto tauq_ = reinterpret_cast<rocmDataType_A*>(tauq);
            auto taup_ = reinterpret_cast<rocmDataType_A*>(taup);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, m, n, a_, lda, d_, e_, tauq_,
                                        taup_);
        });
    });
    return done;
}

#define GEBRD_LAUNCHER_USM(TYPE_A, TYPE_B, ROCSOLVER_ROUTINE)                                      \
    sycl::event gebrd(sycl::queue& queue, std::int64_t m, std::int64_t n, TYPE_A* a,               \
                      std::int64_t lda, TYPE_B* d, TYPE_B* e, TYPE_A* tauq, TYPE_A* taup,          \
                      TYPE_A* scratchpad, std::int64_t scratchpad_size,                            \
                      const std::vector<sycl::event>& dependencies) {                              \
        return gebrd(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, m, n, a, lda, d, e, tauq, taup, \
                     scratchpad, scratchpad_size, dependencies);                                   \
    }

GEBRD_LAUNCHER_USM(float, float, rocsolver_sgebrd)
GEBRD_LAUNCHER_USM(double, double, rocsolver_dgebrd)
GEBRD_LAUNCHER_USM(std::complex<float>, float, rocsolver_cgebrd)
GEBRD_LAUNCHER_USM(std::complex<double>, double, rocsolver_zgebrd)

#undef GEBRD_LAUNCHER_USM

sycl::event gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n, float* a, std::int64_t lda,
                  float* tau, float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "gerqf");
}
sycl::event gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n, double* a, std::int64_t lda,
                  double* tau, double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "gerqf");
}
sycl::event gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<float>* a,
                  std::int64_t lda, std::complex<float>* tau, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "gerqf");
}
sycl::event gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<double>* a,
                  std::int64_t lda, std::complex<double>* tau, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "gerqf");
}

template <typename Func, typename T>
inline sycl::event geqrf(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                         std::int64_t n, T* a, std::int64_t lda, T* tau, T* scratchpad,
                         std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(m, n, lda, scratchpad_size);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType*>(a);
            auto tau_ = reinterpret_cast<rocmDataType*>(tau);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, m, n, a_, lda, tau_);
        });
    });
    return done;
}

#define GEQRF_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                                \
    sycl::event geqrf(sycl::queue& queue, std::int64_t m, std::int64_t n, TYPE* a,                 \
                      std::int64_t lda, TYPE* tau, TYPE* scratchpad, std::int64_t scratchpad_size, \
                      const std::vector<sycl::event>& dependencies) {                              \
        return geqrf(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, m, n, a, lda, tau, scratchpad,  \
                     scratchpad_size, dependencies);                                               \
    }

GEQRF_LAUNCHER_USM(float, rocsolver_sgeqrf)
GEQRF_LAUNCHER_USM(double, rocsolver_dgeqrf)
GEQRF_LAUNCHER_USM(std::complex<float>, rocsolver_cgeqrf)
GEQRF_LAUNCHER_USM(std::complex<double>, rocsolver_zgeqrf)

#undef GEQRF_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event getrf(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                         std::int64_t n, T* a, std::int64_t lda, std::int64_t* ipiv, T* scratchpad,
                         std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(m, n, lda, scratchpad_size);

    // rocsolver legacy api does not accept 64-bit ints.
    // To get around the limitation.
    // Allocate memory with 32-bit ints then copy over results
    std::uint64_t ipiv_size = std::min(n, m);
    int* ipiv32 = (int*)malloc_device(sizeof(int) * ipiv_size, queue);

    int* devInfo = (int*)malloc_device(sizeof(int), queue);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType*>(a);
            auto devInfo_ = reinterpret_cast<int*>(devInfo);
            auto ipiv_ = reinterpret_cast<int*>(ipiv32);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, m, n, a_, lda, ipiv_,
                                        devInfo_);
        });
    });

    // Copy from 32-bit USM to 64-bit
    auto done_casting = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(done);
        cgh.parallel_for(sycl::range<1>{ ipiv_size }, [=](sycl::id<1> index) {
            ipiv[index] = static_cast<std::int64_t>(ipiv32[index]);
        });
    });

    lapack_info_check(queue, devInfo, __func__, func_name);
    free(ipiv32, queue);
    free(devInfo, queue);
    return done_casting;
}

#define GETRF_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                                \
    sycl::event getrf(sycl::queue& queue, std::int64_t m, std::int64_t n, TYPE* a,                 \
                      std::int64_t lda, std::int64_t* ipiv, TYPE* scratchpad,                      \
                      std::int64_t scratchpad_size,                                                \
                      const std::vector<sycl::event>& dependencies) {                              \
        return getrf(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, m, n, a, lda, ipiv, scratchpad, \
                     scratchpad_size, dependencies);                                               \
    }

GETRF_LAUNCHER_USM(float, rocsolver_sgetrf)
GETRF_LAUNCHER_USM(double, rocsolver_dgetrf)
GETRF_LAUNCHER_USM(std::complex<float>, rocsolver_cgetrf)
GETRF_LAUNCHER_USM(std::complex<double>, rocsolver_zgetrf)

#undef GETRF_LAUNCHER_USM

sycl::event getri(sycl::queue& queue, std::int64_t n, std::complex<float>* a, std::int64_t lda,
                  std::int64_t* ipiv, std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getri");
}
sycl::event getri(sycl::queue& queue, std::int64_t n, double* a, std::int64_t lda,
                  std::int64_t* ipiv, double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getri");
}
sycl::event getri(sycl::queue& queue, std::int64_t n, float* a, std::int64_t lda,
                  std::int64_t* ipiv, float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getri");
}
sycl::event getri(sycl::queue& queue, std::int64_t n, std::complex<double>* a, std::int64_t lda,
                  std::int64_t* ipiv, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getri");
}

template <typename Func, typename T>
inline sycl::event getrs(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::math::transpose trans, std::int64_t n, std::int64_t nrhs, T* a,
                         std::int64_t lda, std::int64_t* ipiv, T* b, std::int64_t ldb,
                         T* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, nrhs, lda, ldb, scratchpad_size);

    // rocsolver legacy api does not accept 64-bit ints.
    // To get around the limitation.
    // Create new buffer and convert 64-bit values.
    std::uint64_t ipiv_size = n;
    int* ipiv32 = (int*)malloc_device(sizeof(int) * ipiv_size, queue);

    auto done_casting = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>{ ipiv_size }, [=](sycl::id<1> index) {
            ipiv32[index] = static_cast<std::int32_t>(ipiv[index]);
        });
    });

    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        cgh.depends_on(done_casting);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType*>(a);
            auto ipiv_ = reinterpret_cast<int*>(ipiv32);
            auto b_ = reinterpret_cast<rocmDataType*>(b);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_operation(trans),
                                        n, nrhs, a_, lda, ipiv_, b_, ldb);
        });
    });

    queue.wait();

    free(ipiv32, queue);

    return done;
}

#define GETRS_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                              \
    sycl::event getrs(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,         \
                      std::int64_t nrhs, TYPE* a, std::int64_t lda, std::int64_t* ipiv, TYPE* b, \
                      std::int64_t ldb, TYPE* scratchpad, std::int64_t scratchpad_size,          \
                      const std::vector<sycl::event>& dependencies) {                            \
        return getrs(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, trans, n, nrhs, a, lda, ipiv, \
                     b, ldb, scratchpad, scratchpad_size, dependencies);                         \
    }

GETRS_LAUNCHER_USM(float, rocsolver_sgetrs)
GETRS_LAUNCHER_USM(double, rocsolver_dgetrs)
GETRS_LAUNCHER_USM(std::complex<float>, rocsolver_cgetrs)
GETRS_LAUNCHER_USM(std::complex<double>, rocsolver_zgetrs)

#undef GETRS_LAUNCHER_USM

template <typename Func, typename T_A, typename T_B>
inline sycl::event gesvd(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::math::jobsvd jobu, oneapi::math::jobsvd jobvt, std::int64_t m,
                         std::int64_t n, T_A* a, std::int64_t lda, T_B* s, T_A* u, std::int64_t ldu,
                         T_A* vt, std::int64_t ldvt, T_A* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using rocmDataType_A = typename RocmEquivalentType<T_A>::Type;
    using rocmDataType_B = typename RocmEquivalentType<T_B>::Type;
    overflow_check(m, n, lda, ldu, ldvt, scratchpad_size);
    int* devInfo = (int*)malloc_device(sizeof(int), queue);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType_A*>(a);
            auto s_ = reinterpret_cast<rocmDataType_B*>(s);
            auto u_ = reinterpret_cast<rocmDataType_A*>(u);
            auto vt_ = reinterpret_cast<rocmDataType_A*>(vt);
            auto devInfo_ = reinterpret_cast<int*>(devInfo);
            auto scratch_ = reinterpret_cast<rocmDataType_B*>(scratchpad);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocsolver_jobsvd(jobu),
                                        get_rocsolver_jobsvd(jobvt), m, n, a_, lda, s_, u_, ldu,
                                        vt_, ldvt, scratch_, rocblas_workmode::rocblas_outofplace,
                                        devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
    free(devInfo, queue);
    return done;
}

#define GESVD_LAUNCHER_USM(TYPE_A, TYPE_B, ROCSOLVER_ROUTINE)                                    \
    sycl::event gesvd(sycl::queue& queue, oneapi::math::jobsvd jobu, oneapi::math::jobsvd jobvt, \
                      std::int64_t m, std::int64_t n, TYPE_A* a, std::int64_t lda, TYPE_B* s,    \
                      TYPE_A* u, std::int64_t ldu, TYPE_A* vt, std::int64_t ldvt,                \
                      TYPE_A* scratchpad, std::int64_t scratchpad_size,                          \
                      const std::vector<sycl::event>& dependencies) {                            \
        return gesvd(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, jobu, jobvt, m, n, a, lda, s, \
                     u, ldu, vt, ldvt, scratchpad, scratchpad_size, dependencies);               \
    }

GESVD_LAUNCHER_USM(float, float, rocsolver_sgesvd)
GESVD_LAUNCHER_USM(double, double, rocsolver_dgesvd)
GESVD_LAUNCHER_USM(std::complex<float>, float, rocsolver_cgesvd)
GESVD_LAUNCHER_USM(std::complex<double>, double, rocsolver_zgesvd)

#undef GESVD_LAUNCHER_USM

template <typename Func, typename T_A, typename T_B>
inline sycl::event heevd(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::math::job jobz, oneapi::math::uplo uplo, std::int64_t n, T_A*& a,
                         std::int64_t lda, T_B*& w, T_A*& scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using rocmDataType_A = typename RocmEquivalentType<T_A>::Type;
    using rocmDataType_B = typename RocmEquivalentType<T_B>::Type;
    overflow_check(n, lda, scratchpad_size);
    int* devInfo = (int*)malloc_device(sizeof(int), queue);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType_A*>(a);
            auto w_ = reinterpret_cast<rocmDataType_B*>(w);
            auto devInfo_ = reinterpret_cast<int*>(devInfo);
            auto scratch_ = reinterpret_cast<rocmDataType_B*>(scratchpad);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocsolver_job(jobz),
                                        get_rocblas_fill_mode(uplo), n, a_, lda, w_, scratch_,
                                        devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
    free(devInfo, queue);
    return done;
}

#define HEEVD_LAUNCHER_USM(TYPE_A, TYPE_B, ROCSOLVER_ROUTINE)                                     \
    sycl::event heevd(sycl::queue& queue, oneapi::math::job jobz, oneapi::math::uplo uplo,        \
                      std::int64_t n, TYPE_A* a, std::int64_t lda, TYPE_B* w, TYPE_A* scratchpad, \
                      std::int64_t scratchpad_size,                                               \
                      const std::vector<sycl::event>& dependencies) {                             \
        return heevd(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, jobz, uplo, n, a, lda, w,      \
                     scratchpad, scratchpad_size, dependencies);                                  \
    }

HEEVD_LAUNCHER_USM(std::complex<float>, float, rocsolver_cheevd)
HEEVD_LAUNCHER_USM(std::complex<double>, double, rocsolver_zheevd)

#undef HEEVD_LAUNCHER_USM

template <typename Func, typename T_A, typename T_B>
inline sycl::event hegvd(const char* func_name, Func func, sycl::queue& queue, std::int64_t itype,
                         oneapi::math::job jobz, oneapi::math::uplo uplo, std::int64_t n, T_A*& a,
                         std::int64_t lda, T_A*& b, std::int64_t ldb, T_B*& w, T_A*& scratchpad,
                         std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using rocmDataType_A = typename RocmEquivalentType<T_A>::Type;
    using rocmDataType_B = typename RocmEquivalentType<T_B>::Type;
    overflow_check(n, lda, ldb, scratchpad_size);
    int* devInfo = (int*)malloc_device(sizeof(int), queue);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType_A*>(a);
            auto b_ = reinterpret_cast<rocmDataType_A*>(b);
            auto w_ = reinterpret_cast<rocmDataType_B*>(w);
            auto devInfo_ = reinterpret_cast<int*>(devInfo);
            auto scratch_ = reinterpret_cast<rocmDataType_B*>(scratchpad);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocsolver_itype(itype),
                                        get_rocsolver_job(jobz), get_rocblas_fill_mode(uplo), n, a_,
                                        lda, b_, ldb, w_, scratch_, devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
    free(devInfo, queue);
    return done;
}

#define HEGVD_LAUNCHER_USM(TYPE_A, TYPE_B, ROCSOLVER_ROUTINE)                                    \
    sycl::event hegvd(sycl::queue& queue, std::int64_t itype, oneapi::math::job jobz,            \
                      oneapi::math::uplo uplo, std::int64_t n, TYPE_A* a, std::int64_t lda,      \
                      TYPE_A* b, std::int64_t ldb, TYPE_B* w, TYPE_A* scratchpad,                \
                      std::int64_t scratchpad_size,                                              \
                      const std::vector<sycl::event>& dependencies) {                            \
        return hegvd(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, itype, jobz, uplo, n, a, lda, \
                     b, ldb, w, scratchpad, scratchpad_size, dependencies);                      \
    }

HEGVD_LAUNCHER_USM(std::complex<float>, float, rocsolver_chegvd)
HEGVD_LAUNCHER_USM(std::complex<double>, double, rocsolver_zhegvd)

#undef HEGVD_LAUNCHER_USM

template <typename Func, typename T_A, typename T_B>
inline sycl::event hetrd(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::math::uplo uplo, std::int64_t n, T_A* a, std::int64_t lda, T_B* d,
                         T_B* e, T_A* tau, T_A* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using rocmDataType_A = typename RocmEquivalentType<T_A>::Type;
    using rocmDataType_B = typename RocmEquivalentType<T_B>::Type;
    overflow_check(n, lda, scratchpad_size);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType_A*>(a);
            auto d_ = reinterpret_cast<rocmDataType_B*>(d);
            auto e_ = reinterpret_cast<rocmDataType_B*>(e);
            auto tau_ = reinterpret_cast<rocmDataType_A*>(tau);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_fill_mode(uplo),
                                        n, a_, lda, d_, e_, tau_);
        });
    });
    return done;
}

#define HETRD_LAUNCHER_USM(TYPE_A, TYPE_B, ROCSOLVER_ROUTINE)                                  \
    sycl::event hetrd(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, TYPE_A* a,  \
                      std::int64_t lda, TYPE_B* d, TYPE_B* e, TYPE_A* tau, TYPE_A* scratchpad, \
                      std::int64_t scratchpad_size,                                            \
                      const std::vector<sycl::event>& dependencies) {                          \
        return hetrd(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, a, lda, d, e, tau, \
                     scratchpad, scratchpad_size, dependencies);                               \
    }

HETRD_LAUNCHER_USM(std::complex<float>, float, rocsolver_chetrd)
HETRD_LAUNCHER_USM(std::complex<double>, double, rocsolver_zhetrd)

#undef HETRD_LAUNCHER_USM

sycl::event hetrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                  std::complex<float>* a, std::int64_t lda, std::int64_t* ipiv,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "hetrf");
}
sycl::event hetrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                  std::complex<double>* a, std::int64_t lda, std::int64_t* ipiv,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "hetrf");
}

template <typename Func, typename T>
inline sycl::event orgbr(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::math::generate vec, std::int64_t m, std::int64_t n, std::int64_t k,
                         T* a, std::int64_t lda, T* tau, T* scratchpad,
                         std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, scratchpad_size);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType*>(a);
            auto tau_ = reinterpret_cast<rocmDataType*>(tau);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_generate(vec), m,
                                        n, k, a_, lda, tau_);
        });
    });
    return done;
}

#define ORGBR_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                           \
    sycl::event orgbr(sycl::queue& queue, oneapi::math::generate vec, std::int64_t m,         \
                      std::int64_t n, std::int64_t k, TYPE* a, std::int64_t lda, TYPE* tau,   \
                      TYPE* scratchpad, std::int64_t scratchpad_size,                         \
                      const std::vector<sycl::event>& dependencies) {                         \
        return orgbr(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, vec, m, n, k, a, lda, tau, \
                     scratchpad, scratchpad_size, dependencies);                              \
    }

ORGBR_LAUNCHER_USM(float, rocsolver_sorgbr)
ORGBR_LAUNCHER_USM(double, rocsolver_dorgbr)

#undef ORGBR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event orgqr(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                         std::int64_t n, std::int64_t k, T* a, std::int64_t lda, T* tau,
                         T* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, scratchpad_size);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType*>(a);
            auto tau_ = reinterpret_cast<rocmDataType*>(tau);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, m, n, k, a_, lda, tau_);
        });
    });
    return done;
}

#define ORGQR_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                                \
    sycl::event orgqr(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k, TYPE* a, \
                      std::int64_t lda, TYPE* tau, TYPE* scratchpad, std::int64_t scratchpad_size, \
                      const std::vector<sycl::event>& dependencies) {                              \
        return orgqr(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, m, n, k, a, lda, tau,           \
                     scratchpad, scratchpad_size, dependencies);                                   \
    }

ORGQR_LAUNCHER_USM(float, rocsolver_sorgqr)
ORGQR_LAUNCHER_USM(double, rocsolver_dorgqr)

#undef ORGQR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event orgtr(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::math::uplo uplo, std::int64_t n, T* a, std::int64_t lda, T* tau,
                         T* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType*>(a);
            auto tau_ = reinterpret_cast<rocmDataType*>(tau);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_fill_mode(uplo),
                                        n, a_, lda, tau_);
        });
    });
    return done;
}

#define ORGTR_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                                \
    sycl::event orgtr(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, TYPE* a,        \
                      std::int64_t lda, TYPE* tau, TYPE* scratchpad, std::int64_t scratchpad_size, \
                      const std::vector<sycl::event>& dependencies) {                              \
        return orgtr(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, a, lda, tau,           \
                     scratchpad, scratchpad_size, dependencies);                                   \
    }

ORGTR_LAUNCHER_USM(float, rocsolver_sorgtr)
ORGTR_LAUNCHER_USM(double, rocsolver_dorgtr)

#undef ORGTR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event ormtr(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::math::side side, oneapi::math::uplo uplo,
                         oneapi::math::transpose trans, std::int64_t m, std::int64_t n, T* a,
                         std::int64_t lda, T* tau, T* c, std::int64_t ldc, T* scratchpad,
                         std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldc, scratchpad_size);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType*>(a);
            auto tau_ = reinterpret_cast<rocmDataType*>(tau);
            auto c_ = reinterpret_cast<rocmDataType*>(c);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_side_mode(side),
                                        get_rocblas_fill_mode(uplo), get_rocblas_operation(trans),
                                        m, n, a_, lda, tau_, c_, ldc);
        });
    });
    return done;
}

#define ORMTR_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                             \
    sycl::event ormtr(sycl::queue& queue, oneapi::math::side side, oneapi::math::uplo uplo,     \
                      oneapi::math::transpose trans, std::int64_t m, std::int64_t n, TYPE* a,   \
                      std::int64_t lda, TYPE* tau, TYPE* c, std::int64_t ldc, TYPE* scratchpad, \
                      std::int64_t scratchpad_size,                                             \
                      const std::vector<sycl::event>& dependencies) {                           \
        return ormtr(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, side, uplo, trans, m, n, a,  \
                     lda, tau, c, ldc, scratchpad, scratchpad_size, dependencies);              \
    }

ORMTR_LAUNCHER_USM(float, rocsolver_sormtr)
ORMTR_LAUNCHER_USM(double, rocsolver_dormtr)

#undef ORMTR_LAUNCHER_USM

sycl::event ormrq(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
                  std::int64_t m, std::int64_t n, std::int64_t k, float* a, std::int64_t lda,
                  float* tau, float* c, std::int64_t ldc, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "ormrq");
}
sycl::event ormrq(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
                  std::int64_t m, std::int64_t n, std::int64_t k, double* a, std::int64_t lda,
                  double* tau, double* c, std::int64_t ldc, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "ormrq");
}

template <typename Func, typename T>
inline sycl::event ormqr(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::math::side side, oneapi::math::transpose trans, std::int64_t m,
                         std::int64_t n, std::int64_t k, T* a, std::int64_t lda, T* tau, T* c,
                         std::int64_t ldc, T* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, ldc, scratchpad_size);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType*>(a);
            auto tau_ = reinterpret_cast<rocmDataType*>(tau);
            auto c_ = reinterpret_cast<rocmDataType*>(c);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_side_mode(side),
                                        get_rocblas_operation(trans), m, n, k, a_, lda, tau_, c_,
                                        ldc);
        });
    });
    return done;
}

#define ORMQR_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                               \
    sycl::event ormqr(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans, \
                      std::int64_t m, std::int64_t n, std::int64_t k, TYPE* a, std::int64_t lda,  \
                      TYPE* tau, TYPE* c, std::int64_t ldc, TYPE* scratchpad,                     \
                      std::int64_t scratchpad_size,                                               \
                      const std::vector<sycl::event>& dependencies) {                             \
        return ormqr(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, side, trans, m, n, k, a, lda,  \
                     tau, c, ldc, scratchpad, scratchpad_size, dependencies);                     \
    }

ORMQR_LAUNCHER_USM(float, rocsolver_sormqr)
ORMQR_LAUNCHER_USM(double, rocsolver_dormqr)

#undef ORMQR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event potrf(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::math::uplo uplo, std::int64_t n, T* a, std::int64_t lda,
                         T* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    int* devInfo = (int*)malloc_device(sizeof(int), queue);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType*>(a);
            auto devInfo_ = reinterpret_cast<int*>(devInfo);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_fill_mode(uplo),
                                        n, a_, lda, devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
    free(devInfo, queue);
    return done;
}

#define POTRF_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                             \
    sycl::event potrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, TYPE* a,     \
                      std::int64_t lda, TYPE* scratchpad, std::int64_t scratchpad_size,         \
                      const std::vector<sycl::event>& dependencies) {                           \
        return potrf(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, a, lda, scratchpad, \
                     scratchpad_size, dependencies);                                            \
    }

POTRF_LAUNCHER_USM(float, rocsolver_spotrf)
POTRF_LAUNCHER_USM(double, rocsolver_dpotrf)
POTRF_LAUNCHER_USM(std::complex<float>, rocsolver_cpotrf)
POTRF_LAUNCHER_USM(std::complex<double>, rocsolver_zpotrf)

#undef POTRF_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event potri(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::math::uplo uplo, std::int64_t n, T* a, std::int64_t lda,
                         T* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    int* devInfo = (int*)malloc_device(sizeof(int), queue);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType*>(a);
            auto scratch_ = reinterpret_cast<rocmDataType*>(scratchpad);
            auto devInfo_ = reinterpret_cast<int*>(devInfo);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_fill_mode(uplo),
                                        n, a_, lda, devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
    free(devInfo, queue);
    return done;
}

#define POTRI_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                             \
    sycl::event potri(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, TYPE* a,     \
                      std::int64_t lda, TYPE* scratchpad, std::int64_t scratchpad_size,         \
                      const std::vector<sycl::event>& dependencies) {                           \
        return potri(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, a, lda, scratchpad, \
                     scratchpad_size, dependencies);                                            \
    }

POTRI_LAUNCHER_USM(float, rocsolver_spotri)
POTRI_LAUNCHER_USM(double, rocsolver_dpotri)
POTRI_LAUNCHER_USM(std::complex<float>, rocsolver_cpotri)
POTRI_LAUNCHER_USM(std::complex<double>, rocsolver_zpotri)

#undef POTRI_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event potrs(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs, T* a,
                         std::int64_t lda, T* b, std::int64_t ldb, T* scratchpad,
                         std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, nrhs, lda, ldb, scratchpad_size);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType*>(a);
            auto b_ = reinterpret_cast<rocmDataType*>(b);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_fill_mode(uplo),
                                        n, nrhs, a_, lda, b_, ldb);
        });
    });
    return done;
}

#define POTRS_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                               \
    sycl::event potrs(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,                \
                      std::int64_t nrhs, TYPE* a, std::int64_t lda, TYPE* b, std::int64_t ldb,    \
                      TYPE* scratchpad, std::int64_t scratchpad_size,                             \
                      const std::vector<sycl::event>& dependencies) {                             \
        return potrs(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, nrhs, a, lda, b, ldb, \
                     scratchpad, scratchpad_size, dependencies);                                  \
    }

POTRS_LAUNCHER_USM(float, rocsolver_spotrs)
POTRS_LAUNCHER_USM(double, rocsolver_dpotrs)
POTRS_LAUNCHER_USM(std::complex<float>, rocsolver_cpotrs)
POTRS_LAUNCHER_USM(std::complex<double>, rocsolver_zpotrs)

#undef POTRS_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event syevd(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::math::job jobz, oneapi::math::uplo uplo, std::int64_t n, T* a,
                         std::int64_t lda, T* w, T* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    int* devInfo = (int*)malloc_device(sizeof(int), queue);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType*>(a);
            auto w_ = reinterpret_cast<rocmDataType*>(w);
            auto scratch_ = reinterpret_cast<rocmDataType*>(scratchpad);
            auto devInfo_ = reinterpret_cast<int*>(devInfo);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocsolver_job(jobz),
                                        get_rocblas_fill_mode(uplo), n, a_, lda, w_, scratch_,
                                        devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
    free(devInfo, queue);
    return done;
}

#define SYEVD_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                          \
    sycl::event syevd(sycl::queue& queue, oneapi::math::job jobz, oneapi::math::uplo uplo,   \
                      std::int64_t n, TYPE* a, std::int64_t lda, TYPE* w, TYPE* scratchpad,  \
                      std::int64_t scratchpad_size,                                          \
                      const std::vector<sycl::event>& dependencies) {                        \
        return syevd(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, jobz, uplo, n, a, lda, w, \
                     scratchpad, scratchpad_size, dependencies);                             \
    }

SYEVD_LAUNCHER_USM(float, rocsolver_ssyevd)
SYEVD_LAUNCHER_USM(double, rocsolver_dsyevd)

#undef SYEVD_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event sygvd(const char* func_name, Func func, sycl::queue& queue, std::int64_t itype,
                         oneapi::math::job jobz, oneapi::math::uplo uplo, std::int64_t n, T* a,
                         std::int64_t lda, T* b, std::int64_t ldb, T* w, T* scratchpad,
                         std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, lda, ldb, scratchpad_size);
    int* devInfo = (int*)malloc_device(sizeof(int), queue);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType*>(a);
            auto b_ = reinterpret_cast<rocmDataType*>(b);
            auto w_ = reinterpret_cast<rocmDataType*>(w);
            auto devInfo_ = reinterpret_cast<int*>(devInfo);
            auto scratch_ = reinterpret_cast<rocmDataType*>(scratchpad);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocsolver_itype(itype),
                                        get_rocsolver_job(jobz), get_rocblas_fill_mode(uplo), n, a_,
                                        lda, b_, ldb, w_, scratch_, devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
    free(devInfo, queue);
    return done;
}

#define SYGVD_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                                \
    sycl::event sygvd(sycl::queue& queue, std::int64_t itype, oneapi::math::job jobz,              \
                      oneapi::math::uplo uplo, std::int64_t n, TYPE* a, std::int64_t lda, TYPE* b, \
                      std::int64_t ldb, TYPE* w, TYPE* scratchpad, std::int64_t scratchpad_size,   \
                      const std::vector<sycl::event>& dependencies) {                              \
        return sygvd(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, itype, jobz, uplo, n, a, lda,   \
                     b, ldb, w, scratchpad, scratchpad_size, dependencies);                        \
    }

SYGVD_LAUNCHER_USM(float, rocsolver_ssygvd)
SYGVD_LAUNCHER_USM(double, rocsolver_dsygvd)

#undef SYGVD_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event sytrd(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::math::uplo uplo, std::int64_t n, T* a, std::int64_t lda, T* d,
                         T* e, T* tau, T* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType*>(a);
            auto d_ = reinterpret_cast<rocmDataType*>(d);
            auto e_ = reinterpret_cast<rocmDataType*>(e);
            auto tau_ = reinterpret_cast<rocmDataType*>(tau);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_fill_mode(uplo),
                                        n, a_, lda, d_, e_, tau_);
        });
    });
    return done;
}

#define SYTRD_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                            \
    sycl::event sytrd(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, TYPE* a,    \
                      std::int64_t lda, TYPE* d, TYPE* e, TYPE* tau, TYPE* scratchpad,         \
                      std::int64_t scratchpad_size,                                            \
                      const std::vector<sycl::event>& dependencies) {                          \
        return sytrd(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, a, lda, d, e, tau, \
                     scratchpad, scratchpad_size, dependencies);                               \
    }

SYTRD_LAUNCHER_USM(float, rocsolver_ssytrd)
SYTRD_LAUNCHER_USM(double, rocsolver_dsytrd)

#undef SYTRD_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event sytrf(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::math::uplo uplo, std::int64_t n, T* a, std::int64_t lda,
                         std::int64_t* ipiv, T* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    int* devInfo = (int*)malloc_device(sizeof(int), queue);

    // rocsolver legacy api does not accept 64-bit ints.
    // To get around the limitation.
    // Allocate memory with 32-bit ints then copy over results
    std::uint64_t ipiv_size = n;
    int* ipiv32 = (int*)malloc_device(sizeof(int) * ipiv_size, queue);

    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType*>(a);
            auto ipiv_ = reinterpret_cast<int*>(ipiv32);
            auto devInfo_ = reinterpret_cast<int*>(devInfo);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_fill_mode(uplo),
                                        n, a_, lda, ipiv_, devInfo_);
        });
    });

    // Copy from 32-bit USM to 64-bit
    auto done_casting = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(done);
        cgh.parallel_for(sycl::range<1>{ ipiv_size }, [=](sycl::id<1> index) {
            ipiv[index] = static_cast<std::int64_t>(ipiv32[index]);
        });
    });

    lapack_info_check(queue, devInfo, __func__, func_name);
    free(ipiv32, queue);
    free(devInfo, queue);
    return done_casting;
}

#define SYTRF_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                         \
    sycl::event sytrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, TYPE* a, \
                      std::int64_t lda, std::int64_t* ipiv, TYPE* scratchpad,               \
                      std::int64_t scratchpad_size,                                         \
                      const std::vector<sycl::event>& dependencies) {                       \
        return sytrf(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, a, lda, ipiv,   \
                     scratchpad, scratchpad_size, dependencies);                            \
    }

SYTRF_LAUNCHER_USM(float, rocsolver_ssytrf)
SYTRF_LAUNCHER_USM(double, rocsolver_dsytrf)
SYTRF_LAUNCHER_USM(std::complex<float>, rocsolver_csytrf)
SYTRF_LAUNCHER_USM(std::complex<double>, rocsolver_zsytrf)

#undef SYTRF_LAUNCHER_USM

sycl::event trtrs(sycl::queue& queue, oneapi::math::uplo uplo, oneapi::math::transpose trans,
                  oneapi::math::diag diag, std::int64_t n, std::int64_t nrhs,
                  std::complex<float>* a, std::int64_t lda, std::complex<float>* b,
                  std::int64_t ldb, std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "trtrs");
}
sycl::event trtrs(sycl::queue& queue, oneapi::math::uplo uplo, oneapi::math::transpose trans,
                  oneapi::math::diag diag, std::int64_t n, std::int64_t nrhs, double* a,
                  std::int64_t lda, double* b, std::int64_t ldb, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "trtrs");
}
sycl::event trtrs(sycl::queue& queue, oneapi::math::uplo uplo, oneapi::math::transpose trans,
                  oneapi::math::diag diag, std::int64_t n, std::int64_t nrhs, float* a,
                  std::int64_t lda, float* b, std::int64_t ldb, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "trtrs");
}
sycl::event trtrs(sycl::queue& queue, oneapi::math::uplo uplo, oneapi::math::transpose trans,
                  oneapi::math::diag diag, std::int64_t n, std::int64_t nrhs,
                  std::complex<double>* a, std::int64_t lda, std::complex<double>* b,
                  std::int64_t ldb, std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "trtrs");
}

template <typename Func, typename T>
inline sycl::event ungbr(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::math::generate vec, std::int64_t m, std::int64_t n, std::int64_t k,
                         T* a, std::int64_t lda, T* tau, T* scratchpad,
                         std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType*>(a);
            auto tau_ = reinterpret_cast<rocmDataType*>(tau);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_generate(vec), m,
                                        n, k, a_, lda, tau_);
        });
    });
    return done;
}

#define UNGBR_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                           \
    sycl::event ungbr(sycl::queue& queue, oneapi::math::generate vec, std::int64_t m,         \
                      std::int64_t n, std::int64_t k, TYPE* a, std::int64_t lda, TYPE* tau,   \
                      TYPE* scratchpad, std::int64_t scratchpad_size,                         \
                      const std::vector<sycl::event>& dependencies) {                         \
        return ungbr(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, vec, m, n, k, a, lda, tau, \
                     scratchpad, scratchpad_size, dependencies);                              \
    }

UNGBR_LAUNCHER_USM(std::complex<float>, rocsolver_cungbr)
UNGBR_LAUNCHER_USM(std::complex<double>, rocsolver_zungbr)

#undef UNGBR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event ungqr(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                         std::int64_t n, std::int64_t k, T* a, std::int64_t lda, T* tau,
                         T* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, scratchpad_size);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType*>(a);
            auto tau_ = reinterpret_cast<rocmDataType*>(tau);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, m, n, k, a_, lda, tau_);
        });
    });
    return done;
}

#define UNGQR_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                                \
    sycl::event ungqr(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k, TYPE* a, \
                      std::int64_t lda, TYPE* tau, TYPE* scratchpad, std::int64_t scratchpad_size, \
                      const std::vector<sycl::event>& dependencies) {                              \
        return ungqr(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, m, n, k, a, lda, tau,           \
                     scratchpad, scratchpad_size, dependencies);                                   \
    }

UNGQR_LAUNCHER_USM(std::complex<float>, rocsolver_cungqr)
UNGQR_LAUNCHER_USM(std::complex<double>, rocsolver_zungqr)

#undef UNGQR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event ungtr(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::math::uplo uplo, std::int64_t n, T* a, std::int64_t lda, T* tau,
                         T* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType*>(a);
            auto tau_ = reinterpret_cast<rocmDataType*>(tau);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_fill_mode(uplo),
                                        n, a_, lda, tau_);
        });
    });
    return done;
}

#define UNGTR_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                                \
    sycl::event ungtr(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, TYPE* a,        \
                      std::int64_t lda, TYPE* tau, TYPE* scratchpad, std::int64_t scratchpad_size, \
                      const std::vector<sycl::event>& dependencies) {                              \
        return ungtr(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, a, lda, tau,           \
                     scratchpad, scratchpad_size, dependencies);                                   \
    }

UNGTR_LAUNCHER_USM(std::complex<float>, rocsolver_cungtr)
UNGTR_LAUNCHER_USM(std::complex<double>, rocsolver_zungtr)

#undef UNGTR_LAUNCHER_USM

sycl::event unmrq(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
                  std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float>* a,
                  std::int64_t lda, std::complex<float>* tau, std::complex<float>* c,
                  std::int64_t ldc, std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "unmrq");
}
sycl::event unmrq(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
                  std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double>* a,
                  std::int64_t lda, std::complex<double>* tau, std::complex<double>* c,
                  std::int64_t ldc, std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "unmrq");
}

template <typename Func, typename T>
inline sycl::event unmqr(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::math::side side, oneapi::math::transpose trans, std::int64_t m,
                         std::int64_t n, std::int64_t k, T* a, std::int64_t lda, T* tau, T* c,
                         std::int64_t ldc, T* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType*>(a);
            auto tau_ = reinterpret_cast<rocmDataType*>(tau);
            auto c_ = reinterpret_cast<rocmDataType*>(c);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_side_mode(side),
                                        get_rocblas_operation(trans), m, n, k, a_, lda, tau_, c_,
                                        ldc);
        });
    });
    return done;
}

#define UNMQR_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                               \
    sycl::event unmqr(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans, \
                      std::int64_t m, std::int64_t n, std::int64_t k, TYPE* a, std::int64_t lda,  \
                      TYPE* tau, TYPE* c, std::int64_t ldc, TYPE* scratchpad,                     \
                      std::int64_t scratchpad_size,                                               \
                      const std::vector<sycl::event>& dependencies) {                             \
        return unmqr(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, side, trans, m, n, k, a, lda,  \
                     tau, c, ldc, scratchpad, scratchpad_size, dependencies);                     \
    }

UNMQR_LAUNCHER_USM(std::complex<float>, rocsolver_cunmqr)
UNMQR_LAUNCHER_USM(std::complex<double>, rocsolver_zunmqr)

#undef UNMQR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event unmtr(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::math::side side, oneapi::math::uplo uplo,
                         oneapi::math::transpose trans, std::int64_t m, std::int64_t n, T* a,
                         std::int64_t lda, T* tau, T* c, std::int64_t ldc, T* scratchpad,
                         std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldc, scratchpad_size);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType*>(a);
            auto tau_ = reinterpret_cast<rocmDataType*>(tau);
            auto c_ = reinterpret_cast<rocmDataType*>(c);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_side_mode(side),
                                        get_rocblas_fill_mode(uplo), get_rocblas_operation(trans),
                                        m, n, a_, lda, tau_, c_, ldc);
        });
    });
    return done;
}

#define UNMTR_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                             \
    sycl::event unmtr(sycl::queue& queue, oneapi::math::side side, oneapi::math::uplo uplo,     \
                      oneapi::math::transpose trans, std::int64_t m, std::int64_t n, TYPE* a,   \
                      std::int64_t lda, TYPE* tau, TYPE* c, std::int64_t ldc, TYPE* scratchpad, \
                      std::int64_t scratchpad_size,                                             \
                      const std::vector<sycl::event>& dependencies) {                           \
        return unmtr(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, side, uplo, trans, m, n, a,  \
                     lda, tau, c, ldc, scratchpad, scratchpad_size, dependencies);              \
    }

UNMTR_LAUNCHER_USM(std::complex<float>, rocsolver_cunmtr)
UNMTR_LAUNCHER_USM(std::complex<double>, rocsolver_zunmtr)

#undef UNMTR_LAUNCHER_USM

// SCRATCHPAD APIs

#define GEBRD_LAUNCHER_SCRATCH(TYPE)                                                              \
    template <>                                                                                   \
    std::int64_t gebrd_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t m, std::int64_t n, \
                                             std::int64_t lda) {                                  \
        return 0;                                                                                 \
    }

GEBRD_LAUNCHER_SCRATCH(float)
GEBRD_LAUNCHER_SCRATCH(double)
GEBRD_LAUNCHER_SCRATCH(std::complex<float>)
GEBRD_LAUNCHER_SCRATCH(std::complex<double>)

#undef GEBRD_LAUNCHER_SCRATCH

template <>
std::int64_t gerqf_scratchpad_size<float>(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                          std::int64_t lda) {
    throw unimplemented("lapack", "gerqf_scratchpad_size");
}
template <>
std::int64_t gerqf_scratchpad_size<double>(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                           std::int64_t lda) {
    throw unimplemented("lapack", "gerqf_scratchpad_size");
}
template <>
std::int64_t gerqf_scratchpad_size<std::complex<float>>(sycl::queue& queue, std::int64_t m,
                                                        std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "gerqf_scratchpad_size");
}
template <>
std::int64_t gerqf_scratchpad_size<std::complex<double>>(sycl::queue& queue, std::int64_t m,
                                                         std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "gerqf_scratchpad_size");
}

#define GEQRF_LAUNCHER_SCRATCH(TYPE)                                                              \
    template <>                                                                                   \
    std::int64_t geqrf_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t m, std::int64_t n, \
                                             std::int64_t lda) {                                  \
        return 0;                                                                                 \
    }

GEQRF_LAUNCHER_SCRATCH(float)
GEQRF_LAUNCHER_SCRATCH(double)
GEQRF_LAUNCHER_SCRATCH(std::complex<float>)
GEQRF_LAUNCHER_SCRATCH(std::complex<double>)

#undef GEQRF_LAUNCHER_SCRATCH

#define GESVD_LAUNCHER_SCRATCH(TYPE)                                                             \
    template <>                                                                                  \
    std::int64_t gesvd_scratchpad_size<TYPE>(                                                    \
        sycl::queue & queue, oneapi::math::jobsvd jobu, oneapi::math::jobsvd jobvt,              \
        std::int64_t m, std::int64_t n, std::int64_t lda, std::int64_t ldu, std::int64_t ldvt) { \
        return std::min(m, n) - 1;                                                               \
    }

GESVD_LAUNCHER_SCRATCH(float)
GESVD_LAUNCHER_SCRATCH(double)
GESVD_LAUNCHER_SCRATCH(std::complex<float>)
GESVD_LAUNCHER_SCRATCH(std::complex<double>)

#undef GESVD_LAUNCHER_SCRATCH

#define GETRF_LAUNCHER_SCRATCH(TYPE)                                                              \
    template <>                                                                                   \
    std::int64_t getrf_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t m, std::int64_t n, \
                                             std::int64_t lda) {                                  \
        return 0;                                                                                 \
    }

GETRF_LAUNCHER_SCRATCH(float)
GETRF_LAUNCHER_SCRATCH(double)
GETRF_LAUNCHER_SCRATCH(std::complex<float>)
GETRF_LAUNCHER_SCRATCH(std::complex<double>)

#undef GETRF_LAUNCHER_SCRATCH

template <>
std::int64_t getri_scratchpad_size<float>(sycl::queue& queue, std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "getri_scratchpad_size");
}
template <>
std::int64_t getri_scratchpad_size<double>(sycl::queue& queue, std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "getri_scratchpad_size");
}
template <>
std::int64_t getri_scratchpad_size<std::complex<float>>(sycl::queue& queue, std::int64_t n,
                                                        std::int64_t lda) {
    throw unimplemented("lapack", "getri_scratchpad_size");
}
template <>
std::int64_t getri_scratchpad_size<std::complex<double>>(sycl::queue& queue, std::int64_t n,
                                                         std::int64_t lda) {
    throw unimplemented("lapack", "getri_scratchpad_size");
}

#define GETRS_LAUNCHER_SCRATCH(TYPE)                                                              \
    template <>                                                                                   \
    std::int64_t getrs_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::transpose trans,  \
                                             std::int64_t n, std::int64_t nrhs, std::int64_t lda, \
                                             std::int64_t ldb) {                                  \
        return 0;                                                                                 \
    }

GETRS_LAUNCHER_SCRATCH(float)
GETRS_LAUNCHER_SCRATCH(double)
GETRS_LAUNCHER_SCRATCH(std::complex<float>)
GETRS_LAUNCHER_SCRATCH(std::complex<double>)

#undef GETRS_LAUNCHER_SCRATCH

#define HEEVD_LAUNCHER_SCRATCH(TYPE)                                                      \
    template <>                                                                           \
    std::int64_t heevd_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::job jobz, \
                                             oneapi::math::uplo uplo, std::int64_t n,     \
                                             std::int64_t lda) {                          \
        return n;                                                                         \
    }

HEEVD_LAUNCHER_SCRATCH(std::complex<float>)
HEEVD_LAUNCHER_SCRATCH(std::complex<double>)

#undef HEEVD_LAUNCHER_SCRATCH

#define HEGVD_LAUNCHER_SCRATCH(TYPE)                                                               \
    template <>                                                                                    \
    std::int64_t hegvd_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t itype,              \
                                             oneapi::math::job jobz, oneapi::math::uplo uplo,      \
                                             std::int64_t n, std::int64_t lda, std::int64_t ldb) { \
        return n;                                                                                  \
    }

HEGVD_LAUNCHER_SCRATCH(std::complex<float>)
HEGVD_LAUNCHER_SCRATCH(std::complex<double>)

#undef HEGVD_LAUNCHER_SCRATCH

#define HETRD_LAUNCHER_SCRATCH(TYPE)                                                       \
    template <>                                                                            \
    std::int64_t hetrd_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::uplo uplo, \
                                             std::int64_t n, std::int64_t lda) {           \
        return 0;                                                                          \
    }

HETRD_LAUNCHER_SCRATCH(std::complex<float>)
HETRD_LAUNCHER_SCRATCH(std::complex<double>)

#undef HETRD_LAUNCHER_SCRATCH

template <>
std::int64_t hetrf_scratchpad_size<std::complex<float>>(sycl::queue& queue, oneapi::math::uplo uplo,
                                                        std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "hetrf_scratchpad_size");
}
template <>
std::int64_t hetrf_scratchpad_size<std::complex<double>>(sycl::queue& queue,
                                                         oneapi::math::uplo uplo, std::int64_t n,
                                                         std::int64_t lda) {
    throw unimplemented("lapack", "hetrf_scratchpad_size");
}

#define ORGBR_LAUNCHER_SCRATCH(TYPE)                                                          \
    template <>                                                                               \
    std::int64_t orgbr_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::generate vec, \
                                             std::int64_t m, std::int64_t n, std::int64_t k,  \
                                             std::int64_t lda) {                              \
        return 0;                                                                             \
    }

ORGBR_LAUNCHER_SCRATCH(float)
ORGBR_LAUNCHER_SCRATCH(double)

#undef ORGBR_LAUNCHER_SCRATCH

#define ORGTR_LAUNCHER_SCRATCH(TYPE)                                                       \
    template <>                                                                            \
    std::int64_t orgtr_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::uplo uplo, \
                                             std::int64_t n, std::int64_t lda) {           \
        return 0;                                                                          \
    }

ORGTR_LAUNCHER_SCRATCH(float)
ORGTR_LAUNCHER_SCRATCH(double)

#undef ORGTR_LAUNCHER_SCRATCH

#define ORGQR_LAUNCHER_SCRATCH(TYPE)                                                              \
    template <>                                                                                   \
    std::int64_t orgqr_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t m, std::int64_t n, \
                                             std::int64_t k, std::int64_t lda) {                  \
        return 0;                                                                                 \
    }

ORGQR_LAUNCHER_SCRATCH(float)
ORGQR_LAUNCHER_SCRATCH(double)

#undef ORGQR_LAUNCHER_SCRATCH

template <>
std::int64_t ormrq_scratchpad_size<float>(sycl::queue& queue, oneapi::math::side side,
                                          oneapi::math::transpose trans, std::int64_t m,
                                          std::int64_t n, std::int64_t k, std::int64_t lda,
                                          std::int64_t ldc) {
    throw unimplemented("lapack", "ormrq_scratchpad_size");
}
template <>
std::int64_t ormrq_scratchpad_size<double>(sycl::queue& queue, oneapi::math::side side,
                                           oneapi::math::transpose trans, std::int64_t m,
                                           std::int64_t n, std::int64_t k, std::int64_t lda,
                                           std::int64_t ldc) {
    throw unimplemented("lapack", "ormrq_scratchpad_size");
}

#define ORMQRF_LAUNCHER_SCRATCH(TYPE)                                                         \
    template <>                                                                               \
    std::int64_t ormqr_scratchpad_size<TYPE>(                                                 \
        sycl::queue & queue, oneapi::math::side side, oneapi::math::transpose trans,          \
        std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda, std::int64_t ldc) { \
        return 0;                                                                             \
    }

ORMQRF_LAUNCHER_SCRATCH(float)
ORMQRF_LAUNCHER_SCRATCH(double)

#undef ORMQRF_LAUNCHER_SCRATCH

#define ORMTR_LAUNCHER_SCRATCH(TYPE)                                                               \
    template <>                                                                                    \
    std::int64_t ormtr_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::side side,         \
                                             oneapi::math::uplo uplo,                              \
                                             oneapi::math::transpose trans, std::int64_t m,        \
                                             std::int64_t n, std::int64_t lda, std::int64_t ldc) { \
        return 0;                                                                                  \
    }

ORMTR_LAUNCHER_SCRATCH(float)
ORMTR_LAUNCHER_SCRATCH(double)

#undef ORMTR_LAUNCHER_SCRATCH

#define POTRF_LAUNCHER_SCRATCH(TYPE)                                                       \
    template <>                                                                            \
    std::int64_t potrf_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::uplo uplo, \
                                             std::int64_t n, std::int64_t lda) {           \
        return 0;                                                                          \
    }

POTRF_LAUNCHER_SCRATCH(float)
POTRF_LAUNCHER_SCRATCH(double)
POTRF_LAUNCHER_SCRATCH(std::complex<float>)
POTRF_LAUNCHER_SCRATCH(std::complex<double>)

#undef POTRF_LAUNCHER_SCRATCH

#define POTRS_LAUNCHER_SCRATCH(TYPE)                                                              \
    template <>                                                                                   \
    std::int64_t potrs_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::uplo uplo,        \
                                             std::int64_t n, std::int64_t nrhs, std::int64_t lda, \
                                             std::int64_t ldb) {                                  \
        return 0;                                                                                 \
    }

POTRS_LAUNCHER_SCRATCH(float)
POTRS_LAUNCHER_SCRATCH(double)
POTRS_LAUNCHER_SCRATCH(std::complex<float>)
POTRS_LAUNCHER_SCRATCH(std::complex<double>)

#undef POTRS_LAUNCHER_SCRATCH

#define POTRI_LAUNCHER_SCRATCH(TYPE)                                                       \
    template <>                                                                            \
    std::int64_t potri_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::uplo uplo, \
                                             std::int64_t n, std::int64_t lda) {           \
        return 0;                                                                          \
    }

POTRI_LAUNCHER_SCRATCH(float)
POTRI_LAUNCHER_SCRATCH(double)
POTRI_LAUNCHER_SCRATCH(std::complex<float>)
POTRI_LAUNCHER_SCRATCH(std::complex<double>)

#undef POTRI_LAUNCHER_SCRATCH

#define SYTRF_LAUNCHER_SCRATCH(TYPE)                                                       \
    template <>                                                                            \
    std::int64_t sytrf_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::uplo uplo, \
                                             std::int64_t n, std::int64_t lda) {           \
        return 0;                                                                          \
    }

SYTRF_LAUNCHER_SCRATCH(float)
SYTRF_LAUNCHER_SCRATCH(double)
SYTRF_LAUNCHER_SCRATCH(std::complex<float>)
SYTRF_LAUNCHER_SCRATCH(std::complex<double>)

#undef SYTRF_LAUNCHER_SCRATCH

#define SYEVD_LAUNCHER_SCRATCH(TYPE)                                                      \
    template <>                                                                           \
    std::int64_t syevd_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::job jobz, \
                                             oneapi::math::uplo uplo, std::int64_t n,     \
                                             std::int64_t lda) {                          \
        return n;                                                                         \
    }

SYEVD_LAUNCHER_SCRATCH(float)
SYEVD_LAUNCHER_SCRATCH(double)

#undef SYEVD_LAUNCHER_SCRATCH

#define SYGVD_LAUNCHER_SCRATCH(TYPE)                                                               \
    template <>                                                                                    \
    std::int64_t sygvd_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t itype,              \
                                             oneapi::math::job jobz, oneapi::math::uplo uplo,      \
                                             std::int64_t n, std::int64_t lda, std::int64_t ldb) { \
        return n;                                                                                  \
    }

SYGVD_LAUNCHER_SCRATCH(float)
SYGVD_LAUNCHER_SCRATCH(double)

#undef SYGVD_LAUNCHER_SCRATCH

#define SYTRD_LAUNCHER_SCRATCH(TYPE)                                                       \
    template <>                                                                            \
    std::int64_t sytrd_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::uplo uplo, \
                                             std::int64_t n, std::int64_t lda) {           \
        return 0;                                                                          \
    }

SYTRD_LAUNCHER_SCRATCH(float)
SYTRD_LAUNCHER_SCRATCH(double)

#undef SYTRD_LAUNCHER_SCRATCH

#define TRTRS_LAUNCHER_SCRATCH(TYPE)                                                  \
    template <>                                                                       \
    std::int64_t trtrs_scratchpad_size<TYPE>(                                         \
        sycl::queue & queue, oneapi::math::uplo uplo, oneapi::math::transpose trans,  \
        oneapi::math::diag diag, std::int64_t n, std::int64_t nrhs, std::int64_t lda, \
        std::int64_t ldb) {                                                           \
        return 0;                                                                     \
    }

TRTRS_LAUNCHER_SCRATCH(float)
TRTRS_LAUNCHER_SCRATCH(double)
TRTRS_LAUNCHER_SCRATCH(std::complex<float>)
TRTRS_LAUNCHER_SCRATCH(std::complex<double>)

#undef TRTRS_LAUNCHER_SCRATCH

#define UNGBR_LAUNCHER_SCRATCH(TYPE)                                                          \
    template <>                                                                               \
    std::int64_t ungbr_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::generate vec, \
                                             std::int64_t m, std::int64_t n, std::int64_t k,  \
                                             std::int64_t lda) {                              \
        return 0;                                                                             \
    }

UNGBR_LAUNCHER_SCRATCH(std::complex<float>)
UNGBR_LAUNCHER_SCRATCH(std::complex<double>)

#undef UNGBR_LAUNCHER_SCRATCH

#define UNGQR_LAUNCHER_SCRATCH(TYPE)                                                              \
    template <>                                                                                   \
    std::int64_t ungqr_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t m, std::int64_t n, \
                                             std::int64_t k, std::int64_t lda) {                  \
        return 0;                                                                                 \
    }

UNGQR_LAUNCHER_SCRATCH(std::complex<float>)
UNGQR_LAUNCHER_SCRATCH(std::complex<double>)

#undef UNGQR_LAUNCHER_SCRATCH

#define UNGTR_LAUNCHER_SCRATCH(TYPE)                                                       \
    template <>                                                                            \
    std::int64_t ungtr_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::uplo uplo, \
                                             std::int64_t n, std::int64_t lda) {           \
        return 0;                                                                          \
    }

UNGTR_LAUNCHER_SCRATCH(std::complex<float>)
UNGTR_LAUNCHER_SCRATCH(std::complex<double>)

#undef UNGTR_LAUNCHER_SCRATCH

template <>
std::int64_t unmrq_scratchpad_size<std::complex<float>>(sycl::queue& queue, oneapi::math::side side,
                                                        oneapi::math::transpose trans,
                                                        std::int64_t m, std::int64_t n,
                                                        std::int64_t k, std::int64_t lda,
                                                        std::int64_t ldc) {
    throw unimplemented("lapack", "unmrq_scratchpad_size");
}
template <>
std::int64_t unmrq_scratchpad_size<std::complex<double>>(
    sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans, std::int64_t m,
    std::int64_t n, std::int64_t k, std::int64_t lda, std::int64_t ldc) {
    throw unimplemented("lapack", "unmrq_scratchpad_size");
}

#define UNMQR_LAUNCHER_SCRATCH(TYPE)                                                          \
    template <>                                                                               \
    std::int64_t unmqr_scratchpad_size<TYPE>(                                                 \
        sycl::queue & queue, oneapi::math::side side, oneapi::math::transpose trans,          \
        std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda, std::int64_t ldc) { \
        return 0;                                                                             \
    }

UNMQR_LAUNCHER_SCRATCH(std::complex<float>)
UNMQR_LAUNCHER_SCRATCH(std::complex<double>)

#undef UNMQR_LAUNCHER_SCRATCH

#define UNMTR_LAUNCHER_SCRATCH(TYPE)                                                               \
    template <>                                                                                    \
    std::int64_t unmtr_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::math::side side,         \
                                             oneapi::math::uplo uplo,                              \
                                             oneapi::math::transpose trans, std::int64_t m,        \
                                             std::int64_t n, std::int64_t lda, std::int64_t ldc) { \
        return 0;                                                                                  \
    }

UNMTR_LAUNCHER_SCRATCH(std::complex<float>)
UNMTR_LAUNCHER_SCRATCH(std::complex<double>)

#undef UNMTR_LAUNCHER_SCRATCH

} // namespace rocsolver
} // namespace lapack
} // namespace math
} // namespace oneapi
