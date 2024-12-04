/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#pragma once

#include <complex>
#include <cstdint>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math/types.hpp"
#include "oneapi/math/lapack/types.hpp"
#include "oneapi/math/lapack/exceptions.hpp"
#include "oneapi/math/detail/get_device_id.hpp"
#include "oneapi/math/lapack/detail/lapack_loader.hpp"

namespace oneapi {
namespace math {
namespace lapack {

static inline void gebrd(sycl::queue& queue, std::int64_t m, std::int64_t n,
                         sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<float>& d, sycl::buffer<float>& e,
                         sycl::buffer<std::complex<float>>& tauq,
                         sycl::buffer<std::complex<float>>& taup,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::gebrd(get_device_id(queue), queue, m, n, a, lda, d, e, tauq, taup, scratchpad,
                  scratchpad_size);
}
static inline void gebrd(sycl::queue& queue, std::int64_t m, std::int64_t n,
                         sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& d,
                         sycl::buffer<double>& e, sycl::buffer<double>& tauq,
                         sycl::buffer<double>& taup, sycl::buffer<double>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::gebrd(get_device_id(queue), queue, m, n, a, lda, d, e, tauq, taup, scratchpad,
                  scratchpad_size);
}
static inline void gebrd(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<float>& a,
                         std::int64_t lda, sycl::buffer<float>& d, sycl::buffer<float>& e,
                         sycl::buffer<float>& tauq, sycl::buffer<float>& taup,
                         sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    detail::gebrd(get_device_id(queue), queue, m, n, a, lda, d, e, tauq, taup, scratchpad,
                  scratchpad_size);
}
static inline void gebrd(sycl::queue& queue, std::int64_t m, std::int64_t n,
                         sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<double>& d, sycl::buffer<double>& e,
                         sycl::buffer<std::complex<double>>& tauq,
                         sycl::buffer<std::complex<double>>& taup,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::gebrd(get_device_id(queue), queue, m, n, a, lda, d, e, tauq, taup, scratchpad,
                  scratchpad_size);
}
static inline void gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<float>& a,
                         std::int64_t lda, sycl::buffer<float>& tau,
                         sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    detail::gerqf(get_device_id(queue), queue, m, n, a, lda, tau, scratchpad, scratchpad_size);
}
static inline void gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n,
                         sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& tau,
                         sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    detail::gerqf(get_device_id(queue), queue, m, n, a, lda, tau, scratchpad, scratchpad_size);
}
static inline void gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n,
                         sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>>& tau,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::gerqf(get_device_id(queue), queue, m, n, a, lda, tau, scratchpad, scratchpad_size);
}
static inline void gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n,
                         sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>>& tau,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::gerqf(get_device_id(queue), queue, m, n, a, lda, tau, scratchpad, scratchpad_size);
}
static inline void geqrf(sycl::queue& queue, std::int64_t m, std::int64_t n,
                         sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>>& tau,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::geqrf(get_device_id(queue), queue, m, n, a, lda, tau, scratchpad, scratchpad_size);
}
static inline void geqrf(sycl::queue& queue, std::int64_t m, std::int64_t n,
                         sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& tau,
                         sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    detail::geqrf(get_device_id(queue), queue, m, n, a, lda, tau, scratchpad, scratchpad_size);
}
static inline void geqrf(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<float>& a,
                         std::int64_t lda, sycl::buffer<float>& tau,
                         sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    detail::geqrf(get_device_id(queue), queue, m, n, a, lda, tau, scratchpad, scratchpad_size);
}
static inline void geqrf(sycl::queue& queue, std::int64_t m, std::int64_t n,
                         sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>>& tau,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::geqrf(get_device_id(queue), queue, m, n, a, lda, tau, scratchpad, scratchpad_size);
}
static inline void getrf(sycl::queue& queue, std::int64_t m, std::int64_t n,
                         sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::int64_t>& ipiv,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::getrf(get_device_id(queue), queue, m, n, a, lda, ipiv, scratchpad, scratchpad_size);
}
static inline void getrf(sycl::queue& queue, std::int64_t m, std::int64_t n,
                         sycl::buffer<double>& a, std::int64_t lda,
                         sycl::buffer<std::int64_t>& ipiv, sycl::buffer<double>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::getrf(get_device_id(queue), queue, m, n, a, lda, ipiv, scratchpad, scratchpad_size);
}
static inline void getrf(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<float>& a,
                         std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
                         sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    detail::getrf(get_device_id(queue), queue, m, n, a, lda, ipiv, scratchpad, scratchpad_size);
}
static inline void getrf(sycl::queue& queue, std::int64_t m, std::int64_t n,
                         sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::int64_t>& ipiv,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::getrf(get_device_id(queue), queue, m, n, a, lda, ipiv, scratchpad, scratchpad_size);
}
static inline void getri(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>>& a,
                         std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::getri(get_device_id(queue), queue, n, a, lda, ipiv, scratchpad, scratchpad_size);
}
static inline void getri(sycl::queue& queue, std::int64_t n, sycl::buffer<double>& a,
                         std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
                         sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    detail::getri(get_device_id(queue), queue, n, a, lda, ipiv, scratchpad, scratchpad_size);
}
static inline void getri(sycl::queue& queue, std::int64_t n, sycl::buffer<float>& a,
                         std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
                         sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    detail::getri(get_device_id(queue), queue, n, a, lda, ipiv, scratchpad, scratchpad_size);
}
static inline void getri(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>>& a,
                         std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::getri(get_device_id(queue), queue, n, a, lda, ipiv, scratchpad, scratchpad_size);
}
static inline void getrs(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                         std::int64_t nrhs, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::int64_t>& ipiv, sycl::buffer<std::complex<float>>& b,
                         std::int64_t ldb, sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::getrs(get_device_id(queue), queue, trans, n, nrhs, a, lda, ipiv, b, ldb, scratchpad,
                  scratchpad_size);
}
static inline void getrs(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                         std::int64_t nrhs, sycl::buffer<double>& a, std::int64_t lda,
                         sycl::buffer<std::int64_t>& ipiv, sycl::buffer<double>& b,
                         std::int64_t ldb, sycl::buffer<double>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::getrs(get_device_id(queue), queue, trans, n, nrhs, a, lda, ipiv, b, ldb, scratchpad,
                  scratchpad_size);
}
static inline void getrs(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                         std::int64_t nrhs, sycl::buffer<float>& a, std::int64_t lda,
                         sycl::buffer<std::int64_t>& ipiv, sycl::buffer<float>& b, std::int64_t ldb,
                         sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    detail::getrs(get_device_id(queue), queue, trans, n, nrhs, a, lda, ipiv, b, ldb, scratchpad,
                  scratchpad_size);
}
static inline void getrs(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                         std::int64_t nrhs, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::int64_t>& ipiv, sycl::buffer<std::complex<double>>& b,
                         std::int64_t ldb, sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::getrs(get_device_id(queue), queue, trans, n, nrhs, a, lda, ipiv, b, ldb, scratchpad,
                  scratchpad_size);
}
static inline void gesvd(sycl::queue& queue, oneapi::math::jobsvd jobu, oneapi::math::jobsvd jobvt,
                         std::int64_t m, std::int64_t n, sycl::buffer<double>& a, std::int64_t lda,
                         sycl::buffer<double>& s, sycl::buffer<double>& u, std::int64_t ldu,
                         sycl::buffer<double>& vt, std::int64_t ldvt,
                         sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    detail::gesvd(get_device_id(queue), queue, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,
                  scratchpad, scratchpad_size);
}
static inline void gesvd(sycl::queue& queue, oneapi::math::jobsvd jobu, oneapi::math::jobsvd jobvt,
                         std::int64_t m, std::int64_t n, sycl::buffer<float>& a, std::int64_t lda,
                         sycl::buffer<float>& s, sycl::buffer<float>& u, std::int64_t ldu,
                         sycl::buffer<float>& vt, std::int64_t ldvt,
                         sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    detail::gesvd(get_device_id(queue), queue, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,
                  scratchpad, scratchpad_size);
}
static inline void gesvd(sycl::queue& queue, oneapi::math::jobsvd jobu, oneapi::math::jobsvd jobvt,
                         std::int64_t m, std::int64_t n, sycl::buffer<std::complex<float>>& a,
                         std::int64_t lda, sycl::buffer<float>& s,
                         sycl::buffer<std::complex<float>>& u, std::int64_t ldu,
                         sycl::buffer<std::complex<float>>& vt, std::int64_t ldvt,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::gesvd(get_device_id(queue), queue, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,
                  scratchpad, scratchpad_size);
}
static inline void gesvd(sycl::queue& queue, oneapi::math::jobsvd jobu, oneapi::math::jobsvd jobvt,
                         std::int64_t m, std::int64_t n, sycl::buffer<std::complex<double>>& a,
                         std::int64_t lda, sycl::buffer<double>& s,
                         sycl::buffer<std::complex<double>>& u, std::int64_t ldu,
                         sycl::buffer<std::complex<double>>& vt, std::int64_t ldvt,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::gesvd(get_device_id(queue), queue, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,
                  scratchpad, scratchpad_size);
}
static inline void heevd(sycl::queue& queue, oneapi::math::job jobz, oneapi::math::uplo uplo,
                         std::int64_t n, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<float>& w, sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::heevd(get_device_id(queue), queue, jobz, uplo, n, a, lda, w, scratchpad,
                  scratchpad_size);
}
static inline void heevd(sycl::queue& queue, oneapi::math::job jobz, oneapi::math::uplo uplo,
                         std::int64_t n, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<double>& w, sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::heevd(get_device_id(queue), queue, jobz, uplo, n, a, lda, w, scratchpad,
                  scratchpad_size);
}
static inline void hegvd(sycl::queue& queue, std::int64_t itype, oneapi::math::job jobz,
                         oneapi::math::uplo uplo, std::int64_t n,
                         sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>>& b, std::int64_t ldb,
                         sycl::buffer<float>& w, sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::hegvd(get_device_id(queue), queue, itype, jobz, uplo, n, a, lda, b, ldb, w, scratchpad,
                  scratchpad_size);
}
static inline void hegvd(sycl::queue& queue, std::int64_t itype, oneapi::math::job jobz,
                         oneapi::math::uplo uplo, std::int64_t n,
                         sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>>& b, std::int64_t ldb,
                         sycl::buffer<double>& w, sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::hegvd(get_device_id(queue), queue, itype, jobz, uplo, n, a, lda, b, ldb, w, scratchpad,
                  scratchpad_size);
}
static inline void hetrd(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                         sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<float>& d, sycl::buffer<float>& e,
                         sycl::buffer<std::complex<float>>& tau,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::hetrd(get_device_id(queue), queue, uplo, n, a, lda, d, e, tau, scratchpad,
                  scratchpad_size);
}
static inline void hetrd(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                         sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<double>& d, sycl::buffer<double>& e,
                         sycl::buffer<std::complex<double>>& tau,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::hetrd(get_device_id(queue), queue, uplo, n, a, lda, d, e, tau, scratchpad,
                  scratchpad_size);
}
static inline void hetrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                         sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::int64_t>& ipiv,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::hetrf(get_device_id(queue), queue, uplo, n, a, lda, ipiv, scratchpad, scratchpad_size);
}
static inline void hetrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                         sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::int64_t>& ipiv,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::hetrf(get_device_id(queue), queue, uplo, n, a, lda, ipiv, scratchpad, scratchpad_size);
}
static inline void orgbr(sycl::queue& queue, oneapi::math::generate vec, std::int64_t m,
                         std::int64_t n, std::int64_t k, sycl::buffer<float>& a, std::int64_t lda,
                         sycl::buffer<float>& tau, sycl::buffer<float>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::orgbr(get_device_id(queue), queue, vec, m, n, k, a, lda, tau, scratchpad,
                  scratchpad_size);
}
static inline void orgbr(sycl::queue& queue, oneapi::math::generate vec, std::int64_t m,
                         std::int64_t n, std::int64_t k, sycl::buffer<double>& a, std::int64_t lda,
                         sycl::buffer<double>& tau, sycl::buffer<double>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::orgbr(get_device_id(queue), queue, vec, m, n, k, a, lda, tau, scratchpad,
                  scratchpad_size);
}
static inline void orgqr(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                         sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& tau,
                         sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    detail::orgqr(get_device_id(queue), queue, m, n, k, a, lda, tau, scratchpad, scratchpad_size);
}
static inline void orgqr(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                         sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& tau,
                         sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    detail::orgqr(get_device_id(queue), queue, m, n, k, a, lda, tau, scratchpad, scratchpad_size);
}
static inline void orgtr(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                         sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& tau,
                         sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    detail::orgtr(get_device_id(queue), queue, uplo, n, a, lda, tau, scratchpad, scratchpad_size);
}
static inline void orgtr(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                         sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& tau,
                         sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    detail::orgtr(get_device_id(queue), queue, uplo, n, a, lda, tau, scratchpad, scratchpad_size);
}
static inline void ormtr(sycl::queue& queue, oneapi::math::side side, oneapi::math::uplo uplo,
                         oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
                         sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& tau,
                         sycl::buffer<float>& c, std::int64_t ldc, sycl::buffer<float>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::ormtr(get_device_id(queue), queue, side, uplo, trans, m, n, a, lda, tau, c, ldc,
                  scratchpad, scratchpad_size);
}
static inline void ormtr(sycl::queue& queue, oneapi::math::side side, oneapi::math::uplo uplo,
                         oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
                         sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& tau,
                         sycl::buffer<double>& c, std::int64_t ldc,
                         sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    detail::ormtr(get_device_id(queue), queue, side, uplo, trans, m, n, a, lda, tau, c, ldc,
                  scratchpad, scratchpad_size);
}
static inline void ormrq(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
                         std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<float>& a,
                         std::int64_t lda, sycl::buffer<float>& tau, sycl::buffer<float>& c,
                         std::int64_t ldc, sycl::buffer<float>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::ormrq(get_device_id(queue), queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                  scratchpad, scratchpad_size);
}
static inline void ormrq(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
                         std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<double>& a,
                         std::int64_t lda, sycl::buffer<double>& tau, sycl::buffer<double>& c,
                         std::int64_t ldc, sycl::buffer<double>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::ormrq(get_device_id(queue), queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                  scratchpad, scratchpad_size);
}
static inline void ormqr(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
                         std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<double>& a,
                         std::int64_t lda, sycl::buffer<double>& tau, sycl::buffer<double>& c,
                         std::int64_t ldc, sycl::buffer<double>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::ormqr(get_device_id(queue), queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                  scratchpad, scratchpad_size);
}
static inline void ormqr(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
                         std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<float>& a,
                         std::int64_t lda, sycl::buffer<float>& tau, sycl::buffer<float>& c,
                         std::int64_t ldc, sycl::buffer<float>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::ormqr(get_device_id(queue), queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                  scratchpad, scratchpad_size);
}
static inline void potrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                         sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::potrf(get_device_id(queue), queue, uplo, n, a, lda, scratchpad, scratchpad_size);
}
static inline void potrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                         sycl::buffer<double>& a, std::int64_t lda,
                         sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    detail::potrf(get_device_id(queue), queue, uplo, n, a, lda, scratchpad, scratchpad_size);
}
static inline void potrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                         sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::potrf(get_device_id(queue), queue, uplo, n, a, lda, scratchpad, scratchpad_size);
}
static inline void potrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                         sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::potrf(get_device_id(queue), queue, uplo, n, a, lda, scratchpad, scratchpad_size);
}
static inline void potri(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                         sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::potri(get_device_id(queue), queue, uplo, n, a, lda, scratchpad, scratchpad_size);
}
static inline void potri(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                         sycl::buffer<double>& a, std::int64_t lda,
                         sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    detail::potri(get_device_id(queue), queue, uplo, n, a, lda, scratchpad, scratchpad_size);
}
static inline void potri(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                         sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::potri(get_device_id(queue), queue, uplo, n, a, lda, scratchpad, scratchpad_size);
}
static inline void potri(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                         sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::potri(get_device_id(queue), queue, uplo, n, a, lda, scratchpad, scratchpad_size);
}
static inline void potrs(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                         std::int64_t nrhs, sycl::buffer<float>& a, std::int64_t lda,
                         sycl::buffer<float>& b, std::int64_t ldb, sycl::buffer<float>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::potrs(get_device_id(queue), queue, uplo, n, nrhs, a, lda, b, ldb, scratchpad,
                  scratchpad_size);
}
static inline void potrs(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                         std::int64_t nrhs, sycl::buffer<double>& a, std::int64_t lda,
                         sycl::buffer<double>& b, std::int64_t ldb,
                         sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    detail::potrs(get_device_id(queue), queue, uplo, n, nrhs, a, lda, b, ldb, scratchpad,
                  scratchpad_size);
}
static inline void potrs(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                         std::int64_t nrhs, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>>& b, std::int64_t ldb,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::potrs(get_device_id(queue), queue, uplo, n, nrhs, a, lda, b, ldb, scratchpad,
                  scratchpad_size);
}
static inline void potrs(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                         std::int64_t nrhs, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>>& b, std::int64_t ldb,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::potrs(get_device_id(queue), queue, uplo, n, nrhs, a, lda, b, ldb, scratchpad,
                  scratchpad_size);
}
static inline void syevd(sycl::queue& queue, oneapi::math::job jobz, oneapi::math::uplo uplo,
                         std::int64_t n, sycl::buffer<double>& a, std::int64_t lda,
                         sycl::buffer<double>& w, sycl::buffer<double>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::syevd(get_device_id(queue), queue, jobz, uplo, n, a, lda, w, scratchpad,
                  scratchpad_size);
}
static inline void syevd(sycl::queue& queue, oneapi::math::job jobz, oneapi::math::uplo uplo,
                         std::int64_t n, sycl::buffer<float>& a, std::int64_t lda,
                         sycl::buffer<float>& w, sycl::buffer<float>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::syevd(get_device_id(queue), queue, jobz, uplo, n, a, lda, w, scratchpad,
                  scratchpad_size);
}
static inline void sygvd(sycl::queue& queue, std::int64_t itype, oneapi::math::job jobz,
                         oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<double>& a,
                         std::int64_t lda, sycl::buffer<double>& b, std::int64_t ldb,
                         sycl::buffer<double>& w, sycl::buffer<double>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::sygvd(get_device_id(queue), queue, itype, jobz, uplo, n, a, lda, b, ldb, w, scratchpad,
                  scratchpad_size);
}
static inline void sygvd(sycl::queue& queue, std::int64_t itype, oneapi::math::job jobz,
                         oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<float>& a,
                         std::int64_t lda, sycl::buffer<float>& b, std::int64_t ldb,
                         sycl::buffer<float>& w, sycl::buffer<float>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::sygvd(get_device_id(queue), queue, itype, jobz, uplo, n, a, lda, b, ldb, w, scratchpad,
                  scratchpad_size);
}
static inline void sytrd(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                         sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& d,
                         sycl::buffer<double>& e, sycl::buffer<double>& tau,
                         sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    detail::sytrd(get_device_id(queue), queue, uplo, n, a, lda, d, e, tau, scratchpad,
                  scratchpad_size);
}
static inline void sytrd(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                         sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& d,
                         sycl::buffer<float>& e, sycl::buffer<float>& tau,
                         sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    detail::sytrd(get_device_id(queue), queue, uplo, n, a, lda, d, e, tau, scratchpad,
                  scratchpad_size);
}
static inline void sytrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                         sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
                         sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    detail::sytrf(get_device_id(queue), queue, uplo, n, a, lda, ipiv, scratchpad, scratchpad_size);
}
static inline void sytrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                         sycl::buffer<double>& a, std::int64_t lda,
                         sycl::buffer<std::int64_t>& ipiv, sycl::buffer<double>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::sytrf(get_device_id(queue), queue, uplo, n, a, lda, ipiv, scratchpad, scratchpad_size);
}
static inline void sytrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                         sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::int64_t>& ipiv,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::sytrf(get_device_id(queue), queue, uplo, n, a, lda, ipiv, scratchpad, scratchpad_size);
}
static inline void sytrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                         sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::int64_t>& ipiv,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::sytrf(get_device_id(queue), queue, uplo, n, a, lda, ipiv, scratchpad, scratchpad_size);
}
static inline void trtrs(sycl::queue& queue, oneapi::math::uplo uplo, oneapi::math::transpose trans,
                         oneapi::math::diag diag, std::int64_t n, std::int64_t nrhs,
                         sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>>& b, std::int64_t ldb,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::trtrs(get_device_id(queue), queue, uplo, trans, diag, n, nrhs, a, lda, b, ldb,
                  scratchpad, scratchpad_size);
}
static inline void trtrs(sycl::queue& queue, oneapi::math::uplo uplo, oneapi::math::transpose trans,
                         oneapi::math::diag diag, std::int64_t n, std::int64_t nrhs,
                         sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& b,
                         std::int64_t ldb, sycl::buffer<double>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::trtrs(get_device_id(queue), queue, uplo, trans, diag, n, nrhs, a, lda, b, ldb,
                  scratchpad, scratchpad_size);
}
static inline void trtrs(sycl::queue& queue, oneapi::math::uplo uplo, oneapi::math::transpose trans,
                         oneapi::math::diag diag, std::int64_t n, std::int64_t nrhs,
                         sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& b,
                         std::int64_t ldb, sycl::buffer<float>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::trtrs(get_device_id(queue), queue, uplo, trans, diag, n, nrhs, a, lda, b, ldb,
                  scratchpad, scratchpad_size);
}
static inline void trtrs(sycl::queue& queue, oneapi::math::uplo uplo, oneapi::math::transpose trans,
                         oneapi::math::diag diag, std::int64_t n, std::int64_t nrhs,
                         sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>>& b, std::int64_t ldb,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::trtrs(get_device_id(queue), queue, uplo, trans, diag, n, nrhs, a, lda, b, ldb,
                  scratchpad, scratchpad_size);
}
static inline void ungbr(sycl::queue& queue, oneapi::math::generate vec, std::int64_t m,
                         std::int64_t n, std::int64_t k, sycl::buffer<std::complex<float>>& a,
                         std::int64_t lda, sycl::buffer<std::complex<float>>& tau,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::ungbr(get_device_id(queue), queue, vec, m, n, k, a, lda, tau, scratchpad,
                  scratchpad_size);
}
static inline void ungbr(sycl::queue& queue, oneapi::math::generate vec, std::int64_t m,
                         std::int64_t n, std::int64_t k, sycl::buffer<std::complex<double>>& a,
                         std::int64_t lda, sycl::buffer<std::complex<double>>& tau,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::ungbr(get_device_id(queue), queue, vec, m, n, k, a, lda, tau, scratchpad,
                  scratchpad_size);
}
static inline void ungqr(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                         sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>>& tau,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::ungqr(get_device_id(queue), queue, m, n, k, a, lda, tau, scratchpad, scratchpad_size);
}
static inline void ungqr(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                         sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>>& tau,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::ungqr(get_device_id(queue), queue, m, n, k, a, lda, tau, scratchpad, scratchpad_size);
}
static inline void ungtr(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                         sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>>& tau,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::ungtr(get_device_id(queue), queue, uplo, n, a, lda, tau, scratchpad, scratchpad_size);
}
static inline void ungtr(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                         sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>>& tau,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::ungtr(get_device_id(queue), queue, uplo, n, a, lda, tau, scratchpad, scratchpad_size);
}
static inline void unmrq(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
                         std::int64_t m, std::int64_t n, std::int64_t k,
                         sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>>& tau,
                         sycl::buffer<std::complex<float>>& c, std::int64_t ldc,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::unmrq(get_device_id(queue), queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                  scratchpad, scratchpad_size);
}
static inline void unmrq(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
                         std::int64_t m, std::int64_t n, std::int64_t k,
                         sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>>& tau,
                         sycl::buffer<std::complex<double>>& c, std::int64_t ldc,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::unmrq(get_device_id(queue), queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                  scratchpad, scratchpad_size);
}
static inline void unmqr(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
                         std::int64_t m, std::int64_t n, std::int64_t k,
                         sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>>& tau,
                         sycl::buffer<std::complex<float>>& c, std::int64_t ldc,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::unmqr(get_device_id(queue), queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                  scratchpad, scratchpad_size);
}
static inline void unmqr(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
                         std::int64_t m, std::int64_t n, std::int64_t k,
                         sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>>& tau,
                         sycl::buffer<std::complex<double>>& c, std::int64_t ldc,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::unmqr(get_device_id(queue), queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                  scratchpad, scratchpad_size);
}
static inline void unmtr(sycl::queue& queue, oneapi::math::side side, oneapi::math::uplo uplo,
                         oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
                         sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>>& tau,
                         sycl::buffer<std::complex<float>>& c, std::int64_t ldc,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::unmtr(get_device_id(queue), queue, side, uplo, trans, m, n, a, lda, tau, c, ldc,
                  scratchpad, scratchpad_size);
}
static inline void unmtr(sycl::queue& queue, oneapi::math::side side, oneapi::math::uplo uplo,
                         oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
                         sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>>& tau,
                         sycl::buffer<std::complex<double>>& c, std::int64_t ldc,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    detail::unmtr(get_device_id(queue), queue, side, uplo, trans, m, n, a, lda, tau, c, ldc,
                  scratchpad, scratchpad_size);
}
static inline void geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n,
                               sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                               sycl::buffer<float>& tau, std::int64_t stride_tau,
                               std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                               std::int64_t scratchpad_size) {
    detail::geqrf_batch(get_device_id(queue), queue, m, n, a, lda, stride_a, tau, stride_tau,
                        batch_size, scratchpad, scratchpad_size);
}
static inline void geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n,
                               sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                               sycl::buffer<double>& tau, std::int64_t stride_tau,
                               std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                               std::int64_t scratchpad_size) {
    detail::geqrf_batch(get_device_id(queue), queue, m, n, a, lda, stride_a, tau, stride_tau,
                        batch_size, scratchpad, scratchpad_size);
}
static inline void geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n,
                               sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<std::complex<float>>& tau,
                               std::int64_t stride_tau, std::int64_t batch_size,
                               sycl::buffer<std::complex<float>>& scratchpad,
                               std::int64_t scratchpad_size) {
    detail::geqrf_batch(get_device_id(queue), queue, m, n, a, lda, stride_a, tau, stride_tau,
                        batch_size, scratchpad, scratchpad_size);
}
static inline void geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n,
                               sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<std::complex<double>>& tau,
                               std::int64_t stride_tau, std::int64_t batch_size,
                               sycl::buffer<std::complex<double>>& scratchpad,
                               std::int64_t scratchpad_size) {
    detail::geqrf_batch(get_device_id(queue), queue, m, n, a, lda, stride_a, tau, stride_tau,
                        batch_size, scratchpad, scratchpad_size);
}
static inline void getri_batch(sycl::queue& queue, std::int64_t n, sycl::buffer<float>& a,
                               std::int64_t lda, std::int64_t stride_a,
                               sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                               std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                               std::int64_t scratchpad_size) {
    detail::getri_batch(get_device_id(queue), queue, n, a, lda, stride_a, ipiv, stride_ipiv,
                        batch_size, scratchpad, scratchpad_size);
}
static inline void getri_batch(sycl::queue& queue, std::int64_t n, sycl::buffer<double>& a,
                               std::int64_t lda, std::int64_t stride_a,
                               sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                               std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                               std::int64_t scratchpad_size) {
    detail::getri_batch(get_device_id(queue), queue, n, a, lda, stride_a, ipiv, stride_ipiv,
                        batch_size, scratchpad, scratchpad_size);
}
static inline void getri_batch(sycl::queue& queue, std::int64_t n,
                               sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                               std::int64_t stride_ipiv, std::int64_t batch_size,
                               sycl::buffer<std::complex<float>>& scratchpad,
                               std::int64_t scratchpad_size) {
    detail::getri_batch(get_device_id(queue), queue, n, a, lda, stride_a, ipiv, stride_ipiv,
                        batch_size, scratchpad, scratchpad_size);
}
static inline void getri_batch(sycl::queue& queue, std::int64_t n,
                               sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                               std::int64_t stride_ipiv, std::int64_t batch_size,
                               sycl::buffer<std::complex<double>>& scratchpad,
                               std::int64_t scratchpad_size) {
    detail::getri_batch(get_device_id(queue), queue, n, a, lda, stride_a, ipiv, stride_ipiv,
                        batch_size, scratchpad, scratchpad_size);
}
static inline void getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                               std::int64_t nrhs, sycl::buffer<float>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                               std::int64_t stride_ipiv, sycl::buffer<float>& b, std::int64_t ldb,
                               std::int64_t stride_b, std::int64_t batch_size,
                               sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    detail::getrs_batch(get_device_id(queue), queue, trans, n, nrhs, a, lda, stride_a, ipiv,
                        stride_ipiv, b, ldb, stride_b, batch_size, scratchpad, scratchpad_size);
}
static inline void getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                               std::int64_t nrhs, sycl::buffer<double>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                               std::int64_t stride_ipiv, sycl::buffer<double>& b, std::int64_t ldb,
                               std::int64_t stride_b, std::int64_t batch_size,
                               sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    detail::getrs_batch(get_device_id(queue), queue, trans, n, nrhs, a, lda, stride_a, ipiv,
                        stride_ipiv, b, ldb, stride_b, batch_size, scratchpad, scratchpad_size);
}
static inline void getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                               std::int64_t nrhs, sycl::buffer<std::complex<float>>& a,
                               std::int64_t lda, std::int64_t stride_a,
                               sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                               sycl::buffer<std::complex<float>>& b, std::int64_t ldb,
                               std::int64_t stride_b, std::int64_t batch_size,
                               sycl::buffer<std::complex<float>>& scratchpad,
                               std::int64_t scratchpad_size) {
    detail::getrs_batch(get_device_id(queue), queue, trans, n, nrhs, a, lda, stride_a, ipiv,
                        stride_ipiv, b, ldb, stride_b, batch_size, scratchpad, scratchpad_size);
}
static inline void getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                               std::int64_t nrhs, sycl::buffer<std::complex<double>>& a,
                               std::int64_t lda, std::int64_t stride_a,
                               sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                               sycl::buffer<std::complex<double>>& b, std::int64_t ldb,
                               std::int64_t stride_b, std::int64_t batch_size,
                               sycl::buffer<std::complex<double>>& scratchpad,
                               std::int64_t scratchpad_size) {
    detail::getrs_batch(get_device_id(queue), queue, trans, n, nrhs, a, lda, stride_a, ipiv,
                        stride_ipiv, b, ldb, stride_b, batch_size, scratchpad, scratchpad_size);
}
static inline void getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n,
                               sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                               sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                               std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                               std::int64_t scratchpad_size) {
    detail::getrf_batch(get_device_id(queue), queue, m, n, a, lda, stride_a, ipiv, stride_ipiv,
                        batch_size, scratchpad, scratchpad_size);
}
static inline void getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n,
                               sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                               sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                               std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                               std::int64_t scratchpad_size) {
    detail::getrf_batch(get_device_id(queue), queue, m, n, a, lda, stride_a, ipiv, stride_ipiv,
                        batch_size, scratchpad, scratchpad_size);
}
static inline void getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n,
                               sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                               std::int64_t stride_ipiv, std::int64_t batch_size,
                               sycl::buffer<std::complex<float>>& scratchpad,
                               std::int64_t scratchpad_size) {
    detail::getrf_batch(get_device_id(queue), queue, m, n, a, lda, stride_a, ipiv, stride_ipiv,
                        batch_size, scratchpad, scratchpad_size);
}
static inline void getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n,
                               sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                               std::int64_t stride_ipiv, std::int64_t batch_size,
                               sycl::buffer<std::complex<double>>& scratchpad,
                               std::int64_t scratchpad_size) {
    detail::getrf_batch(get_device_id(queue), queue, m, n, a, lda, stride_a, ipiv, stride_ipiv,
                        batch_size, scratchpad, scratchpad_size);
}
static inline void orgqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                               sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                               sycl::buffer<float>& tau, std::int64_t stride_tau,
                               std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                               std::int64_t scratchpad_size) {
    detail::orgqr_batch(get_device_id(queue), queue, m, n, k, a, lda, stride_a, tau, stride_tau,
                        batch_size, scratchpad, scratchpad_size);
}
static inline void orgqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                               sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                               sycl::buffer<double>& tau, std::int64_t stride_tau,
                               std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                               std::int64_t scratchpad_size) {
    detail::orgqr_batch(get_device_id(queue), queue, m, n, k, a, lda, stride_a, tau, stride_tau,
                        batch_size, scratchpad, scratchpad_size);
}
static inline void potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                               sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                               std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                               std::int64_t scratchpad_size) {
    detail::potrf_batch(get_device_id(queue), queue, uplo, n, a, lda, stride_a, batch_size,
                        scratchpad, scratchpad_size);
}
static inline void potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                               sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                               std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                               std::int64_t scratchpad_size) {
    detail::potrf_batch(get_device_id(queue), queue, uplo, n, a, lda, stride_a, batch_size,
                        scratchpad, scratchpad_size);
}
static inline void potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                               sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                               std::int64_t stride_a, std::int64_t batch_size,
                               sycl::buffer<std::complex<float>>& scratchpad,
                               std::int64_t scratchpad_size) {
    detail::potrf_batch(get_device_id(queue), queue, uplo, n, a, lda, stride_a, batch_size,
                        scratchpad, scratchpad_size);
}
static inline void potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                               sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                               std::int64_t stride_a, std::int64_t batch_size,
                               sycl::buffer<std::complex<double>>& scratchpad,
                               std::int64_t scratchpad_size) {
    detail::potrf_batch(get_device_id(queue), queue, uplo, n, a, lda, stride_a, batch_size,
                        scratchpad, scratchpad_size);
}
static inline void potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                               std::int64_t nrhs, sycl::buffer<float>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<float>& b, std::int64_t ldb,
                               std::int64_t stride_b, std::int64_t batch_size,
                               sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    detail::potrs_batch(get_device_id(queue), queue, uplo, n, nrhs, a, lda, stride_a, b, ldb,
                        stride_b, batch_size, scratchpad, scratchpad_size);
}
static inline void potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                               std::int64_t nrhs, sycl::buffer<double>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<double>& b, std::int64_t ldb,
                               std::int64_t stride_b, std::int64_t batch_size,
                               sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    detail::potrs_batch(get_device_id(queue), queue, uplo, n, nrhs, a, lda, stride_a, b, ldb,
                        stride_b, batch_size, scratchpad, scratchpad_size);
}
static inline void potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                               std::int64_t nrhs, sycl::buffer<std::complex<float>>& a,
                               std::int64_t lda, std::int64_t stride_a,
                               sycl::buffer<std::complex<float>>& b, std::int64_t ldb,
                               std::int64_t stride_b, std::int64_t batch_size,
                               sycl::buffer<std::complex<float>>& scratchpad,
                               std::int64_t scratchpad_size) {
    detail::potrs_batch(get_device_id(queue), queue, uplo, n, nrhs, a, lda, stride_a, b, ldb,
                        stride_b, batch_size, scratchpad, scratchpad_size);
}
static inline void potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                               std::int64_t nrhs, sycl::buffer<std::complex<double>>& a,
                               std::int64_t lda, std::int64_t stride_a,
                               sycl::buffer<std::complex<double>>& b, std::int64_t ldb,
                               std::int64_t stride_b, std::int64_t batch_size,
                               sycl::buffer<std::complex<double>>& scratchpad,
                               std::int64_t scratchpad_size) {
    detail::potrs_batch(get_device_id(queue), queue, uplo, n, nrhs, a, lda, stride_a, b, ldb,
                        stride_b, batch_size, scratchpad, scratchpad_size);
}
static inline void ungqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                               sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<std::complex<float>>& tau,
                               std::int64_t stride_tau, std::int64_t batch_size,
                               sycl::buffer<std::complex<float>>& scratchpad,
                               std::int64_t scratchpad_size) {
    detail::ungqr_batch(get_device_id(queue), queue, m, n, k, a, lda, stride_a, tau, stride_tau,
                        batch_size, scratchpad, scratchpad_size);
}
static inline void ungqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                               sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<std::complex<double>>& tau,
                               std::int64_t stride_tau, std::int64_t batch_size,
                               sycl::buffer<std::complex<double>>& scratchpad,
                               std::int64_t scratchpad_size) {
    detail::ungqr_batch(get_device_id(queue), queue, m, n, k, a, lda, stride_a, tau, stride_tau,
                        batch_size, scratchpad, scratchpad_size);
}
static inline sycl::event gebrd(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                std::complex<float>* a, std::int64_t lda, float* d, float* e,
                                std::complex<float>* tauq, std::complex<float>* taup,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::gebrd(get_device_id(queue), queue, m, n, a, lda, d, e, tauq, taup, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event gebrd(sycl::queue& queue, std::int64_t m, std::int64_t n, double* a,
                                std::int64_t lda, double* d, double* e, double* tauq, double* taup,
                                double* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::gebrd(get_device_id(queue), queue, m, n, a, lda, d, e, tauq, taup, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event gebrd(sycl::queue& queue, std::int64_t m, std::int64_t n, float* a,
                                std::int64_t lda, float* d, float* e, float* tauq, float* taup,
                                float* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::gebrd(get_device_id(queue), queue, m, n, a, lda, d, e, tauq, taup, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event gebrd(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                std::complex<double>* a, std::int64_t lda, double* d, double* e,
                                std::complex<double>* tauq, std::complex<double>* taup,
                                std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::gebrd(get_device_id(queue), queue, m, n, a, lda, d, e, tauq, taup, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n, float* a,
                                std::int64_t lda, float* tau, float* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::gerqf(get_device_id(queue), queue, m, n, a, lda, tau, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n, double* a,
                                std::int64_t lda, double* tau, double* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::gerqf(get_device_id(queue), queue, m, n, a, lda, tau, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                std::complex<float>* a, std::int64_t lda, std::complex<float>* tau,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::gerqf(get_device_id(queue), queue, m, n, a, lda, tau, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                std::complex<double>* a, std::int64_t lda,
                                std::complex<double>* tau, std::complex<double>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::gerqf(get_device_id(queue), queue, m, n, a, lda, tau, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event geqrf(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                std::complex<float>* a, std::int64_t lda, std::complex<float>* tau,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::geqrf(get_device_id(queue), queue, m, n, a, lda, tau, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event geqrf(sycl::queue& queue, std::int64_t m, std::int64_t n, double* a,
                                std::int64_t lda, double* tau, double* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::geqrf(get_device_id(queue), queue, m, n, a, lda, tau, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event geqrf(sycl::queue& queue, std::int64_t m, std::int64_t n, float* a,
                                std::int64_t lda, float* tau, float* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::geqrf(get_device_id(queue), queue, m, n, a, lda, tau, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event geqrf(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                std::complex<double>* a, std::int64_t lda,
                                std::complex<double>* tau, std::complex<double>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::geqrf(get_device_id(queue), queue, m, n, a, lda, tau, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event getrf(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                std::complex<float>* a, std::int64_t lda, std::int64_t* ipiv,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::getrf(get_device_id(queue), queue, m, n, a, lda, ipiv, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event getrf(sycl::queue& queue, std::int64_t m, std::int64_t n, double* a,
                                std::int64_t lda, std::int64_t* ipiv, double* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::getrf(get_device_id(queue), queue, m, n, a, lda, ipiv, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event getrf(sycl::queue& queue, std::int64_t m, std::int64_t n, float* a,
                                std::int64_t lda, std::int64_t* ipiv, float* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::getrf(get_device_id(queue), queue, m, n, a, lda, ipiv, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event getrf(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                std::complex<double>* a, std::int64_t lda, std::int64_t* ipiv,
                                std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::getrf(get_device_id(queue), queue, m, n, a, lda, ipiv, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event getri(sycl::queue& queue, std::int64_t n, std::complex<float>* a,
                                std::int64_t lda, std::int64_t* ipiv,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::getri(get_device_id(queue), queue, n, a, lda, ipiv, scratchpad, scratchpad_size,
                         dependencies);
}
static inline sycl::event getri(sycl::queue& queue, std::int64_t n, double* a, std::int64_t lda,
                                std::int64_t* ipiv, double* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::getri(get_device_id(queue), queue, n, a, lda, ipiv, scratchpad, scratchpad_size,
                         dependencies);
}
static inline sycl::event getri(sycl::queue& queue, std::int64_t n, float* a, std::int64_t lda,
                                std::int64_t* ipiv, float* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::getri(get_device_id(queue), queue, n, a, lda, ipiv, scratchpad, scratchpad_size,
                         dependencies);
}
static inline sycl::event getri(sycl::queue& queue, std::int64_t n, std::complex<double>* a,
                                std::int64_t lda, std::int64_t* ipiv,
                                std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::getri(get_device_id(queue), queue, n, a, lda, ipiv, scratchpad, scratchpad_size,
                         dependencies);
}
static inline sycl::event getrs(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                                std::int64_t nrhs, std::complex<float>* a, std::int64_t lda,
                                std::int64_t* ipiv, std::complex<float>* b, std::int64_t ldb,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::getrs(get_device_id(queue), queue, trans, n, nrhs, a, lda, ipiv, b, ldb,
                         scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getrs(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                                std::int64_t nrhs, double* a, std::int64_t lda, std::int64_t* ipiv,
                                double* b, std::int64_t ldb, double* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::getrs(get_device_id(queue), queue, trans, n, nrhs, a, lda, ipiv, b, ldb,
                         scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getrs(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                                std::int64_t nrhs, float* a, std::int64_t lda, std::int64_t* ipiv,
                                float* b, std::int64_t ldb, float* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::getrs(get_device_id(queue), queue, trans, n, nrhs, a, lda, ipiv, b, ldb,
                         scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getrs(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                                std::int64_t nrhs, std::complex<double>* a, std::int64_t lda,
                                std::int64_t* ipiv, std::complex<double>* b, std::int64_t ldb,
                                std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::getrs(get_device_id(queue), queue, trans, n, nrhs, a, lda, ipiv, b, ldb,
                         scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event gesvd(sycl::queue& queue, oneapi::math::jobsvd jobu,
                                oneapi::math::jobsvd jobvt, std::int64_t m, std::int64_t n,
                                double* a, std::int64_t lda, double* s, double* u, std::int64_t ldu,
                                double* vt, std::int64_t ldvt, double* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::gesvd(get_device_id(queue), queue, jobu, jobvt, m, n, a, lda, s, u, ldu, vt,
                         ldvt, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event gesvd(sycl::queue& queue, oneapi::math::jobsvd jobu,
                                oneapi::math::jobsvd jobvt, std::int64_t m, std::int64_t n,
                                float* a, std::int64_t lda, float* s, float* u, std::int64_t ldu,
                                float* vt, std::int64_t ldvt, float* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::gesvd(get_device_id(queue), queue, jobu, jobvt, m, n, a, lda, s, u, ldu, vt,
                         ldvt, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event gesvd(sycl::queue& queue, oneapi::math::jobsvd jobu,
                                oneapi::math::jobsvd jobvt, std::int64_t m, std::int64_t n,
                                std::complex<float>* a, std::int64_t lda, float* s,
                                std::complex<float>* u, std::int64_t ldu, std::complex<float>* vt,
                                std::int64_t ldvt, std::complex<float>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::gesvd(get_device_id(queue), queue, jobu, jobvt, m, n, a, lda, s, u, ldu, vt,
                         ldvt, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event gesvd(sycl::queue& queue, oneapi::math::jobsvd jobu,
                                oneapi::math::jobsvd jobvt, std::int64_t m, std::int64_t n,
                                std::complex<double>* a, std::int64_t lda, double* s,
                                std::complex<double>* u, std::int64_t ldu, std::complex<double>* vt,
                                std::int64_t ldvt, std::complex<double>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::gesvd(get_device_id(queue), queue, jobu, jobvt, m, n, a, lda, s, u, ldu, vt,
                         ldvt, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event heevd(sycl::queue& queue, oneapi::math::job jobz, oneapi::math::uplo uplo,
                                std::int64_t n, std::complex<float>* a, std::int64_t lda, float* w,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::heevd(get_device_id(queue), queue, jobz, uplo, n, a, lda, w, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event heevd(sycl::queue& queue, oneapi::math::job jobz, oneapi::math::uplo uplo,
                                std::int64_t n, std::complex<double>* a, std::int64_t lda,
                                double* w, std::complex<double>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::heevd(get_device_id(queue), queue, jobz, uplo, n, a, lda, w, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event hegvd(sycl::queue& queue, std::int64_t itype, oneapi::math::job jobz,
                                oneapi::math::uplo uplo, std::int64_t n, std::complex<float>* a,
                                std::int64_t lda, std::complex<float>* b, std::int64_t ldb,
                                float* w, std::complex<float>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::hegvd(get_device_id(queue), queue, itype, jobz, uplo, n, a, lda, b, ldb, w,
                         scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event hegvd(sycl::queue& queue, std::int64_t itype, oneapi::math::job jobz,
                                oneapi::math::uplo uplo, std::int64_t n, std::complex<double>* a,
                                std::int64_t lda, std::complex<double>* b, std::int64_t ldb,
                                double* w, std::complex<double>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::hegvd(get_device_id(queue), queue, itype, jobz, uplo, n, a, lda, b, ldb, w,
                         scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event hetrd(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                std::complex<float>* a, std::int64_t lda, float* d, float* e,
                                std::complex<float>* tau, std::complex<float>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::hetrd(get_device_id(queue), queue, uplo, n, a, lda, d, e, tau, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event hetrd(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                std::complex<double>* a, std::int64_t lda, double* d, double* e,
                                std::complex<double>* tau, std::complex<double>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::hetrd(get_device_id(queue), queue, uplo, n, a, lda, d, e, tau, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event hetrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                std::complex<float>* a, std::int64_t lda, std::int64_t* ipiv,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::hetrf(get_device_id(queue), queue, uplo, n, a, lda, ipiv, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event hetrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                std::complex<double>* a, std::int64_t lda, std::int64_t* ipiv,
                                std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::hetrf(get_device_id(queue), queue, uplo, n, a, lda, ipiv, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event orgbr(sycl::queue& queue, oneapi::math::generate vec, std::int64_t m,
                                std::int64_t n, std::int64_t k, float* a, std::int64_t lda,
                                float* tau, float* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::orgbr(get_device_id(queue), queue, vec, m, n, k, a, lda, tau, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event orgbr(sycl::queue& queue, oneapi::math::generate vec, std::int64_t m,
                                std::int64_t n, std::int64_t k, double* a, std::int64_t lda,
                                double* tau, double* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::orgbr(get_device_id(queue), queue, vec, m, n, k, a, lda, tau, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event orgqr(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                                double* a, std::int64_t lda, double* tau, double* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::orgqr(get_device_id(queue), queue, m, n, k, a, lda, tau, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event orgqr(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                                float* a, std::int64_t lda, float* tau, float* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::orgqr(get_device_id(queue), queue, m, n, k, a, lda, tau, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event orgtr(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                float* a, std::int64_t lda, float* tau, float* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::orgtr(get_device_id(queue), queue, uplo, n, a, lda, tau, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event orgtr(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                double* a, std::int64_t lda, double* tau, double* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::orgtr(get_device_id(queue), queue, uplo, n, a, lda, tau, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event ormtr(sycl::queue& queue, oneapi::math::side side,
                                oneapi::math::uplo uplo, oneapi::math::transpose trans,
                                std::int64_t m, std::int64_t n, float* a, std::int64_t lda,
                                float* tau, float* c, std::int64_t ldc, float* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::ormtr(get_device_id(queue), queue, side, uplo, trans, m, n, a, lda, tau, c, ldc,
                         scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event ormtr(sycl::queue& queue, oneapi::math::side side,
                                oneapi::math::uplo uplo, oneapi::math::transpose trans,
                                std::int64_t m, std::int64_t n, double* a, std::int64_t lda,
                                double* tau, double* c, std::int64_t ldc, double* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::ormtr(get_device_id(queue), queue, side, uplo, trans, m, n, a, lda, tau, c, ldc,
                         scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event ormrq(sycl::queue& queue, oneapi::math::side side,
                                oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
                                std::int64_t k, float* a, std::int64_t lda, float* tau, float* c,
                                std::int64_t ldc, float* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::ormrq(get_device_id(queue), queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                         scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event ormrq(sycl::queue& queue, oneapi::math::side side,
                                oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
                                std::int64_t k, double* a, std::int64_t lda, double* tau, double* c,
                                std::int64_t ldc, double* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::ormrq(get_device_id(queue), queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                         scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event ormqr(sycl::queue& queue, oneapi::math::side side,
                                oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
                                std::int64_t k, double* a, std::int64_t lda, double* tau, double* c,
                                std::int64_t ldc, double* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::ormqr(get_device_id(queue), queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                         scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event ormqr(sycl::queue& queue, oneapi::math::side side,
                                oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
                                std::int64_t k, float* a, std::int64_t lda, float* tau, float* c,
                                std::int64_t ldc, float* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::ormqr(get_device_id(queue), queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                         scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event potrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                float* a, std::int64_t lda, float* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::potrf(get_device_id(queue), queue, uplo, n, a, lda, scratchpad, scratchpad_size,
                         dependencies);
}
static inline sycl::event potrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                double* a, std::int64_t lda, double* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::potrf(get_device_id(queue), queue, uplo, n, a, lda, scratchpad, scratchpad_size,
                         dependencies);
}
static inline sycl::event potrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                std::complex<float>* a, std::int64_t lda,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::potrf(get_device_id(queue), queue, uplo, n, a, lda, scratchpad, scratchpad_size,
                         dependencies);
}
static inline sycl::event potrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                std::complex<double>* a, std::int64_t lda,
                                std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::potrf(get_device_id(queue), queue, uplo, n, a, lda, scratchpad, scratchpad_size,
                         dependencies);
}
static inline sycl::event potri(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                float* a, std::int64_t lda, float* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::potri(get_device_id(queue), queue, uplo, n, a, lda, scratchpad, scratchpad_size,
                         dependencies);
}
static inline sycl::event potri(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                double* a, std::int64_t lda, double* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::potri(get_device_id(queue), queue, uplo, n, a, lda, scratchpad, scratchpad_size,
                         dependencies);
}
static inline sycl::event potri(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                std::complex<float>* a, std::int64_t lda,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::potri(get_device_id(queue), queue, uplo, n, a, lda, scratchpad, scratchpad_size,
                         dependencies);
}
static inline sycl::event potri(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                std::complex<double>* a, std::int64_t lda,
                                std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::potri(get_device_id(queue), queue, uplo, n, a, lda, scratchpad, scratchpad_size,
                         dependencies);
}
static inline sycl::event potrs(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                std::int64_t nrhs, float* a, std::int64_t lda, float* b,
                                std::int64_t ldb, float* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::potrs(get_device_id(queue), queue, uplo, n, nrhs, a, lda, b, ldb, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event potrs(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                std::int64_t nrhs, double* a, std::int64_t lda, double* b,
                                std::int64_t ldb, double* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::potrs(get_device_id(queue), queue, uplo, n, nrhs, a, lda, b, ldb, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event potrs(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                std::int64_t nrhs, std::complex<float>* a, std::int64_t lda,
                                std::complex<float>* b, std::int64_t ldb,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::potrs(get_device_id(queue), queue, uplo, n, nrhs, a, lda, b, ldb, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event potrs(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                std::int64_t nrhs, std::complex<double>* a, std::int64_t lda,
                                std::complex<double>* b, std::int64_t ldb,
                                std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::potrs(get_device_id(queue), queue, uplo, n, nrhs, a, lda, b, ldb, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event syevd(sycl::queue& queue, oneapi::math::job jobz, oneapi::math::uplo uplo,
                                std::int64_t n, double* a, std::int64_t lda, double* w,
                                double* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::syevd(get_device_id(queue), queue, jobz, uplo, n, a, lda, w, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event syevd(sycl::queue& queue, oneapi::math::job jobz, oneapi::math::uplo uplo,
                                std::int64_t n, float* a, std::int64_t lda, float* w,
                                float* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::syevd(get_device_id(queue), queue, jobz, uplo, n, a, lda, w, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event sygvd(sycl::queue& queue, std::int64_t itype, oneapi::math::job jobz,
                                oneapi::math::uplo uplo, std::int64_t n, double* a,
                                std::int64_t lda, double* b, std::int64_t ldb, double* w,
                                double* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::sygvd(get_device_id(queue), queue, itype, jobz, uplo, n, a, lda, b, ldb, w,
                         scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event sygvd(sycl::queue& queue, std::int64_t itype, oneapi::math::job jobz,
                                oneapi::math::uplo uplo, std::int64_t n, float* a, std::int64_t lda,
                                float* b, std::int64_t ldb, float* w, float* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::sygvd(get_device_id(queue), queue, itype, jobz, uplo, n, a, lda, b, ldb, w,
                         scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event sytrd(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                double* a, std::int64_t lda, double* d, double* e, double* tau,
                                double* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::sytrd(get_device_id(queue), queue, uplo, n, a, lda, d, e, tau, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event sytrd(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                float* a, std::int64_t lda, float* d, float* e, float* tau,
                                float* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::sytrd(get_device_id(queue), queue, uplo, n, a, lda, d, e, tau, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event sytrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                float* a, std::int64_t lda, std::int64_t* ipiv, float* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::sytrf(get_device_id(queue), queue, uplo, n, a, lda, ipiv, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event sytrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                double* a, std::int64_t lda, std::int64_t* ipiv, double* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::sytrf(get_device_id(queue), queue, uplo, n, a, lda, ipiv, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event sytrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                std::complex<float>* a, std::int64_t lda, std::int64_t* ipiv,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::sytrf(get_device_id(queue), queue, uplo, n, a, lda, ipiv, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event sytrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                std::complex<double>* a, std::int64_t lda, std::int64_t* ipiv,
                                std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::sytrf(get_device_id(queue), queue, uplo, n, a, lda, ipiv, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event trtrs(sycl::queue& queue, oneapi::math::uplo uplo,
                                oneapi::math::transpose trans, oneapi::math::diag diag,
                                std::int64_t n, std::int64_t nrhs, std::complex<float>* a,
                                std::int64_t lda, std::complex<float>* b, std::int64_t ldb,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::trtrs(get_device_id(queue), queue, uplo, trans, diag, n, nrhs, a, lda, b, ldb,
                         scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event trtrs(sycl::queue& queue, oneapi::math::uplo uplo,
                                oneapi::math::transpose trans, oneapi::math::diag diag,
                                std::int64_t n, std::int64_t nrhs, double* a, std::int64_t lda,
                                double* b, std::int64_t ldb, double* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::trtrs(get_device_id(queue), queue, uplo, trans, diag, n, nrhs, a, lda, b, ldb,
                         scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event trtrs(sycl::queue& queue, oneapi::math::uplo uplo,
                                oneapi::math::transpose trans, oneapi::math::diag diag,
                                std::int64_t n, std::int64_t nrhs, float* a, std::int64_t lda,
                                float* b, std::int64_t ldb, float* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::trtrs(get_device_id(queue), queue, uplo, trans, diag, n, nrhs, a, lda, b, ldb,
                         scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event trtrs(sycl::queue& queue, oneapi::math::uplo uplo,
                                oneapi::math::transpose trans, oneapi::math::diag diag,
                                std::int64_t n, std::int64_t nrhs, std::complex<double>* a,
                                std::int64_t lda, std::complex<double>* b, std::int64_t ldb,
                                std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::trtrs(get_device_id(queue), queue, uplo, trans, diag, n, nrhs, a, lda, b, ldb,
                         scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event ungbr(sycl::queue& queue, oneapi::math::generate vec, std::int64_t m,
                                std::int64_t n, std::int64_t k, std::complex<float>* a,
                                std::int64_t lda, std::complex<float>* tau,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::ungbr(get_device_id(queue), queue, vec, m, n, k, a, lda, tau, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event ungbr(sycl::queue& queue, oneapi::math::generate vec, std::int64_t m,
                                std::int64_t n, std::int64_t k, std::complex<double>* a,
                                std::int64_t lda, std::complex<double>* tau,
                                std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::ungbr(get_device_id(queue), queue, vec, m, n, k, a, lda, tau, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event ungqr(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                                std::complex<float>* a, std::int64_t lda, std::complex<float>* tau,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::ungqr(get_device_id(queue), queue, m, n, k, a, lda, tau, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event ungqr(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                                std::complex<double>* a, std::int64_t lda,
                                std::complex<double>* tau, std::complex<double>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::ungqr(get_device_id(queue), queue, m, n, k, a, lda, tau, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event ungtr(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                std::complex<float>* a, std::int64_t lda, std::complex<float>* tau,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::ungtr(get_device_id(queue), queue, uplo, n, a, lda, tau, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event ungtr(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                std::complex<double>* a, std::int64_t lda,
                                std::complex<double>* tau, std::complex<double>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::ungtr(get_device_id(queue), queue, uplo, n, a, lda, tau, scratchpad,
                         scratchpad_size, dependencies);
}
static inline sycl::event unmrq(sycl::queue& queue, oneapi::math::side side,
                                oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
                                std::int64_t k, std::complex<float>* a, std::int64_t lda,
                                std::complex<float>* tau, std::complex<float>* c, std::int64_t ldc,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::unmrq(get_device_id(queue), queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                         scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event unmrq(sycl::queue& queue, oneapi::math::side side,
                                oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
                                std::int64_t k, std::complex<double>* a, std::int64_t lda,
                                std::complex<double>* tau, std::complex<double>* c,
                                std::int64_t ldc, std::complex<double>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::unmrq(get_device_id(queue), queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                         scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event unmqr(sycl::queue& queue, oneapi::math::side side,
                                oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
                                std::int64_t k, std::complex<float>* a, std::int64_t lda,
                                std::complex<float>* tau, std::complex<float>* c, std::int64_t ldc,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::unmqr(get_device_id(queue), queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                         scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event unmqr(sycl::queue& queue, oneapi::math::side side,
                                oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
                                std::int64_t k, std::complex<double>* a, std::int64_t lda,
                                std::complex<double>* tau, std::complex<double>* c,
                                std::int64_t ldc, std::complex<double>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::unmqr(get_device_id(queue), queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                         scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event unmtr(sycl::queue& queue, oneapi::math::side side,
                                oneapi::math::uplo uplo, oneapi::math::transpose trans,
                                std::int64_t m, std::int64_t n, std::complex<float>* a,
                                std::int64_t lda, std::complex<float>* tau, std::complex<float>* c,
                                std::int64_t ldc, std::complex<float>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::unmtr(get_device_id(queue), queue, side, uplo, trans, m, n, a, lda, tau, c, ldc,
                         scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event unmtr(sycl::queue& queue, oneapi::math::side side,
                                oneapi::math::uplo uplo, oneapi::math::transpose trans,
                                std::int64_t m, std::int64_t n, std::complex<double>* a,
                                std::int64_t lda, std::complex<double>* tau,
                                std::complex<double>* c, std::int64_t ldc,
                                std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return detail::unmtr(get_device_id(queue), queue, side, uplo, trans, m, n, a, lda, tau, c, ldc,
                         scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, float* a,
                                      std::int64_t lda, std::int64_t stride_a, float* tau,
                                      std::int64_t stride_tau, std::int64_t batch_size,
                                      float* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::geqrf_batch(get_device_id(queue), queue, m, n, a, lda, stride_a, tau, stride_tau,
                               batch_size, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, double* a,
                                      std::int64_t lda, std::int64_t stride_a, double* tau,
                                      std::int64_t stride_tau, std::int64_t batch_size,
                                      double* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::geqrf_batch(get_device_id(queue), queue, m, n, a, lda, stride_a, tau, stride_tau,
                               batch_size, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                      std::complex<float>* a, std::int64_t lda,
                                      std::int64_t stride_a, std::complex<float>* tau,
                                      std::int64_t stride_tau, std::int64_t batch_size,
                                      std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::geqrf_batch(get_device_id(queue), queue, m, n, a, lda, stride_a, tau, stride_tau,
                               batch_size, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                      std::complex<double>* a, std::int64_t lda,
                                      std::int64_t stride_a, std::complex<double>* tau,
                                      std::int64_t stride_tau, std::int64_t batch_size,
                                      std::complex<double>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::geqrf_batch(get_device_id(queue), queue, m, n, a, lda, stride_a, tau, stride_tau,
                               batch_size, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event geqrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                      float** a, std::int64_t* lda, float** tau,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      float* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::geqrf_batch(get_device_id(queue), queue, m, n, a, lda, tau, group_count,
                               group_sizes, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event geqrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                      double** a, std::int64_t* lda, double** tau,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      double* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::geqrf_batch(get_device_id(queue), queue, m, n, a, lda, tau, group_count,
                               group_sizes, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event geqrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                      std::complex<float>** a, std::int64_t* lda,
                                      std::complex<float>** tau, std::int64_t group_count,
                                      std::int64_t* group_sizes, std::complex<float>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::geqrf_batch(get_device_id(queue), queue, m, n, a, lda, tau, group_count,
                               group_sizes, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event geqrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                      std::complex<double>** a, std::int64_t* lda,
                                      std::complex<double>** tau, std::int64_t group_count,
                                      std::int64_t* group_sizes, std::complex<double>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::geqrf_batch(get_device_id(queue), queue, m, n, a, lda, tau, group_count,
                               group_sizes, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, float* a,
                                      std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                                      std::int64_t stride_ipiv, std::int64_t batch_size,
                                      float* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::getrf_batch(get_device_id(queue), queue, m, n, a, lda, stride_a, ipiv,
                               stride_ipiv, batch_size, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, double* a,
                                      std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                                      std::int64_t stride_ipiv, std::int64_t batch_size,
                                      double* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::getrf_batch(get_device_id(queue), queue, m, n, a, lda, stride_a, ipiv,
                               stride_ipiv, batch_size, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                      std::complex<float>* a, std::int64_t lda,
                                      std::int64_t stride_a, std::int64_t* ipiv,
                                      std::int64_t stride_ipiv, std::int64_t batch_size,
                                      std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::getrf_batch(get_device_id(queue), queue, m, n, a, lda, stride_a, ipiv,
                               stride_ipiv, batch_size, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                      std::complex<double>* a, std::int64_t lda,
                                      std::int64_t stride_a, std::int64_t* ipiv,
                                      std::int64_t stride_ipiv, std::int64_t batch_size,
                                      std::complex<double>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::getrf_batch(get_device_id(queue), queue, m, n, a, lda, stride_a, ipiv,
                               stride_ipiv, batch_size, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                      float** a, std::int64_t* lda, std::int64_t** ipiv,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      float* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::getrf_batch(get_device_id(queue), queue, m, n, a, lda, ipiv, group_count,
                               group_sizes, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                      double** a, std::int64_t* lda, std::int64_t** ipiv,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      double* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::getrf_batch(get_device_id(queue), queue, m, n, a, lda, ipiv, group_count,
                               group_sizes, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                      std::complex<float>** a, std::int64_t* lda,
                                      std::int64_t** ipiv, std::int64_t group_count,
                                      std::int64_t* group_sizes, std::complex<float>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::getrf_batch(get_device_id(queue), queue, m, n, a, lda, ipiv, group_count,
                               group_sizes, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                      std::complex<double>** a, std::int64_t* lda,
                                      std::int64_t** ipiv, std::int64_t group_count,
                                      std::int64_t* group_sizes, std::complex<double>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::getrf_batch(get_device_id(queue), queue, m, n, a, lda, ipiv, group_count,
                               group_sizes, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getri_batch(sycl::queue& queue, std::int64_t n, float* a,
                                      std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                                      std::int64_t stride_ipiv, std::int64_t batch_size,
                                      float* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::getri_batch(get_device_id(queue), queue, n, a, lda, stride_a, ipiv, stride_ipiv,
                               batch_size, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getri_batch(sycl::queue& queue, std::int64_t n, double* a,
                                      std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                                      std::int64_t stride_ipiv, std::int64_t batch_size,
                                      double* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::getri_batch(get_device_id(queue), queue, n, a, lda, stride_a, ipiv, stride_ipiv,
                               batch_size, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getri_batch(sycl::queue& queue, std::int64_t n, std::complex<float>* a,
                                      std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                                      std::int64_t stride_ipiv, std::int64_t batch_size,
                                      std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::getri_batch(get_device_id(queue), queue, n, a, lda, stride_a, ipiv, stride_ipiv,
                               batch_size, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getri_batch(sycl::queue& queue, std::int64_t n, std::complex<double>* a,
                                      std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                                      std::int64_t stride_ipiv, std::int64_t batch_size,
                                      std::complex<double>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::getri_batch(get_device_id(queue), queue, n, a, lda, stride_a, ipiv, stride_ipiv,
                               batch_size, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getri_batch(sycl::queue& queue, std::int64_t* n, float** a,
                                      std::int64_t* lda, std::int64_t** ipiv,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      float* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::getri_batch(get_device_id(queue), queue, n, a, lda, ipiv, group_count,
                               group_sizes, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getri_batch(sycl::queue& queue, std::int64_t* n, double** a,
                                      std::int64_t* lda, std::int64_t** ipiv,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      double* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::getri_batch(get_device_id(queue), queue, n, a, lda, ipiv, group_count,
                               group_sizes, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getri_batch(sycl::queue& queue, std::int64_t* n, std::complex<float>** a,
                                      std::int64_t* lda, std::int64_t** ipiv,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::getri_batch(get_device_id(queue), queue, n, a, lda, ipiv, group_count,
                               group_sizes, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getri_batch(sycl::queue& queue, std::int64_t* n, std::complex<double>** a,
                                      std::int64_t* lda, std::int64_t** ipiv,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      std::complex<double>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::getri_batch(get_device_id(queue), queue, n, a, lda, ipiv, group_count,
                               group_sizes, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose trans,
                                      std::int64_t n, std::int64_t nrhs, float* a, std::int64_t lda,
                                      std::int64_t stride_a, std::int64_t* ipiv,
                                      std::int64_t stride_ipiv, float* b, std::int64_t ldb,
                                      std::int64_t stride_b, std::int64_t batch_size,
                                      float* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::getrs_batch(get_device_id(queue), queue, trans, n, nrhs, a, lda, stride_a, ipiv,
                               stride_ipiv, b, ldb, stride_b, batch_size, scratchpad,
                               scratchpad_size, dependencies);
}
static inline sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose trans,
                                      std::int64_t n, std::int64_t nrhs, double* a,
                                      std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                                      std::int64_t stride_ipiv, double* b, std::int64_t ldb,
                                      std::int64_t stride_b, std::int64_t batch_size,
                                      double* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::getrs_batch(get_device_id(queue), queue, trans, n, nrhs, a, lda, stride_a, ipiv,
                               stride_ipiv, b, ldb, stride_b, batch_size, scratchpad,
                               scratchpad_size, dependencies);
}
static inline sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose trans,
                                      std::int64_t n, std::int64_t nrhs, std::complex<float>* a,
                                      std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                                      std::int64_t stride_ipiv, std::complex<float>* b,
                                      std::int64_t ldb, std::int64_t stride_b,
                                      std::int64_t batch_size, std::complex<float>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::getrs_batch(get_device_id(queue), queue, trans, n, nrhs, a, lda, stride_a, ipiv,
                               stride_ipiv, b, ldb, stride_b, batch_size, scratchpad,
                               scratchpad_size, dependencies);
}
static inline sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose trans,
                                      std::int64_t n, std::int64_t nrhs, std::complex<double>* a,
                                      std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                                      std::int64_t stride_ipiv, std::complex<double>* b,
                                      std::int64_t ldb, std::int64_t stride_b,
                                      std::int64_t batch_size, std::complex<double>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::getrs_batch(get_device_id(queue), queue, trans, n, nrhs, a, lda, stride_a, ipiv,
                               stride_ipiv, b, ldb, stride_b, batch_size, scratchpad,
                               scratchpad_size, dependencies);
}
static inline sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose* trans,
                                      std::int64_t* n, std::int64_t* nrhs, float** a,
                                      std::int64_t* lda, std::int64_t** ipiv, float** b,
                                      std::int64_t* ldb, std::int64_t group_count,
                                      std::int64_t* group_sizes, float* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::getrs_batch(get_device_id(queue), queue, trans, n, nrhs, a, lda, ipiv, b, ldb,
                               group_count, group_sizes, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose* trans,
                                      std::int64_t* n, std::int64_t* nrhs, double** a,
                                      std::int64_t* lda, std::int64_t** ipiv, double** b,
                                      std::int64_t* ldb, std::int64_t group_count,
                                      std::int64_t* group_sizes, double* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::getrs_batch(get_device_id(queue), queue, trans, n, nrhs, a, lda, ipiv, b, ldb,
                               group_count, group_sizes, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose* trans,
                                      std::int64_t* n, std::int64_t* nrhs, std::complex<float>** a,
                                      std::int64_t* lda, std::int64_t** ipiv,
                                      std::complex<float>** b, std::int64_t* ldb,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::getrs_batch(get_device_id(queue), queue, trans, n, nrhs, a, lda, ipiv, b, ldb,
                               group_count, group_sizes, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose* trans,
                                      std::int64_t* n, std::int64_t* nrhs, std::complex<double>** a,
                                      std::int64_t* lda, std::int64_t** ipiv,
                                      std::complex<double>** b, std::int64_t* ldb,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      std::complex<double>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::getrs_batch(get_device_id(queue), queue, trans, n, nrhs, a, lda, ipiv, b, ldb,
                               group_count, group_sizes, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event orgqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                      std::int64_t k, float* a, std::int64_t lda,
                                      std::int64_t stride_a, float* tau, std::int64_t stride_tau,
                                      std::int64_t batch_size, float* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::orgqr_batch(get_device_id(queue), queue, m, n, k, a, lda, stride_a, tau,
                               stride_tau, batch_size, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event orgqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                      std::int64_t k, double* a, std::int64_t lda,
                                      std::int64_t stride_a, double* tau, std::int64_t stride_tau,
                                      std::int64_t batch_size, double* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::orgqr_batch(get_device_id(queue), queue, m, n, k, a, lda, stride_a, tau,
                               stride_tau, batch_size, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event orgqr_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                      std::int64_t* k, float** a, std::int64_t* lda, float** tau,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      float* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::orgqr_batch(get_device_id(queue), queue, m, n, k, a, lda, tau, group_count,
                               group_sizes, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event orgqr_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                      std::int64_t* k, double** a, std::int64_t* lda, double** tau,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      double* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::orgqr_batch(get_device_id(queue), queue, m, n, k, a, lda, tau, group_count,
                               group_sizes, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                      float* a, std::int64_t lda, std::int64_t stride_a,
                                      std::int64_t batch_size, float* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::potrf_batch(get_device_id(queue), queue, uplo, n, a, lda, stride_a, batch_size,
                               scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                      double* a, std::int64_t lda, std::int64_t stride_a,
                                      std::int64_t batch_size, double* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::potrf_batch(get_device_id(queue), queue, uplo, n, a, lda, stride_a, batch_size,
                               scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                      std::complex<float>* a, std::int64_t lda,
                                      std::int64_t stride_a, std::int64_t batch_size,
                                      std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::potrf_batch(get_device_id(queue), queue, uplo, n, a, lda, stride_a, batch_size,
                               scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                      std::complex<double>* a, std::int64_t lda,
                                      std::int64_t stride_a, std::int64_t batch_size,
                                      std::complex<double>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::potrf_batch(get_device_id(queue), queue, uplo, n, a, lda, stride_a, batch_size,
                               scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n,
                                      float** a, std::int64_t* lda, std::int64_t group_count,
                                      std::int64_t* group_sizes, float* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::potrf_batch(get_device_id(queue), queue, uplo, n, a, lda, group_count,
                               group_sizes, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n,
                                      double** a, std::int64_t* lda, std::int64_t group_count,
                                      std::int64_t* group_sizes, double* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::potrf_batch(get_device_id(queue), queue, uplo, n, a, lda, group_count,
                               group_sizes, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n,
                                      std::complex<float>** a, std::int64_t* lda,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::potrf_batch(get_device_id(queue), queue, uplo, n, a, lda, group_count,
                               group_sizes, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n,
                                      std::complex<double>** a, std::int64_t* lda,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      std::complex<double>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::potrf_batch(get_device_id(queue), queue, uplo, n, a, lda, group_count,
                               group_sizes, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                      std::int64_t nrhs, float* a, std::int64_t lda,
                                      std::int64_t stride_a, float* b, std::int64_t ldb,
                                      std::int64_t stride_b, std::int64_t batch_size,
                                      float* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::potrs_batch(get_device_id(queue), queue, uplo, n, nrhs, a, lda, stride_a, b, ldb,
                               stride_b, batch_size, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                      std::int64_t nrhs, double* a, std::int64_t lda,
                                      std::int64_t stride_a, double* b, std::int64_t ldb,
                                      std::int64_t stride_b, std::int64_t batch_size,
                                      double* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::potrs_batch(get_device_id(queue), queue, uplo, n, nrhs, a, lda, stride_a, b, ldb,
                               stride_b, batch_size, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                      std::int64_t nrhs, std::complex<float>* a, std::int64_t lda,
                                      std::int64_t stride_a, std::complex<float>* b,
                                      std::int64_t ldb, std::int64_t stride_b,
                                      std::int64_t batch_size, std::complex<float>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::potrs_batch(get_device_id(queue), queue, uplo, n, nrhs, a, lda, stride_a, b, ldb,
                               stride_b, batch_size, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                      std::int64_t nrhs, std::complex<double>* a, std::int64_t lda,
                                      std::int64_t stride_a, std::complex<double>* b,
                                      std::int64_t ldb, std::int64_t stride_b,
                                      std::int64_t batch_size, std::complex<double>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::potrs_batch(get_device_id(queue), queue, uplo, n, nrhs, a, lda, stride_a, b, ldb,
                               stride_b, batch_size, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event potrs_batch(sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n,
                                      std::int64_t* nrhs, float** a, std::int64_t* lda, float** b,
                                      std::int64_t* ldb, std::int64_t group_count,
                                      std::int64_t* group_sizes, float* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::potrs_batch(get_device_id(queue), queue, uplo, n, nrhs, a, lda, b, ldb,
                               group_count, group_sizes, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event potrs_batch(sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n,
                                      std::int64_t* nrhs, double** a, std::int64_t* lda, double** b,
                                      std::int64_t* ldb, std::int64_t group_count,
                                      std::int64_t* group_sizes, double* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::potrs_batch(get_device_id(queue), queue, uplo, n, nrhs, a, lda, b, ldb,
                               group_count, group_sizes, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event potrs_batch(sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n,
                                      std::int64_t* nrhs, std::complex<float>** a,
                                      std::int64_t* lda, std::complex<float>** b, std::int64_t* ldb,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::potrs_batch(get_device_id(queue), queue, uplo, n, nrhs, a, lda, b, ldb,
                               group_count, group_sizes, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event potrs_batch(sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n,
                                      std::int64_t* nrhs, std::complex<double>** a,
                                      std::int64_t* lda, std::complex<double>** b,
                                      std::int64_t* ldb, std::int64_t group_count,
                                      std::int64_t* group_sizes, std::complex<double>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::potrs_batch(get_device_id(queue), queue, uplo, n, nrhs, a, lda, b, ldb,
                               group_count, group_sizes, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event ungqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                      std::int64_t k, std::complex<float>* a, std::int64_t lda,
                                      std::int64_t stride_a, std::complex<float>* tau,
                                      std::int64_t stride_tau, std::int64_t batch_size,
                                      std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::ungqr_batch(get_device_id(queue), queue, m, n, k, a, lda, stride_a, tau,
                               stride_tau, batch_size, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event ungqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                      std::int64_t k, std::complex<double>* a, std::int64_t lda,
                                      std::int64_t stride_a, std::complex<double>* tau,
                                      std::int64_t stride_tau, std::int64_t batch_size,
                                      std::complex<double>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::ungqr_batch(get_device_id(queue), queue, m, n, k, a, lda, stride_a, tau,
                               stride_tau, batch_size, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event ungqr_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                      std::int64_t* k, std::complex<float>** a, std::int64_t* lda,
                                      std::complex<float>** tau, std::int64_t group_count,
                                      std::int64_t* group_sizes, std::complex<float>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::ungqr_batch(get_device_id(queue), queue, m, n, k, a, lda, tau, group_count,
                               group_sizes, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event ungqr_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                      std::int64_t* k, std::complex<double>** a, std::int64_t* lda,
                                      std::complex<double>** tau, std::int64_t group_count,
                                      std::int64_t* group_sizes, std::complex<double>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return detail::ungqr_batch(get_device_id(queue), queue, m, n, k, a, lda, tau, group_count,
                               group_sizes, scratchpad, scratchpad_size, dependencies);
}

template <typename fp_type, internal::is_floating_point<fp_type> = nullptr>
std::int64_t gebrd_scratchpad_size(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                   std::int64_t lda) {
    return detail::gebrd_scratchpad_size<fp_type>(get_device_id(queue), queue, m, n, lda);
}
template <typename fp_type, oneapi::math::lapack::internal::is_floating_point<fp_type> = nullptr>
std::int64_t gerqf_scratchpad_size(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                   std::int64_t lda) {
    return detail::gerqf_scratchpad_size<fp_type>(get_device_id(queue), queue, m, n, lda);
}
template <typename fp_type, internal::is_floating_point<fp_type> = nullptr>
std::int64_t geqrf_scratchpad_size(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                   std::int64_t lda) {
    return detail::geqrf_scratchpad_size<fp_type>(get_device_id(queue), queue, m, n, lda);
}
template <typename fp_type, internal::is_real_floating_point<fp_type> = nullptr>
std::int64_t gesvd_scratchpad_size(sycl::queue& queue, oneapi::math::jobsvd jobu,
                                   oneapi::math::jobsvd jobvt, std::int64_t m, std::int64_t n,
                                   std::int64_t lda, std::int64_t ldu, std::int64_t ldvt) {
    return detail::gesvd_scratchpad_size<fp_type>(get_device_id(queue), queue, jobu, jobvt, m, n,
                                                  lda, ldu, ldvt);
}
template <typename fp_type, internal::is_complex_floating_point<fp_type> = nullptr>
std::int64_t gesvd_scratchpad_size(sycl::queue& queue, oneapi::math::jobsvd jobu,
                                   oneapi::math::jobsvd jobvt, std::int64_t m, std::int64_t n,
                                   std::int64_t lda, std::int64_t ldu, std::int64_t ldvt) {
    return detail::gesvd_scratchpad_size<fp_type>(get_device_id(queue), queue, jobu, jobvt, m, n,
                                                  lda, ldu, ldvt);
}
template <typename fp_type, internal::is_floating_point<fp_type> = nullptr>
std::int64_t getrf_scratchpad_size(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                   std::int64_t lda) {
    return detail::getrf_scratchpad_size<fp_type>(get_device_id(queue), queue, m, n, lda);
}
template <typename fp_type, internal::is_floating_point<fp_type> = nullptr>
std::int64_t getri_scratchpad_size(sycl::queue& queue, std::int64_t n, std::int64_t lda) {
    return detail::getri_scratchpad_size<fp_type>(get_device_id(queue), queue, n, lda);
}
template <typename fp_type, internal::is_floating_point<fp_type> = nullptr>
std::int64_t getrs_scratchpad_size(sycl::queue& queue, oneapi::math::transpose trans,
                                   std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                   std::int64_t ldb) {
    return detail::getrs_scratchpad_size<fp_type>(get_device_id(queue), queue, trans, n, nrhs, lda,
                                                  ldb);
}
template <typename fp_type, internal::is_complex_floating_point<fp_type> = nullptr>
std::int64_t heevd_scratchpad_size(sycl::queue& queue, oneapi::math::job jobz,
                                   oneapi::math::uplo uplo, std::int64_t n, std::int64_t lda) {
    return detail::heevd_scratchpad_size<fp_type>(get_device_id(queue), queue, jobz, uplo, n, lda);
}
template <typename fp_type, internal::is_complex_floating_point<fp_type> = nullptr>
std::int64_t hegvd_scratchpad_size(sycl::queue& queue, std::int64_t itype, oneapi::math::job jobz,
                                   oneapi::math::uplo uplo, std::int64_t n, std::int64_t lda,
                                   std::int64_t ldb) {
    return detail::hegvd_scratchpad_size<fp_type>(get_device_id(queue), queue, itype, jobz, uplo, n,
                                                  lda, ldb);
}
template <typename fp_type, internal::is_complex_floating_point<fp_type> = nullptr>
std::int64_t hetrd_scratchpad_size(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                   std::int64_t lda) {
    return detail::hetrd_scratchpad_size<fp_type>(get_device_id(queue), queue, uplo, n, lda);
}
template <typename fp_type,
          oneapi::math::lapack::internal::is_complex_floating_point<fp_type> = nullptr>
std::int64_t hetrf_scratchpad_size(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                   std::int64_t lda) {
    return detail::hetrf_scratchpad_size<fp_type>(get_device_id(queue), queue, uplo, n, lda);
}
template <typename fp_type,
          oneapi::math::lapack::internal::is_real_floating_point<fp_type> = nullptr>
std::int64_t orgbr_scratchpad_size(sycl::queue& queue, oneapi::math::generate vect, std::int64_t m,
                                   std::int64_t n, std::int64_t k, std::int64_t lda) {
    return detail::orgbr_scratchpad_size<fp_type>(get_device_id(queue), queue, vect, m, n, k, lda);
}
template <typename fp_type,
          oneapi::math::lapack::internal::is_real_floating_point<fp_type> = nullptr>
std::int64_t orgtr_scratchpad_size(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                   std::int64_t lda) {
    return detail::orgtr_scratchpad_size<fp_type>(get_device_id(queue), queue, uplo, n, lda);
}
template <typename fp_type, internal::is_real_floating_point<fp_type> = nullptr>
std::int64_t orgqr_scratchpad_size(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                   std::int64_t k, std::int64_t lda) {
    return detail::orgqr_scratchpad_size<fp_type>(get_device_id(queue), queue, m, n, k, lda);
}
template <typename fp_type,
          oneapi::math::lapack::internal::is_real_floating_point<fp_type> = nullptr>
std::int64_t ormrq_scratchpad_size(sycl::queue& queue, oneapi::math::side side,
                                   oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
                                   std::int64_t k, std::int64_t lda, std::int64_t ldc) {
    return detail::ormrq_scratchpad_size<fp_type>(get_device_id(queue), queue, side, trans, m, n, k,
                                                  lda, ldc);
}
template <typename fp_type, internal::is_real_floating_point<fp_type> = nullptr>
std::int64_t ormqr_scratchpad_size(sycl::queue& queue, oneapi::math::side side,
                                   oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
                                   std::int64_t k, std::int64_t lda, std::int64_t ldc) {
    return detail::ormqr_scratchpad_size<fp_type>(get_device_id(queue), queue, side, trans, m, n, k,
                                                  lda, ldc);
}
template <typename fp_type,
          oneapi::math::lapack::internal::is_real_floating_point<fp_type> = nullptr>
std::int64_t ormtr_scratchpad_size(sycl::queue& queue, oneapi::math::side side,
                                   oneapi::math::uplo uplo, oneapi::math::transpose trans,
                                   std::int64_t m, std::int64_t n, std::int64_t lda,
                                   std::int64_t ldc) {
    return detail::ormtr_scratchpad_size<fp_type>(get_device_id(queue), queue, side, uplo, trans, m,
                                                  n, lda, ldc);
}
template <typename fp_type, internal::is_floating_point<fp_type> = nullptr>
std::int64_t potrf_scratchpad_size(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                   std::int64_t lda) {
    return detail::potrf_scratchpad_size<fp_type>(get_device_id(queue), queue, uplo, n, lda);
}
template <typename fp_type, oneapi::math::lapack::internal::is_floating_point<fp_type> = nullptr>
std::int64_t potrs_scratchpad_size(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                   std::int64_t nrhs, std::int64_t lda, std::int64_t ldb) {
    return detail::potrs_scratchpad_size<fp_type>(get_device_id(queue), queue, uplo, n, nrhs, lda,
                                                  ldb);
}
template <typename fp_type, oneapi::math::lapack::internal::is_floating_point<fp_type> = nullptr>
std::int64_t potri_scratchpad_size(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                   std::int64_t lda) {
    return detail::potri_scratchpad_size<fp_type>(get_device_id(queue), queue, uplo, n, lda);
}
template <typename fp_type, oneapi::math::lapack::internal::is_floating_point<fp_type> = nullptr>
std::int64_t sytrf_scratchpad_size(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                   std::int64_t lda) {
    return detail::sytrf_scratchpad_size<fp_type>(get_device_id(queue), queue, uplo, n, lda);
}
template <typename fp_type, internal::is_real_floating_point<fp_type> = nullptr>
std::int64_t syevd_scratchpad_size(sycl::queue& queue, oneapi::math::job jobz,
                                   oneapi::math::uplo uplo, std::int64_t n, std::int64_t lda) {
    return detail::syevd_scratchpad_size<fp_type>(get_device_id(queue), queue, jobz, uplo, n, lda);
}
template <typename fp_type, internal::is_real_floating_point<fp_type> = nullptr>
std::int64_t sygvd_scratchpad_size(sycl::queue& queue, std::int64_t itype, oneapi::math::job jobz,
                                   oneapi::math::uplo uplo, std::int64_t n, std::int64_t lda,
                                   std::int64_t ldb) {
    return detail::sygvd_scratchpad_size<fp_type>(get_device_id(queue), queue, itype, jobz, uplo, n,
                                                  lda, ldb);
}
template <typename fp_type, internal::is_real_floating_point<fp_type> = nullptr>
std::int64_t sytrd_scratchpad_size(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                   std::int64_t lda) {
    return detail::sytrd_scratchpad_size<fp_type>(get_device_id(queue), queue, uplo, n, lda);
}
template <typename fp_type, internal::is_floating_point<fp_type> = nullptr>
std::int64_t trtrs_scratchpad_size(sycl::queue& queue, oneapi::math::uplo uplo,
                                   oneapi::math::transpose trans, oneapi::math::diag diag,
                                   std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                   std::int64_t ldb) {
    return detail::trtrs_scratchpad_size<fp_type>(get_device_id(queue), queue, uplo, trans, diag, n,
                                                  nrhs, lda, ldb);
}
template <typename fp_type,
          oneapi::math::lapack::internal::is_complex_floating_point<fp_type> = nullptr>
std::int64_t ungbr_scratchpad_size(sycl::queue& queue, oneapi::math::generate vect, std::int64_t m,
                                   std::int64_t n, std::int64_t k, std::int64_t lda) {
    return detail::ungbr_scratchpad_size<fp_type>(get_device_id(queue), queue, vect, m, n, k, lda);
}
template <typename fp_type, internal::is_complex_floating_point<fp_type> = nullptr>
std::int64_t ungqr_scratchpad_size(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                   std::int64_t k, std::int64_t lda) {
    return detail::ungqr_scratchpad_size<fp_type>(get_device_id(queue), queue, m, n, k, lda);
}
template <typename fp_type,
          oneapi::math::lapack::internal::is_complex_floating_point<fp_type> = nullptr>
std::int64_t ungtr_scratchpad_size(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                   std::int64_t lda) {
    return detail::ungtr_scratchpad_size<fp_type>(get_device_id(queue), queue, uplo, n, lda);
}
template <typename fp_type,
          oneapi::math::lapack::internal::is_complex_floating_point<fp_type> = nullptr>
std::int64_t unmrq_scratchpad_size(sycl::queue& queue, oneapi::math::side side,
                                   oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
                                   std::int64_t k, std::int64_t lda, std::int64_t ldc) {
    return detail::unmrq_scratchpad_size<fp_type>(get_device_id(queue), queue, side, trans, m, n, k,
                                                  lda, ldc);
}
template <typename fp_type, internal::is_complex_floating_point<fp_type> = nullptr>
std::int64_t unmqr_scratchpad_size(sycl::queue& queue, oneapi::math::side side,
                                   oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
                                   std::int64_t k, std::int64_t lda, std::int64_t ldc) {
    return detail::unmqr_scratchpad_size<fp_type>(get_device_id(queue), queue, side, trans, m, n, k,
                                                  lda, ldc);
}
template <typename fp_type,
          oneapi::math::lapack::internal::is_complex_floating_point<fp_type> = nullptr>
std::int64_t unmtr_scratchpad_size(sycl::queue& queue, oneapi::math::side side,
                                   oneapi::math::uplo uplo, oneapi::math::transpose trans,
                                   std::int64_t m, std::int64_t n, std::int64_t lda,
                                   std::int64_t ldc) {
    return detail::unmtr_scratchpad_size<fp_type>(get_device_id(queue), queue, side, uplo, trans, m,
                                                  n, lda, ldc);
}
template <typename fp_type, oneapi::math::lapack::internal::is_floating_point<fp_type> = nullptr>
std::int64_t getrf_batch_scratchpad_size(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                         std::int64_t lda, std::int64_t stride_a,
                                         std::int64_t stride_ipiv, std::int64_t batch_size) {
    return detail::getrf_batch_scratchpad_size<fp_type>(get_device_id(queue), queue, m, n, lda,
                                                        stride_a, stride_ipiv, batch_size);
}
template <typename fp_type, oneapi::math::lapack::internal::is_floating_point<fp_type> = nullptr>
std::int64_t getri_batch_scratchpad_size(sycl::queue& queue, std::int64_t n, std::int64_t lda,
                                         std::int64_t stride_a, std::int64_t stride_ipiv,
                                         std::int64_t batch_size) {
    return detail::getri_batch_scratchpad_size<fp_type>(get_device_id(queue), queue, n, lda,
                                                        stride_a, stride_ipiv, batch_size);
}
template <typename fp_type, oneapi::math::lapack::internal::is_floating_point<fp_type> = nullptr>
std::int64_t getrs_batch_scratchpad_size(sycl::queue& queue, oneapi::math::transpose trans,
                                         std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                         std::int64_t stride_a, std::int64_t stride_ipiv,
                                         std::int64_t ldb, std::int64_t stride_b,
                                         std::int64_t batch_size) {
    return detail::getrs_batch_scratchpad_size<fp_type>(get_device_id(queue), queue, trans, n, nrhs,
                                                        lda, stride_a, stride_ipiv, ldb, stride_b,
                                                        batch_size);
}
template <typename fp_type, oneapi::math::lapack::internal::is_floating_point<fp_type> = nullptr>
std::int64_t geqrf_batch_scratchpad_size(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                         std::int64_t lda, std::int64_t stride_a,
                                         std::int64_t stride_tau, std::int64_t batch_size) {
    return detail::geqrf_batch_scratchpad_size<fp_type>(get_device_id(queue), queue, m, n, lda,
                                                        stride_a, stride_tau, batch_size);
}
template <typename fp_type, oneapi::math::lapack::internal::is_floating_point<fp_type> = nullptr>
std::int64_t potrf_batch_scratchpad_size(sycl::queue& queue, oneapi::math::uplo uplo,
                                         std::int64_t n, std::int64_t lda, std::int64_t stride_a,
                                         std::int64_t batch_size) {
    return detail::potrf_batch_scratchpad_size<fp_type>(get_device_id(queue), queue, uplo, n, lda,
                                                        stride_a, batch_size);
}
template <typename fp_type, oneapi::math::lapack::internal::is_floating_point<fp_type> = nullptr>
std::int64_t potrs_batch_scratchpad_size(sycl::queue& queue, oneapi::math::uplo uplo,
                                         std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                         std::int64_t stride_a, std::int64_t ldb,
                                         std::int64_t stride_b, std::int64_t batch_size) {
    return detail::potrs_batch_scratchpad_size<fp_type>(get_device_id(queue), queue, uplo, n, nrhs,
                                                        lda, stride_a, ldb, stride_b, batch_size);
}
template <typename fp_type,
          oneapi::math::lapack::internal::is_real_floating_point<fp_type> = nullptr>
std::int64_t orgqr_batch_scratchpad_size(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                         std::int64_t k, std::int64_t lda, std::int64_t stride_a,
                                         std::int64_t stride_tau, std::int64_t batch_size) {
    return detail::orgqr_batch_scratchpad_size<fp_type>(get_device_id(queue), queue, m, n, k, lda,
                                                        stride_a, stride_tau, batch_size);
}
template <typename fp_type,
          oneapi::math::lapack::internal::is_complex_floating_point<fp_type> = nullptr>
std::int64_t ungqr_batch_scratchpad_size(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                         std::int64_t k, std::int64_t lda, std::int64_t stride_a,
                                         std::int64_t stride_tau, std::int64_t batch_size) {
    return detail::ungqr_batch_scratchpad_size<fp_type>(get_device_id(queue), queue, m, n, k, lda,
                                                        stride_a, stride_tau, batch_size);
}
template <typename fp_type, oneapi::math::lapack::internal::is_floating_point<fp_type> = nullptr>
std::int64_t getrf_batch_scratchpad_size(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                         std::int64_t* lda, std::int64_t group_count,
                                         std::int64_t* group_sizes) {
    return detail::getrf_batch_scratchpad_size<fp_type>(get_device_id(queue), queue, m, n, lda,
                                                        group_count, group_sizes);
}
template <typename fp_type, oneapi::math::lapack::internal::is_floating_point<fp_type> = nullptr>
std::int64_t getri_batch_scratchpad_size(sycl::queue& queue, std::int64_t* n, std::int64_t* lda,
                                         std::int64_t group_count, std::int64_t* group_sizes) {
    return detail::getri_batch_scratchpad_size<fp_type>(get_device_id(queue), queue, n, lda,
                                                        group_count, group_sizes);
}
template <typename fp_type, oneapi::math::lapack::internal::is_floating_point<fp_type> = nullptr>
std::int64_t getrs_batch_scratchpad_size(sycl::queue& queue, oneapi::math::transpose* trans,
                                         std::int64_t* n, std::int64_t* nrhs, std::int64_t* lda,
                                         std::int64_t* ldb, std::int64_t group_count,
                                         std::int64_t* group_sizes) {
    return detail::getrs_batch_scratchpad_size<fp_type>(get_device_id(queue), queue, trans, n, nrhs,
                                                        lda, ldb, group_count, group_sizes);
}
template <typename fp_type, oneapi::math::lapack::internal::is_floating_point<fp_type> = nullptr>
std::int64_t geqrf_batch_scratchpad_size(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                         std::int64_t* lda, std::int64_t group_count,
                                         std::int64_t* group_sizes) {
    return detail::geqrf_batch_scratchpad_size<fp_type>(get_device_id(queue), queue, m, n, lda,
                                                        group_count, group_sizes);
}
template <typename fp_type,
          oneapi::math::lapack::internal::is_real_floating_point<fp_type> = nullptr>
std::int64_t orgqr_batch_scratchpad_size(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                         std::int64_t* k, std::int64_t* lda,
                                         std::int64_t group_count, std::int64_t* group_sizes) {
    return detail::orgqr_batch_scratchpad_size<fp_type>(get_device_id(queue), queue, m, n, k, lda,
                                                        group_count, group_sizes);
}
template <typename fp_type, oneapi::math::lapack::internal::is_floating_point<fp_type> = nullptr>
std::int64_t potrf_batch_scratchpad_size(sycl::queue& queue, oneapi::math::uplo* uplo,
                                         std::int64_t* n, std::int64_t* lda,
                                         std::int64_t group_count, std::int64_t* group_sizes) {
    return detail::potrf_batch_scratchpad_size<fp_type>(get_device_id(queue), queue, uplo, n, lda,
                                                        group_count, group_sizes);
}
template <typename fp_type, oneapi::math::lapack::internal::is_floating_point<fp_type> = nullptr>
std::int64_t potrs_batch_scratchpad_size(sycl::queue& queue, oneapi::math::uplo* uplo,
                                         std::int64_t* n, std::int64_t* nrhs, std::int64_t* lda,
                                         std::int64_t* ldb, std::int64_t group_count,
                                         std::int64_t* group_sizes) {
    return detail::potrs_batch_scratchpad_size<fp_type>(get_device_id(queue), queue, uplo, n, nrhs,
                                                        lda, ldb, group_count, group_sizes);
}
template <typename fp_type,
          oneapi::math::lapack::internal::is_complex_floating_point<fp_type> = nullptr>
std::int64_t ungqr_batch_scratchpad_size(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                         std::int64_t* k, std::int64_t* lda,
                                         std::int64_t group_count, std::int64_t* group_sizes) {
    return detail::ungqr_batch_scratchpad_size<fp_type>(get_device_id(queue), queue, m, n, k, lda,
                                                        group_count, group_sizes);
}

} // namespace lapack
} // namespace math
} // namespace oneapi
