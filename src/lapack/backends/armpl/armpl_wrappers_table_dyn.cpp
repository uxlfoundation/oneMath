//
// generated file
//
#include "armpl_common.hpp"
#include "oneapi/math/lapack/detail/armpl/onemath_lapack_armpl.hxx"
#include "lapack/function_table.hpp"

#define WRAPPER_VERSION 1

extern "C" lapack_function_table_t onemath_lapack_table = {
    WRAPPER_VERSION,
#define LAPACK_BACKEND armpl
    oneapi::math::lapack::armpl::gebrd,
    oneapi::math::lapack::armpl::gebrd,
    oneapi::math::lapack::armpl::gebrd,
    oneapi::math::lapack::armpl::gebrd,
    oneapi::math::lapack::armpl::gerqf,
    oneapi::math::lapack::armpl::gerqf,
    oneapi::math::lapack::armpl::gerqf,
    oneapi::math::lapack::armpl::gerqf,
    oneapi::math::lapack::armpl::geqrf,
    oneapi::math::lapack::armpl::geqrf,
    oneapi::math::lapack::armpl::geqrf,
    oneapi::math::lapack::armpl::geqrf,
    oneapi::math::lapack::armpl::getrf,
    oneapi::math::lapack::armpl::getrf,
    oneapi::math::lapack::armpl::getrf,
    oneapi::math::lapack::armpl::getrf,
    oneapi::math::lapack::armpl::getri,
    oneapi::math::lapack::armpl::getri,
    oneapi::math::lapack::armpl::getri,
    oneapi::math::lapack::armpl::getri,
    oneapi::math::lapack::armpl::getrs,
    oneapi::math::lapack::armpl::getrs,
    oneapi::math::lapack::armpl::getrs,
    oneapi::math::lapack::armpl::getrs,
    oneapi::math::lapack::armpl::gesvd,
    oneapi::math::lapack::armpl::gesvd,
    oneapi::math::lapack::armpl::gesvd,
    oneapi::math::lapack::armpl::gesvd,
    oneapi::math::lapack::armpl::heevd,
    oneapi::math::lapack::armpl::heevd,
    oneapi::math::lapack::armpl::hegvd,
    oneapi::math::lapack::armpl::hegvd,
    oneapi::math::lapack::armpl::hetrd,
    oneapi::math::lapack::armpl::hetrd,
    oneapi::math::lapack::armpl::hetrf,
    oneapi::math::lapack::armpl::hetrf,
    oneapi::math::lapack::armpl::orgbr,
    oneapi::math::lapack::armpl::orgbr,
    oneapi::math::lapack::armpl::orgqr,
    oneapi::math::lapack::armpl::orgqr,
    oneapi::math::lapack::armpl::orgtr,
    oneapi::math::lapack::armpl::orgtr,
    oneapi::math::lapack::armpl::ormtr,
    oneapi::math::lapack::armpl::ormtr,
    oneapi::math::lapack::armpl::ormrq,
    oneapi::math::lapack::armpl::ormrq,
    oneapi::math::lapack::armpl::ormqr,
    oneapi::math::lapack::armpl::ormqr,
    oneapi::math::lapack::armpl::potrf,
    oneapi::math::lapack::armpl::potrf,
    oneapi::math::lapack::armpl::potrf,
    oneapi::math::lapack::armpl::potrf,
    oneapi::math::lapack::armpl::potri,
    oneapi::math::lapack::armpl::potri,
    oneapi::math::lapack::armpl::potri,
    oneapi::math::lapack::armpl::potri,
    oneapi::math::lapack::armpl::potrs,
    oneapi::math::lapack::armpl::potrs,
    oneapi::math::lapack::armpl::potrs,
    oneapi::math::lapack::armpl::potrs,
    oneapi::math::lapack::armpl::syevd,
    oneapi::math::lapack::armpl::syevd,
    oneapi::math::lapack::armpl::sygvd,
    oneapi::math::lapack::armpl::sygvd,
    oneapi::math::lapack::armpl::sytrd,
    oneapi::math::lapack::armpl::sytrd,
    oneapi::math::lapack::armpl::sytrf,
    oneapi::math::lapack::armpl::sytrf,
    oneapi::math::lapack::armpl::sytrf,
    oneapi::math::lapack::armpl::sytrf,
    oneapi::math::lapack::armpl::trtrs,
    oneapi::math::lapack::armpl::trtrs,
    oneapi::math::lapack::armpl::trtrs,
    oneapi::math::lapack::armpl::trtrs,
    oneapi::math::lapack::armpl::ungbr,
    oneapi::math::lapack::armpl::ungbr,
    oneapi::math::lapack::armpl::ungqr,
    oneapi::math::lapack::armpl::ungqr,
    oneapi::math::lapack::armpl::ungtr,
    oneapi::math::lapack::armpl::ungtr,
    oneapi::math::lapack::armpl::unmrq,
    oneapi::math::lapack::armpl::unmrq,
    oneapi::math::lapack::armpl::unmqr,
    oneapi::math::lapack::armpl::unmqr,
    oneapi::math::lapack::armpl::unmtr,
    oneapi::math::lapack::armpl::unmtr,
    oneapi::math::lapack::armpl::gebrd,
    oneapi::math::lapack::armpl::gebrd,
    oneapi::math::lapack::armpl::gebrd,
    oneapi::math::lapack::armpl::gebrd,
    oneapi::math::lapack::armpl::gerqf,
    oneapi::math::lapack::armpl::gerqf,
    oneapi::math::lapack::armpl::gerqf,
    oneapi::math::lapack::armpl::gerqf,
    oneapi::math::lapack::armpl::geqrf,
    oneapi::math::lapack::armpl::geqrf,
    oneapi::math::lapack::armpl::geqrf,
    oneapi::math::lapack::armpl::geqrf,
    oneapi::math::lapack::armpl::getrf,
    oneapi::math::lapack::armpl::getrf,
    oneapi::math::lapack::armpl::getrf,
    oneapi::math::lapack::armpl::getrf,
    oneapi::math::lapack::armpl::getri,
    oneapi::math::lapack::armpl::getri,
    oneapi::math::lapack::armpl::getri,
    oneapi::math::lapack::armpl::getri,
    oneapi::math::lapack::armpl::getrs,
    oneapi::math::lapack::armpl::getrs,
    oneapi::math::lapack::armpl::getrs,
    oneapi::math::lapack::armpl::getrs,
    oneapi::math::lapack::armpl::gesvd,
    oneapi::math::lapack::armpl::gesvd,
    oneapi::math::lapack::armpl::gesvd,
    oneapi::math::lapack::armpl::gesvd,
    oneapi::math::lapack::armpl::heevd,
    oneapi::math::lapack::armpl::heevd,
    oneapi::math::lapack::armpl::hegvd,
    oneapi::math::lapack::armpl::hegvd,
    oneapi::math::lapack::armpl::hetrd,
    oneapi::math::lapack::armpl::hetrd,
    oneapi::math::lapack::armpl::hetrf,
    oneapi::math::lapack::armpl::hetrf,
    oneapi::math::lapack::armpl::orgbr,
    oneapi::math::lapack::armpl::orgbr,
    oneapi::math::lapack::armpl::orgqr,
    oneapi::math::lapack::armpl::orgqr,
    oneapi::math::lapack::armpl::orgtr,
    oneapi::math::lapack::armpl::orgtr,
    oneapi::math::lapack::armpl::ormtr,
    oneapi::math::lapack::armpl::ormtr,
    oneapi::math::lapack::armpl::ormrq,
    oneapi::math::lapack::armpl::ormrq,
    oneapi::math::lapack::armpl::ormqr,
    oneapi::math::lapack::armpl::ormqr,
    oneapi::math::lapack::armpl::potrf,
    oneapi::math::lapack::armpl::potrf,
    oneapi::math::lapack::armpl::potrf,
    oneapi::math::lapack::armpl::potrf,
    oneapi::math::lapack::armpl::potri,
    oneapi::math::lapack::armpl::potri,
    oneapi::math::lapack::armpl::potri,
    oneapi::math::lapack::armpl::potri,
    oneapi::math::lapack::armpl::potrs,
    oneapi::math::lapack::armpl::potrs,
    oneapi::math::lapack::armpl::potrs,
    oneapi::math::lapack::armpl::potrs,
    oneapi::math::lapack::armpl::syevd,
    oneapi::math::lapack::armpl::syevd,
    oneapi::math::lapack::armpl::sygvd,
    oneapi::math::lapack::armpl::sygvd,
    oneapi::math::lapack::armpl::sytrd,
    oneapi::math::lapack::armpl::sytrd,
    oneapi::math::lapack::armpl::sytrf,
    oneapi::math::lapack::armpl::sytrf,
    oneapi::math::lapack::armpl::sytrf,
    oneapi::math::lapack::armpl::sytrf,
    oneapi::math::lapack::armpl::trtrs,
    oneapi::math::lapack::armpl::trtrs,
    oneapi::math::lapack::armpl::trtrs,
    oneapi::math::lapack::armpl::trtrs,
    oneapi::math::lapack::armpl::ungbr,
    oneapi::math::lapack::armpl::ungbr,
    oneapi::math::lapack::armpl::ungqr,
    oneapi::math::lapack::armpl::ungqr,
    oneapi::math::lapack::armpl::ungtr,
    oneapi::math::lapack::armpl::ungtr,
    oneapi::math::lapack::armpl::unmrq,
    oneapi::math::lapack::armpl::unmrq,
    oneapi::math::lapack::armpl::unmqr,
    oneapi::math::lapack::armpl::unmqr,
    oneapi::math::lapack::armpl::unmtr,
    oneapi::math::lapack::armpl::unmtr,
    oneapi::math::lapack::armpl::geqrf_batch,
    oneapi::math::lapack::armpl::geqrf_batch,
    oneapi::math::lapack::armpl::geqrf_batch,
    oneapi::math::lapack::armpl::geqrf_batch,
    oneapi::math::lapack::armpl::getri_batch,
    oneapi::math::lapack::armpl::getri_batch,
    oneapi::math::lapack::armpl::getri_batch,
    oneapi::math::lapack::armpl::getri_batch,
    oneapi::math::lapack::armpl::getrs_batch,
    oneapi::math::lapack::armpl::getrs_batch,
    oneapi::math::lapack::armpl::getrs_batch,
    oneapi::math::lapack::armpl::getrs_batch,
    oneapi::math::lapack::armpl::getrf_batch,
    oneapi::math::lapack::armpl::getrf_batch,
    oneapi::math::lapack::armpl::getrf_batch,
    oneapi::math::lapack::armpl::getrf_batch,
    oneapi::math::lapack::armpl::orgqr_batch,
    oneapi::math::lapack::armpl::orgqr_batch,
    oneapi::math::lapack::armpl::potrf_batch,
    oneapi::math::lapack::armpl::potrf_batch,
    oneapi::math::lapack::armpl::potrf_batch,
    oneapi::math::lapack::armpl::potrf_batch,
    oneapi::math::lapack::armpl::potrs_batch,
    oneapi::math::lapack::armpl::potrs_batch,
    oneapi::math::lapack::armpl::potrs_batch,
    oneapi::math::lapack::armpl::potrs_batch,
    oneapi::math::lapack::armpl::ungqr_batch,
    oneapi::math::lapack::armpl::ungqr_batch,
    oneapi::math::lapack::armpl::geqrf_batch,
    oneapi::math::lapack::armpl::geqrf_batch,
    oneapi::math::lapack::armpl::geqrf_batch,
    oneapi::math::lapack::armpl::geqrf_batch,
    oneapi::math::lapack::armpl::getrf_batch,
    oneapi::math::lapack::armpl::getrf_batch,
    oneapi::math::lapack::armpl::getrf_batch,
    oneapi::math::lapack::armpl::getrf_batch,
    oneapi::math::lapack::armpl::getri_batch,
    oneapi::math::lapack::armpl::getri_batch,
    oneapi::math::lapack::armpl::getri_batch,
    oneapi::math::lapack::armpl::getri_batch,
    oneapi::math::lapack::armpl::getrs_batch,
    oneapi::math::lapack::armpl::getrs_batch,
    oneapi::math::lapack::armpl::getrs_batch,
    oneapi::math::lapack::armpl::getrs_batch,
    oneapi::math::lapack::armpl::orgqr_batch,
    oneapi::math::lapack::armpl::orgqr_batch,
    oneapi::math::lapack::armpl::potrf_batch,
    oneapi::math::lapack::armpl::potrf_batch,
    oneapi::math::lapack::armpl::potrf_batch,
    oneapi::math::lapack::armpl::potrf_batch,
    oneapi::math::lapack::armpl::potrs_batch,
    oneapi::math::lapack::armpl::potrs_batch,
    oneapi::math::lapack::armpl::potrs_batch,
    oneapi::math::lapack::armpl::potrs_batch,
    oneapi::math::lapack::armpl::ungqr_batch,
    oneapi::math::lapack::armpl::ungqr_batch,
    oneapi::math::lapack::armpl::geqrf_batch,
    oneapi::math::lapack::armpl::geqrf_batch,
    oneapi::math::lapack::armpl::geqrf_batch,
    oneapi::math::lapack::armpl::geqrf_batch,
    oneapi::math::lapack::armpl::getrf_batch,
    oneapi::math::lapack::armpl::getrf_batch,
    oneapi::math::lapack::armpl::getrf_batch,
    oneapi::math::lapack::armpl::getrf_batch,
    oneapi::math::lapack::armpl::getri_batch,
    oneapi::math::lapack::armpl::getri_batch,
    oneapi::math::lapack::armpl::getri_batch,
    oneapi::math::lapack::armpl::getri_batch,
    oneapi::math::lapack::armpl::getrs_batch,
    oneapi::math::lapack::armpl::getrs_batch,
    oneapi::math::lapack::armpl::getrs_batch,
    oneapi::math::lapack::armpl::getrs_batch,
    oneapi::math::lapack::armpl::orgqr_batch,
    oneapi::math::lapack::armpl::orgqr_batch,
    oneapi::math::lapack::armpl::potrf_batch,
    oneapi::math::lapack::armpl::potrf_batch,
    oneapi::math::lapack::armpl::potrf_batch,
    oneapi::math::lapack::armpl::potrf_batch,
    oneapi::math::lapack::armpl::potrs_batch,
    oneapi::math::lapack::armpl::potrs_batch,
    oneapi::math::lapack::armpl::potrs_batch,
    oneapi::math::lapack::armpl::potrs_batch,
    oneapi::math::lapack::armpl::ungqr_batch,
    oneapi::math::lapack::armpl::ungqr_batch,
    oneapi::math::lapack::armpl::gebrd_scratchpad_size<float>,
    oneapi::math::lapack::armpl::gebrd_scratchpad_size<double>,
    oneapi::math::lapack::armpl::gebrd_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::gebrd_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::gerqf_scratchpad_size<float>,
    oneapi::math::lapack::armpl::gerqf_scratchpad_size<double>,
    oneapi::math::lapack::armpl::gerqf_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::gerqf_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::geqrf_scratchpad_size<float>,
    oneapi::math::lapack::armpl::geqrf_scratchpad_size<double>,
    oneapi::math::lapack::armpl::geqrf_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::geqrf_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::gesvd_scratchpad_size<float>,
    oneapi::math::lapack::armpl::gesvd_scratchpad_size<double>,
    oneapi::math::lapack::armpl::gesvd_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::gesvd_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::getrf_scratchpad_size<float>,
    oneapi::math::lapack::armpl::getrf_scratchpad_size<double>,
    oneapi::math::lapack::armpl::getrf_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::getrf_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::getri_scratchpad_size<float>,
    oneapi::math::lapack::armpl::getri_scratchpad_size<double>,
    oneapi::math::lapack::armpl::getri_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::getri_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::getrs_scratchpad_size<float>,
    oneapi::math::lapack::armpl::getrs_scratchpad_size<double>,
    oneapi::math::lapack::armpl::getrs_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::getrs_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::heevd_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::heevd_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::hegvd_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::hegvd_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::hetrd_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::hetrd_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::hetrf_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::hetrf_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::orgbr_scratchpad_size<float>,
    oneapi::math::lapack::armpl::orgbr_scratchpad_size<double>,
    oneapi::math::lapack::armpl::orgtr_scratchpad_size<float>,
    oneapi::math::lapack::armpl::orgtr_scratchpad_size<double>,
    oneapi::math::lapack::armpl::orgqr_scratchpad_size<float>,
    oneapi::math::lapack::armpl::orgqr_scratchpad_size<double>,
    oneapi::math::lapack::armpl::ormrq_scratchpad_size<float>,
    oneapi::math::lapack::armpl::ormrq_scratchpad_size<double>,
    oneapi::math::lapack::armpl::ormqr_scratchpad_size<float>,
    oneapi::math::lapack::armpl::ormqr_scratchpad_size<double>,
    oneapi::math::lapack::armpl::ormtr_scratchpad_size<float>,
    oneapi::math::lapack::armpl::ormtr_scratchpad_size<double>,
    oneapi::math::lapack::armpl::potrf_scratchpad_size<float>,
    oneapi::math::lapack::armpl::potrf_scratchpad_size<double>,
    oneapi::math::lapack::armpl::potrf_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::potrf_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::potrs_scratchpad_size<float>,
    oneapi::math::lapack::armpl::potrs_scratchpad_size<double>,
    oneapi::math::lapack::armpl::potrs_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::potrs_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::potri_scratchpad_size<float>,
    oneapi::math::lapack::armpl::potri_scratchpad_size<double>,
    oneapi::math::lapack::armpl::potri_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::potri_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::sytrf_scratchpad_size<float>,
    oneapi::math::lapack::armpl::sytrf_scratchpad_size<double>,
    oneapi::math::lapack::armpl::sytrf_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::sytrf_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::syevd_scratchpad_size<float>,
    oneapi::math::lapack::armpl::syevd_scratchpad_size<double>,
    oneapi::math::lapack::armpl::sygvd_scratchpad_size<float>,
    oneapi::math::lapack::armpl::sygvd_scratchpad_size<double>,
    oneapi::math::lapack::armpl::sytrd_scratchpad_size<float>,
    oneapi::math::lapack::armpl::sytrd_scratchpad_size<double>,
    oneapi::math::lapack::armpl::trtrs_scratchpad_size<float>,
    oneapi::math::lapack::armpl::trtrs_scratchpad_size<double>,
    oneapi::math::lapack::armpl::trtrs_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::trtrs_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::ungbr_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::ungbr_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::ungqr_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::ungqr_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::ungtr_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::ungtr_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::unmrq_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::unmrq_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::unmqr_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::unmqr_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::unmtr_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::unmtr_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::getrf_batch_scratchpad_size<float>,
    oneapi::math::lapack::armpl::getrf_batch_scratchpad_size<double>,
    oneapi::math::lapack::armpl::getrf_batch_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::getrf_batch_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::getri_batch_scratchpad_size<float>,
    oneapi::math::lapack::armpl::getri_batch_scratchpad_size<double>,
    oneapi::math::lapack::armpl::getri_batch_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::getri_batch_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::getrs_batch_scratchpad_size<float>,
    oneapi::math::lapack::armpl::getrs_batch_scratchpad_size<double>,
    oneapi::math::lapack::armpl::getrs_batch_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::getrs_batch_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::geqrf_batch_scratchpad_size<float>,
    oneapi::math::lapack::armpl::geqrf_batch_scratchpad_size<double>,
    oneapi::math::lapack::armpl::geqrf_batch_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::geqrf_batch_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::potrf_batch_scratchpad_size<float>,
    oneapi::math::lapack::armpl::potrf_batch_scratchpad_size<double>,
    oneapi::math::lapack::armpl::potrf_batch_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::potrf_batch_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::potrs_batch_scratchpad_size<float>,
    oneapi::math::lapack::armpl::potrs_batch_scratchpad_size<double>,
    oneapi::math::lapack::armpl::potrs_batch_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::potrs_batch_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::orgqr_batch_scratchpad_size<float>,
    oneapi::math::lapack::armpl::orgqr_batch_scratchpad_size<double>,
    oneapi::math::lapack::armpl::ungqr_batch_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::ungqr_batch_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::getrf_batch_scratchpad_size<float>,
    oneapi::math::lapack::armpl::getrf_batch_scratchpad_size<double>,
    oneapi::math::lapack::armpl::getrf_batch_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::getrf_batch_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::getri_batch_scratchpad_size<float>,
    oneapi::math::lapack::armpl::getri_batch_scratchpad_size<double>,
    oneapi::math::lapack::armpl::getri_batch_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::getri_batch_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::getrs_batch_scratchpad_size<float>,
    oneapi::math::lapack::armpl::getrs_batch_scratchpad_size<double>,
    oneapi::math::lapack::armpl::getrs_batch_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::getrs_batch_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::geqrf_batch_scratchpad_size<float>,
    oneapi::math::lapack::armpl::geqrf_batch_scratchpad_size<double>,
    oneapi::math::lapack::armpl::geqrf_batch_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::geqrf_batch_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::orgqr_batch_scratchpad_size<float>,
    oneapi::math::lapack::armpl::orgqr_batch_scratchpad_size<double>,
    oneapi::math::lapack::armpl::potrf_batch_scratchpad_size<float>,
    oneapi::math::lapack::armpl::potrf_batch_scratchpad_size<double>,
    oneapi::math::lapack::armpl::potrf_batch_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::potrf_batch_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::potrs_batch_scratchpad_size<float>,
    oneapi::math::lapack::armpl::potrs_batch_scratchpad_size<double>,
    oneapi::math::lapack::armpl::potrs_batch_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::potrs_batch_scratchpad_size<std::complex<double>>,
    oneapi::math::lapack::armpl::ungqr_batch_scratchpad_size<std::complex<float>>,
    oneapi::math::lapack::armpl::ungqr_batch_scratchpad_size<std::complex<double>>
#undef LAPACK_BACKEND
};
