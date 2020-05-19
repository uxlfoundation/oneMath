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
#include "blas/function_table.hpp"
#include "onemkl/blas/detail/cublas/onemkl_blas_cublas.hpp"

#define WRAPPER_VERSION 1

extern "C" function_table_t mkl_blas_table = {
    WRAPPER_VERSION,
    onemkl::cublas::asum,
    onemkl::cublas::asum,
    onemkl::cublas::asum,
    onemkl::cublas::asum,
    onemkl::cublas::axpy,
    onemkl::cublas::axpy,
    onemkl::cublas::axpy,
    onemkl::cublas::axpy,
    onemkl::cublas::copy,
    onemkl::cublas::copy,
    onemkl::cublas::copy,
    onemkl::cublas::copy,
    onemkl::cublas::dot,
    onemkl::cublas::dot,
    onemkl::cublas::dot,
    onemkl::cublas::dotc,
    onemkl::cublas::dotc,
    onemkl::cublas::dotu,
    onemkl::cublas::dotu,
    onemkl::cublas::iamin,
    onemkl::cublas::iamin,
    onemkl::cublas::iamin,
    onemkl::cublas::iamin,
    onemkl::cublas::iamax,
    onemkl::cublas::iamax,
    onemkl::cublas::iamax,
    onemkl::cublas::iamax,
    onemkl::cublas::nrm2,
    onemkl::cublas::nrm2,
    onemkl::cublas::nrm2,
    onemkl::cublas::nrm2,
    onemkl::cublas::rot,
    onemkl::cublas::rot,
    onemkl::cublas::rot,
    onemkl::cublas::rot,
    onemkl::cublas::rotg,
    onemkl::cublas::rotg,
    onemkl::cublas::rotg,
    onemkl::cublas::rotg,
    onemkl::cublas::rotm,
    onemkl::cublas::rotm,
    onemkl::cublas::rotmg,
    onemkl::cublas::rotmg,
    onemkl::cublas::scal,
    onemkl::cublas::scal,
    onemkl::cublas::scal,
    onemkl::cublas::scal,
    onemkl::cublas::scal,
    onemkl::cublas::scal,
    onemkl::cublas::sdsdot,
    onemkl::cublas::swap,
    onemkl::cublas::swap,
    onemkl::cublas::swap,
    onemkl::cublas::swap,
    onemkl::cublas::gbmv,
    onemkl::cublas::gbmv,
    onemkl::cublas::gbmv,
    onemkl::cublas::gbmv,
    onemkl::cublas::gemv,
    onemkl::cublas::gemv,
    onemkl::cublas::gemv,
    onemkl::cublas::gemv,
    onemkl::cublas::ger,
    onemkl::cublas::ger,
    onemkl::cublas::gerc,
    onemkl::cublas::gerc,
    onemkl::cublas::geru,
    onemkl::cublas::geru,
    onemkl::cublas::hbmv,
    onemkl::cublas::hbmv,
    onemkl::cublas::hemv,
    onemkl::cublas::hemv,
    onemkl::cublas::her,
    onemkl::cublas::her,
    onemkl::cublas::her2,
    onemkl::cublas::her2,
    onemkl::cublas::hpmv,
    onemkl::cublas::hpmv,
    onemkl::cublas::hpr,
    onemkl::cublas::hpr,
    onemkl::cublas::hpr2,
    onemkl::cublas::hpr2,
    onemkl::cublas::sbmv,
    onemkl::cublas::sbmv,
    onemkl::cublas::spmv,
    onemkl::cublas::spmv,
    onemkl::cublas::spr,
    onemkl::cublas::spr,
    onemkl::cublas::spr2,
    onemkl::cublas::spr2,
    onemkl::cublas::symv,
    onemkl::cublas::symv,
    onemkl::cublas::syr,
    onemkl::cublas::syr,
    onemkl::cublas::syr2,
    onemkl::cublas::syr2,
    onemkl::cublas::tbmv,
    onemkl::cublas::tbmv,
    onemkl::cublas::tbmv,
    onemkl::cublas::tbmv,
    onemkl::cublas::tbsv,
    onemkl::cublas::tbsv,
    onemkl::cublas::tbsv,
    onemkl::cublas::tbsv,
    onemkl::cublas::tpmv,
    onemkl::cublas::tpmv,
    onemkl::cublas::tpmv,
    onemkl::cublas::tpmv,
    onemkl::cublas::tpsv,
    onemkl::cublas::tpsv,
    onemkl::cublas::tpsv,
    onemkl::cublas::tpsv,
    onemkl::cublas::trmv,
    onemkl::cublas::trmv,
    onemkl::cublas::trmv,
    onemkl::cublas::trmv,
    onemkl::cublas::trsv,
    onemkl::cublas::trsv,
    onemkl::cublas::trsv,
    onemkl::cublas::trsv,
    onemkl::cublas::gemm,
    onemkl::cublas::gemm,
    onemkl::cublas::gemm,
    onemkl::cublas::gemm,
    onemkl::cublas::gemm,
    onemkl::cublas::hemm,
    onemkl::cublas::hemm,
    onemkl::cublas::herk,
    onemkl::cublas::herk,
    onemkl::cublas::her2k,
    onemkl::cublas::her2k,
    onemkl::cublas::symm,
    onemkl::cublas::symm,
    onemkl::cublas::symm,
    onemkl::cublas::symm,
    onemkl::cublas::syrk,
    onemkl::cublas::syrk,
    onemkl::cublas::syrk,
    onemkl::cublas::syrk,
    onemkl::cublas::syr2k,
    onemkl::cublas::syr2k,
    onemkl::cublas::syr2k,
    onemkl::cublas::syr2k,
    onemkl::cublas::trmm,
    onemkl::cublas::trmm,
    onemkl::cublas::trmm,
    onemkl::cublas::trmm,
    onemkl::cublas::trsm,
    onemkl::cublas::trsm,
    onemkl::cublas::trsm,
    onemkl::cublas::trsm,
    onemkl::cublas::gemm_batch,
    onemkl::cublas::gemm_batch,
    onemkl::cublas::gemm_batch,
    onemkl::cublas::gemm_batch,
    onemkl::cublas::trsm_batch,
    onemkl::cublas::trsm_batch,
    onemkl::cublas::trsm_batch,
    onemkl::cublas::trsm_batch,
    onemkl::cublas::gemmt,
    onemkl::cublas::gemmt,
    onemkl::cublas::gemmt,
    onemkl::cublas::gemmt,
    onemkl::cublas::gemm_ext,
    onemkl::cublas::gemm_ext,
    onemkl::cublas::gemm_ext,
    onemkl::cublas::gemm_ext,
    onemkl::cublas::gemm_ext,
    onemkl::cublas::gemm_ext,
    onemkl::cublas::gemm_ext,
    onemkl::cublas::asum,
    onemkl::cublas::asum,
    onemkl::cublas::asum,
    onemkl::cublas::asum,
    onemkl::cublas::axpy,
    onemkl::cublas::axpy,
    onemkl::cublas::axpy,
    onemkl::cublas::axpy,
    onemkl::cublas::copy,
    onemkl::cublas::copy,
    onemkl::cublas::copy,
    onemkl::cublas::copy,
    onemkl::cublas::dot,
    onemkl::cublas::dot,
    onemkl::cublas::dot,
    onemkl::cublas::dotc,
    onemkl::cublas::dotc,
    onemkl::cublas::dotu,
    onemkl::cublas::dotu,
    onemkl::cublas::iamin,
    onemkl::cublas::iamin,
    onemkl::cublas::iamin,
    onemkl::cublas::iamin,
    onemkl::cublas::iamax,
    onemkl::cublas::iamax,
    onemkl::cublas::iamax,
    onemkl::cublas::iamax,
    onemkl::cublas::nrm2,
    onemkl::cublas::nrm2,
    onemkl::cublas::nrm2,
    onemkl::cublas::nrm2,
    onemkl::cublas::rot,
    onemkl::cublas::rot,
    onemkl::cublas::rot,
    onemkl::cublas::rot,
    onemkl::cublas::rotg,
    onemkl::cublas::rotg,
    onemkl::cublas::rotg,
    onemkl::cublas::rotg,
    onemkl::cublas::rotm,
    onemkl::cublas::rotm,
    onemkl::cublas::rotmg,
    onemkl::cublas::rotmg,
    onemkl::cublas::scal,
    onemkl::cublas::scal,
    onemkl::cublas::scal,
    onemkl::cublas::scal,
    onemkl::cublas::scal,
    onemkl::cublas::scal,
    onemkl::cublas::sdsdot,
    onemkl::cublas::swap,
    onemkl::cublas::swap,
    onemkl::cublas::swap,
    onemkl::cublas::swap,
    onemkl::cublas::gbmv,
    onemkl::cublas::gbmv,
    onemkl::cublas::gbmv,
    onemkl::cublas::gbmv,
    onemkl::cublas::gemv,
    onemkl::cublas::gemv,
    onemkl::cublas::gemv,
    onemkl::cublas::gemv,
    onemkl::cublas::ger,
    onemkl::cublas::ger,
    onemkl::cublas::gerc,
    onemkl::cublas::gerc,
    onemkl::cublas::geru,
    onemkl::cublas::geru,
    onemkl::cublas::hbmv,
    onemkl::cublas::hbmv,
    onemkl::cublas::hemv,
    onemkl::cublas::hemv,
    onemkl::cublas::her,
    onemkl::cublas::her,
    onemkl::cublas::her2,
    onemkl::cublas::her2,
    onemkl::cublas::hpmv,
    onemkl::cublas::hpmv,
    onemkl::cublas::hpr,
    onemkl::cublas::hpr,
    onemkl::cublas::hpr2,
    onemkl::cublas::hpr2,
    onemkl::cublas::sbmv,
    onemkl::cublas::sbmv,
    onemkl::cublas::spmv,
    onemkl::cublas::spmv,
    onemkl::cublas::spr,
    onemkl::cublas::spr,
    onemkl::cublas::spr2,
    onemkl::cublas::spr2,
    onemkl::cublas::symv,
    onemkl::cublas::symv,
    onemkl::cublas::syr,
    onemkl::cublas::syr,
    onemkl::cublas::syr2,
    onemkl::cublas::syr2,
    onemkl::cublas::tbmv,
    onemkl::cublas::tbmv,
    onemkl::cublas::tbmv,
    onemkl::cublas::tbmv,
    onemkl::cublas::tbsv,
    onemkl::cublas::tbsv,
    onemkl::cublas::tbsv,
    onemkl::cublas::tbsv,
    onemkl::cublas::tpmv,
    onemkl::cublas::tpmv,
    onemkl::cublas::tpmv,
    onemkl::cublas::tpmv,
    onemkl::cublas::tpsv,
    onemkl::cublas::tpsv,
    onemkl::cublas::tpsv,
    onemkl::cublas::tpsv,
    onemkl::cublas::trmv,
    onemkl::cublas::trmv,
    onemkl::cublas::trmv,
    onemkl::cublas::trmv,
    onemkl::cublas::trsv,
    onemkl::cublas::trsv,
    onemkl::cublas::trsv,
    onemkl::cublas::trsv,
    onemkl::cublas::gemm,
    onemkl::cublas::gemm,
    onemkl::cublas::gemm,
    onemkl::cublas::gemm,
    onemkl::cublas::hemm,
    onemkl::cublas::hemm,
    onemkl::cublas::herk,
    onemkl::cublas::herk,
    onemkl::cublas::her2k,
    onemkl::cublas::her2k,
    onemkl::cublas::symm,
    onemkl::cublas::symm,
    onemkl::cublas::symm,
    onemkl::cublas::symm,
    onemkl::cublas::syrk,
    onemkl::cublas::syrk,
    onemkl::cublas::syrk,
    onemkl::cublas::syrk,
    onemkl::cublas::syr2k,
    onemkl::cublas::syr2k,
    onemkl::cublas::syr2k,
    onemkl::cublas::syr2k,
    onemkl::cublas::trmm,
    onemkl::cublas::trmm,
    onemkl::cublas::trmm,
    onemkl::cublas::trmm,
    onemkl::cublas::trsm,
    onemkl::cublas::trsm,
    onemkl::cublas::trsm,
    onemkl::cublas::trsm,
    onemkl::cublas::gemm_batch,
    onemkl::cublas::gemm_batch,
    onemkl::cublas::gemm_batch,
    onemkl::cublas::gemm_batch,
    onemkl::cublas::gemm_batch,
    onemkl::cublas::gemm_batch,
    onemkl::cublas::gemm_batch,
    onemkl::cublas::gemm_batch,
    onemkl::cublas::gemmt,
    onemkl::cublas::gemmt,
    onemkl::cublas::gemmt,
    onemkl::cublas::gemmt,
};
