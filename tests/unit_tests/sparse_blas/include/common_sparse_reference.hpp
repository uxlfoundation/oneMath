/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef _COMMON_SPARSE_REFERENCE_HPP__
#define _COMMON_SPARSE_REFERENCE_HPP__

#include <stdexcept>
#include <string>
#include <tuple>

#include "oneapi/math.hpp"

#include "test_common.hpp"

template <typename T>
inline T conjugate(T) {
    static_assert(false, "Unsupported type");
}
template <>
inline float conjugate(float t) {
    return t;
}
template <>
inline double conjugate(double t) {
    return t;
}
template <>
inline std::complex<float> conjugate(std::complex<float> t) {
    return std::conj(t);
}
template <>
inline std::complex<double> conjugate(std::complex<double> t) {
    return std::conj(t);
}

template <typename T>
inline T opVal(const T t, const bool isConj) {
    return (isConj ? conjugate(t) : t);
};

template <typename fpType, typename intType, typename accIntType, typename accFpType>
void do_csr_transpose(const oneapi::math::transpose opA, intType* ia_t, intType* ja_t, fpType* a_t,
                      intType a_nrows, intType a_ncols, intType indexing, accIntType& ia,
                      accIntType& ja, accFpType& a, const bool structOnlyFlag = false) {
    const bool isConj = (opA == oneapi::math::transpose::conjtrans);

    // initialize ia_t to zero
    for (intType i = 0; i < a_ncols + 1; ++i) {
        ia_t[i] = 0;
    }

    // fill ia_t with counts of columns
    for (intType i = 0; i < a_nrows; ++i) {
        const intType st = ia[i] - indexing;
        const intType en = ia[i + 1] - indexing;
        for (intType j = st; j < en; ++j) {
            const intType col = ja[j] - indexing;
            ia_t[col + 1]++;
        }
    }
    // prefix sum to get official ia_t counts
    ia_t[0] = indexing;
    for (intType i = 0; i < a_ncols; ++i) {
        ia_t[i + 1] += ia_t[i];
    }

    // second pass through data to fill transpose structure
    for (intType i = 0; i < a_nrows; ++i) {
        const intType st = ia[i] - indexing;
        const intType en = ia[i + 1] - indexing;
        for (intType j = st; j < en; ++j) {
            const intType col = ja[j] - indexing;
            const intType j_in_a_t = ia_t[col] - indexing;
            ia_t[col]++;
            ja_t[j_in_a_t] = i + indexing;
            if (!structOnlyFlag) {
                const fpType val = a[j];
                a_t[j_in_a_t] = opVal(val, isConj);
            }
        }
    }

    // adjust ia_t back to original state after filling structure
    for (intType i = a_ncols; i > 0; --i) {
        ia_t[i] = ia_t[i - 1];
    }
    ia_t[0] = indexing;
}

// Transpose the given sparse matrix if needed
template <typename fpType, typename intType>
auto sparse_transpose_if_needed(const intType* ia, const intType* ja, const fpType* a,
                                intType a_nrows, intType a_ncols, std::size_t nnz, intType indexing,
                                oneapi::math::transpose transpose_val) {
    std::vector<intType> iopa;
    std::vector<intType> jopa;
    std::vector<fpType> opa;
    if (transpose_val == oneapi::math::transpose::nontrans) {
        iopa.assign(ia, ia + a_nrows + 1);
        jopa.assign(ja, ja + nnz);
        opa.assign(a, a + nnz);
    }
    else if (transpose_val == oneapi::math::transpose::trans ||
             transpose_val == oneapi::math::transpose::conjtrans) {
        iopa.resize(static_cast<std::size_t>(a_ncols + 1));
        jopa.resize(nnz);
        opa.resize(nnz);
        do_csr_transpose(transpose_val, iopa.data(), jopa.data(), opa.data(), a_nrows, a_ncols,
                         indexing, ia, ja, a);
    }
    else {
        throw std::runtime_error("unsupported transpose_val=" +
                                 std::to_string(static_cast<char>(transpose_val)));
    }
    return std::make_tuple(iopa, jopa, opa);
}

/// Reduce the leading dimension to the minimum and transpose the matrix if needed
/// The outputted matrix always uses row major layout
template <typename fpType>
auto extract_dense_matrix(const fpType* x, std::size_t nrows, std::size_t ncols, std::size_t ld,
                          oneapi::math::transpose transpose_val,
                          oneapi::math::layout dense_matrix_layout) {
    const bool is_row_major = dense_matrix_layout == oneapi::math::layout::row_major;
    const bool is_transposed = transpose_val != oneapi::math::transpose::nontrans;
    const bool apply_conjugate = transpose_val == oneapi::math::transpose::conjtrans;
    const bool swap_ld = is_row_major != is_transposed;
    if (swap_ld && ncols > ld) {
        throw std::runtime_error("Expected ncols <= ld");
    }
    if (!swap_ld && nrows > ld) {
        throw std::runtime_error("Expected nrows <= ld");
    }

    // Copy with a default leading dimension and transpose if needed
    std::vector<fpType> opx(nrows * ncols);
    for (std::size_t i = 0; i < nrows; ++i) {
        for (std::size_t j = 0; j < ncols; ++j) {
            auto val = swap_ld ? x[i * ld + j] : x[j * ld + i];
            opx[i * ncols + j] = opVal(val, apply_conjugate);
        }
    }
    return opx;
}

/// Convert the sparse matrix in the given format to a dense matrix A in row major layout applied with A_view.
template <typename fpType, typename intType>
std::vector<fpType> sparse_to_dense(sparse_matrix_format_t format, const intType* ia,
                                    const intType* ja, const fpType* a, std::size_t a_nrows,
                                    std::size_t a_ncols, std::size_t nnz, intType indexing,
                                    oneapi::math::transpose transpose_val,
                                    oneapi::math::sparse::matrix_view A_view) {
    oneapi::math::sparse::matrix_descr type_view = A_view.type_view;
    oneapi::math::uplo uplo_val = A_view.uplo_view;
    const bool is_symmetric_or_hermitian_view =
        type_view == oneapi::math::sparse::matrix_descr::symmetric ||
        type_view == oneapi::math::sparse::matrix_descr::hermitian;
    const bool apply_conjugate = transpose_val == oneapi::math::transpose::conjtrans;
    std::vector<fpType> dense_a(a_nrows * a_ncols, fpType(0));

    auto write_to_dense_if_needed = [&](std::size_t a_idx, std::size_t row, std::size_t col) {
        if ((type_view == oneapi::math::sparse::matrix_descr::triangular ||
             is_symmetric_or_hermitian_view) &&
            ((uplo_val == oneapi::math::uplo::lower && col > row) ||
             (uplo_val == oneapi::math::uplo::upper && col < row))) {
            // Read only the upper or lower part of the sparse matrix
            return;
        }
        if (type_view == oneapi::math::sparse::matrix_descr::diagonal && col != row) {
            // Read only the diagonal
            return;
        }
        // Do not transpose symmetric matrices to simplify the propagation of the symmetric values
        std::size_t dense_a_idx =
            (!is_symmetric_or_hermitian_view && transpose_val != oneapi::math::transpose::nontrans)
                ? col * a_nrows + row
                : row * a_ncols + col;
        fpType val = opVal(a[a_idx], apply_conjugate);
        dense_a[dense_a_idx] = val;
    };

    if (format == sparse_matrix_format_t::CSR) {
        for (std::size_t row = 0; row < a_nrows; row++) {
            for (intType i = ia[row] - indexing; i < ia[row + 1] - indexing; i++) {
                std::size_t iu = static_cast<std::size_t>(i);
                std::size_t col = static_cast<std::size_t>(ja[iu] - indexing);
                write_to_dense_if_needed(iu, row, col);
            }
        }
    }
    else if (format == sparse_matrix_format_t::COO) {
        for (std::size_t i = 0; i < nnz; i++) {
            std::size_t row = static_cast<std::size_t>(ia[i] - indexing);
            std::size_t col = static_cast<std::size_t>(ja[i] - indexing);
            write_to_dense_if_needed(i, row, col);
        }
    }

    // Write unit diagonal
    if (A_view.diag_view == oneapi::math::diag::unit && a_nrows == a_ncols) {
        for (std::size_t i = 0; i < a_nrows; i++) {
            dense_a[i * a_nrows + i] = fpType(1);
        }
    }

    // Propagate the rest of the symmetric matrix
    if (is_symmetric_or_hermitian_view) {
        for (std::size_t i = 0; i < a_nrows; ++i) {
            for (std::size_t j = i + 1; j < a_ncols; ++j) {
                if (uplo_val == oneapi::math::uplo::lower) {
                    dense_a[i * a_ncols + j] = dense_a[j * a_nrows + i];
                }
                else {
                    dense_a[j * a_nrows + i] = dense_a[i * a_ncols + j];
                }
            }
        }
    }
    return dense_a;
}

#endif // _COMMON_SPARSE_REFERENCE_HPP__
