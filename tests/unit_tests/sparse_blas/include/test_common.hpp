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

#ifndef _TEST_COMMON_HPP__
#define _TEST_COMMON_HPP__

#include <complex>
#include <iostream>
#include <memory>
#include <limits>
#include <vector>
#include <set>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "test_helper.hpp"

// Sparse BLAS domain needs to call more functions per test so we use this macro helper to select between runtime and compile dispatch for each function
#ifdef CALL_RT_API
#define CALL_RT_OR_CT(FUNC, QUEUE, ...) FUNC(QUEUE, __VA_ARGS__)
#else
#define CALL_RT_OR_CT(FUNC, QUEUE, ...) TEST_RUN_SPARSE_CT_SELECT(QUEUE, FUNC, __VA_ARGS__);
#endif

template <typename T>
struct complex_info {
    using real_type = T;
    static const bool is_complex = false;
};

template <typename T>
struct complex_info<std::complex<T>> {
    using real_type = T;
    static const bool is_complex = true;
};

enum sparse_matrix_format_t {
    CSR,
    COO,
};

inline std::set<oneapi::mkl::sparse::matrix_property> get_default_matrix_properties(
    sycl::queue queue, sparse_matrix_format_t format) {
    auto vendor_id = oneapi::mkl::get_device_id(queue);
    if (vendor_id == oneapi::mkl::device::nvidiagpu && format == sparse_matrix_format_t::COO) {
        return { oneapi::mkl::sparse::matrix_property::sorted_by_rows };
    }
    return {};
}

/// Return the combinations of matrix_properties to test other than the default
inline std::vector<std::set<oneapi::mkl::sparse::matrix_property>>
get_all_matrix_properties_combinations(sycl::queue queue, sparse_matrix_format_t format) {
    auto vendor_id = oneapi::mkl::get_device_id(queue);
    if (vendor_id == oneapi::mkl::device::nvidiagpu && format == sparse_matrix_format_t::COO) {
        // Ensure all the sets have the sorted or sorted_by_rows properties
        return { { oneapi::mkl::sparse::matrix_property::sorted },
                 { oneapi::mkl::sparse::matrix_property::sorted_by_rows,
                   oneapi::mkl::sparse::matrix_property::symmetric },
                 { oneapi::mkl::sparse::matrix_property::sorted,
                   oneapi::mkl::sparse::matrix_property::symmetric } };
    }

    std::vector<std::set<oneapi::mkl::sparse::matrix_property>> properties_combinations{
        { oneapi::mkl::sparse::matrix_property::sorted },
        { oneapi::mkl::sparse::matrix_property::symmetric },
        { oneapi::mkl::sparse::matrix_property::sorted,
          oneapi::mkl::sparse::matrix_property::symmetric }
    };
    if (format == sparse_matrix_format_t::COO) {
        properties_combinations.push_back({ oneapi::mkl::sparse::matrix_property::sorted_by_rows });
    }
    return properties_combinations;
}

void print_error_code(sycl::exception const& e);

// Catch asynchronous exceptions.
struct exception_handler_t {
    void operator()(sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const& e) {
                std::cout << "Caught asynchronous SYCL exception:\n" << e.what() << std::endl;
                print_error_code(e);
            }
        }
    }
};

struct UsmDeleter {
    sycl::queue q;
    UsmDeleter(sycl::queue _q) : q(_q) {}
    void operator()(void* ptr) {
        sycl::free(ptr, q);
    }
};

// Use a unique_ptr to automatically free device memory on unique_ptr destruction.
template <class T>
auto malloc_device_uptr(sycl::queue q, std::size_t num_elts) {
    return std::unique_ptr<T, UsmDeleter>(sycl::malloc_device<T>(num_elts, q), UsmDeleter(q));
}

// SYCL buffer creation helper.
template <typename vec>
sycl::buffer<typename vec::value_type, 1> make_buffer(const vec& v) {
    sycl::buffer<typename vec::value_type, 1> buf(v.data(), sycl::range<1>(v.size()));
    return buf;
}

template <typename T>
void copy_host_to_buffer(sycl::queue queue, const std::vector<T>& src, sycl::buffer<T, 1> dst) {
    queue.submit([&](sycl::handler& cgh) {
        auto dst_acc = dst.template get_access<sycl::access::mode::discard_write>(
            cgh, sycl::range<1>(src.size()));
        cgh.copy(src.data(), dst_acc);
    });
}

template <typename OutT, typename XT, typename YT>
std::pair<OutT, OutT> swap_if_cond(bool swap, XT x, YT y) {
    if (swap) {
        return { static_cast<OutT>(y), static_cast<OutT>(x) };
    }
    else {
        return { static_cast<OutT>(x), static_cast<OutT>(y) };
    }
}

template <typename T>
auto swap_if_cond(bool swap, T x, T y) {
    return swap_if_cond<T, T, T>(swap, x, y);
}

template <typename OutT, typename XT, typename YT>
auto swap_if_transposed(oneapi::mkl::transpose op, XT x, YT y) {
    return swap_if_cond<OutT, XT, YT>(op != oneapi::mkl::transpose::nontrans, x, y);
}

template <typename T>
auto swap_if_transposed(oneapi::mkl::transpose op, T x, T y) {
    return swap_if_transposed<T, T, T>(op, x, y);
}

template <typename fpType>
struct set_fp_value {
    inline fpType operator()(fpType real, fpType /*imag*/) {
        return real;
    }
};

template <typename scalarType>
struct set_fp_value<std::complex<scalarType>> {
    inline auto operator()(scalarType real, scalarType imag) {
        return std::complex<scalarType>(real, imag);
    }
};

template <typename fpType>
struct rand_scalar {
    inline fpType operator()(double min, double max) {
        return (fpType(std::rand()) / fpType(RAND_MAX)) * fpType(max - min) + fpType(min);
    }
};

template <typename fpType>
struct rand_scalar<std::complex<fpType>> {
    inline std::complex<fpType> operator()(double min, double max) {
        rand_scalar<fpType> rand;
        return std::complex<fpType>(rand(min, max), rand(min, max));
    }
};

template <typename fpType>
void rand_vector(std::vector<fpType>& v, std::size_t n) {
    using fpRealType = typename complex_info<fpType>::real_type;
    v.resize(n);
    rand_scalar<fpType> rand;
    for (std::size_t i = 0; i < n; i++) {
        v[i] = rand(fpRealType(-0.5), fpRealType(0.5));
    }
}

template <typename fpType>
void rand_matrix(std::vector<fpType>& m, oneapi::mkl::layout layout_val, std::size_t nrows,
                 std::size_t ncols, std::size_t ld,
                 oneapi::mkl::transpose transpose_val = oneapi::mkl::transpose::nontrans) {
    using fpRealType = typename complex_info<fpType>::real_type;
    auto [op_nrows, op_cols] = swap_if_transposed(transpose_val, nrows, ncols);
    auto [outer_size, inner_size] =
        swap_if_cond(layout_val == oneapi::mkl::layout::row_major, op_cols, op_nrows);
    if (inner_size > ld) {
        throw std::runtime_error("Expected inner_size <= ld");
    }
    m.resize(outer_size * ld);
    rand_scalar<fpType> rand;
    for (std::size_t i = 0; i < outer_size; ++i) {
        std::size_t j = 0;
        for (; j < inner_size; ++j) {
            m[i * ld + j] = rand(fpRealType(-0.5), fpRealType(0.5));
        }
        for (; j < ld; ++j) {
            m[i * ld + j] = set_fp_value<fpType>()(-1.f, 0.f);
        }
    }
}

/// Generate random value in the range [-0.5, 0.5]
/// The amplitude is guaranteed to be >= 0.1 if is_diag is true
template <typename fpType>
fpType generate_data(bool is_diag) {
    rand_scalar<fpType> rand_data;
    if (is_diag) {
        // Guarantee a large amplitude
        fpType sign = (std::rand() % 2) * 2 - 1;
        return rand_data(10, 20) * sign;
    }
    return rand_data(-0.5, 0.5);
}

/// Populate the 3 arrays of a random sparse matrix in CSR representation (ia, ja, values)
/// with the given density in range [0, 1] and values in range [-0.5, 0.5].
/// ja is sorted.
/// require_diagonal means all diagonal entries guaranteed to be nonzero.
template <typename fpType, typename intType>
intType generate_random_csr_matrix(const intType nrows, const intType ncols,
                                   const double density_val, intType indexing,
                                   std::vector<intType>& ia, std::vector<intType>& ja,
                                   std::vector<fpType>& a, bool is_symmetric,
                                   bool require_diagonal = false) {
    intType nnz = 0;
    rand_scalar<double> rand_density;

    ia.push_back(indexing); // starting index of row0.
    for (intType i = 0; i < nrows; i++) {
        if (is_symmetric) {
            // Fill the lower triangular part based on the previously filled upper triangle
            // This ensures that the ja indices are always sorted
            for (intType j = 0; j < i; ++j) {
                // Check if element at row j and column i has been added, assuming ja is sorted
                intType row_offset_j = ia[static_cast<std::size_t>(j)];
                intType num_elts_row_j = ia.at(static_cast<std::size_t>(j) + 1) - row_offset_j;
                intType ja_idx = 0;
                while (ja_idx < num_elts_row_j &&
                       ja[static_cast<std::size_t>(row_offset_j + ja_idx)] < i) {
                    ++ja_idx;
                }
                auto symmetric_idx = static_cast<std::size_t>(row_offset_j + ja_idx);
                if (ja_idx < num_elts_row_j && ja[symmetric_idx] == i) {
                    a.push_back(a[symmetric_idx]);
                    ja.push_back(j + indexing);
                    nnz++;
                }
            }
        }
        // Loop through the upper triangular to fill a symmetric matrix
        const intType j_start = is_symmetric ? i : 0;
        for (intType j = j_start; j < ncols; j++) {
            const bool is_diag = require_diagonal && i == j;
            const bool force_last_nnz = nnz == 0 && i == nrows - 1 && j == ncols - 1;
            if (force_last_nnz || is_diag || (rand_density(0.0, 1.0) <= density_val)) {
                a.push_back(generate_data<fpType>(is_diag));
                ja.push_back(j + indexing);
                nnz++;
            }
        }
        ia.push_back(nnz + indexing); // ending index of row_i
    }
    return nnz;
}

/// Populate the 3 arrays of a random sparse matrix in COO representation (ia, ja, values)
/// with the given density in range [0, 1] and values in range [-0.5, 0.5].
/// Indices are sorted by row (ia) then by column (ja).
/// require_diagonal means all diagonal entries guaranteed to be nonzero.
template <typename fpType, typename intType>
intType generate_random_coo_matrix(const intType nrows, const intType ncols,
                                   const double density_val, intType indexing,
                                   std::vector<intType>& ia, std::vector<intType>& ja,
                                   std::vector<fpType>& a, bool is_symmetric,
                                   bool require_diagonal = false) {
    rand_scalar<double> rand_density;

    for (intType i = 0; i < nrows; i++) {
        if (is_symmetric) {
            // Fill the lower triangular part based on the previously filled upper triangle
            // This ensures that the ja indices are always sorted
            for (intType j = 0; j < i; ++j) {
                // Check if element at row j and column i has been added, assuming ia and ja are sorted
                std::size_t idx = 0;
                while (idx < ia.size() && ia[idx] - indexing <= j && ja[idx] - indexing < i) {
                    ++idx;
                }
                if (idx < ia.size() && ia[idx] - indexing == j && ja[idx] - indexing == i) {
                    a.push_back(a[idx]);
                    ia.push_back(i + indexing);
                    ja.push_back(j + indexing);
                }
            }
        }
        // Loop through the upper triangular to fill a symmetric matrix
        const intType j_start = is_symmetric ? i : 0;
        for (intType j = j_start; j < ncols; j++) {
            const bool is_diag = require_diagonal && i == j;
            const bool force_last_nnz = a.size() == 0 && i == nrows - 1 && j == ncols - 1;
            if (force_last_nnz || is_diag || (rand_density(0.0, 1.0) <= density_val)) {
                a.push_back(generate_data<fpType>(is_diag));
                ia.push_back(i + indexing);
                ja.push_back(j + indexing);
            }
        }
    }
    return static_cast<intType>(a.size());
}

// Populate the 3 arrays of a random sparse matrix in CSR or COO representation
// with the given density in range [0, 1] and values in range [-0.5, 0.5].
// require_diagonal means all diagonal entries guaranteed to be nonzero
template <typename fpType, typename intType>
intType generate_random_matrix(sparse_matrix_format_t format, const intType nrows,
                               const intType ncols, const double density_val, intType indexing,
                               std::vector<intType>& ia, std::vector<intType>& ja,
                               std::vector<fpType>& a, bool is_symmetric,
                               bool require_diagonal = false) {
    ia.clear();
    ja.clear();
    a.clear();
    if (format == sparse_matrix_format_t::CSR) {
        return generate_random_csr_matrix(nrows, ncols, density_val, indexing, ia, ja, a,
                                          is_symmetric, require_diagonal);
    }
    else if (format == sparse_matrix_format_t::COO) {
        return generate_random_coo_matrix(nrows, ncols, density_val, indexing, ia, ja, a,
                                          is_symmetric, require_diagonal);
    }
    throw std::runtime_error("Unsupported sparse format");
}

/// Shuffle the 3arrays CSR or COO representation (ia, ja, values)
/// of any sparse matrix.
/// In CSR format, the elements within a row are shuffled without changing ia.
/// In COO format, all the elements are shuffled.
template <typename fpType, typename intType>
void shuffle_sparse_matrix_if_needed(
    sparse_matrix_format_t format,
    const std::set<oneapi::mkl::sparse::matrix_property>& matrix_properties, intType indexing,
    intType* ia, intType* ja, fpType* a, intType nnz, std::size_t nrows) {
    const bool is_sorted = matrix_properties.find(oneapi::mkl::sparse::matrix_property::sorted) !=
                           matrix_properties.cend();
    if (is_sorted) {
        return;
    }
    const bool is_sorted_by_rows =
        matrix_properties.find(oneapi::mkl::sparse::matrix_property::sorted_by_rows) !=
        matrix_properties.cend();
    if (format == sparse_matrix_format_t::CSR) {
        for (std::size_t i = 0; i < nrows; ++i) {
            intType nnz_row = ia[i + 1] - ia[i];
            for (intType j = ia[i] - indexing; j < ia[i + 1] - indexing; ++j) {
                intType q = ia[i] - indexing + std::rand() % nnz_row;
                // Swap elements j and q
                std::swap(ja[q], ja[j]);
                std::swap(a[q], a[j]);
            }
        }
        // sorted_by_rows does not impact CSR
    }
    else if (format == sparse_matrix_format_t::COO) {
        if (is_sorted_by_rows) {
            std::size_t linear_idx = 0;
            for (std::size_t i = 0; i < nrows; ++i) {
                // Count the number of non-zero elements for the given row
                std::size_t nnz_row = 1;
                while (linear_idx + nnz_row < static_cast<std::size_t>(nnz) &&
                       ia[linear_idx] == ia[linear_idx + nnz_row]) {
                    ++nnz_row;
                }
                for (std::size_t j = 0; j < nnz_row; ++j) {
                    // Swap elements within the same row
                    std::size_t q = linear_idx + (static_cast<std::size_t>(std::rand()) % nnz_row);
                    // Swap elements j and q
                    std::swap(ja[q], ja[linear_idx + j]);
                    std::swap(a[q], a[linear_idx + j]);
                }
                linear_idx += nnz_row;
            }
        }
        else {
            for (std::size_t i = 0; i < static_cast<std::size_t>(nnz); ++i) {
                intType q = std::rand() % nnz;
                // Swap elements i and q
                std::swap(ia[q], ia[i]);
                std::swap(ja[q], ja[i]);
                std::swap(a[q], a[i]);
            }
        }
    }
    else {
        throw oneapi::mkl::exception("sparse_blas", "shuffle_sparse_matrix_if_needed",
                                     "Internal error: unsupported format");
    }
}

/// Initialize a sparse matrix specified by the given format
template <typename ContainerValueT, typename ContainerIndexT>
void init_sparse_matrix(sycl::queue& queue, sparse_matrix_format_t format,
                        oneapi::mkl::sparse::matrix_handle_t* p_smhandle, std::int64_t num_rows,
                        std::int64_t num_cols, std::int64_t nnz, oneapi::mkl::index_base index,
                        ContainerIndexT rows, ContainerIndexT cols, ContainerValueT vals) {
    if (format == sparse_matrix_format_t::CSR) {
        CALL_RT_OR_CT(oneapi::mkl::sparse::init_csr_matrix, queue, p_smhandle, num_rows, num_cols,
                      nnz, index, rows, cols, vals);
    }
    else if (format == sparse_matrix_format_t::COO) {
        CALL_RT_OR_CT(oneapi::mkl::sparse::init_coo_matrix, queue, p_smhandle, num_rows, num_cols,
                      nnz, index, rows, cols, vals);
    }
    else {
        throw oneapi::mkl::exception("sparse_blas", "init_sparse_matrix",
                                     "Internal error: unsupported format");
    }
}

/// Reset the data of a sparse matrix specified by the given format
template <typename ContainerValueT, typename ContainerIndexT>
void set_matrix_data(sycl::queue& queue, sparse_matrix_format_t format,
                     oneapi::mkl::sparse::matrix_handle_t smhandle, std::int64_t num_rows,
                     std::int64_t num_cols, std::int64_t nnz, oneapi::mkl::index_base index,
                     ContainerIndexT rows, ContainerIndexT cols, ContainerValueT vals) {
    if (format == sparse_matrix_format_t::CSR) {
        CALL_RT_OR_CT(oneapi::mkl::sparse::set_csr_matrix_data, queue, smhandle, num_rows, num_cols,
                      nnz, index, rows, cols, vals);
    }
    else if (format == sparse_matrix_format_t::COO) {
        CALL_RT_OR_CT(oneapi::mkl::sparse::set_coo_matrix_data, queue, smhandle, num_rows, num_cols,
                      nnz, index, rows, cols, vals);
    }
    else {
        throw oneapi::mkl::exception("sparse_blas", "set_matrix_data",
                                     "Internal error: unsupported format");
    }
}

template <typename... HandlesT>
inline void free_handles(sycl::queue& queue, const std::vector<sycl::event> dependencies,
                         HandlesT&&... handles) {
    // Fold expression so that handles expands to each value one after the other.
    (
        [&] {
            if (!handles) {
                return;
            }
            sycl::event event;
            if constexpr (std::is_same_v<decltype(handles),
                                         oneapi::mkl::sparse::dense_vector_handle_t>) {
                CALL_RT_OR_CT(event = oneapi::mkl::sparse::release_dense_vector, queue, handles,
                              dependencies);
            }
            else if constexpr (std::is_same_v<decltype(handles),
                                              oneapi::mkl::sparse::dense_matrix_handle_t>) {
                CALL_RT_OR_CT(event = oneapi::mkl::sparse::release_dense_matrix, queue, handles,
                              dependencies);
            }
            else if constexpr (std::is_same_v<decltype(handles),
                                              oneapi::mkl::sparse::matrix_handle_t>) {
                CALL_RT_OR_CT(event = oneapi::mkl::sparse::release_sparse_matrix, queue, handles,
                              dependencies);
            }
            event.wait();
        }(),
        ...);
}

template <typename... HandlesT>
inline void free_handles(sycl::queue& queue, HandlesT&&... handles) {
    free_handles(queue, {}, handles...);
}

template <typename... HandlesT>
inline void wait_and_free_handles(sycl::queue& queue, HandlesT&&... handles) {
    queue.wait();
    free_handles(queue, handles...);
}

inline bool require_square_matrix(
    oneapi::mkl::sparse::matrix_view A_view,
    const std::set<oneapi::mkl::sparse::matrix_property>& matrix_properties) {
    const bool is_symmetric =
        matrix_properties.find(oneapi::mkl::sparse::matrix_property::symmetric) !=
        matrix_properties.cend();
    return A_view.type_view != oneapi::mkl::sparse::matrix_descr::general || is_symmetric;
}

template <typename fpType>
bool check_equal(fpType x, fpType x_ref, double abs_error_margin, double rel_error_margin,
                 std::ostream& out) {
    using fpRealType = typename complex_info<fpType>::real_type;
    static_assert(std::is_floating_point_v<fpRealType>,
                  "Expected floating-point real or complex type.");

    const fpRealType epsilon = std::numeric_limits<fpRealType>::epsilon();
    const auto abs_bound = static_cast<fpRealType>(abs_error_margin) * epsilon;
    const auto rel_bound = static_cast<fpRealType>(rel_error_margin) * epsilon;

    const auto aerr = std::abs(x - x_ref);
    const auto rerr = aerr / std::abs(x_ref);
    const bool valid = (rerr <= rel_bound) || (aerr <= abs_bound);
    if (!valid) {
        out << "Mismatching results: actual = " << x << " vs. reference = " << x_ref << "\n";
        out << " relative error = " << rerr << " absolute error = " << aerr
            << " relative bound = " << rel_bound << " absolute bound = " << abs_bound << "\n";
    }
    return valid;
}

template <typename vecType1, typename vecType2>
bool check_equal_vector(const vecType1& v, const vecType2& v_ref, double abs_error_factor = 10.0,
                        double rel_error_factor = 200.0, std::ostream& out = std::cout) {
    using T = typename vecType2::value_type;
    std::size_t n = v.size();
    if (n != v_ref.size()) {
        out << "Mismatching size got " << n << " expected " << v_ref.size() << "\n";
        return false;
    }
    if (n == 0) {
        return true;
    }

    auto max_norm_ref =
        *std::max_element(std::begin(v_ref), std::end(v_ref),
                          [](const T& a, const T& b) { return std::abs(a) < std::abs(b); });
    // Heuristic for the average-case error margins
    double abs_error_margin =
        abs_error_factor * std::abs(max_norm_ref) * std::log2(static_cast<double>(n));
    double rel_error_margin = rel_error_factor * std::log2(static_cast<double>(n));

    constexpr int max_print = 20;
    int count = 0;
    bool valid = true;

    for (std::size_t i = 0; i < n; ++i) {
        // Allow to convert the unsigned index `i` to a signed one to keep this function generic and allow for `v` and `v_ref` to be a vector, a pointer or a random access iterator.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
        auto res = v[i];
        auto ref = v_ref[i];
#pragma clang diagnostic pop
        if (!check_equal(res, ref, abs_error_margin, rel_error_margin, out)) {
            out << " at index i =" << i << "\n";
            valid = false;
            ++count;
            if (count > max_print) {
                return valid;
            }
        }
    }

    return valid;
}

#endif // _TEST_COMMON_HPP__
