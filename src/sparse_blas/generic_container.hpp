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

#ifndef _ONEMATH_SRC_SPARSE_BLAS_GENERIC_CONTAINER_HPP_
#define _ONEMATH_SRC_SPARSE_BLAS_GENERIC_CONTAINER_HPP_

#include <memory>
#include <string>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math/exceptions.hpp"
#include "oneapi/math/sparse_blas/types.hpp"
#include "enum_data_types.hpp"

namespace oneapi::math::sparse::detail {

/// Represent a non-templated container for USM or buffer.
struct generic_container {
    // USM pointer, nullptr if the provided data is a buffer.
    void* usm_ptr;

    // Buffer pointer, nullptr if the provided data is a USM pointer. The buffer
    // is needed to properly handle the dependencies when the handle is used.
    // Use a void* type for the buffer to avoid using template arguments in
    // every function using data handles. Using `sycl::buffer::reinterpret` does
    // not solve the issue as the returned buffer needs the type of the original
    // buffer for the aligned_allocator.
    std::shared_ptr<void> buffer_ptr;

    // Underlying USM or buffer data type
    data_type data_type;

    generic_container() : usm_ptr(nullptr), buffer_ptr(), data_type(data_type::none) {}

    template <typename T>
    generic_container(T* ptr) : usm_ptr(ptr),
                                buffer_ptr(),
                                data_type(get_data_type<T>()) {}

    template <typename T>
    generic_container(const sycl::buffer<T, 1> buffer)
            : usm_ptr(nullptr),
              buffer_ptr(std::make_shared<sycl::buffer<T, 1>>(buffer)),
              data_type(get_data_type<T>()) {}

    bool use_buffer() const {
        return static_cast<bool>(buffer_ptr);
    }

    template <typename T>
    void set_usm_ptr(T* ptr) {
        usm_ptr = ptr;
        data_type = get_data_type<T>();
    }

    template <typename T>
    void set_buffer_untyped(const sycl::buffer<T, 1> buffer) {
        buffer_ptr = std::make_shared<sycl::buffer<T, 1>>(buffer);
        // Do not set data_type if T is meant as a generic byte type.
    }

    template <typename T>
    void set_buffer(const sycl::buffer<T, 1> buffer) {
        set_buffer_untyped(buffer);
        data_type = get_data_type<T>();
    }

    template <typename T>
    T* get_usm_ptr() {
        return static_cast<T*>(usm_ptr);
    }

    template <typename T>
    auto& get_buffer() {
        return *reinterpret_cast<sycl::buffer<T, 1>*>(buffer_ptr.get());
    }
};

/// Common type for dense vector and matrix handles
template <typename BackendHandleT>
struct generic_dense_handle {
    BackendHandleT backend_handle;

    generic_container value_container;

    template <typename T>
    generic_dense_handle(BackendHandleT backend_handle, T* value_ptr)
            : backend_handle(backend_handle),
              value_container(generic_container(value_ptr)) {}

    template <typename T>
    generic_dense_handle(BackendHandleT backend_handle, const sycl::buffer<T, 1> value_buffer)
            : backend_handle(backend_handle),
              value_container(value_buffer) {}

    bool all_use_buffer() const {
        return value_container.use_buffer();
    }

    data_type get_value_type() const {
        return value_container.data_type;
    }

    data_type get_int_type() const {
        return data_type::none;
    }

    template <typename T>
    T* get_usm_ptr() {
        return value_container.get_usm_ptr<T>();
    }

    template <typename T>
    auto& get_buffer() {
        return value_container.get_buffer<T>();
    }

    template <typename T>
    void set_usm_ptr(T* ptr) {
        value_container.set_usm_ptr(ptr);
    }

    template <typename T>
    void set_buffer(const sycl::buffer<T, 1> buffer) {
        value_container.set_buffer(buffer);
    }
};

/// Generic dense_vector_handle used by all backends
template <typename BackendHandleT>
struct generic_dense_vector_handle : public detail::generic_dense_handle<BackendHandleT> {
    std::int64_t size;

    template <typename T>
    generic_dense_vector_handle(BackendHandleT backend_handle, T* value_ptr, std::int64_t size)
            : generic_dense_handle<BackendHandleT>(backend_handle, value_ptr),
              size(size) {}

    template <typename T>
    generic_dense_vector_handle(BackendHandleT backend_handle,
                                const sycl::buffer<T, 1> value_buffer, std::int64_t size)
            : generic_dense_handle<BackendHandleT>(backend_handle, value_buffer),
              size(size) {
        if (value_buffer.size() < static_cast<std::size_t>(size)) {
            throw oneapi::math::invalid_argument(
                "sparse_blas", "init_dense_vector",
                "Buffer size too small, expected at least " + std::to_string(size) + " but got " +
                    std::to_string(value_buffer.size()) + " elements.");
        }
    }
};

/// Generic dense_matrix_handle used by all backends
template <typename BackendHandleT>
struct generic_dense_matrix_handle : public detail::generic_dense_handle<BackendHandleT> {
    std::int64_t num_rows;
    std::int64_t num_cols;
    std::int64_t ld;
    oneapi::math::layout dense_layout;

    template <typename T>
    generic_dense_matrix_handle(BackendHandleT backend_handle, T* value_ptr, std::int64_t num_rows,
                                std::int64_t num_cols, std::int64_t ld, layout dense_layout)
            : generic_dense_handle<BackendHandleT>(backend_handle, value_ptr),
              num_rows(num_rows),
              num_cols(num_cols),
              ld(ld),
              dense_layout(dense_layout) {}

    template <typename T>
    generic_dense_matrix_handle(BackendHandleT backend_handle,
                                const sycl::buffer<T, 1> value_buffer, std::int64_t num_rows,
                                std::int64_t num_cols, std::int64_t ld, layout dense_layout)
            : generic_dense_handle<BackendHandleT>(backend_handle, value_buffer),
              num_rows(num_rows),
              num_cols(num_cols),
              ld(ld),
              dense_layout(dense_layout) {
        std::size_t minimum_size = static_cast<std::size_t>(
            (dense_layout == oneapi::math::layout::row_major ? num_rows : num_cols) * ld);
        if (value_buffer.size() < minimum_size) {
            throw oneapi::math::invalid_argument(
                "sparse_blas", "init_dense_matrix",
                "Buffer size too small, expected at least " + std::to_string(minimum_size) +
                    " but got " + std::to_string(value_buffer.size()) + " elements.");
        }
    }
};

enum class sparse_format { CSR, COO };

/// Generic sparse_matrix_handle used by all backends
template <typename BackendHandleT>
struct generic_sparse_handle {
    BackendHandleT backend_handle;

    generic_container row_container;
    generic_container col_container;
    generic_container value_container;

    sparse_format format;
    std::int64_t num_rows;
    std::int64_t num_cols;
    std::int64_t nnz;
    index_base index;
    std::int32_t properties_mask;
    bool can_be_reset;

    template <typename fpType, typename intType>
    generic_sparse_handle(BackendHandleT backend_handle, intType* row_ptr, intType* col_ptr,
                          fpType* value_ptr, sparse_format format, std::int64_t num_rows,
                          std::int64_t num_cols, std::int64_t nnz, index_base index)
            : backend_handle(backend_handle),
              row_container(generic_container(row_ptr)),
              col_container(generic_container(col_ptr)),
              value_container(generic_container(value_ptr)),
              format(format),
              num_rows(num_rows),
              num_cols(num_cols),
              nnz(nnz),
              index(index),
              properties_mask(0),
              can_be_reset(true) {}

    template <typename fpType, typename intType>
    generic_sparse_handle(BackendHandleT backend_handle, const sycl::buffer<intType, 1> row_buffer,
                          const sycl::buffer<intType, 1> col_buffer,
                          const sycl::buffer<fpType, 1> value_buffer, sparse_format format,
                          std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,
                          index_base index)
            : backend_handle(backend_handle),
              row_container(row_buffer),
              col_container(col_buffer),
              value_container(value_buffer),
              format(format),
              num_rows(num_rows),
              num_cols(num_cols),
              nnz(nnz),
              index(index),
              properties_mask(0),
              can_be_reset(true) {}

    bool all_use_buffer() const {
        return value_container.use_buffer() && row_container.use_buffer() &&
               col_container.use_buffer();
    }

    data_type get_value_type() const {
        return value_container.data_type;
    }

    data_type get_int_type() const {
        return row_container.data_type;
    }

    void set_matrix_property(matrix_property property) {
        properties_mask |= matrix_property_to_mask(property);
    }

    bool has_matrix_property(matrix_property property) {
        return properties_mask & matrix_property_to_mask(property);
    }

private:
    std::int32_t matrix_property_to_mask(matrix_property property) {
        switch (property) {
            case matrix_property::symmetric: return 1 << 0;
            case matrix_property::sorted: return 1 << 1;
            case matrix_property::sorted_by_rows: return 1 << 2;
            default:
                throw oneapi::math::invalid_argument(
                    "sparse_blas", "set_matrix_property",
                    "Unsupported matrix property " + std::to_string(static_cast<int>(property)));
        }
    }
};

inline void throw_incompatible_container(const std::string& function_name) {
    throw oneapi::math::invalid_argument(
        "sparse_blas", function_name,
        "Incompatible container types. All inputs and outputs must use the same container: buffer or USM");
}

/**
 * Check that all internal containers use the same container.
*/
template <typename ContainerT, typename... Ts>
void check_all_containers_use_buffers(const std::string& function_name,
                                      ContainerT first_internal_container,
                                      Ts... internal_containers) {
    bool first_use_buffer = first_internal_container->all_use_buffer();
    for (const auto internal_container : { internal_containers... }) {
        if (internal_container->all_use_buffer() != first_use_buffer) {
            throw_incompatible_container(function_name);
        }
    }
}

/**
 * Check that all internal containers use the same container type, data type and integer type.
 * The integer type can be 'none' if the internal container does not store any integer (i.e. for dense handles).
 * The first internal container is used to determine what container and types the other internal containers should use.
*/
template <typename ContainerT, typename... Ts>
void check_all_containers_compatible(const std::string& function_name,
                                     ContainerT first_internal_container,
                                     Ts... internal_containers) {
    check_all_containers_use_buffers(function_name, first_internal_container,
                                     internal_containers...);
    data_type first_value_type = first_internal_container->get_value_type();
    data_type first_int_type = first_internal_container->get_int_type();
    for (const auto internal_container : { internal_containers... }) {
        const data_type other_value_type = internal_container->get_value_type();
        if (other_value_type != first_value_type) {
            throw oneapi::math::invalid_argument(
                "sparse_blas", function_name,
                "Incompatible data types expected " + data_type_to_str(first_value_type) +
                    " but got " + data_type_to_str(other_value_type));
        }
        const data_type other_int_type = internal_container->get_int_type();
        if (other_int_type != data_type::none && other_int_type != first_int_type) {
            throw oneapi::math::invalid_argument(
                "sparse_blas", function_name,
                "Incompatible integer types expected " + data_type_to_str(first_int_type) +
                    " but got " + data_type_to_str(other_int_type));
        }
    }
}

template <typename fpType, typename InternalHandleT>
void check_can_reset_value_handle(const std::string& function_name,
                                  InternalHandleT* internal_handle, bool expect_buffer) {
    if (internal_handle->get_value_type() != detail::get_data_type<fpType>()) {
        throw oneapi::math::invalid_argument(
            "sparse_blas", function_name,
            "Incompatible data types expected " +
                data_type_to_str(internal_handle->get_value_type()) + " but got " +
                data_type_to_str(detail::get_data_type<fpType>()));
    }
    if (internal_handle->all_use_buffer() != expect_buffer) {
        throw oneapi::math::invalid_argument(
            "sparse_blas", function_name, "Cannot change the container type between buffer or USM");
    }
}

template <typename fpType, typename intType, typename InternalHandleT>
void check_can_reset_sparse_handle(const std::string& function_name,
                                   InternalHandleT* internal_smhandle, bool expect_buffer) {
    check_can_reset_value_handle<fpType>(function_name, internal_smhandle, expect_buffer);
    if (internal_smhandle->get_int_type() != detail::get_data_type<intType>()) {
        throw oneapi::math::invalid_argument(
            "sparse_blas", function_name,
            "Incompatible data types expected " +
                data_type_to_str(internal_smhandle->get_int_type()) + " but got " +
                data_type_to_str(detail::get_data_type<intType>()));
    }
    if (!internal_smhandle->can_be_reset) {
        throw oneapi::math::unimplemented(
            "sparse_blas", function_name,
            "The backend does not support reseting the matrix handle's data after it was used in a computation.");
    }
}

} // namespace oneapi::math::sparse::detail

#endif // _ONEMATH_SRC_SPARSE_BLAS_GENERIC_CONTAINER_HPP_
