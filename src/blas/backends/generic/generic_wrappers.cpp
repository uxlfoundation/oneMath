//
// generated file
//

#include "blas/function_table.hpp"

#include "oneapi/math/blas/detail/generic/onemath_blas_generic.hpp"

#define WRAPPER_VERSION 1

extern "C" ONEMATH_EXPORT blas_function_table_t onemath_blas_table = {
    WRAPPER_VERSION,
#define BACKEND generic
#define MAJOR   column_major
#include "../backend_wrappers.cxx"
#undef MAJOR
#define MAJOR row_major
#include "../backend_wrappers.cxx"
#undef MAJOR
#undef BACKEND
};
