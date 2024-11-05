.. _onemath_sparse_linear_algebra:

Sparse Linear Algebra
=====================

See the latest specification for the sparse domain `here
<https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemath/source/domains/spblas/spblas>`_.

This page documents implementation specific or backend specific details of the
sparse domain.

OneMKL Intel CPU and GPU backends
---------------------------------

Currently known limitations:

- All operations' algorithms except ``no_optimize_alg`` map to the default
  algorithm.
- The required external workspace size is always 0 bytes.
- ``oneapi::math::sparse::set_csr_data`` and
  ``oneapi::math::sparse::set_coo_data`` functions cannot be used on a handle
  that has already been used for an operation or its optimize function. Doing so
  will throw a ``oneapi::math::unimplemented`` exception.
- Using ``spsv`` with the ``oneapi::math::sparse::spsv_alg::no_optimize_alg`` and
  a sparse matrix that does not have the
  ``oneapi::math::sparse::matrix_property::sorted`` property will throw a
  ``oneapi::math::unimplemented`` exception.
- Using ``spmm`` on Intel GPU with a sparse matrix that is
  ``oneapi::math::transpose::conjtrans`` and has the
  ``oneapi::math::sparse::matrix_property::symmetric`` property will throw a
  ``oneapi::math::unimplemented`` exception.
- Using ``spmv`` with a sparse matrix that is
  ``oneapi::math::transpose::conjtrans`` with a ``type_view``
  ``matrix_descr::symmetric`` or ``matrix_descr::hermitian`` will throw a
  ``oneapi::math::unimplemented`` exception.
- Using ``spsv`` on Intel GPU with a sparse matrix that is
  ``oneapi::math::transpose::conjtrans`` and will throw a
  ``oneapi::math::unimplemented`` exception.
- Scalar parameters ``alpha`` and ``beta`` should be host pointers to prevent
  synchronizations and copies to the host.


cuSPARSE backend
----------------

Currently known limitations:

- The COO format requires the indices to be sorted by row. See the `cuSPARSE
  documentation
  <https://docs.nvidia.com/cuda/cusparse/index.html#coordinate-coo>`_. Sparse
  operations using matrices with the COO format without the property
  ``matrix_property::sorted_by_rows`` or ``matrix_property::sorted`` will throw
  a ``oneapi::math::unimplemented`` exception.
- Using ``spmm`` with the algorithm ``spmm_alg::csr_alg3`` and an ``opA`` other
  than ``transpose::nontrans`` or an ``opB`` ``transpose::conjtrans`` will throw
  a ``oneapi::math::unimplemented`` exception.
- Using ``spmm`` with the algorithm ``spmm_alg::csr_alg3``,
  ``opB=transpose::trans`` and real fp64 precision will throw a
  ``oneapi::math::unimplemented`` exception. This configuration can fail as of
  CUDA 12.6.2, see the related issue
  `here<https://forums.developer.nvidia.com/t/cusparse-spmm-sample-failing-with-misaligned-address/311022>`_.
- Using ``spmv`` with a ``type_view`` other than ``matrix_descr::general`` will
  throw a ``oneapi::math::unimplemented`` exception.
- Using ``spsv`` with the algorithm ``spsv_alg::no_optimize_alg`` may still
  perform some mandatory preprocessing.
- oneMath does not provide a way to use non-default algorithms without
  calling preprocess functions such as ``cusparseSpMM_preprocess`` or
  ``cusparseSpMV_preprocess``. Feel free to create an issue if this is needed.


Operation algorithms mapping
----------------------------

The following tables describe how a oneMath algorithm maps to the backend's
algorithms. Refer to the backend's documentation for a more detailed explanation
of the algorithms.

Backends with no equivalent algorithms will fallback to the backend's default
behavior.


spmm
^^^^

.. list-table::
   :header-rows: 1
   :widths: 10 30 45

   * - ``spmm_alg`` value
     - MKLCPU/MKLGPU
     - cuSPARSE
   * - ``default_alg``
     - none
     - ``CUSPARSE_SPMM_ALG_DEFAULT``
   * - ``no_optimize_alg``
     - none
     - ``CUSPARSE_SPMM_ALG_DEFAULT``
   * - ``coo_alg1``
     - none
     - ``CUSPARSE_SPMM_COO_ALG1``
   * - ``coo_alg2``
     - none
     - ``CUSPARSE_SPMM_COO_ALG2``
   * - ``coo_alg3``
     - none
     - ``CUSPARSE_SPMM_COO_ALG3``
   * - ``coo_alg4``
     - none
     - ``CUSPARSE_SPMM_COO_ALG4``
   * - ``csr_alg1``
     - none
     - ``CUSPARSE_SPMM_CSR_ALG1``
   * - ``csr_alg2``
     - none
     - ``CUSPARSE_SPMM_CSR_ALG2``
   * - ``csr_alg3``
     - none
     - ``CUSPARSE_SPMM_CSR_ALG3``


spmv
^^^^

.. list-table::
   :header-rows: 1
   :widths: 10 30 45

   * - ``spmv_alg`` value
     - MKLCPU/MKLGPU
     - cuSPARSE
   * - ``default_alg``
     - none
     - ``CUSPARSE_SPMV_ALG_DEFAULT``
   * - ``no_optimize_alg``
     - none
     - ``CUSPARSE_SPMV_ALG_DEFAULT``
   * - ``coo_alg1``
     - none
     - ``CUSPARSE_SPMV_COO_ALG1``
   * - ``coo_alg2``
     - none
     - ``CUSPARSE_SPMV_COO_ALG2``
   * - ``csr_alg1``
     - none
     - ``CUSPARSE_SPMV_CSR_ALG1``
   * - ``csr_alg2``
     - none
     - ``CUSPARSE_SPMV_CSR_ALG2``
   * - ``csr_alg3``
     - none
     - ``CUSPARSE_SPMV_ALG_DEFAULT``


spsv
^^^^

.. list-table::
   :header-rows: 1
   :widths: 10 30 45

   * - ``spsv_alg`` value
     - MKLCPU/MKLGPU
     - cuSPARSE
   * - ``default_alg``
     - none
     - ``CUSPARSE_SPSV_ALG_DEFAULT``
   * - ``no_optimize_alg``
     - none
     - ``CUSPARSE_SPSV_ALG_DEFAULT``
