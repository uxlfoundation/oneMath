.. _onemath_blas_developer_reference:

BLAS
====

See the :ref:`BLAS specification <onemath_blas>` for the full API reference.

This page documents implementation specific or backend specific details of the
BLAS domain.

rocBLAS
-------

Currently known limitations:

- ``gemm_batch`` with ``int8`` inputs and a ``float`` output accumulates in an
  ``int32`` workspace, because rocBLAS has no native int8-to-float compute type.
  Every int8 magnitude is at most 128, so a reduction length ``k`` larger than
  ``INT32_MAX / (128 * 128)`` (that is, ``k > 131071``) would wrap the
  accumulator and is reported as ``unimplemented``. The cuBLAS backend
  accumulates this combination in ``float`` and has no such ceiling, so the
  same call succeeds on NVIDIA and throws on AMD once ``k`` exceeds that limit.
