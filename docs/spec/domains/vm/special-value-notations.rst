.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_vm_special_value_notations:

Special Value Notations
=======================


.. container::


   This defines notations of special values for complex functions. The
   definitions are provided in text, tables, or formulas.


   - z, z1, z2, etc. denote complex numbers.


   - i, ``i2=-1`` is the imaginary unit.


   - x, X, x1, x2, etc. denote real imaginary parts.


   - y, Y, y1, y2, etc. denote imaginary parts.


   - X and Y represent any finite positive IEEE-754 floating point
      values, if not stated otherwise.


   - Quiet NaN and signaling NaN are denoted with QNAN and SNAN,
      respectively.


   - The IEEE-754 positive infinities or floating-point numbers are
      denoted with a ``+`` sign before X, Y, etc.


   - The IEEE-754 negative infinities or floating-point numbers are
      denoted with a ``-`` sign before X, Y, etc.


   ``CONJ(z)`` and ``CIS(z)`` are defined as follows:


   ``CONJ(x+i·y)=x-i·y``


   ``CIS(y)=cos(y)+i·sin(y)``.


   The special value tables show the result of the function for the z
   argument at the intersection of the RE(z) column and the i*IM(z) row.
   If the function encounters a special point for the argument z, the lower part
   of this cell shows the special point and the VM status value for this element. An
   empty cell indicates that this argument is normal and the result is
   well-defined computationally.


**Parent topic:** :ref:`onemath_vm`

