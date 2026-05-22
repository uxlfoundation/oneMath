.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_vm_mathematical_functions:

VM Mathematical Functions
=========================


.. container::


   This section describes VM functions that compute values of
   mathematical functions on real and complex vector arguments with unit
   increment.


   Each function is introduced by its short name, a brief description of
   its purpose, and the calling sequence for each type of data, as well
   as a description of the input/output arguments.


   The input range of parameters is equal to the mathematical range of
   the input data type, unless the function description specifies input
   threshold values, which mark off the precision overflow, as follows:


   - ``FLT_MAX`` denotes the maximum number representable in single precision real
     data type


   - ``DBL_MAX`` denotes the maximum number representable in double precision real
     data type


   The following tables list the available mathematical functions grouped by
   category.


   .. container:: tablenoborder

      .. tabularcolumns:: |p{0.4\textwidth}|p{0.6\textwidth}|

      .. list-table::
         :header-rows: 1
         :class: longtable

         * - Arithmetic Routines
           - Description
         * - :ref:`onemath_vm_add`
           - Adds vector elements
         * - :ref:`onemath_vm_sub`
           - Subtracts vector elements
         * - :ref:`onemath_vm_sqr`
           - Squares vector elements
         * - :ref:`onemath_vm_mul`
           - Multiplies vector elements
         * - :ref:`onemath_vm_mulbyconj`
           - Multiplies elements of one vector by conjugated elements of the
             second vector
         * - :ref:`onemath_vm_conj`
           - Conjugates vector elements
         * - :ref:`onemath_vm_abs`
           - Computes the absolute value of vector elements
         * - :ref:`onemath_vm_arg`
           - Computes the argument of vector elements
         * - :ref:`onemath_vm_linearfrac`
           - Performs linear fraction transformation of vectors
         * - :ref:`onemath_vm_fmod`
           - Performs element by element computation of the modulus function of
             vector ``a`` with respect to vector ``b``
         * - :ref:`onemath_vm_remainder`
           - Performs element by element computation of the remainder function
             on the elements of vector ``a`` and the corresponding elements of
             vector ``b``

      .. tabularcolumns:: |p{0.4\textwidth}|p{0.6\textwidth}|

      .. list-table::
         :header-rows: 1
         :class: longtable

         * - Power and Root Routines
           - Description
         * - :ref:`onemath_vm_inv`
           - Inverts vector elements
         * - :ref:`onemath_vm_div`
           - Divides elements of one vector by elements of the second vector
         * - :ref:`onemath_vm_sqrt`
           - Computes the square root of vector elements
         * - :ref:`onemath_vm_invsqrt`
           - Computes the inverse square root of vector elements
         * - :ref:`onemath_vm_cbrt`
           - Computes the cube root of vector elements
         * - :ref:`onemath_vm_invcbrt`
           - Computes the inverse cube root of vector elements
         * - :ref:`onemath_vm_pow2o3`
           - Computes the cube root of the square of each vector element
         * - :ref:`onemath_vm_pow3o2`
           - Computes the square root of the cube of each vector element
         * - :ref:`onemath_vm_pow`
           - Raises each vector element to the specified power
         * - :ref:`onemath_vm_powx`
           - Raises each vector element to the constant power
         * - :ref:`onemath_vm_powr`
           - Computes ``a`` to the power ``b`` for elements of two vectors,
             where the elements of vector argument ``a`` are all non-negative
         * - :ref:`onemath_vm_hypot`
           - Computes the square root of sum of squares

      .. tabularcolumns:: |p{0.4\textwidth}|p{0.6\textwidth}|

      .. list-table::
         :header-rows: 1
         :class: longtable

         * - Exponential and Logarithmic Routines
           - Description
         * - :ref:`onemath_vm_exp`
           - Computes the base ``e`` exponential of vector elements
         * - :ref:`onemath_vm_exp2`
           - Computes the base 2 exponential of vector elements
         * - :ref:`onemath_vm_exp10`
           - Computes the base 10 exponential of vector elements
         * - :ref:`onemath_vm_expm1`
           - Computes the base ``e`` exponential of vector elements decreased
             by 1
         * - :ref:`onemath_vm_ln`
           - Computes the natural logarithm of vector elements
         * - :ref:`onemath_vm_log2`
           - Computes the base 2 logarithm of vector elements
         * - :ref:`onemath_vm_log10`
           - Computes the base 10 logarithm of vector elements
         * - :ref:`onemath_vm_log1p`
           - Computes the natural logarithm of vector elements that are
             increased by 1
         * - :ref:`onemath_vm_logb`
           - Computes the exponents of the elements of input vector ``a``

      .. tabularcolumns:: |p{0.4\textwidth}|p{0.6\textwidth}|

      .. list-table::
         :header-rows: 1
         :class: longtable

         * - Trigonometric Routines
           - Description
         * - :ref:`onemath_vm_cos`
           - Computes the cosine of vector elements
         * - :ref:`onemath_vm_sin`
           - Computes the sine of vector elements
         * - :ref:`onemath_vm_sincos`
           - Computes the sine and cosine of vector elements
         * - :ref:`onemath_vm_cis`
           - Computes the complex exponent of vector elements (cosine and sine
             combined to complex value)
         * - :ref:`onemath_vm_tan`
           - Computes the tangent of vector elements
         * - :ref:`onemath_vm_acos`
           - Computes the inverse cosine of vector elements
         * - :ref:`onemath_vm_asin`
           - Computes the inverse sine of vector elements
         * - :ref:`onemath_vm_atan`
           - Computes the inverse tangent of vector elements
         * - :ref:`onemath_vm_atan2`
           - Computes the four-quadrant inverse tangent of ratios of the
             elements of two vectors
         * - :ref:`onemath_vm_cospi`
           - Computes the cosine of vector elements multiplied by ``π``
         * - :ref:`onemath_vm_sinpi`
           - Computes the sine of vector elements multiplied by ``π``
         * - :ref:`onemath_vm_tanpi`
           - Computes the tangent of vector elements multiplied by ``π``
         * - :ref:`onemath_vm_acospi`
           - Computes the inverse cosine of vector elements divided by ``π``
         * - :ref:`onemath_vm_asinpi`
           - Computes the inverse sine of vector elements divided by ``π``
         * - :ref:`onemath_vm_atanpi`
           - Computes the inverse tangent of vector elements divided by ``π``
         * - :ref:`onemath_vm_atan2pi`
           - Computes the four-quadrant inverse tangent of the ratios of the
             corresponding elements of two vectors divided by ``π``
         * - :ref:`onemath_vm_cosd`
           - Computes the cosine of vector elements multiplied by ``π``/180
         * - :ref:`onemath_vm_sind`
           - Computes the sine of vector elements multiplied by ``π``/180
         * - :ref:`onemath_vm_tand`
           - Computes the tangent of vector elements multiplied by ``π``/180

      .. tabularcolumns:: |p{0.4\textwidth}|p{0.6\textwidth}|

      .. list-table::
         :header-rows: 1
         :class: longtable

         * - Hyperbolic Routines
           - Description
         * - :ref:`onemath_vm_cosh`
           - Computes the hyperbolic cosine of vector elements
         * - :ref:`onemath_vm_sinh`
           - Computes the hyperbolic sine of vector elements
         * - :ref:`onemath_vm_tanh`
           - Computes the hyperbolic tangent of vector elements
         * - :ref:`onemath_vm_acosh`
           - Computes the inverse hyperbolic cosine of vector elements
         * - :ref:`onemath_vm_asinh`
           - Computes the inverse hyperbolic sine of vector elements
         * - :ref:`onemath_vm_atanh`
           - Computes the inverse hyperbolic tangent of vector elements.

      .. tabularcolumns:: |p{0.4\textwidth}|p{0.6\textwidth}|

      .. list-table::
         :header-rows: 1
         :class: longtable

         * - Special Routines
           - Description
         * - :ref:`onemath_vm_erf`
           - Computes the error function value of vector elements
         * - :ref:`onemath_vm_erfc`
           - Computes the complementary error function value of vector elements
         * - :ref:`onemath_vm_cdfnorm`
           - Computes the cumulative normal distribution function value of
             vector elements
         * - :ref:`onemath_vm_erfinv`
           - Computes the inverse error function value of vector elements
         * - :ref:`onemath_vm_erfcinv`
           - Computes the inverse complementary error function value of vector
             elements
         * - :ref:`onemath_vm_cdfnorminv`
           - Computes the inverse cumulative normal distribution function value
             of vector elements
         * - :ref:`onemath_vm_lgamma`
           - Computes the natural logarithm for the absolute value of the gamma
             function of vector elements
         * - :ref:`onemath_vm_tgamma`
           - Computes the gamma function of vector elements
         * - :ref:`onemath_vm_expint1`
           - Computes the exponential integral of vector elements

      .. tabularcolumns:: |p{0.4\textwidth}|p{0.6\textwidth}|

      .. list-table::
         :header-rows: 1
         :class: longtable

         * - Rounding Routines
           - Description
         * - :ref:`onemath_vm_floor`
           - Rounds towards minus infinity
         * - :ref:`onemath_vm_ceil`
           - Rounds towards plus infinity
         * - :ref:`onemath_vm_trunc`
           - Rounds towards zero infinity
         * - :ref:`onemath_vm_round`
           - Rounds to nearest integer
         * - :ref:`onemath_vm_nearbyint`
           - Rounds according to current mode
         * - :ref:`onemath_vm_rint`
           - Rounds according to current mode and reports inexact result
             status
         * - :ref:`onemath_vm_modf`
           - Computes the integer and fractional parts
         * - :ref:`onemath_vm_frac`
           - Computes the fractional part

      .. tabularcolumns:: |p{0.4\textwidth}|p{0.6\textwidth}|

      .. list-table::
         :header-rows: 1
         :class: longtable

         * - Miscellaneous Routines
           - Description
         * - :ref:`onemath_vm_copysign`
           - Returns vector of elements of one argument with signs changed to
             match other argument elements
         * - :ref:`onemath_vm_nextafter`
           - Returns vector of elements containing the next representable
             floating-point values following the values from the elements of
             one vector in the direction of the corresponding elements of
             another vector
         * - :ref:`onemath_vm_fdim`
           - Returns vector containing the differences of the corresponding
             elements of the vector arguments if the first is larger and +0
             otherwise
         * - :ref:`onemath_vm_fmax`
           - Returns the larger of each pair of elements of the two vector
             arguments
         * - :ref:`onemath_vm_fmin`
           - Returns the smaller of each pair of elements of the two vector
             arguments
         * - :ref:`onemath_vm_maxmag`
           - Returns the element with the larger magnitude between each pair of
             elements of the two vector arguments
         * - :ref:`onemath_vm_minmag`
           - Returns the element with the smaller magnitude between each pair
             of elements of the two vector arguments


**Parent topic:** :ref:`onemath_vm`

.. toctree::
    :hidden:

    abs
    acos
    acosh
    acospi
    add
    arg
    asin
    asinh
    asinpi
    atan
    atan2
    atan2pi
    atanh
    atanpi
    cbrt
    cdfnorm
    cdfnorminv
    ceil
    cis
    conj
    copysign
    cos
    cosd
    cosh
    cospi
    div
    erf
    erfc
    erfcinv
    erfinv
    exp
    exp10
    exp2
    expint1
    expm1
    fdim
    floor
    fmax
    fmin
    fmod
    frac
    hypot
    inv
    invcbrt
    invsqrt
    lgamma
    linearfrac
    ln
    log10
    log1p
    log2
    logb
    maxmag
    minmag
    modf
    mul
    mulbyconj
    nearbyint
    nextafter
    pow
    pow2o3
    pow3o2
    powr
    powx
    remainder
    rint
    round
    sin
    sincos
    sind
    sinh
    sinpi
    sqr
    sqrt
    sub
    tan
    tand
    tanh
    tanpi
    tgamma
    trunc
