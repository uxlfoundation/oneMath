.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_vm_pow:

pow
===


.. container::


   Computes ``a`` to the power ``b`` for elements of two vectors.


   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      Buffer API:


      .. code-block:: cpp


            namespace oneapi::math::vm {

            sycl::event pow(
                    sycl::queue& exec_queue,
                    std::int64_t n,
                    sycl::buffer<T,1>& a,
                    sycl::buffer<T,1>& b,
                    sycl::buffer<T,1>& y,
                    oneapi::math::vm::mode mode = oneapi::math::vm::mode::not_defined,
                    oneapi::math::vm::error_handler<T> errhandler = {});

            } // namespace oneapi::math::vm



      USM API:


      .. code-block:: cpp


            namespace oneapi::math::vm {

            sycl::event pow(
                    sycl::queue& exec_queue,
                    std::int64_t n,
                    const T *a,
                    const T *b,
                    T* y,
                    std::vector<sycl::event> const & depends = {},
                    oneapi::math::vm::mode mode = oneapi::math::vm::mode::not_defined,
                    oneapi::math::vm::error_handler<T> errhandler = {});

            } // namespace oneapi::math::vm



      ``pow`` supports the following precisions.


      .. list-table::
         :header-rows: 1

         * - T
         * - ``float``
         * - ``double``
         * - ``std::complex<float>``
         * - ``std::complex<double>``




.. container:: section


   .. rubric:: Description
      :class: sectiontitle


   The pow(a, b) function computes ``a`` to the power ``b`` for elements
   of two vectors.


   The real function pow has certain limitations on the input range of
   ``a`` and ``b`` parameters. Specifically, if ``a``\ [i] is positive,
   then ``b``\ [i] may be arbitrary. For negative ``a``\ [i], the value
   of ``b``\ [i] must be an integer (either positive or negative).


   The complex function pow has no input range limitations.


   .. container:: tablenoborder


      .. list-table::
         :header-rows: 1

         * - Argument 1
           - Argument 2
           - Result
           - Status code
         * - +0
           - neg. odd integer
           - +∞
           - ``oneapi::math::vm::status::errdom``
         * - -0
           - neg. odd integer
           - -∞
           - ``oneapi::math::vm::status::errdom``
         * - +0
           - neg. even integer
           - +∞
           - ``oneapi::math::vm::status::errdom``
         * - -0
           - neg. even integer
           - +∞
           - ``oneapi::math::vm::status::errdom``
         * - +0
           - neg. non-integer
           - +∞
           - ``oneapi::math::vm::status::errdom``
         * - -0
           - neg. non-integer
           - +∞
           - ``oneapi::math::vm::status::errdom``
         * - -0
           - pos. odd integer
           - +0
           -  
         * - -0
           - pos. odd integer
           - -0
           -  
         * - +0
           - pos. even integer
           - +0
           -  
         * - -0
           - pos. even integer
           - +0
           -  
         * - +0
           - pos. non-integer
           - +0
           -  
         * - -0
           - pos. non-integer
           - +0
           -  
         * - -1
           - +∞
           - +1
           -  
         * - -1
           - -∞
           - +1
           -  
         * - +1
           - any value
           - +1
           -  
         * - +1
           - +0
           - +1
           -  
         * - +1
           - -0
           - +1
           -  
         * - +1
           - +∞
           - +1
           -  
         * - +1
           - -∞
           - +1
           -  
         * - +1
           - QNAN
           - +1
           -  
         * - any value
           - +0
           - +1
           -  
         * - +0
           - +0
           - +1
           -  
         * - -0
           - +0
           - +1
           -  
         * - +∞
           - +0
           - +1
           -  
         * - -∞
           - +0
           - +1
           -  
         * - QNAN
           - +0
           - +1
           -  
         * - any value
           - -0
           - +1
           -  
         * - +0
           - -0
           - +1
           -  
         * - -0
           - -0
           - +1
           -  
         * - +∞
           - -0
           - +1
           -  
         * - -∞
           - -0
           - +1
           -  
         * - QNAN
           - -0
           - +1
           -  
         * - a < +0
           - non-integer
           - QNAN
           - ``oneapi::math::vm::status::errdom``
         * - \|a\| < 1
           - -∞
           - +∞
           -  
         * - +0
           - -∞
           - +∞
           - ``oneapi::math::vm::status::errdom``
         * - -0
           - -∞
           - +∞
           - ``oneapi::math::vm::status::errdom``
         * - \|a\| > 1
           - -∞
           - +0
           -  
         * - +∞
           - -∞
           - +0
           -  
         * - -∞
           - -∞
           - +0
           -  
         * - \|a\| < 1
           - +∞
           - +0
           -  
         * - +0
           - +∞
           - +0
           -  
         * - -0
           - +∞
           - +0
           -  
         * - \|a\| > 1
           - +∞
           - +∞
           -  
         * - +∞
           - +∞
           - +∞
           -  
         * - -∞
           - +∞
           - +∞
           -  
         * - -∞
           - neg. odd integer
           - -0
           -  
         * - -∞
           - neg. even integer
           - +0
           -  
         * - -∞
           - neg. non-integer
           - +0
           -  
         * - -∞
           - pos. odd integer
           - -∞
           -  
         * - -∞
           - pos. even integer
           - +∞
           -  
         * - -∞
           - pos. non-integer
           - +∞
           -  
         * - +∞
           - b < +0
           - +0
           -  
         * - +∞
           - b > +0
           - +∞
           -  
         * - Big finite value\*
           - Big finite value\*
           - +/-∞
           - ``oneapi::math::vm::status::overflow``
         * - QNAN
           - QNAN
           - QNAN
           -  
         * - QNAN
           - SNAN
           - QNAN
           -  
         * - SNAN
           - QNAN
           - QNAN
           -  
         * - SNAN
           - SNAN
           - QNAN
           -  




   \* Overflow in a real function is supported only in the HA/LA accuracy
   modes. The overflow occurs when x and y are finite numbers, but the
   result is too large to fit the target precision. In this case, the
   function:


   #. Returns ∞ in the result.


   #. Sets the VM status code to ``oneapi::math::vm::status::overflow``.


   Overflow in a complex function occurs (supported in the HA/LA
   accuracy modes only) when all RE(x), RE(y), IM(x), IM(y) arguments
   are finite numbers, but the real or imaginary part of the computed
   result is so large that it does not fit the target precision. In this
   case, the function returns ∞ in that part of the result, and sets the
   VM status code to ``oneapi::math::vm::status::overflow`` (overriding any possible
   ``oneapi::math::vm::status::accuracy_warning`` status).


   The complex double precision versions of this function are
   implemented in the EP accuracy mode only. If used in HA or LA mode,
   the functions set the VM status code to
   ``oneapi::math::vm::status::accuracy_warning``.


.. container:: section


   .. rubric:: Input Parameters
      :class: sectiontitle


   Buffer API:


   exec_queue
      The queue where the routine should be executed.


   n
      Specifies the number of elements to be calculated.


   a
      The buffer ``a`` containing 1st input vector of size ``n``.


   b
      The buffer ``b`` containing 2nd input vector of size ``n``.


   mode
      Overrides the global VM mode setting for this function call. See
      :ref:`onemath_vm_setmode`
      function for possible values and their description. This is an
      optional parameter. The default value is ``oneapi::math::vm::mode::not_defined``.


   errhandler
      Sets local error handling mode for this function call. See the
      :ref:`onemath_vm_create_error_handler`
      function for arguments and their descriptions. This is an optional
      parameter. The local error handler is disabled by default.


   USM API:


   exec_queue
      The queue where the routine should be executed.


   n
      Specifies the number of elements to be calculated.


   a
      Pointer ``a`` to the 1st input vector of size ``n``.


   b
      Pointer ``b`` to the 2nd input vector of size ``n``.


   depends
      Vector of dependent events (to wait for input data to be ready).


   mode
      Overrides the global VM mode setting for this function call. See
      the :ref:`onemath_vm_setmode`
      function for possible values and their description. This is an
      optional parameter. The default value is ``oneapi::math::vm::mode::not_defined``.


   errhandler
      Sets local error handling mode for this function call. See the
      :ref:`onemath_vm_create_error_handler`
      function for arguments and their descriptions. This is an optional
      parameter. The local error handler is disabled by default.


.. container:: section


   .. rubric:: Output Parameters
      :class: sectiontitle


   Buffer API:


   y
      The buffer ``y`` containing the output vector of size ``n``.


   USM API:


   y
      Pointer ``y`` to the output vector of size ``n``.


   return value (event)
      Event, signifying availability of computed output and status code(s).

.. container:: section


    .. rubric:: Exceptions
        :class: sectiontitle

    For list of generated exceptions please refer to  :ref:`onemath_vm_exceptions`


.. container:: familylinks


   .. container:: parentlink

      **Parent topic:** :ref:`onemath_vm_mathematical_functions`


