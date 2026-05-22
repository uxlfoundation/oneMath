.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_vm_sinh:

sinh
====


.. container::


   Computes hyperbolic sine of vector elements.


   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      Buffer API:


      .. code-block:: cpp


            namespace oneapi::math::vm {

            sycl::event sinh(
                    sycl::queue& exec_queue,
                    std::int64_t n,
                    sycl::buffer<T,1>& a,
                    sycl::buffer<T,1>& y,
                    oneapi::math::vm::mode mode = oneapi::math::vm::mode::not_defined,
                    oneapi::math::vm::error_handler<T> errhandler = {});

            } // namespace oneapi::math::vm



      USM API:


      .. code-block:: cpp


            namespace oneapi::math::vm {

            sycl::event sinh(
                    sycl::queue& exec_queue,
                    std::int64_t n,
                    const T *a,
                    T* y,
                    std::vector<sycl::event> const & depends = {},
                    oneapi::math::vm::mode mode = oneapi::math::vm::mode::not_defined,
                    oneapi::math::vm::error_handler<T> errhandler = {});

            } // namespace oneapi::math::vm



      ``sinh`` supports the following precisions.


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


   The sinh(a) function computes hyperbolic sine of vector elements.


   .. container:: tablenoborder


      .. list-table::
         :header-rows: 1

         * - Data Type
           - Threshold Limitations on Input Parameters
         * - single precision
           - ``-Log(FLT_MAX)-Log(2) <a[i] < Log(FLT_MAX)+Log(2)``
         * - double precision
           - ``-Log(DBL_MAX)-Log(2) <a[i] < Log(DBL_MAX)+Log(2)``




   .. container:: tablenoborder


      .. list-table::
         :header-rows: 1

         * - Argument
           - Result
           - Status code
         * - +0
           - +0
           -  
         * - -0
           - -0
           -  
         * - a > overflow
           - +∞
           - ``oneapi::math::vm::status::overflow``
         * - a < -overflow
           - -∞
           - ``oneapi::math::vm::status::overflow``
         * - +∞
           - +∞
           -  
         * - -∞
           - -∞
           -  
         * - QNAN
           - QNAN
           -  
         * - SNAN
           - QNAN
           -  




   .. container:: tablenoborder


      .. list-table::
         :header-rows: 1

         * -
           -
           -
           -
           -
           -
           -
           -
         * - +i·∞
           - -∞+i·QNAN
           - QNAN+i·QNAN
           - -0+i·QNAN
           - +0+i·QNAN
           - QNAN+i·QNAN
           - +∞+i·QNAN
           - QNAN+i·QNAN
         * - +i·Y
           - -∞·Cos(Y)+ i·∞·Sin(Y)
           -  
           -  
           -  
           -  
           - +∞·CIS(Y)
           - QNAN+i·QNAN
         * - +i·0
           - -∞+i·0
           -  
           - -0+i·0
           - +0+i·0
           -  
           - +∞+i·0
           - QNAN+i·0
         * - -i·0
           - -∞-i·0
           -  
           - -0-i·0
           - +0-i·0
           -  
           - +∞-i·0
           - QNAN-i·0
         * - -i·Y
           - -∞·Cos(Y)+ i·∞·Sin(Y)
           -  
           -  
           -  
           -  
           - +∞·CIS(Y)
           - QNAN+i·QNAN
         * - -i·∞
           - -∞+i·QNAN
           - QNAN+i·QNAN
           - -0+i·QNAN
           - +0+i·QNAN
           - QNAN+i·QNAN
           - +∞+i·QNAN
           - QNAN+i·QNAN
         * - +i·NAN
           - -∞+i·QNAN
           - QNAN+i·QNAN
           - -0+i·QNAN
           - +0+i·QNAN
           - QNAN+i·QNAN
           - +∞+i·QNAN
           - QNAN+i·QNAN




   Notes:


   - The complex sinh(a) function sets the VM status code to
      oneapi::math::vm::status::overflow in the case of overflow, that is, when RE(a),
      IM(a) are finite non-zero numbers, but the real or imaginary part
      of the exact result is so large that it does not meet the target
      precision.


   - ``sinh(CONJ(a))=CONJ(sinh(a))``


   - ``sinh(-a)=-sinh(a)``.


.. container:: section


   .. rubric:: Input Parameters
      :class: sectiontitle


   Buffer API:


   exec_queue
      The queue where the routine should be executed.


   n
      Specifies the number of elements to be calculated.


   a
      The buffer ``a`` containing input vector of size ``n``.


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
      Pointer ``a`` to the input vector of size ``n``.


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


