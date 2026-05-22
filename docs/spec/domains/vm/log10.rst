.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_vm_log10:

log10
=====


.. container::


   Computes the base 10 logarithm of vector elements.


   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      Buffer API:


      .. code-block:: cpp


            namespace oneapi::math::vm {

            sycl::event log10(
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

            sycl::event log10(
                    sycl::queue& exec_queue,
                    std::int64_t n,
                    const T *a,
                    T* y,
                    std::vector<sycl::event> const & depends = {},
                    oneapi::math::vm::mode mode = oneapi::math::vm::mode::not_defined,
                    oneapi::math::vm::error_handler<T> errhandler = {});

            } // namespace oneapi::math::vm



      ``log10`` supports the following precisions.


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


   The log10(a) function computes the base 10 logarithm of vector
   elements.


   .. container:: tablenoborder


      .. list-table::
         :header-rows: 1

         * - Argument
           - Result
           - Status code
         * - +1
           - +0
           -  
         * - a <+0
           - QNAN
           - ``oneapi::math::vm::status::errdom``
         * - +0
           - -∞
           - ``oneapi::math::vm::status::sing``
         * - -0
           - -∞
           - ``oneapi::math::vm::status::sing``
         * - -∞
           - QNAN
           - ``oneapi::math::vm::status::errdom``
         * - +∞
           - +∞
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

         * - RE(a) i·IM(a)
           - -∞
           - -X  
           - -0
           - +0
           - +X  
           - +∞  
           - NAN  
         * - +i·∞
           - |math0|
           - |math1|
           - |math1|
           - |math1|
           - |math1|
           - :math:`+\infty + i \frac{\pi}{4} \frac{1}{\ln 10}`
           - +∞+i·QNAN
         * - +i·Y
           - |math2|
           -  
           -  
           -  
           -  
           - +∞+i·0
           - QNAN+i·QNAN
         * - +i·0
           - |math2|
           -  
           - :math:`-\infty + i \frac{\pi}{\ln 10}`
           - -∞+i·0
           -  
           - +∞+i·0
           - QNAN+i·QNAN
         * - -i·0
           - |math3|
           -  
           - :math:`-\infty - i \frac{\pi}{\ln 10}`
           - -∞-i·0
           -  
           - +∞-i·0
           - QNAN-i·QNAN
         * - -i·Y
           - |math3|
           -  
           -  
           -  
           -  
           - +∞-i·0
           - QNAN+i·QNAN
         * - -i·∞
           - |math0|
           - |math4|
           - |math4|
           - |math4|
           - |math4|
           - :math:`+\infty - i \frac{\pi}{4} \frac{1}{\ln 10}`
           - +∞+i·QNAN
         * - +i·NAN
           - +∞+i·QNAN
           - QNAN+i·QNAN
           - QNAN+i·QNAN
           - QNAN+i·QNAN
           - QNAN+i·QNAN
           - +∞+i·QNAN
           - QNAN+i·QNAN




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



.. |math0| replace:: :math:`+\infty + i \frac{3}{4} \frac{\pi}{\ln 10}`
.. |math1| replace:: :math:`+\infty + i \frac{\pi}{2} \frac{1}{\ln 10}`
.. |math2| replace:: :math:`+\infty + i \frac{\pi}{\ln 10}`
.. |math3| replace:: :math:`+\infty - i \frac{\pi}{\ln 10}`
.. |math4| replace:: :math:`+\infty - i \frac{\pi}{2} \frac{1}{\ln 10}`
