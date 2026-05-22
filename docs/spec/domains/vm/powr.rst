.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_vm_powr:

powr
====


.. container::


   Computes ``a`` to the power ``b`` for elements of two vectors, where
   the elements of vector argument ``a`` are all non-negative.


   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      Buffer API:


      .. code-block:: cpp


            namespace oneapi::math::vm {

            sycl::event powr(
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

            sycl::event powr(
                    sycl::queue& exec_queue,
                    std::int64_t n,
                    const T *a,
                    const T *b,
                    T* y,
                    std::vector<sycl::event> const & depends = {},
                    oneapi::math::vm::mode mode = oneapi::math::vm::mode::not_defined,
                    oneapi::math::vm::error_handler<T> errhandler = {});

            } // namespace oneapi::math::vm



      ``powr`` supports the following precisions.


      .. list-table::
         :header-rows: 1

         * - T
         * - ``float``
         * - ``double``




.. container:: section


   .. rubric:: Description
      :class: sectiontitle


   The powr(a, b) function raises each element of vector ``a`` by the
   corresponding element of vector ``b``. The elements of ``a`` are all
   nonnegative (``a``\ :sub:`i`\ ≥ 0).


   .. container:: tablenoborder


      .. list-table::
         :header-rows: 1

         * - Data Type
           - Threshold Limitations on Input Parameters
         * - single precision
           - ``a``\ :sub:`i` < (FLT_MAX)\ :sup:`1/``b``\ i`
         * - double precision
           - ``a``\ :sub:`i` < (DBL_MAX)\ :sup:`1/``b``\ i`




   Special values and VM status code treatment for v?Powr function are
   the same as for pow, unless otherwise indicated in this table:


   .. container:: tablenoborder


      .. list-table::
         :header-rows: 1

         * - Argument 1
           - Argument 2
           - Result
           - Status code
         * - ``a`` < 0
           - any value ``b``
           - NAN
           - ``oneapi::math::vm::status::errdom``
         * - 0 < ``a`` < ∞
           - ±0
           - 1
           -  
         * - ±0
           - -∞ < ``b`` < 0
           - +∞
           -  
         * - ±0
           - -∞
           - +∞
           -  
         * - ±0
           - ``b`` > 0
           - +0
           -  
         * - 1
           - -∞ < ``b`` < ∞
           - 1
           -  
         * - ±0
           - ±0
           - NAN
           -  
         * - +∞
           - ±0
           - NAN
           -  
         * - 1
           - +∞
           - NAN
           -  
         * - ``a``\ ≥ 0
           - NAN
           - NAN
           -  
         * - NAN
           - any value ``b``
           - NAN
           -  
         * - 0 < ``a`` <1
           - -∞
           - +∞
           -  
         * - ``a`` > 1
           - -∞
           - +0
           -  
         * - 0 ≤\ ``a`` < 1
           - +∞
           - +0
           -  
         * - ``a`` > 1
           - +∞
           - +∞
           -  
         * - +∞
           - ``b`` < +0
           - +0
           -  
         * - +∞
           - ``b`` > +0
           - +∞
           -  
         * - QNAN
           - QNAN
           - QNAN
           - ``oneapi::math::vm::status::errdom``
         * - QNAN
           - SNAN
           - QNAN
           - ``oneapi::math::vm::status::errdom``
         * - SNAN
           - QNAN
           - QNAN
           - ``oneapi::math::vm::status::errdom``
         * - SNAN
           - SNAN
           - QNAN
           - ``oneapi::math::vm::status::errdom``




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


