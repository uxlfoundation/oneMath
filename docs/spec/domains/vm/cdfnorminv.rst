.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_vm_cdfnorminv:

cdfnorminv
==========


.. container::


   Computes the inverse cumulative normal distribution function values
   of vector elements.


   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      Buffer API:


      .. code-block:: cpp


            namespace oneapi::math::vm {

            sycl::event cdfnorminv(
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

            sycl::event cdfnorminv(
                    sycl::queue& exec_queue,
                    std::int64_t n,
                    const T *a,
                    T* y,
                    std::vector<sycl::event> const & depends = {},
                    oneapi::math::vm::mode mode = oneapi::math::vm::mode::not_defined,
                    oneapi::math::vm::error_handler<T> errhandler = {});

            } // namespace oneapi::math::vm



      ``cdfnorminv`` supports the following precisions.


      .. list-table::
         :header-rows: 1

         * - T
         * - ``float``
         * - ``double``




.. container:: section


   .. rubric:: Description
      :class: sectiontitle


   The cdfnorminv(a) function computes the inverse cumulative normal
   distribution function values for elements of the input vector ``a``
   and writes them to the output vector ``y``.


   The inverse cumulative normal distribution function is defined as
   given by:

   .. math::
      \operatorname{cdfnorminv}(x) = \operatorname{cdfnorm}^{-1}(x)


   where :math:`\operatorname{cdfnorm}(x)` denotes the cumulative normal distribution
   function.


   Useful relations:

   .. math::
      \operatorname{erfcinv}(x) = \operatorname{erfinv}(1 - x)


   .. math::
      \operatorname{cdfnorminv}(x) &= \sqrt{2} \operatorname{erfinv}(2x - 1) \\
                                   &= \sqrt{2} \operatorname{erfcinv}(2 - 2x)


   where :math:`\operatorname{erfinv}(x)` denotes the inverse error function and
   :math:`\operatorname{erfcinv}(x)` denotes the inverse complementary error function.

   The following figure illustrates the relationships among erfinv
   family functions (erfinv, erfcinv, cdfnorminv).


   .. container:: figtop


      cdfnorminv Family Functions Relationship
      |


      .. container:: imagecenter


         |image0|


   .. container:: tablenoborder


      .. list-table::
         :header-rows: 1

         * - Argument
           - Result
           - Status code
         * - +0.5
           - +0
           -  
         * - +1
           - +∞
           - ``oneapi::math::vm::status::sing``
         * - -0
           - -∞
           - ``oneapi::math::vm::status::sing``
         * - +0
           - -∞
           - ``oneapi::math::vm::status::sing``
         * - a < -0
           - QNAN
           - ``oneapi::math::vm::status::errdom``
         * - a > +1
           - QNAN
           - ``oneapi::math::vm::status::errdom``
         * - +∞
           - QNAN
           - ``oneapi::math::vm::status::errdom``
         * - -∞
           - QNAN
           - ``oneapi::math::vm::status::errdom``
         * - QNAN
           - QNAN
           -  
         * - SNAN
           - QNAN
           -  




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



.. |image0| image:: ../equations/inverse_error_functions_plot.jpg
