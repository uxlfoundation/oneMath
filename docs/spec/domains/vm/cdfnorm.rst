.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_vm_cdfnorm:

cdfnorm
=======


.. container::


   Computes the cumulative normal distribution function values of vector
   elements.


   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      Buffer API:


      .. code-block:: cpp


            namespace oneapi::math::vm {

            sycl::event cdfnorm(
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

            sycl::event cdfnorm(
                    sycl::queue& exec_queue,
                    std::int64_t n,
                    const T *a,
                    T* y,
                    std::vector<sycl::event> const & depends = {},
                    oneapi::math::vm::mode mode = oneapi::math::vm::mode::not_defined,
                    oneapi::math::vm::error_handler<T> errhandler = {});

            } // namespace oneapi::math::vm



      ``cdfnorm`` supports the following precisions.


      .. list-table::
         :header-rows: 1

         * - T
         * - ``float``
         * - ``double``




.. container:: section


   .. rubric:: Description
      :class: sectiontitle


   The cdfnorm function computes the cumulative normal distribution
   function values for elements of the input vector ``a`` and writes
   them to the output vector ``y``.


   The cumulative normal distribution function is defined as given by:

   .. math::
      \operatorname{cdfnorm}(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{x} e^{-\frac{t^2}{2}}
      \operatorname{d \!} x

   Useful relations for these functions:

   .. math::
      \operatorname{erf}(x) + \operatorname{erfc}(x) = 1


   .. math::
      \operatorname{cdfnorm}(x) &= \frac{1}{2} \left( 1 + \operatorname{erf}\left(\frac{x}{\sqrt{2}}\right) \right) \\
                  &= 1 - \frac{1}{2} \operatorname{erfc}\left(\frac{x}{\sqrt{2}}\right)


   where :math:`\operatorname{erf}` and :math:`\operatorname{erfc}` are the error and complementary error
   functions, respectively.

   The following figure illustrates the relationships among the family of error
   functions (erf, erfc, cdfnorm).


   .. container:: figtop


      cdfnorm Family Functions Relationship
      |


      .. container:: imagecenter


         |image0|


   .. container:: tablenoborder


      .. list-table::
         :header-rows: 1

         * - Argument
           - Result
           - Status code
         * - a < underflow
           - +0
           - ``oneapi::math::vm::status::underflow``
         * - +∞
           - +1
           -  
         * - -∞
           - +0
           -  
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

.. |image0| image:: ../equations/error_functions_plot.jpg
