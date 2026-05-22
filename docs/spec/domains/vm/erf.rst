.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_vm_erf:

erf
===


.. container::


   Computes the error function value of vector elements.


   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      Buffer API:


      .. code-block:: cpp


            namespace oneapi::math::vm {

            sycl::event erf(
                    sycl::queue& exec_queue,
                    std::int64_t n,
                    sycl::buffer<T,1>& a,
                    sycl::buffer<T,1>& y,
                    oneapi::math::vm::mode mode = oneapi::math::vm::mode::not_defined);

            } // namespace oneapi::math::vm



      USM API:


      .. code-block:: cpp


            namespace oneapi::math::vm {

            sycl::event erf(
                    sycl::queue& exec_queue,
                    std::int64_t n,
                    const T *a,
                    T* y,
                    std::vector<sycl::event> const & depends = {},
                    oneapi::math::vm::mode mode = oneapi::math::vm::mode::not_defined);

            } // namespace oneapi::math::vm



      ``erf`` supports the following precisions.


      .. list-table::
         :header-rows: 1

         * - T
         * - ``float``
         * - ``double``




.. container:: section


   .. rubric:: Description
      :class: sectiontitle


   The erf function computes the error function values for elements of
   the input vector ``a`` and writes them to the output vector ``y``.


   The error function is defined as given by:


   .. container:: imagecenter


      .. math::
         \operatorname{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} \operatorname{d \!} t


   Useful relations:

   .. math::
      \operatorname{erfc}(x) = 1 - \operatorname{erf}(x)


   where :math:`\operatorname{erfc}` is the complementary error function.


   .. math::
      \Phi(x) = \frac{1}{2} \left( 1 + \operatorname{erf}\left(x / \sqrt{2}\right) \right)


   where


   .. math::
      \Phi(x) = \frac{1}{\sqrt{2\pi}} \int_0^x \exp(-t^2/2) \operatorname{d \!} t


   is the cumulative normal distribution function.


   .. math::
      \Phi^{-1}(x) = \sqrt{2} \operatorname{erf}^{-1}(2x - 1)


   where :math:`\Phi^{-1}(x)` and :math:`\operatorname{erf}^{-1}(x)` are the inverses to
   :math:`\Phi(x)` and :math:`\operatorname{erf}(x)`, respectively.

   The following figure illustrates the relationships among erf family
   functions (erf, erfc, cdfnorm).


   .. container:: figtop


      erf Family Functions Relationship
      |


      .. container:: imagecenter


         |image0|


   Useful relations for these functions:

   .. math::
      \operatorname{erf}(x) + \operatorname{erfc}(x) = 1

   |


   .. container:: imagecenter


      .. math::
         \operatorname{cdfnorm}(x) &= \frac{1}{2} \left(1 + \operatorname{erf}\left(\frac{x}{\sqrt{2}}\right) \right) \\
                     &= 1 - \frac{1}{2} \operatorname{erfc}\left(\frac{x}{\sqrt{2}}\right)


   .. container:: tablenoborder


      .. list-table::
         :header-rows: 1

         * - Argument
           - Result
           - Status code
         * - +∞
           - +1
           -  
         * - -∞
           - -1
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
