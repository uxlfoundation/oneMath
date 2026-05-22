.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_vm_arg:

arg
===


.. container::


   Computes argument of vector elements.


   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      Buffer API:


      .. code-block:: cpp


            namespace oneapi::math::vm {

            sycl::event arg(
                    sycl::queue& exec_queue,
                    std::int64_t n,
                    sycl::buffer<T,1>& a,
                    sycl::buffer<R,1>& y,
                    oneapi::math::vm::mode mode = oneapi::math::vm::mode::not_defined);

            } // namespace oneapi::math::vm



      USM API:


      .. code-block:: cpp


            namespace oneapi::math::vm {

            sycl::event arg(
                    sycl::queue& exec_queue,
                    std::int64_t n,
                    const T *a,
                    R* y,
                    std::vector<sycl::event> const & depends = {},
                    oneapi::math::vm::mode mode = oneapi::math::vm::mode::not_defined);

            } // namespace oneapi::math::vm



      ``arg`` supports the following precisions.


      .. list-table::
         :header-rows: 1

         * - T
           - R
         * - ``std::complex<float>``
           - ``float``
         * - ``std::complex<double>``
           - ``double``


.. container:: section


   .. rubric:: Description
      :class: sectiontitle


   The arg(a) function computes argument of vector elements.


   See :ref:`onemath_vm_special_value_notations` for the conventions used in the
   table below.


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
           - +3·π/4
           - +π/2
           - +π/2
           - +π/2
           - +π/2
           - +π/4
           - NAN
         * - +i·Y
           - +π
           -  
           - +π/2
           - +π/2
           -  
           - +0
           - NAN
         * - +i·0
           - +π
           - +π
           - +π
           - +0
           - +0
           - +0
           - NAN
         * - -i·0
           - -π
           - -π
           - -π
           - -0
           - -0
           - -0
           - NAN
         * - -i·Y
           - -π
           -  
           - -π/2
           - -π/2
           -  
           - -0
           - NAN
         * - -i·∞
           - -3·π/4
           - -π/2
           - -π/2
           - -π/2
           - -π/2
           - -π/4
           - NAN
         * - +i·NAN
           - NAN
           - NAN
           - NAN
           - NAN
           - NAN
           - NAN
           - NAN




   .. container:: Note


      .. rubric:: Note
         :class: NoteTipHead


      ``arg(a)=atan2(IM(a), RE(a))``


   The arg function does not generate any errors.


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


