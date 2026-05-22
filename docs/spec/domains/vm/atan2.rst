.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_vm_atan2:

atan2
=====


.. container::


   Computes four-quadrant inverse tangent of elements of two vectors.


   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      Buffer API:


      .. code-block:: cpp


            namespace oneapi::math::vm {

            sycl::event atan2(
                    sycl::queue& exec_queue,
                    std::int64_t n,
                    sycl::buffer<T,1>& a,
                    sycl::buffer<T,1>& b,
                    sycl::buffer<T,1>& y,
                    oneapi::math::vm::mode mode = oneapi::math::vm::mode::not_defined);

            } // namespace oneapi::math::vm



      USM API:


      .. code-block:: cpp


            namespace oneapi::math::vm {

            sycl::event atan2(
                    sycl::queue& exec_queue,
                    std::int64_t n,
                    const T *a,
                    const T *b,
                    T* y,
                    std::vector<sycl::event> const & depends = {},
                    oneapi::math::vm::mode mode = oneapi::math::vm::mode::not_defined);

            } // namespace oneapi::math::vm



      ``ad2d`` supports the following precisions.


      .. list-table::
         :header-rows: 1

         * - T
         * - ``float``
         * - ``double``




.. container:: section


   .. rubric:: Description
      :class: sectiontitle


   The atan2(a, b) function computes four-quadrant inverse tangent of
   elements of two vectors.


   The elements of the output vector are computed as the four-quadrant
   arctangent of ``a[i] / b[i]``.


   .. container:: tablenoborder


      .. list-table::
         :header-rows: 1

         * - Argument 1
           - Argument 2
           - Result
           - Status code
         * - -∞
           - -∞
           - -3\*\ ``π``/4
           -  
         * - -∞
           - b < +0
           - -``π``/2
           -  
         * - -∞
           - -0
           - -``π``/2
           -  
         * - -∞
           - +0
           - -``π``/2
           -  
         * - -∞
           - b > +0
           - -``π``/2
           -  
         * - -∞
           - +∞
           - -``π``/4
           -  
         * - a < +0
           - -∞
           - -``π``
           -  
         * - a < +0
           - -0
           - -``π``/2
           -  
         * - a < +0
           - +0
           - -``π``/2
           -  
         * - a < +0
           - +∞
           - -0
           -  
         * - -0
           - -∞
           - -``π``
           -  
         * - -0
           - b < +0
           - -``π``
           -  
         * - -0
           - -0
           - -``π``
           -  
         * - -0
           - +0
           - -0
           -  
         * - -0
           - b > +0
           - -0
           -  
         * - -0
           - +∞
           - -0
           -  
         * - +0
           - -∞
           - +\ ``π``
           -  
         * - +0
           - b < +0
           - +\ ``π``
           -  
         * - +0
           - -0
           - +\ ``π``
           -  
         * - +0
           - +0
           - +0
           -  
         * - +0
           - b > +0
           - +0
           -  
         * - +0
           - +∞
           - +0
           -  
         * - a > +0
           - -∞
           - +\ ``π``
           -  
         * - a > +0
           - -0
           - +\ ``π``/2
           -  
         * - a > +0
           - +0
           - +\ ``π``/2
           -  
         * - a > +0
           - +∞
           - +0
           -  
         * - +∞
           - -∞
           - +3\*\ ``π``/4
           -  
         * - +∞
           - b < +0
           - +\ ``π``/2
           -  
         * - +∞
           - -0
           - +\ ``π``/2
           -  
         * - +∞
           - +0
           - +\ ``π``/2
           -  
         * - +∞
           - b > +0
           - +\ ``π``/2
           -  
         * - +∞
           - +∞
           - +\ ``π``/4
           -  
         * - a > +0
           - QNAN
           - QNAN
           -  
         * - a > +0
           - SNAN
           - QNAN
           -  
         * - QNAN
           - b > +0
           - QNAN
           -  
         * - SNAN
           - b > +0
           - QNAN
           -  
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




   The atan2(a, b) function does not generate any errors.


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


