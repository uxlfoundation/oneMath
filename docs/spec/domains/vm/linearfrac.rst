.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_vm_linearfrac:

linearfrac
==========


.. container::


   Performs linear fraction transformation of vectors ``a`` and ``b``
   with scalar parameters.


   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      Buffer API:


      .. code-block:: cpp


            namespace oneapi::math::vm {

            sycl::event linearfrac(
                    sycl::queue& exec_queue,
                    std::int64_t n,
                    sycl::buffer<T,1>& a,
                    sycl::buffer<T,1>& b,
                    T scalea,
                    T shifta,
                    T scaleb,
                    T shiftb,
                    sycl::buffer<T,1>& y,
                    oneapi::math::vm::mode mode = oneapi::math::vm::mode::not_defined,
                    oneapi::math::vm::error_handler<T> errhandler = {});

            } // namespace oneapi::math::vm



      USM API:


      .. code-block:: cpp


            namespace oneapi::math::vm {

            sycl::event linearfrac(
                    sycl::queue& exec_queue,
                    std::int64_t n,
                    const T *a,
                    const T *b,
                    T scalea,
                    T shifta,
                    T scaleb,
                    T shiftb,
                    T* y,
                    std::vector<sycl::event> const & depends = {},
                    oneapi::math::vm::mode mode = oneapi::math::vm::mode::not_defined,
                    oneapi::math::vm::error_handler<T> errhandler = {});

            } // namespace oneapi::math::vm



      ``linearfrac`` supports the following precisions.


      .. list-table::
         :header-rows: 1

         * - T
         * - ``float``
         * - ``double``




.. container:: section


   .. rubric:: Description
      :class: sectiontitle


   The linearfrac(a, b, scalea, shifta, scaleb, shiftb) function performs a
   linear fraction transformation of vector ``a`` by vector ``b`` with
   scalar parameters: scaling multipliers scalea, scaleb and shifting
   addends shifta, shiftb:


   ``y[i]=(scalea·a[i]+shifta)/(scaleb·b[i]+shiftb)``, ``i``\ =1,2 … n


   The linearfrac function is implemented in the EP accuracy mode only,
   therefore no special values are defined for this function. If used in
   HA or LA mode, linearfrac sets the VM status code to
   ``oneapi::math::vm::status::accuracy_warning``. Correctness is guaranteed within the
   threshold limitations defined for each input parameter (see the table
   below); otherwise, the behavior is unspecified.


    



   .. container:: tablenoborder


      .. list-table::
         :header-rows: 1

         * - Threshold Limitations on Input Parameters
         * - ``2EMIN/2≤ |scalea| ≤ 2(EMAX-2)/2``
         * - ``2EMIN/2≤ |scaleb| ≤ 2(EMAX-2)/2``
         * - ``|shifta| ≤ 2EMAX-2``
         * - ``|shiftb| ≤ 2EMAX-2``
         * - ``2EMIN/2≤a[i] ≤ 2(EMAX-2)/2``
         * - ``2EMIN/2≤b[i] ≤ 2(EMAX-2)/2``
         * - ``a[i] ≠ - (shifta/scalea)*(1-δ1), |δ1| ≤ 21-(p-1)/2``
         * - ``b[i] ≠ - (shiftb/scaleb)*(1-δ2), |δ2| ≤ 21-(p-1)/2``




   ``EMIN``\ and ``EMAX`` are the minimum and maximum exponents and
   ``p`` is the number of significant bits (precision) for the
   corresponding data type according to the ANSI/IEEE Standard 754-2008
   ([:ref:`onemath_vm_bibliography`]):


   - for single precision\ ``EMIN`` = -126, ``EMAX`` = 127, ``p`` = 24


   - for double precision\ ``EMIN`` = -1022, ``EMAX`` = 1023, ``p`` =
      53


   The thresholds become less strict for common cases with scalea=0
   and/or scaleb=0:


   - ifscalea=0, there are no limitations for the values of ``a[i]``
      and shifta.


   - ifscaleb=0, there are no limitations for the values of ``b[i]``
      and shiftb.


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


   scalea
      Constant value for scaling multipliers of vector ``a``


   shifta
      Constant value for shifting addend of vector ``a``


   scaleb
      Constant value for scaling multipliers of vector ``b``


   shiftb
      Constant value for shifting addend of vector ``b``


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
      The pointer ``a`` to the 1st input vector of size ``n``.


   b
      The pointer ``b`` to the 2nd input vector of size ``n``.


   scalea
      Constant value for scaling multipliers of vector ``a``


   shifta
      Constant value for shifting addend of vector ``a``


   scaleb
      Constant value for scaling multipliers of vector ``b``


   shiftb
      Constant value for shifting addend of vector ``b``


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


