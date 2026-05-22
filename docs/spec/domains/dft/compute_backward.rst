.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_dft_compute_backward:

The ``compute_backward`` function templates
===========================================

The ``oneapi::math::dft::compute_backward`` function templates enable the
computation of backward DFT(s), as defined by a (committed)
``oneapi::math::dft::descriptor`` object, on user-provided data. These function
templates are declared in the ``oneapi::math::dft`` namespace; the usage of
prepended namespace specifiers ``oneapi::math::dft`` is omitted below for conciseness.

.. _onemath_dft_compute_backward_description:

.. rubric:: Description

Given a successfully-committed ``descriptor`` object whose configuration is not
inconsistent with backward DFT calculations, the backward DFT it defines can be
computed using a specialization of (some of) the ``compute_backward`` function
templates.

The ``compute_backward`` functions require a successfully-committed ``descriptor``
object and one, two or four "data container" arguments (depending on the
configuration of the ``descriptor`` object). If using USM allocations as data
containers, these functions may also be provided with an
``std::vector<sycl::event>`` object collecting dependencies to be observed by
the desired DFT calculations and return a ``sycl::event`` instance tracking the
progress of the enqueued DFT calculations.

.. note::
   The ``compute_backward`` functions may need to access the internals and
   private/protected members of (some) ``descriptor`` classes.  This could be
   done, for instance, by labeling them as friend functions to the
   ``descriptor`` :ref:`class template<onemath_dft_descriptor>`.

.. onemath_dft_compute_backward_buffer:

``compute_backward`` (Buffer versions)
--------------------------------------

.. rubric:: Syntax (in-place transform, except for complex descriptors with ``config_value::REAL_REAL`` for ``config_param::COMPLEX_STORAGE``)

.. code-block:: cpp

   namespace oneapi::math::dft {

      template <typename descriptor_type, typename data_type>
      void compute_backward(descriptor_type              &desc,
                            sycl::buffer<data_type, 1>   &inout);
   }


.. rubric:: Syntax (in-place transform, for complex descriptors with ``config_value::REAL_REAL`` for ``config_param::COMPLEX_STORAGE``)

.. code-block:: cpp

   namespace oneapi::math::dft {

      template <typename descriptor_type typename data_type>
      void compute_backward(descriptor_type              &desc,
                            sycl::buffer<data_type, 1>   &inout_re,
                            sycl::buffer<data_type, 1>   &inout_im);
   }


.. rubric:: Syntax (out-of-place transform, except for complex descriptors with ``config_value::REAL_REAL`` for ``config_param::COMPLEX_STORAGE``)

.. code-block:: cpp

   namespace oneapi::math::dft {
   
      template <typename descriptor_type, typename input_type, typename output_type>
      void compute_backward(descriptor_type              &desc,
                            sycl::buffer<input_type, 1>  &in,
                            sycl::buffer<output_type, 1> &out);
   }

.. rubric:: Syntax (out-of-place transform, for complex descriptors with ``config_value::REAL_REAL`` for ``config_param::COMPLEX_STORAGE``)

.. code-block:: cpp

   namespace oneapi::math::dft {

      template <typename descriptor_type, typename input_type, typename output_type>
      void compute_backward(descriptor_type              &desc,
                            sycl::buffer<input_type, 1>  &in_re,
                            sycl::buffer<input_type, 1>  &in_im,
                            sycl::buffer<output_type, 1> &out_re,
                            sycl::buffer<output_type, 1> &out_im);
   }

.. container:: section

   .. rubric:: Input Parameters

   ``desc``
      A fully-configured and committed ``descriptor`` object, whose
      configuration is not inconsistent with backward DFT calculations.

   ``inout``
      ``sycl::buffer`` object of sufficient capacity to store the elements
      defining all the relevant data sequences, as configured by ``desc``
      (configured for in-place operations and not with ``config_value::REAL_REAL``
      for ``config_param::COMPLEX_STORAGE``, if complex).

   ``inout_re``
      ``sycl::buffer`` object of sufficient capacity to store the elements
      defining the real parts of all the relevant data sequences, as configured
      by ``desc``. ``data_type`` must be ``float`` or ``double``,
      consistently with the ``desc``'s configuration value for
      ``config_param::PRECISION``. Only with complex descriptors configured for
      in-place operations with ``config_value::REAL_REAL`` for
      ``config_param::COMPLEX_STORAGE``.

   ``inout_im``
      ``sycl::buffer`` object of sufficient capacity to store the elements
      defining the imaginary parts of all the relevant data sequences, as
      configured by ``desc``. ``data_type`` must be ``float`` or ``double``,
      consistently with the ``desc``'s configuration value for
      ``config_param::PRECISION``. Only with complex descriptors configured for
      in-place operations with ``config_value::REAL_REAL`` for
      ``config_param::COMPLEX_STORAGE``.

   ``in``
      ``sycl::buffer`` object of sufficient capacity to store the elements
      defining all the relevant backward-domain data sequences, as
      configured by ``desc`` (configured for out-of-place operations and not
      with ``config_value::REAL_REAL`` for ``config_param::COMPLEX_STORAGE``,
      if complex).

   ``in_re``
      ``sycl::buffer`` object of sufficient capacity to store the elements
      defining the real parts of all the relevant backward-domain data sequences,
      as configured by ``desc``. Only with complex descriptors configured for
      out-of-place operations with ``config_value::REAL_REAL`` for
      ``config_param::COMPLEX_STORAGE``.

   ``in_im``
      ``sycl::buffer`` object of sufficient capacity to store the elements
      defining the imaginary parts of all the relevant backward-domain data
      sequences, as configured by ``desc``. Only with complex descriptors
      configured for out-of-place operations with ``config_value::REAL_REAL``
      for ``config_param::COMPLEX_STORAGE``.

.. container:: section

   .. rubric:: Output Parameters

   ``inout``
      ``sycl::buffer`` object of sufficient capacity to store the elements
      defining all the relevant data sequences, as configured by ``desc``
      (configured for in-place operations and not with ``config_value::REAL_REAL``
      for ``config_param::COMPLEX_STORAGE``, if complex).

   ``inout_re``
      ``sycl::buffer`` object of sufficient capacity to store the elements
      defining the real parts of all the relevant data sequences, as configured
      by ``desc``. ``data_type`` must be ``float`` or ``double``,
      consistently with the ``desc``'s configuration value for
      ``config_param::PRECISION``. Only with complex descriptors configured for
      in-place operations with ``config_value::REAL_REAL`` for
      ``config_param::COMPLEX_STORAGE``.

   ``inout_im``
      ``sycl::buffer`` object of sufficient capacity to store the elements
      defining the imaginary parts of all the relevant data sequences, as
      configured by ``desc``. ``data_type`` must be ``float`` or ``double``,
      consistently with the ``desc``'s configuration value for
      ``config_param::PRECISION``. Only with complex descriptors configured for
      in-place operations with ``config_value::REAL_REAL`` for
      ``config_param::COMPLEX_STORAGE``.

   ``out``
      ``sycl::buffer`` object of sufficient capacity to store the elements
      defining all the relevant forward-domain data sequences, as configured by
      ``desc`` (configured for out-of-place operations and not with
      ``config_value::REAL_REAL`` for ``config_param::COMPLEX_STORAGE``, if complex).

   ``out_re``
      ``sycl::buffer`` object of sufficient capacity to store the elements
      defining the real parts of all the relevant forward-domain data sequences,
      as configured by ``desc``. Only with complex descriptors configured for
      out-of-place operations with ``config_value::REAL_REAL`` for
      ``config_param::COMPLEX_STORAGE``.

   ``out_im``
      ``sycl::buffer`` object of sufficient capacity to store the elements
      defining the imaginary parts of all the relevant forward-domain data
      sequences, as configured by ``desc``. Only with complex descriptors configured
      for out-of-place operations with ``config_value::REAL_REAL`` for
      ``config_param::COMPLEX_STORAGE``.

.. container:: section

   .. rubric:: Throws

   The ``compute_backward`` functions shall throw the following
   :ref:`exception<onemath_common_exceptions>` if the associated condition is
   detected. An implementation may throw additional implementation-specific
   exception(s) in case of error conditions not covered here:

   ``oneapi::math::invalid_argument()``
      If ``desc`` is invalid. For instance, if its configuration value
      associated with configuration parameter ``config_param::COMMIT_STATUS`` is
      not ``config_param::COMMITTED``.

.. onemath_dft_compute_backward_usm:

``compute_backward`` (USM versions)
-----------------------------------

.. rubric:: Syntax (in-place transform, except for complex descriptors with ``config_value::REAL_REAL`` for ``config_param::COMPLEX_STORAGE``)

.. code-block:: cpp

   namespace oneapi::math::dft {

      template <typename descriptor_type, typename data_type>
      sycl::event compute_backward(descriptor_type                &desc,
                                   data_type                      *inout,
                                   const std::vector<sycl::event> &dependencies = {});
   }

.. rubric:: Syntax (in-place transform, for complex descriptors with ``config_value::REAL_REAL`` for ``config_param::COMPLEX_STORAGE``)

.. code-block:: cpp

   namespace oneapi::math::dft {

      template <typename descriptor_type, typename data_type>
      sycl::event compute_backward(descriptor_type                &desc,
                                   data_type                      *inout_re,
                                   data_type                      *inout_im,
                                   const std::vector<sycl::event> &dependencies = {});
   }

.. rubric:: Syntax (out-of-place transform, except for complex descriptors with ``config_value::REAL_REAL`` for ``config_param::COMPLEX_STORAGE``)

.. code-block:: cpp

   namespace oneapi::math::dft {

      template <typename descriptor_type, typename input_type, typename output_type>
      sycl::event compute_backward(descriptor_type                &desc,
                                   input_type                     *in,
                                   output_type                    *out,
                                   const std::vector<sycl::event> &dependencies = {});
   }

.. rubric:: Syntax (out-of-place transform, for complex descriptors with ``config_value::REAL_REAL`` for ``config_param::COMPLEX_STORAGE``)

.. code-block:: cpp

   namespace oneapi::math::dft {

      template <typename descriptor_type, typename input_type, typename output_type>
      sycl::event compute_backward(descriptor_type                &desc,
                                   input_type                     *in_re,
                                   input_type                     *in_im,
                                   output_type                    *out_re,
                                   output_type                    *out_im,
                                   const std::vector<sycl::event> &dependencies = {});
   }

.. container:: section

   .. rubric:: Input Parameters

   ``desc``
      A fully-configured and committed ``descriptor`` object, whose
      configuration is not inconsistent with backward DFT calculations.

   ``inout``
      Pointer to USM allocation of sufficient capacity to store the elements
      defining all the relevant data sequences, as configured by ``desc``
      (configured for in-place operations and not with ``config_value::REAL_REAL``
      for ``config_param::COMPLEX_STORAGE``, if complex).
      

   ``inout_re``
      Pointer to USM allocation of sufficient capacity to store the elements
      defining the real parts of all the relevant data sequences, as configured
      by ``desc``. ``data_type`` must be ``float`` or ``double``, consistently
      with the ``desc``'s configuration value for ``config_param::PRECISION``.
      Only with complex descriptors configured for in-place operations with
      ``config_value::REAL_REAL`` for ``config_param::COMPLEX_STORAGE``.

   ``inout_im``
      Pointer to USM allocation of sufficient capacity to store the elements
      defining the imaginary parts of all the relevant data sequences, as
      configured by ``desc``. ``data_type`` must be ``float`` or ``double``,
      consistently with the ``desc``'s configuration value for
      ``config_param::PRECISION``. Only with complex descriptors configured for
      in-place operations with ``config_value::REAL_REAL`` for
      ``config_param::COMPLEX_STORAGE``.

   ``in``
      Pointer to USM allocation of sufficient capacity to store the elements
      defining all the relevant backward-domain data sequences, as configured by
      ``desc`` (configured for out-of-place operations and not with
      ``config_value::REAL_REAL`` for ``config_param::COMPLEX_STORAGE``, if complex).

   ``in_re``
      Pointer to USM allocation of sufficient capacity to store the elements
      defining the real parts of all the relevant backward-domain data sequences,
      as configured by ``desc``. Only with complex descriptors configured for
      out-of-place operations with ``config_value::REAL_REAL`` for
      ``config_param::COMPLEX_STORAGE``.

   ``in_im``
      Pointer to USM allocation of sufficient capacity to store the elements
      defining the imaginary parts of all the relevant backward-domain data
      sequences, as configured by ``desc``. Only with complex descriptors
      configured for out-of-place operations with ``config_value::REAL_REAL``
      for ``config_param::COMPLEX_STORAGE``.

   ``dependencies``
      An ``std::vector<sycl::event>`` object collecting the events returned by
      previously enqueued tasks that must be finished before the desired
      transform can be calculated.

.. container:: section

   .. rubric:: Output Parameters

   ``inout``
      Pointer to USM allocation of sufficient capacity to store the elements
      defining all the relevant data sequences, as configured by ``desc``
      (configured for in-place operations and not with ``config_value::REAL_REAL``
      for ``config_param::COMPLEX_STORAGE``, if complex).

   ``inout_re``
      Pointer to USM allocation of sufficient capacity to store the elements
      defining the real parts of all the relevant data sequences, as configured
      by ``desc``. ``data_type`` must be ``float`` or ``double``, consistently
      with the ``desc``'s configuration value for ``config_param::PRECISION``.
      Only with complex descriptors configured for in-place operations with
      ``config_value::REAL_REAL`` for ``config_param::COMPLEX_STORAGE``.

   ``inout_im``
      Pointer to USM allocation of sufficient capacity to store the elements
      defining the imaginary parts of all the relevant data sequences, as
      configured by ``desc``. ``data_type`` must be ``float`` or ``double``,
      consistently with the ``desc``'s configuration value for
      ``config_param::PRECISION``. Only with complex descriptors configured for
      in-place operations with ``config_value::REAL_REAL`` for
      ``config_param::COMPLEX_STORAGE``.

   ``out``
      Pointer to USM allocation of sufficient capacity to store the elements
      defining all the relevant forward-domain data sequences, as configured by
      ``desc`` (configured for out-of-place operations and not with
      ``config_value::REAL_REAL`` for ``config_param::COMPLEX_STORAGE``, if complex).

   ``out_re``
      Pointer to USM allocation of sufficient capacity to store the elements
      defining the real parts of all the relevant forward-domain data sequences,
      as configured by ``desc``. Only with complex descriptors configured for
      out-of-place operations with ``config_value::REAL_REAL`` for
      ``config_param::COMPLEX_STORAGE``.

   ``out_im``
      Pointer to USM allocation of sufficient capacity to store the elements
      defining the imaginary parts of all the relevant forward-domain data
      sequences, as configured by ``desc``. Only with complex descriptors
      configured for out-of-place operations with ``config_value::REAL_REAL``
      for ``config_param::COMPLEX_STORAGE``.

.. container:: section

   .. rubric:: Throws

   The ``compute_backward`` functions shall throw the following
   :ref:`exception<onemath_common_exceptions>` if the associated condition is
   detected. An implementation may throw additional implementation-specific
   exception(s) in case of error conditions not covered here:

   ``oneapi::math::invalid_argument()``
      If ``desc`` is invalid. For instance, if its configuration value
      associated with configuration parameter ``config_param::COMMIT_STATUS`` is
      not ``config_param::COMMITTED``. It will also be thrown if any required
      input/output pointer is ``nullptr``.

.. container:: section

   .. rubric:: Return Values

   These functions return a ``sycl::event`` object that enables tracking
   progress of the backward DFT, and that can be passed as a dependency to other
   routines, which may depend on the result of the backward transform(s) before
   proceeding with other operations.

**Parent topic:** :ref:`onemath_dft`
