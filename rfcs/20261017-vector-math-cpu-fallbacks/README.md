# CPU Vector Math with `exp`

## Revision

| Date | Revision | Comments |
| --- | --- | --- |
| 20261125 | 1.0 | Initial version. |

## Motivation

Mathematical functions represent a significant part of AI and scientific
computing workloads.  Applications computing mathematical functions over long
arrays need a portable way to invoke CPU implementations through oneMath.

This RFC is motivated by the absence of an implementation of the VM
component. We believe there are areas of the existing VM specification that
require discussion too.

## Outline

1. [Introduction](#introduction)
2. [Proposal](#proposal)
3. [Changes in the Project](#changes-in-the-project)
4. [Validation](#validation)
5. [Examples](#examples)
6. [User impact](#user-impact)
7. [Open questions](#open-questions)

## Introduction

This RFC proposes an initial CPU implementation of the oneMath Vector Math
(VM) component. It is the first in a planned series of proposals covering
accuracy modes, reproducibility, function coverage and numerical validation.
We would appreciate feedback on the approach and scope described below.

## Proposal

We propose adding a CPU implementation of VM, starting with `exp` for `float`.
For simplicity, the initial implementation would target Linux, limiting
library-loading and platform-validation work to one operating system.
The initial implementation would offer three backends (also called providers):

- System libm (or better) as a generic fallback,
- MKL for an existing long-vector interface on
  [supported x86-64 systems][source-mkl-platforms], and
- [ArmPL][source-armpl] for optimized execution on AArch64.

It would also use oneMath's existing dispatch structure so that further
routines could follow the same approach.

`exp` is a popular math function, which is available in most math libraries,
but it is an arbitrary choice.  The proposed interface should primarily
demonstrate the public interface, provider selection and short-vector
adaptation on one function, with one generic and two vendor backends.

### Interface and behavior

The existing [VM specification][source-exp] is the target. The abbreviated
signature below illustrates the proposed array calls.

```cpp
namespace oneapi {
namespace math {
namespace vm {
template <typename T>
sycl::event exp(sycl::queue& queue, std::int64_t n,
                const T* input, T* output,
                const std::vector<sycl::event>& depends = {});
} // namespace vm
} // namespace math
} // namespace oneapi
```

Under this proposal, a call would compute `output[i] = exp(input[i])` for `n`
contiguous elements. It would submit work to a CPU queue after the supplied
dependencies and return an event for completion.

The prototype accepts ordinary host memory, host USM or shared USM.  USM
allocations would have to belong to the queue's context. In-place operation
would be supported, while partial overlap would initially be rejected.

We propose that a queue call selects ArmPL when available for an AArch64 CPU
queue, or MKL when available for a supported x86-64 CPU queue, otherwise system
libm. Callers could also select any provider explicitly through the existing
`backend_selector` interface. An unavailable explicit selection would raise an
exception.

Reusing the queue, event and contiguous-array interface allows future CPU
providers and accelerator backends to follow the same model.

### Floating-point environment

We propose that a CPU task temporarily use the default floating-point
environment, then restore the worker's previous state and `errno` on completion.
This would prevent settings left by earlier work on a reused thread from
affecting the result and avoid leaving changes to that thread's state.

### Generic backend

We propose to loop over generic scalar or short-vector calls for the
generic backend.  System-libm is a naive choice, that has a few advantages

- avoids another library dependency,
- has sufficient coverage for now, and
- is aiming for correctly rounded, not yet broadly available in distributions.

### Vendor backends

The MKL backend would call an existing long-vector interface such as
[`vsExp`][source-mkl-vsexp] for `float` arrays. This is a direct integration,
with array traversal and numerical special cases handled by the library.  The
wrapper would still integrate queue submission and completion, and preserve the
public API's array-length and in-place behavior.

The ArmPL backend would adapt [short-vector operations][source-armpl-math] to
the array interface, which may be practical as a basis for other vendors
lacking a long-vector interface.

### Other Considered Approaches

It is acknowledged that system-libm may not be practical if results or coverage
remain platform-dependent.  Therefore, a library dependency might be the price
to pay for portability, across platforms and devices, but also consistent/full
coverage.

Best alternatives would be [CORE-MATH](https://core-math.gitlabpages.inria.fr/)
or [LLVM libc](https://libc.llvm.org/math/). The former provides full coverage
today but lacks performance and portability, the latter lacks coverage today
but will provide best performance and cross-platform support and consistency in
the long term.

## Changes in the Project

The work would affect VM headers, CPU providers, dispatch integration,
build and installation rules, examples and tests. Implementation will be
reviewed separately; no release date is proposed.

### Implementation approach

The implementation would reuse oneMath's existing queue, event,
backend-selector and exception mechanisms. Synchronous calls to CPU
libraries would execute within tasks submitted to the queue, following
the precedent of the [OpenBLAS integration][source-openblas-level1].

A first implementation would support submission, event dependencies/completion
and regular oneMath/SYCL exception handling. Other features such as error
callbacks, VM status reporting and accuracy modes would be addressed later.

## Validation

Tests should cover accuracy, special values, vector tails, in-place calls, but
also dependencies, provider selection and fallback.

### Accuracy checks

We propose comparing all three providers against MPFR, initially using a
provisional test threshold of 4 Units in the Last Place (ULP) and exhaustively
searching the range of floats. This provisional numerical test threshold does
not establish support for a specified accuracy mode. Special inputs should be
injected into the test array, and matched against predefined results defined in
VM specification.

A numerical-validation RFC will propose the final error definition, reference
methodology and values/metrics to report.

### Other tests

If backends' public interfaces overlap, e.g. ArmPL and Libm, it may be a good
idea to check that loading vendor backends leaves global resolution of libm
symbol unchanged.

## Examples

For example, application code could use ordinary host memory on a CPU queue:

```cpp
sycl::queue queue{ sycl::cpu_selector{} };
std::vector<float> input{ -1.0f, 0.0f, 1.0f };
std::vector<float> output(input.size());
oneapi::math::vm::exp(queue, static_cast<std::int64_t>(input.size()),
                      input.data(), output.data()).wait_and_throw();
```

## User impact

User impact of this PR is limited since we are still in the enablement phase.
This is more of an opportunity to initiate a discussion and work towards a
first solution.

## Open questions

### Generic backend

- Is system-libm portable enough? Could be a no today, but ok next year.
- Should we accept a dependency for more portability and consistent accuracy?

### Vendor backends

- Should providers with scalar or short-vector interfaces temporarily be
  supported through array adapters?
- Or are vendors expected to provide a long-vector interface, similar to MKL?

### Follow-up work

Planned companion RFCs:

- Error handling, callbacks and VM status reporting
- Accuracy modes
- Reproducibility requirements
- Function coverage and support queries
- Numerical validation

[source-exp]: https://github.com/uxlfoundation/oneMath/blob/1308c5e979daf0c80e3d6ae61c1e2de5ad0f2e4a/docs/spec/domains/vm/exp.rst
[source-openblas-level1]: https://github.com/uxlfoundation/oneMath/blob/1308c5e979daf0c80e3d6ae61c1e2de5ad0f2e4a/src/blas/backends/openblas/openblas_level1.cxx
[source-mkl-vsexp]: https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/v-exp.html
[source-mkl-platforms]: https://www.intel.com/content/www/us/en/developer/articles/release-notes/onemkl/2026.html
[source-armpl]: https://developer.arm.com/Tools%20and%20Software/Arm%20Performance%20Libraries
[source-armpl-math]: https://support.arm.com/documentation/101004/2607/Arm-Performance-Libraries-Math-Functions?lang=en
