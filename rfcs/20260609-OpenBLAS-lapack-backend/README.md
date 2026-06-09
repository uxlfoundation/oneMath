# Design Document

### Revision

| Date     | Revision | Comments        |
| -------- | -------- | --------------- |
| 20260609 | 1.0      | Initial version |

---

## Motivation

We aim to integrate OpenBLAS LAPACK support as a backend for the LAPACK domain in the oneAPI Math Library (oneMath) to provide a high-performance, open-source alternative to existing LAPACK implementations. The integration will target CPU devices on x86 and ARM architectures, ensuring optimized performance across supported platforms.

### What problem do we solve?

Currently, oneMath employs Intel oneMKL for x86 platforms, Arm Performance Libraries (ArmPL) for ARM architectures, and Netlib LAPACK as a reference backend. While these libraries provide LAPACK functionality, vendor-specific backends are tied to particular hardware ecosystems, and Netlib LAPACK primarily serves as a correctness reference rather than a performance-oriented implementation.

To improve performance portability and broaden platform support, we propose integrating OpenBLAS LAPACK as an additional backend. OpenBLAS provides LAPACK functionality through the LAPACKE interface and includes architecture-aware optimizations for both x86 and ARM CPUs. Its integration will strengthen oneMath's portability, performance, and accessibility across diverse CPU environments.

### What is the user impact if we don't do this?

Users will face reduced backend flexibility, limited portability, and continued dependence on vendor-specific LAPACK implementations, potentially restricting adoption of oneMath in open-source and vendor-neutral HPC environments.

### What is the timeline?

Implementation, build system integration, validation, and testing are targeted for completion within six months, with initial focus on LAPACK functionality and support across AdaptiveCpp and DPC++ compilers.

---

## Outline

1. [Introduction](#introduction)
2. [Proposal](#proposal)
3. [Changes in the Project](#changes-in-the-project)

---

## Introduction

The oneAPI Math Library (oneMath) provides a unified interface for high-performance mathematical operations across heterogeneous platforms. It currently supports LAPACK functionality through Intel oneMKL, Arm Performance Libraries (ArmPL), and Netlib LAPACK backends.

These backends provide support for dense linear algebra operations such as:

* Linear system solvers
* LU, QR, and Cholesky factorizations
* Eigenvalue and eigenvector computations
* Singular value decomposition (SVD)
* Least-squares solvers

While oneMKL and ArmPL provide highly optimized implementations, they are closely associated with specific hardware ecosystems. Netlib LAPACK serves as a portable reference implementation but does not provide the performance required for modern HPC workloads.

To address these limitations, this proposal introduces OpenBLAS LAPACK support as an additional backend for the LAPACK domain. OpenBLAS provides a widely adopted, open-source LAPACK implementation through LAPACKE and supports both x86 and ARM architectures. Its integration into oneMath will improve portability, reduce dependence on vendor-specific libraries, and expand backend flexibility while preserving the existing oneMath programming model.

---

## Proposal

This proposal outlines the integration of OpenBLAS LAPACK support as a backend for the oneAPI Math Library (oneMath).

The primary objective is to enhance oneMath's modular and extensible design by enabling support for OpenBLAS LAPACK through a SYCL-compatible interface. This integration aims to improve performance portability across diverse CPU architectures by leveraging OpenBLAS's optimized implementations while preserving a unified, vendor-agnostic programming model.

The proposed integration introduces OpenBLAS LAPACK support within the oneMath library through a structured enhancement of its interface and implementation layers. The key steps in this process include:

### LAPACK Wrapper Header Definition

New headers will be added under the `include/` directory to declare OpenBLAS LAPACK-specific interfaces. These headers will conform to the existing oneMath API structure and act as the entry point for invoking OpenBLAS-based LAPACK functionality.

### Compile-Time Interface Binding

The compile-time dispatch layer will be updated to route LAPACK API calls directly to OpenBLAS LAPACK implementations. This ensures type-safe, efficient linkage without runtime overhead, adhering to oneMath's modular dispatch model.

### SYCL-Compatible Wrapper Implementation

The `src/` directory will contain SYCL-compliant wrappers that internally call OpenBLAS LAPACK routines through the LAPACKE interface. These wrappers will translate oneMath abstractions into formats compatible with OpenBLAS while maintaining portability and compatibility with supported SYCL implementations.

### CMake Build System Integration

The build configuration will be extended to detect and optionally enable the OpenBLAS LAPACK backend. This includes defining CMake options, linking rules, dependency resolution, and backend registration to support seamless integration into the oneMath build process.

### Unit Test Integration

The testing infrastructure will be expanded to validate the correctness of OpenBLAS LAPACK-based implementations. Tests will be incorporated into the existing functional and unit test pipelines to ensure compatibility, numerical correctness, and overall reliability.

---

## Changes in the Project

### Header-Level Integration

New headers will be introduced to declare SYCL-facing function prototypes for the OpenBLAS LAPACK backend. These will follow oneMath's modular design principles and enable seamless dispatch integration through backend-specific routing support.

### Backend Implementation

The backend layer will implement wrapper functions that interface between oneMath LAPACK APIs and native OpenBLAS LAPACK routines.

These implementations will:

* Support existing oneMath LAPACK interfaces.
* Translate oneMath LAPACK calls into LAPACKE-compatible invocations.
* Handle matrix layouts and parameter conversions.
* Manage workspace requirements where applicable.
* Maintain portability across supported platforms.
* Preserve compatibility with existing SYCL execution models.

### Build System Enhancements

The CMake build infrastructure will be extended to support OpenBLAS LAPACK as an optional backend. This includes:

* Introducing dedicated build flags.
* Defining linkage behavior.
* Locating OpenBLAS and LAPACKE libraries during configuration.
* Registering backend-specific build targets.
* Supporting flexible deployment across multiple platforms.

### Testing and Validation

The test infrastructure will be updated to validate OpenBLAS LAPACK backend functionality. This includes:

* Enabling conditional compilation.
* Adjusting test linkage.
* Verifying backend dispatch behavior.
* Ensuring numerical correctness.
* Exercising existing LAPACK functionality through unit and functional test workflows.
* Validation using supported SYCL implementations, including AdaptiveCpp and DPC++.

---

## Timeline

The proposed work—including implementation, integration, build configuration, and testing—is expected to be completed within six months.

---

## Examples

The proposed integration of OpenBLAS LAPACK does not introduce any changes to the user-facing API. Developers continue to use oneMath's standardized LAPACK interfaces without requiring modifications to their application code.

For example, LU factorization can continue to be invoked through the existing oneMath interface:

```cpp
getrf_done = oneapi::math::lapack::getrf(
    queue,
    m,
    n,
    a,
    lda,
    ipiv,
    scratchpad,
    scratchpad_size);
```

This interface remains consistent regardless of the underlying backend. When OpenBLAS LAPACK is enabled and selected, the computation is transparently routed through the OpenBLAS LAPACKE implementation, allowing users to benefit from optimized performance while preserving portability.

---

## User Impact

### Portability

Applications can leverage OpenBLAS LAPACK as a backend without requiring modifications to existing user code, ensuring consistent behavior across supported x86 and ARM platforms.

### Performance

The integration benefits from OpenBLAS's optimized BLAS kernels and LAPACK implementations, including architecture-specific optimizations available on modern processors.

### Open-Source Accessibility

Users gain access to a fully open-source LAPACK backend, reducing dependency on proprietary software stacks and simplifying deployment in HPC, research, and academic environments.

### Modularity

OpenBLAS LAPACK support can be enabled through a configurable option within the oneMath build system, promoting clean separation of concerns and backend extensibility.

---
