# oneAPI Math Kernel Library (oneMKL) Interfaces Examples 
oneAPI Math Kernel Library (oneMKL) Interfaces offers examples with the following routines: 
- blas: level3/gemm_usm  
- rng: uniform_usm  
- lapack: getrs_usm
- dft: complex_fwd_usm, real_fwd_usm
- sparse_blas: sparse_spmv_usm

Each routine has one run-time dispatching example and one compile-time dispatching example (which uses both mklcpu and cuda backends), located in `example/<$domain>/run_time_dispatching` and `example/<$domain>/compile_time_dispatching` subfolders, respectively.

To build examples, use cmake build option `-DBUILD_EXAMPLES=true`.  
Compile_time_dispatching will be built if `-DBUILD_EXAMPLES=true` and cuda backend is enabled, because the compile-time dispatching example runs on both mklcpu and cuda backends.
Run_time_dispatching will be built if `-DBUILD_EXAMPLES=true` and `-DBUILD_SHARED_LIBS=true`.

The example executable naming convention follows `example_<$domain>_<$routine>_<$backend>` for compile-time dispatching examples 
  or `example_<$domain>_<$routine>` for run-time dispatching examples. 
  E.g. `example_blas_gemm_usm_mklcpu_cublas `  `example_blas_gemm_usm`

## Example outputs (blas, rng, lapack, dft, sparse_blas)
  
## blas

Run-time dispatching examples with mklcpu backend
```
$ export ONEAPI_DEVICE_SELECTOR="opencl:cpu"
$ ./bin/example_blas_gemm_usm

########################################################################
# General Matrix-Matrix Multiplication using Unified Shared Memory Example:
#
# C = alpha * A * B + beta * C
#
# where A, B and C are general dense matrices and alpha, beta are
# floating point type precision scalars.
#
# Using apis:
#   gemm
#
# Using single precision (float) data type
#
# Device will be selected during runtime.
# The environment variable ONEAPI_DEVICE_SELECTOR can be used to specify
# available devices
#
########################################################################

Running BLAS GEMM USM example on CPU device.
Device name is: Intel(R) Core(TM) i7-6770HQ CPU @ 2.60GHz
Running with single precision real data type:

                GEMM parameters:
                        transA = trans, transB = nontrans
                        m = 45, n = 98, k = 67
                        lda = 103, ldB = 105, ldC = 106
                        alpha = 2, beta = 3

                Outputting 2x2 block of A,B,C matrices:

                        A = [ 0.340188, 0.260249, ...
                            [ -0.105617, 0.0125354, ...
                            [ ...


                        B = [ -0.326421, -0.192968, ...
                            [ 0.363891, 0.251295, ...
                            [ ...


                        C = [ 0.00698781, 0.525862, ...
                            [ 0.585167, 1.59017, ...
                            [ ...

BLAS GEMM USM example ran OK.

```
Run-time dispatching examples with mklgpu backend
```
$ export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
$ ./bin/example_blas_gemm_usm

########################################################################
# General Matrix-Matrix Multiplication using Unified Shared Memory Example:
#
# C = alpha * A * B + beta * C
#
# where A, B and C are general dense matrices and alpha, beta are
# floating point type precision scalars.
#
# Using apis:
#   gemm
#
# Using single precision (float) data type
#
# Device will be selected during runtime.
# The environment variable ONEAPI_DEVICE_SELECTOR can be used to specify
# available devices
#
########################################################################

Running BLAS GEMM USM example on GPU device.
Device name is: Intel(R) Iris(R) Pro Graphics 580 [0x193b]
Running with single precision real data type:

                GEMM parameters:
                        transA = trans, transB = nontrans
                        m = 45, n = 98, k = 67
                        lda = 103, ldB = 105, ldC = 106
                        alpha = 2, beta = 3

                Outputting 2x2 block of A,B,C matrices:

                        A = [ 0.340188, 0.260249, ...
                            [ -0.105617, 0.0125354, ...
                            [ ...


                        B = [ -0.326421, -0.192968, ...
                            [ 0.363891, 0.251295, ...
                            [ ...


                        C = [ 0.00698781, 0.525862, ...
                            [ 0.585167, 1.59017, ...
                            [ ...

BLAS GEMM USM example ran OK.
```
Compile-time dispatching example with both mklcpu and cublas backend

(Note that the mklcpu and cublas result matrices have a small difference. This is expected due to precision limitation of `float`)
```
./bin/example_blas_gemm_usm_mklcpu_cublas

########################################################################
# General Matrix-Matrix Multiplication using Unified Shared Memory Example:
#
# C = alpha * A * B + beta * C
#
# where A, B and C are general dense matrices and alpha, beta are
# floating point type precision scalars.
#
# Using apis:
#   gemm
#
# Using single precision (float) data type
#
# Running on both Intel CPU and Nvidia GPU devices
#
########################################################################

Running BLAS GEMM USM example
Running with single precision real data type on:
        CPU device: Intel(R) Core(TM) i9-7920X CPU @ 2.90GHz
        GPU device: TITAN RTX

                GEMM parameters:
                        transA = trans, transB = nontrans
                        m = 45, n = 98, k = 67
                        lda = 103, ldB = 105, ldC = 106
                        alpha = 2, beta = 3

                Outputting 2x2 block of A,B,C matrices:

                        A = [ 0.340188, 0.260249, ...
                            [ -0.105617, 0.0125354, ...
                            [ ...


                        B = [ -0.326421, -0.192968, ...
                            [ 0.363891, 0.251295, ...
                            [ ...


                        (CPU) C = [ 0.00698781, 0.525862, ...
                            [ 0.585167, 1.59017, ...
                            [ ...


                        (GPU) C = [ 0.00698793, 0.525862, ...
                            [ 0.585168, 1.59017, ...
                            [ ...

BLAS GEMM USM example ran OK on MKLCPU and CUBLAS

```
 
## lapack 
Run-time dispatching example with mklgpu backend:
```
$ export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
$ ./bin/example_lapack_getrs_usm

########################################################################
# LU Factorization and Solve Example:
#
# Computes LU Factorization A = P * L * U
# and uses it to solve for X in a system of linear equations:
#   AX = B
# where A is a general dense matrix and B is a matrix whose columns
# are the right-hand sides for the systems of equations.
#
# Using apis:
#   getrf and getrs
#
# Using single precision (float) data type
#
# Device will be selected during runtime.
# The environment variable ONEAPI_DEVICE_SELECTOR can be used to specify
# available devices
#
########################################################################

Running LAPACK getrs example on GPU device.
Device name is: Intel(R) Iris(R) Pro Graphics 580 [0x193b]
Running with single precision real data type:

                GETRF and GETRS parameters:
                        trans = nontrans
                        m = 23, n = 23, nrhs = 23
                        lda = 32, ldb = 32

                Outputting 2x2 block of A and X matrices:

                        A = [ 0.340188, 0.304177, ...
                            [ -0.105617, -0.343321, ...
                            [ ...


                        X = [ -1.1748, 1.84793, ...
                            [ 1.47856, 0.189481, ...
                            [ ...

LAPACK GETRS USM example ran OK
```

Compile-time dispatching example with both mklcpu and cusolver backend
```
$ ./bin/example_lapack_getrs_usm_mklcpu_cusolver

########################################################################
# LU Factorization and Solve Example:
#
# Computes LU Factorization A = P * L * U
# and uses it to solve for X in a system of linear equations:
#   AX = B
# where A is a general dense matrix and B is a matrix whose columns
# are the right-hand sides for the systems of equations.
#
# Using apis:
#   getrf and getrs
#
# Using single precision (float) data type
#
# Running on both Intel CPU and NVIDIA GPU devices
#
########################################################################

Running LAPACK GETRS USM example
Running with single precision real data type on:
        CPU device :Intel(R) Core(TM) i9-7920X CPU @ 2.90GHz
        GPU device :TITAN RTX

                GETRF and GETRS parameters:
                        trans = nontrans
                        m = 23, n = 23, nrhs = 23
                        lda = 32, ldb = 32

                Outputting 2x2 block of A,B,X matrices:

                        A = [ 0.340188, 0.304177, ...
                            [ -0.105617, -0.343321, ...
                            [ ...


                        (CPU) X = [ -1.1748, 1.84793, ...
                            [ 1.47856, 0.189481, ...
                            [ ...


                        (GPU) X = [ -1.1748, 1.84793, ...
                            [ 1.47856, 0.189481, ...
                            [ ...

LAPACK GETRS USM example ran OK on MKLCPU and CUSOLVER

```

## rng
Run-time dispatching example with mklgpu backend:
```
$ export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
$ ./bin/example_rng_uniform_usm

########################################################################
# Generate uniformly distributed random numbers with philox4x32x10
# generator example:
#
# Using APIs:
#   default_engine uniform
#
# Using single precision (float) data type
#
# Device will be selected during runtime.
# The environment variable ONEAPI_DEVICE_SELECTOR can be used to specify
# available devices
#
########################################################################

Running RNG uniform usm example on GPU device
Device name is: Intel(R) Iris(R) Pro Graphics 580 [0x193b]
Running with single precision real data type:
                generation parameters:
                        seed = 777, a = 0, b = 10
                Output of generator:
                        first 10 numbers of 1000:
8.52971 1.76033 6.04753 3.68079 9.04039 2.61014 3.75788 3.94859 7.93444 8.60436
Random number generator with uniform distribution ran OK

```

Compile-time dispatching example with both mklcpu and curand backend
```
$ ./bin/example_rng_uniform_usm_mklcpu_curand

########################################################################
# Generate uniformly distributed random numbers with philox4x32x10
# generator example:
#
# Using APIs:
#   default_engine uniform
#
# Using single precision (float) data type
#
# Running on both Intel CPU and Nvidia GPU devices
#
########################################################################

Running RNG uniform usm example
Running with single precision real data type:
        CPU device: Intel(R) Core(TM) i9-7920X CPU @ 2.90GHz
        GPU device: TITAN RTX
                generation parameters:
                        seed = 777, a = 0, b = 10
                Output of generator on CPU device:
                        first 10 numbers of 1000:
8.52971 1.76033 6.04753 3.68079 9.04039 2.61014 3.75788 3.94859 7.93444 8.60436
                Output of generator on GPU device:
                        first 10 numbers of 1000:
3.52971 6.76033 1.04753 8.68079 4.48229 0.501966 6.78265 8.99091 6.39516 9.67955
Random number generator example with uniform distribution ran OK on MKLCPU and CURAND

```

## dft

Compile-time dispatching example with MKLGPU backend

```none
$ ONEAPI_DEVICE_SELECTOR="level_zero:gpu" ./bin/example_dft_complex_fwd_buffer_mklgpu

########################################################################
# Complex out-of-place forward transform for Buffer API's example:
#
# Using APIs:
#   Compile-time dispatch API
#   Buffer forward complex out-of-place
#
# Using single precision (float) data type
#
# For Intel GPU with Intel MKLGPU backend.
#
# The environment variable ONEAPI_DEVICE_SELECTOR can be used to specify
# available devices
########################################################################

Running DFT Complex forward out-of-place buffer example
Using compile-time dispatch API with MKLGPU.
Running with single precision real data type on:
	GPU device :Intel(R) UHD Graphics 750 [0x4c8a]
DFT Complex USM example ran OK on MKLGPU
```

Runtime dispatching example with MKLGPU, cuFFT, rocFFT and portFFT backends:

```none
$ ONEAPI_DEVICE_SELECTOR="level_zero:gpu" ./bin/example_dft_real_fwd_usm

########################################################################
# DFT complex in-place forward transform with USM API example:
#
# Using APIs:
#   USM forward complex in-place
#   Run-time dispatch
#
# Using single precision (float) data type
#
# Device will be selected during runtime.
# The environment variable ONEAPI_DEVICE_SELECTOR can be used to specify
# available devices
#
########################################################################

Running DFT complex forward example on GPU device
Device name is: Intel(R) UHD Graphics 750 [0x4c8a]
Running with single precision real data type:
DFT example run_time dispatch
DFT example ran OK
```

```none
$ ONEAPI_DEVICE_SELECTOR="level_zero:gpu" ./bin/example_dft_real_fwd_usm

########################################################################
# DFT complex in-place forward transform with USM API example:
#
# Using APIs:
#   USM forward complex in-place
#   Run-time dispatch
#
# Using single precision (float) data type
#
# Device will be selected during runtime.
# The environment variable ONEAPI_DEVICE_SELECTOR can be used to specify
# available devices
#
########################################################################

Running DFT complex forward example on GPU device
Device name is: NVIDIA A100-PCIE-40GB
Running with single precision real data type:
DFT example run_time dispatch
DFT example ran OK
```

```none
$ ./bin/example_dft_real_fwd_usm

########################################################################
# DFT complex in-place forward transform with USM API example:
#
# Using APIs:
#   USM forward complex in-place
#   Run-time dispatch
#
# Using single precision (float) data type
#
# Device will be selected during runtime.
# The environment variable ONEAPI_DEVICE_SELECTOR can be used to specify
# available devices
#
########################################################################

Running DFT complex forward example on GPU device
Device name is: AMD Radeon PRO W6800
Running with single precision real data type:
DFT example run_time dispatch
DFT example ran OK
```

```none
$ LD_LIBRARY_PATH=lib/:$LD_LIBRARY_PATH ./bin/example_dft_real_fwd_usm
########################################################################
# DFT complex in-place forward transform with USM API example:
#
# Using APIs:
#   USM forward complex in-place
#   Run-time dispatch
#
# Using single precision (float) data type
#
# Device will be selected during runtime.
# The environment variable ONEAPI_DEVICE_SELECTOR can be used to specify
# available devices
#
########################################################################

Running DFT complex forward example on GPU device
Device name is: Intel(R) UHD Graphics 750
Running with single precision real data type:
DFT example run_time dispatch
Unsupported Configuration:
	oneMKL: dft/backends/portfft/commit: function is not implemented portFFT only supports complex to complex transforms
```

## sparse_blas

Run-time dispatching examples with mklcpu backend
```
$ export ONEAPI_DEVICE_SELECTOR="opencl:cpu"
$ ./bin/example_sparse_blas_spmv_usm

########################################################################
# Sparse Matrix-Vector Multiply Example:
#
# y = alpha * op(A) * x + beta * y
#
# where A is a sparse matrix in CSR format, x and y are dense vectors
# and alpha, beta are floating point type precision scalars.
#
# Using apis:
#   sparse::spmv
#
# Using single precision (float) data type
#
# Device will be selected during runtime.
# The environment variable ONEAPI_DEVICE_SELECTOR can be used to specify
# available devices
#
########################################################################

Running Sparse BLAS SPMV USM example on CPU device.
Device name is: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
Running with single precision real data type:

                sparse::spmv parameters:
                        transA = nontrans
                        nrows = 64
                        alpha = 1, beta = 0

                 sparse::spmv example passed
        Finished
Sparse BLAS SPMV USM example ran OK.
```

Run-time dispatching examples with mklgpu backend
```
$ export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
$ ./bin/example_sparse_blas_spmv_usm

########################################################################
# Sparse Matrix-Vector Multiply Example:
#
# y = alpha * op(A) * x + beta * y
#
# where A is a sparse matrix in CSR format, x and y are dense vectors
# and alpha, beta are floating point type precision scalars.
#
# Using apis:
#   sparse::spmv
#
# Using single precision (float) data type
#
# Device will be selected during runtime.
# The environment variable ONEAPI_DEVICE_SELECTOR can be used to specify
# available devices
#
########################################################################

Running Sparse BLAS SPMV USM example on GPU device.
Device name is: Intel(R) HD Graphics 530 [0x1912]
Running with single precision real data type:

                sparse::spmv parameters:
                        transA = nontrans
                        nrows = 64
                        alpha = 1, beta = 0

                 sparse::spmv example passed
        Finished
Sparse BLAS SPMV USM example ran OK.
```

Compile-time dispatching example with both mklcpu and cusparse backend
```
$ ./bin/sparse_blas_spmv_usm_mklcpu_cusparse

########################################################################
# Sparse Matrix-Vector Multiply Example:
#
# y = alpha * op(A) * x + beta * y
#
# where A is a sparse matrix in COO format, x and y are dense vectors
# and alpha, beta are floating point type precision scalars.
#
# Using apis:
#   sparse::spmv
#
# Using single precision (float) data type
#
# Running on both Intel CPU and Nvidia GPU devices
#
########################################################################

Running Sparse BLAS SPMV USM example on:
        CPU device: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
        GPU device: NVIDIA A100-PCIE-40GB
Running with single precision real data type:

                sparse::spmv parameters:
                        transA = nontrans
                        size = 8
                        alpha = 1, beta = 0

                 sparse::spmv example passed
        Finished

                sparse::spmv parameters:
                        transA = nontrans
                        size = 8
                        alpha = 1, beta = 0

                 sparse::spmv example passed
        Finished
Sparse BLAS SPMV USM example ran OK on MKLCPU and CUSPARSE.
```
