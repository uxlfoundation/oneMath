.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _matrix-storage:

Matrix Storage
==============


.. container::


   The oneMath BLAS and LAPACK routines for DPC++ use several matrix and
   vector storage formats. These are the same formats used in
   traditional Fortran BLAS/LAPACK. LAPACK routines require column
   major layout.

   .. container:: section

      .. rubric:: General Matrix
         :name: general-matrix
         :class: sectiontitle

      A general matrix ``A`` of ``m`` rows and ``n`` columns with
      leading dimension ``lda`` is represented as a one dimensional
      array ``a`` of size of at least ``lda`` \* ``n`` if column major
      layout is used and at least ``lda`` \* ``m`` if row major layout
      is used.  Before entry in any BLAS function using a general
      matrix, the leading ``m`` by ``n`` part of the array ``a`` must
      contain the matrix ``A``. For column (respectively row) major
      layout, the elements of each column (respectively row) are
      contiguous in memory while the elements of each row
      (respectively column) are at distance ``lda`` from the element
      in the same row (respectively column) and the previous column
      (respectively row).

      Visually, the matrix

      .. math::
            
         A = \begin{bmatrix}
             A_{11} & A_{12} & A_{13} & \ldots & A_{1n}\\
             A_{21} & A_{22} & A_{23} & \ldots & A_{2n}\\
             A_{31} & A_{32} & A_{33} & \ldots & A_{3n}\\
             \vdots & \vdots & \vdots & \ddots & \vdots\\
             A_{m1} & A_{m2} & A_{m3} & \ldots & A_{mn}
             \end{bmatrix}

      is stored in memory as an array

      - For column major layout,

      .. math::
         
         \scriptstyle a = 
            [\underbrace{\underbrace{A_{11},A_{21},A_{31},...,A_{m1},*,...,*}_\text{lda},
                         \underbrace{A_{12},A_{22},A_{32},...,A_{m2},*,...,*}_\text{lda},
                         ...,
                         \underbrace{A_{1n},A_{2n},A_{3n},...,A_{mn},*,...,*}_\text{lda}}
                         _\text{lda x n}]
      
      - For row major layout,

      .. math::
         
         \scriptstyle a = 
            [\underbrace{\underbrace{A_{11},A_{12},A_{13},...,A_{1n},*,...,*}_\text{lda},
                         \underbrace{A_{21},A_{22},A_{23},...,A_{2n},*,...,*}_\text{lda},
                         ...,
                         \underbrace{A_{m1},A_{m2},A_{m3},...,A_{mn},*,...,*}_\text{lda}}
                         _\text{m x lda}]

   .. container:: section

      .. rubric:: Triangular Matrix
         :name: triangular-matrix
         :class: sectiontitle

      A triangular matrix ``A`` of ``n`` rows and ``n`` columns with
      leading dimension ``lda`` is represented as a one dimensional
      array ``a``, of a size of at least ``lda`` \* ``n``. When column
      (respectively row) major layout is used, the elements of each
      column (respectively row) are contiguous in memory while the
      elements of each row (respectively column) are at distance
      ``lda`` from the element in the same row (respectively column)
      and the previous column (respectively row).

      Before entry in any BLAS function using a triangular matrix,

      -  If ``upper_lower = uplo::upper``, the leading ``n`` by ``n``
         upper triangular part of the array ``a`` must contain the upper
         triangular part of the matrix ``A``. The strictly lower
         triangular part of the array ``a`` is not referenced. In other
         words, the matrix

         .. math::

            A = \begin{bmatrix}
                A_{11} & A_{12} & A_{13} & \ldots & A_{1n}\\
                *      & A_{22} & A_{23} & \ldots & A_{2n}\\
                *      & *      & A_{33} & \ldots & A_{3n}\\
                \vdots & \vdots & \vdots & \ddots & \vdots\\
                *      & *      & *      & \ldots & A_{nn}
                \end{bmatrix}

         is stored in memory as the array

         - For column major layout,

         .. math::
            
            \scriptstyle a = 
               [\underbrace{\underbrace{A_{11},*,...,*}_\text{lda},
                            \underbrace{A_{12},A_{22},*,...,*}_\text{lda},
                            ...,
                            \underbrace{A_{1n},A_{2n},A_{3n},...,A_{nn},*,...,*}_\text{lda}}
                            _\text{lda x n}]

         - For row major layout,

         .. math::
            
            \scriptstyle a = 
               [\underbrace{\underbrace{A_{11},A_{12},A_{13},...,A_{1n},*,...,*}_\text{lda},
                            \underbrace{*,A_{22},A_{23},...,A_{2n},*,...,*}_\text{lda},
                            ...,
                            \underbrace{*,...,*,A_{nn},*,...,*}_\text{lda}}
                            _\text{lda x n}]

      -  If ``upper_lower = uplo::lower``, the leading ``n`` by ``n``
         lower triangular part of the array ``a`` must contain the lower
         triangular part of the matrix ``A``. The strictly upper
         triangular part of the array ``a`` is not referenced. That is,
         the matrix

         .. math::

            A = \begin{bmatrix}
                A_{11} & *      & *      & \ldots & *     \\
                A_{21} & A_{22} & *      & \ldots & *     \\
                A_{31} & A_{32} & A_{33} & \ldots & *     \\
                \vdots & \vdots & \vdots & \ddots & \vdots\\
                A_{n1} & A_{n2} & A_{n3} & \ldots & A_{nn}
                \end{bmatrix}

         is stored in memory as the array

         - For column major layout,
      
         .. math::
                  
            \scriptstyle a = 
               [\underbrace{\underbrace{A_{11},A_{21},A_{31},..,A_{n1},*,...,*}_\text{lda},
                            \underbrace{*,A_{22},A_{32},...,A_{n2},*,...,*}_\text{lda},
                            ...,
                            \underbrace{*,...,*,A_{nn},*,...,*}_\text{lda}}
                            _\text{lda x n}]

         - For row major layout,

         .. math::
                  
            \scriptstyle a = 
               [\underbrace{\underbrace{A_{11},*,...,*}_\text{lda},
                            \underbrace{A_{21},A_{22},*,...,*}_\text{lda},
                            ...,
                            \underbrace{A_{n1},A_{n2},A_{n3},...,A_{nn},*,...,*}_\text{lda}}
                            _\text{lda x n}]

   .. container:: section

      .. rubric:: Band Matrix
         :name: band-matrix
         :class: sectiontitle

      A general band matrix ``A`` of ``m`` rows and ``n`` columns with
      ``kl`` sub-diagonals, ``ku`` super-diagonals, and leading
      dimension ``lda`` is represented as a one dimensional array
      ``a`` of a size of at least ``lda`` \* ``n`` (respectively
      ``lda`` \* ``m``) if column (respectively row) major layout is
      used.

      Before entry in any BLAS function using a general band matrix,
      the leading (``kl`` + ``ku`` + 1\ ``)`` by ``n`` (respectively
      ``m``) part of the array ``a`` must contain the matrix
      ``A``. This matrix must be supplied column-by-column
      (respectively row-by-row), with the main diagonal of the matrix
      in row ``ku`` (respectively ``kl``) of the array (0-based
      indexing), the first super-diagonal starting at position 1
      (respectively 0) in row (``ku`` - 1) (respectively column
      (``kl`` + 1)), the first sub-diagonal starting at position 0
      (respectively 1) in row (``ku`` + 1) (respectively column
      (``kl`` - 1)), and so on. Elements in the array ``a`` that do
      not correspond to elements in the band matrix (such as the top
      left ``ku`` by ``ku`` triangle) are not referenced.

      Visually, the matrix ``A``

      .. math::

         A = \left[\begin{smallmatrix}
             A_{11}     & A_{12}     & A_{13}     & \ldots & A_{1,ku+1} & *          & \ldots     & \ldots     & \ldots & \ldots    & \ldots    & *         \\
             A_{21}     & A_{22}     & A_{23}     & A_{24} & \ldots     & A_{2,ku+2} & *          & \ldots     & \ldots & \ldots    & \ldots    & *         \\
             A_{31}     & A_{32}     & A_{33}     & A_{34} & A_{35}     & \ldots     & A_{3,ku+3} & *          & \ldots & \ldots    & \ldots    & *         \\
             \vdots     & A_{42}     & A_{43}     & \ddots & \ddots     & \ddots     & \ddots     & \ddots     & *      & \ldots    & \ldots    & \vdots    \\
             A_{kl+1,1} & \vdots     & A_{53}     & \ddots & \ddots     & \ddots     & \ddots     & \ddots     & \ddots & *         & \ldots    & \vdots    \\
             *          & A_{kl+2,2} & \vdots     & \ddots & \ddots     & \ddots     & \ddots     & \ddots     & \ddots & \ddots    & \ddots    & \vdots    \\
             \vdots     & *          & A_{kl+3,3} & \ddots & \ddots     & \ddots     & \ddots     & \ddots     & \ddots & \ddots    & \ddots    & *         \\
             \vdots     & \vdots     & *          & \ddots & \ddots     & \ddots     & \ddots     & \ddots     & \ddots & \ddots    & \ddots    & A_{n-ku,n}\\
             \vdots     & \vdots     & \vdots     & *      & \ddots     & \ddots     & \ddots     & \ddots     & \ddots & \ddots    & \ddots    & \vdots    \\
             \vdots     & \vdots     & \vdots     & \vdots & *          & \ddots     & \ddots     & \ddots     & \ddots & \ddots    & \ddots    & A_{m-2,n} \\
             \vdots     & \vdots     & \vdots     & \vdots & \vdots     & \ddots     & \ddots     & \ddots     & \ddots & \ddots    & \ddots    & A_{m-1,n} \\
             *          & *          & *          & \ldots & \ldots     & \ldots     & *          & A_{m,m-kl} & \ldots & A_{m,n-2} & A_{m,n-1} & A_{m,n} 
             \end{smallmatrix}\right]


      is stored in memory as an array

      - For column major layout,
        
      .. math::
               
         \scriptscriptstyle a = 
            [\underbrace{
             \underbrace{\underbrace{*,...,*}_\text{ku},A_{11}, A_{12},...,A_{min(kl+1,m),1},*,...,*}_\text{lda},
             \underbrace{\underbrace{*,...,*}_\text{ku-1},A_{max(1,2-ku),2},...,A_{min(kl+2,m),2},*,...*}_\text{lda},
             ...,
             \underbrace{\underbrace{*,...,*}_\text{max(0,ku-n+1)},A_{max(1,n-ku),n},...,A_{min(kl+n,m),n},*,...*}_\text{lda}
             }_\text{lda x n}]


      - For row major layout,

      .. math::
               
         \scriptscriptstyle a = 
            [\underbrace{
             \underbrace{\underbrace{*,...,*}_\text{kl},A_{11}, A_{12},...,A_{1,min(ku+1,n)},*,...,*}_\text{lda},
             \underbrace{\underbrace{*,...,*}_\text{kl-1},A_{2,max(1,2-kl)},...,A_{2,min(ku+2,n)},*,...*}_\text{lda},
             ...,
             \underbrace{\underbrace{*,...,*}_\text{max(0,kl-m+1)},A_{m,max(1,m-kl)},...,A_{m,min(ku+m,n)},*,...*}_\text{lda}
             }_\text{lda x m}]

      The following program segment transfers a band matrix from
      conventional full matrix storage (variable ``matrix``, with
      leading dimension ``ldm``) to band storage (variable ``a``, with
      leading dimension ``lda``):


      - Using matrices stored with column major layout,
        
      ::

         for (j = 0; j < n; j++) {
             k = ku – j;
             for (i = max(0, j – ku); i < min(m, j + kl + 1); i++) {
                 a[(k + i) + j * lda] = matrix[i + j * ldm];
             }
         }

      - Using matrices stored with row major layout,

      ::

         for (i = 0; i < m; i++) {
             k = kl – i;
             for (j = max(0, i – kl); j < min(n, i + ku + 1); j++) {
                 a[(k + j) + i * lda] = matrix[j + i * ldm];
             }
         }
        

   .. container:: section

      .. rubric:: Triangular Band Matrix
         :name: triangular-band-matrix
         :class: sectiontitle

      A triangular band matrix ``A`` of ``n`` rows and ``n`` columns
      with ``k`` sub/super-diagonals and leading dimension ``lda`` is
      represented as a one dimensional array ``a`` of size at least
      ``lda`` \* ``n``.

      Before entry in any BLAS function using a triangular band matrix,


      - If ``upper_lower = uplo::upper``, the leading (``k`` + 1) by ``n``
        part of the array ``a`` must contain the upper
        triangular band part of the matrix ``A``. When using column
        major layout, this matrix must be supplied column-by-column
        (respectively row-by-row) with the main diagonal of the
        matrix in row (``k``) (respectively column 0) of the array,
        the first super-diagonal starting at position 1
        (respectively 0) in row (``k`` - 1) (respectively column 1),
        and so on. Elements in the array ``a`` that do not correspond
        to elements in the triangular band matrix (such as the top
        left ``k`` by ``k`` triangle) are not referenced.

        Visually, the matrix

        .. math::

           A = \left[\begin{smallmatrix}
               A_{11}     & A_{12}     & A_{13}     & \ldots & A_{1,k+1} & *          & \ldots      & \ldots     & \ldots & \ldots    & \ldots    & *         \\
               *          & A_{22}     & A_{23}     & A_{24} & \ldots     & A_{2,k+2} & *           & \ldots     & \ldots & \ldots    & \ldots    & *         \\
               \vdots     & *          & A_{33}     & A_{34} & A_{35}     & \ldots     & A_{3,k+3}  & *          & \ldots & \ldots    & \ldots    & *         \\
               \vdots     & \vdots     & *          & \ddots & \ddots     & \ddots     & \ddots     & \ddots     & *      & \ldots    & \ldots    & \vdots    \\
               \vdots     & \vdots     & \vdots     & \ddots & \ddots     & \ddots     & \ddots     & \ddots     & \ddots & *         & \ldots    & \vdots    \\
               \vdots     & \vdots     & \vdots     & \vdots & \ddots     & \ddots     & \ddots     & \ddots     & \ddots & \ddots    & \ddots    & \vdots    \\
               \vdots     & \vdots     & \vdots     & \vdots & \vdots     & \ddots     & \ddots     & \ddots     & \ddots & \ddots    & \ddots    & *         \\
               \vdots     & \vdots     & \vdots     & \vdots & \vdots     & \vdots     & \ddots     & \ddots     & \ddots & \ddots    & \ddots    & A_{n-k,n}\\
               \vdots     & \vdots     & \vdots     & \vdots & \vdots     & \vdots     & \vdots     & \ddots     & \ddots & \ddots    & \ddots    & \vdots    \\
               \vdots     & \vdots     & \vdots     & \vdots & \vdots     & \vdots     & \vdots     & \vdots     & \ddots & \ddots    & \ddots    & A_{n-2,n} \\
               \vdots     & \vdots     & \vdots     & \vdots & \vdots     & \vdots     & \vdots     & \vdots     & \vdots & \ddots    & \ddots    & A_{n-1,n} \\
               *          & *          & *          & \ldots & \ldots     & \ldots     & \ldots     & \ldots     & \ldots & \ldots    & *         & A_{n,n} 
               \end{smallmatrix}\right]

        is stored as an array

      .. container:: fignone
                            
         - For column major layout,
                
            .. math::
                     
               \scriptstyle a = 
                  [\underbrace{
                   \underbrace{\underbrace{*,...,*}_\text{ku},A_{11},*,...,*}_\text{lda},
                   \underbrace{\underbrace{*,...,*}_\text{ku-1},A_{max(1,2-k),2},...,A_{2,2},*,...*}_\text{lda},
                   ...,
                   \underbrace{\underbrace{*,...,*}_\text{max(0,k-n+1)},A_{max(1,n-k),n},...,A_{n,n},*,...*}_\text{lda}
                   }_\text{lda x n}]


         - For row major layout,
            
            .. math::
                     
               \scriptstyle a = 
                  [\underbrace{
                   \underbrace{A_{11},A_{21},...,A_{min(k+1,n),1},*,...,*}_\text{lda},
                   \underbrace{A_{2,2},...,A_{min(k+2,n),2},*,...,*}_\text{lda},
                   ...,
                   \underbrace{A_{n,n},*,...*}_\text{lda}
                   }_\text{lda x n}]

         The following program segment transfers a band matrix from
         conventional full matrix storage (variable ``matrix``, with
         leading dimension ``ldm``) to band storage (variable ``a``,
         with leading dimension ``lda``):

         - Using matrices stored with column major layout,

         ::

            for (j = 0; j < n; j++) {
                m = k – j;
                for (i = max(0, j – k); i <= j; i++) {
                    a[(m + i) + j * lda] = matrix[i + j * ldm];
                }
            }

         - Using matrices stored with row major layout,

         ::

            for (i = 0; i < n; i++) {
                m = –i;
                for (j = i; j < min(n, i + k + 1); j++) {
                    a[(m + j) + i * lda] = matrix[j + i * ldm];
                }
            }

      - If ``upper_lower = uplo::lower``, the leading (``k`` + 1) by ``n``
        part of the array ``a`` must contain the upper triangular
        band part of the matrix ``A``. This matrix must be supplied
        column-by-column with the main diagonal of the matrix in row 0
        of the array, the first sub-diagonal starting at position 0 in
        row 1, and so on. Elements in the array ``a`` that do not
        correspond to elements in the triangular band matrix (such as
        the bottom right ``k`` by ``k`` triangle) are not referenced.

        That is, the matrix

        .. math::

           A = \left[\begin{smallmatrix}
               A_{11}     & *          & \ldots     & \ldots & \ldots     & \ldots    & \ldots     & \ldots     & \ldots & \ldots    & \ldots    & *         \\
               A_{21}     & A_{22}     & *          & \ldots & \ldots     & \ldots    & \ldots     & \ldots     & \ldots & \ldots    & \ldots    & *         \\
               A_{31}     & A_{32}     & A_{33}     & *      & \ldots     & \ldots    & \ldots     & \ldots     & \ldots & \ldots    & \ldots    & *         \\
               \vdots     & A_{42}     & A_{43}     & \ddots & \ddots     & \ldots    & \ldots     & \ldots     & \ldots & \ldots    & \ldots    & \vdots    \\
               A_{k+1,1}  & \vdots     & A_{53}     & \ddots & \ddots     & \ddots    & \ldots     & \ldots     & \ldots & \ldots    & \ldots    & \vdots    \\
               *          & A_{k+2,2}  & \vdots     & \ddots & \ddots     & \ddots    & \ddots     & \ldots     & \ldots & \ldots    & \ldots    & \vdots    \\
               \vdots     & *          & A_{k+3,3}  & \ddots & \ddots     & \ddots    & \ddots     & \ddots     & \ldots & \ldots    & \ldots    & \vdots    \\
               \vdots     & \vdots     & *          & \ddots & \ddots     & \ddots    & \ddots     & \ddots     & \ddots & \ldots    & \ldots    & \vdots    \\
               \vdots     & \vdots     & \vdots     & *      & \ddots     & \ddots    & \ddots     & \ddots     & \ddots & \ddots    & \ldots    & \vdots    \\
               \vdots     & \vdots     & \vdots     & \vdots & *          & \ddots    & \ddots     & \ddots     & \ddots & \ddots    & \ddots    & \vdots    \\
               \vdots     & \vdots     & \vdots     & \vdots & \vdots     & \ddots    & \ddots     & \ddots     & \ddots & \ddots    & \ddots    & *         \\
               *          & *          & *          & \ldots & \ldots     & \ldots    & *          & A_{n,n-k}  & \ldots & A_{n,n-2} & A_{n,n-1} & A_{n,n} 
               \end{smallmatrix}\right]


        is stored as the array


      .. container:: fignone

         - For column major layout,

           .. math::
                    
              \scriptstyle a = 
                 [\underbrace{
                  \underbrace{A_{11},A_{21},...,A_{min(k+1,n),1},*,...,*}_\text{lda},
                  \underbrace{A_{2,2},...,A_{min(k+2,n),2},*,...,*}_\text{lda},
                  ...,
                  \underbrace{A_{n,n},*,...*}_\text{lda}
                  }_\text{lda x n}]

         - For row major layout,
        
            .. math::
                     
               \scriptstyle a = 
                  [\underbrace{
                   \underbrace{\underbrace{*,...,*}_\text{k},A_{11},*,...,*}_\text{lda},
                   \underbrace{\underbrace{*,...,*}_\text{k-1},A_{max(1,2-k),2},...,A_{2,2},*,...*}_\text{lda},
                   ...,
                   \underbrace{\underbrace{*,...,*}_\text{max(0,k-n+1)},A_{max(1,n-k),n},...,A_{n,n},*,...*}_\text{lda}
                   }_\text{lda x n}]


         The following program segment transfers a band matrix from
         conventional full matrix storage (variable ``matrix``, with
         leading dimension ``ldm``) to band storage (variable ``a``,
         with leading dimension ``lda``):

         - Using matrices stored with column major layout,
           
         ::

            for (j = 0; j < n; j++) {
                m = –j;
                for (i = j; i < min(n, j + k + 1); i++) {
                    a[(m + i) + j * lda] = matrix[i + j * ldm];
                }
            }

         - Using matrices stored with row major layout,

         ::

            for (i = 0; i < n; i++) {
                m = k – i;
                for (j = max(0, i – k); j <= i; j++) {
                    a[(m + j) + i * lda] = matrix[j + i * ldm];
                }
            }


   .. container:: section

      .. rubric:: Packed Triangular Matrix
         :name: packed-triangular-matrix
         :class: sectiontitle

      A triangular matrix ``A`` of ``n`` rows and ``n`` columns is
      represented in packed format as a one dimensional array ``a`` of
      size at least (``n``\ \*(``n`` + 1))/2. All elements in the upper
      or lower part of the matrix ``A`` are stored contiguously in the
      array ``a``.

      Before entry in any BLAS function using a triangular packed
      matrix,

      - If ``upper_lower = uplo::upper``, if column (respectively row)
        major layout is used, the first (``n``\ \*(``n`` + 1))/2
        elements in the array ``a`` must contain the upper triangular
        part of the matrix ``A`` packed sequentially, column by column
        (respectively row by row) so that ``a``\ [0] contains ``A``\
        :sub:`11`, ``a``\ [1] and ``a``\ [2] contain ``A``\ :sub:`12`
        and ``A``\ :sub:`22` (respectively ``A``\ :sub:`13`)
        respectively, and so on. Hence, the matrix

        .. math::
              
           A = \begin{bmatrix}
               A_{11} & A_{12} & A_{13} & \ldots & A_{1n}\\
               *      & A_{22} & A_{23} & \ldots & A_{2n}\\
               *      & *      & A_{33} & \ldots & A_{3n}\\
               \vdots & \vdots & \vdots & \ddots & \vdots\\
               *      & *      & *      & \ldots & A_{nn}
               \end{bmatrix}

        is stored as the array

        - For column major layout,

          .. math::
             
             \scriptstyle a = [A_{11},A_{12},A_{22},A_{13},A_{23},A_{33},...,A_{(n-1),n},A_{nn}]

        - For row major layout,

          .. math::
             
             \scriptstyle a = [A_{11},A_{12},A_{13},...,A_{1n},
                  A_{22},A_{23},...,A_{2n},...,
                  A_{(n-1),(n-1)},A_{(n-1),n},A_{nn}]

      - If ``upper_lower = uplo::lower``, if column (respectively row)
        major layout is used, the first (``n``\ \*(``n`` + 1))/2
        elements in the array ``a`` must contain the lower triangular
        part of the matrix ``A`` packed sequentially, column by column
        (row by row) so that ``a``\ [0] contains ``A``\ :sub:`11`,
        ``a``\ [1] and ``a``\ [2] contain ``A``\ :sub:`21` and ``A``\
        :sub:`31` (respectively ``A``\ :sub:`22`) respectively, and so
        on. The matrix

         .. math::
               
            A = \begin{bmatrix}
                A_{11} & *      & *      & \ldots & *     \\
                A_{21} & A_{22} & *      & \ldots & *     \\
                A_{31} & A_{32} & A_{33} & \ldots & *     \\
                \vdots & \vdots & \vdots & \ddots & \vdots\\
                A_{n1} & A_{n2} & A_{n3} & \ldots & A_{nn}
                \end{bmatrix}

         is stored as the array

         - For column major layout,

          .. math::
             
             \scriptstyle a = [A_{11},A_{21},A_{31},...,A_{n1},
                  A_{22},A_{32},...,A_{n2},...,
                  A_{(n-1),(n-1)},A_{n,(n-1)},A_{nn}]

         - For row major layout,

          .. math::
             
             \scriptstyle a = [A_{11},A_{21},A_{22},A_{31},A_{32},A_{33},...,A_{n,(n-1)},A_{nn}]

   .. container:: section

      .. rubric:: Vector
         :name: vector
         :class: sectiontitle

      A vector ``X`` of ``n`` elements with increment ``incx`` is
      represented as a one dimensional array ``x`` of size at least (1 +
      (``n`` - 1) \* abs(``incx``)).

      Visually, the vector

      .. math::
            
            X = (X_{1},X_{2}, X_{3},...,X_{n})

      is stored in memory as an array


      .. math::
               
         \scriptstyle x = [\underbrace{
             \underbrace{X_{1},*,...,*}_\text{incx},
             \underbrace{X_{2},*,...,*}_\text{incx},
             ...,
             \underbrace{X_{n-1},*,...,*}_\text{incx},X_{n}
             }_\text{1 + (n-1) x incx}] \quad if \:incx \:> \:0 

      .. math::
               
         \scriptstyle x = [\underbrace{
             \underbrace{X_{n},*,...,*}_\text{|incx|},
             \underbrace{X_{n-1},*,...,*}_\text{|incx|},
             ...,
             \underbrace{X_{2},*,...,*}_\text{|incx|},X_{1}
             }_\text{1 + (1-n) x incx}] \quad if \:incx \:< \:0 


      **Parent topic:** :ref:`onemath_dense_linear_algebra`

