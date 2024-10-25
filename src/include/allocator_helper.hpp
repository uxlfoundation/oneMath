/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: Apache-2.0
*******************************************************************************/

#ifndef __ALLOCATOR_HELPER_HPP
#define __ALLOCATOR_HELPER_HPP

#ifdef _WIN64
#include <malloc.h>
#else
#include <stdlib.h>
#endif

namespace oneapi {
namespace mkl {

static inline void* aligned_alloc(size_t align, size_t size) {
#ifdef _WIN64
    return ::_aligned_malloc(size, align);
#else
    return ::aligned_alloc(align, size);
#endif
}

static inline void aligned_free(void* p) {
#ifdef _WIN64
    ::_aligned_free(p);
#else
    ::free(p);
#endif
}
} // namespace mkl
} // namespace oneapi

#endif // __ALLOCATOR_HELPER_HPP
