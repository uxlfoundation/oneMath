/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  For your convenience, a copy of the License has been included in this
*  repository.
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
**************************************************************************/
#include "cublas_scope_handle.hpp"

namespace oneapi {
namespace math {
namespace blas {
namespace cublas {

/**
 * Inserts a new element in the map if its key is unique. This new element
 * is constructed in place using args as the arguments for the construction
 * of a value_type (which is an object of a pair type). The insertion only
 * takes place if no other element in the container has a key equivalent to
 * the one being emplaced (keys in a map container are unique).
 */
thread_local cublas_handle CublasScopedContextHandler::handle_helper = cublas_handle{};

CublasScopedContextHandler::CublasScopedContextHandler(sycl::interop_handle& ih) : ih(ih) {
    // Initialize streamID member to a CUstream associated with the queue `ih`
    // has been submitted to.
    streamId = ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();

    // Initialize the `cublasHandle_t` member `nativeHandle`
    CUdevice device = ih.get_native_device<sycl::backend::ext_oneapi_cuda>();
    auto it = handle_helper.cublas_handle_mapper_.find(device);
    if (it != handle_helper.cublas_handle_mapper_.end()) {
        // Use existing handle if one already exists for the device, but update
        // the native stream.
        nativeHandle = it->second;
        cudaStream_t currentStreamId;
        cublasStatus_t err;
        CUBLAS_ERROR_FUNC(cublasGetStream, err, nativeHandle, &currentStreamId);
        if (currentStreamId != streamId) {
            CUBLAS_ERROR_FUNC(cublasSetStream, err, nativeHandle, streamId);
        }
    }
    else {
        // Create a new handle if one doesn't already exist for the device
        cublasStatus_t err;
        CUBLAS_ERROR_FUNC(cublasCreate, err, &nativeHandle);
        CUBLAS_ERROR_FUNC(cublasSetStream, err, nativeHandle, streamId);
        handle_helper.cublas_handle_mapper_.insert(std::make_pair(device, nativeHandle));
    }
}

void CublasScopedContextHandler::begin_recording_if_graph() {
// interop_handle graph methods only available from extension version 2
#if SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND >= 2
    if (!ih.ext_codeplay_has_graph()) {
        return;
    }

    CUresult err;
#if CUDA_VERSION >= 12030
    // After CUDA 12.3 we can use cuStreamBeginCaptureToGraph to capture
    // the stream directly in the native graph, rather than needing to
    // instantiate the stream capture as a new graph.
    auto graph = ih.ext_codeplay_get_native_graph<sycl::backend::ext_oneapi_cuda>();
    CUDA_ERROR_FUNC(cuStreamBeginCaptureToGraph, err, streamId, graph, nullptr, nullptr, 0,
                    CU_STREAM_CAPTURE_MODE_GLOBAL);
#else
    CUDA_ERROR_FUNC(cuStreamBeginCapture, err, streamId, CU_STREAM_CAPTURE_MODE_GLOBAL);
#endif // CUDA_VERSION
#endif // SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND >= 2
}

void CublasScopedContextHandler::end_recording_if_graph() {
// interop_handle graph methods only available from extension version 2
#if SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND >= 2
    if (!ih.ext_codeplay_has_graph()) {
        return;
    }

    auto graph = ih.ext_codeplay_get_native_graph<sycl::backend::ext_oneapi_cuda>();
    CUresult err;
#if CUDA_VERSION >= 12030
    CUDA_ERROR_FUNC(cuStreamEndCapture, err, streamId, &graph);
#else
    // cuStreamEndCapture returns a new graph, if we overwrite
    // "graph" it won't be picked up by the SYCL runtime, as
    // "ext_codeplay_get_native_graph" returns a passed-by-value pointer.
    CUgraph recorded_graph;
    CUDA_ERROR_FUNC(cuStreamEndCapture, err, streamId, &recorded_graph);

    // Add graph to native graph as a child node
    // Need to return a node object for the node to be created,
    // can't be nullptr.
    CUgraphNode node;
    CUDA_ERROR_FUNC(cuGraphAddChildGraphNode, err, &node, graph, nullptr, 0, recorded_graph);
#endif // CUDA_VERSION
#endif // SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND >= 2
}
} // namespace cublas
} // namespace blas
} // namespace math
} // namespace oneapi
