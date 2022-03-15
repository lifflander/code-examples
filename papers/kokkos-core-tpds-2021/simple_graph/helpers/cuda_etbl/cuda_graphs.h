/*
 * Copyright 2017-2020 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_graphs_h__
#define __cuda_etbl_graphs_h__

#include "../cuda_uuid.h"
#include "cuda.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

CU_DEFINE_UUID(CU_ETID_CudaGraphs,
    0x12fde351, 0x86f, 0x44c8, 0xa2, 0xad, 0x23, 0x57, 0xb7, 0xc3, 0x52, 0x21);

#define CU_GRAPH_NODE_TYPE_EX_CONDITIONAL ((CUgraphNodeType)100)

typedef struct CUetblCudaGraphs_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    void *reserved[33];

    /**
     * \brief Creates a dynamic control flow node and adds it to a graph
     *
     * Creates a new dynamic control node and adds it to \p hGraph with \p numDependencies
     * dependencies specified via \p dependencies.
     * It is possible for \p numDependencies to be 0, in which case the node will be placed
     * at the root of the graph. \p dependencies may not have any duplicate entries.
     * A handle to the new node will be returned in \p phGraphNode.
     *
     * A dynamic control node may be used to implement a conditional execution path or loop
     * inside of a graph. The body of the conditional or loop is provided as a graph, and is
     * subject to the following constraints:
     *
     * - Allowed node types in the body are kernel nodes, empty nodes, child graphs, and
     *   conditionals. This applies recursively to child graphs and conditional bodies.
     * - All kernels, including kernels in nested conditionals or child graphs at any level,
     *   must belong to the same CUDA context.
     *
     * The semantics of the dynamic control flow is analogous to the following:
     *
     * \code
     *   bool control = false;
     *   // user-specified code
     *   while (control) {
     *       control = false;
     *       // user-specified body
     *   }
     * \endcode
     *
     * The "control" value starts out as false when the graph is launched. It may be set at
     * any time before the conditional node executes, by the method described below. When the
     * graph reaches the conditional node, if the control value is true, it will be reset to
     * false and the body will execute. The conditional node is in effect evaluated again
     * after executing the body. The control value may be set again to true inside the body
     * to cause it to execute again. When the conditional node is evaluated with a false
     * control value, the conditional node is then completed from the perspective of its
     * parent graph and downstream nodes may execute.
     *
     * To set the control value:
     *
     * - After the graph containing the conditional node is instantiated, obtain a handle for
     *   the node with ::cuGraphExecGetConditionalNodeSchedulingHandle.
     * - In a kernel or kernels at appropriate locations in the graph, insert a call to
     *   `void __cuda_syscall_cuGraphSetConditional(unsigned long long handle, bool value)`.
     *
     * The body graph is cloned in this call. The clone may be obtained with
     * ::cuGraphConditionalNodeGetBody.
     *
     * \param pNode           - Returns newly created node
     * \param graph           - Graph to which to add the node
     * \param dependencies    - Dependencies of the node
     * \param numDependencies - Number of dependencies
     * \param body            - The graph to use as the body
     *
     * \return
     * ::CUDA_SUCCESS,
     * ::CUDA_ERROR_DEINITIALIZED,
     * ::CUDA_ERROR_NOT_INITIALIZED,
     * ::CUDA_ERROR_INVALID_VALUE
     * \note_graph_thread_safety
     */
    CUresult (CUDAAPI *cuGraphAddConditionalNode)(CUgraphNode *pNode, CUgraph graph, const CUgraphNode *dependencies, size_t numDependencies, CUgraph body);

    /**
     * \brief Get the handle of a dynamic control flow node in an instantiated graph
     *
     * Returns the handle needed to modify the control value of a dynamic control flow
     * node in a graph. For more information, see ::cuGraphAddConditionalNode.
     *
     * \return
     * ::CUDA_SUCCESS,
     * ::CUDA_ERROR_DEINITIALIZED,
     * ::CUDA_ERROR_NOT_INITIALIZED,
     * ::CUDA_ERROR_INVALID_VALUE
     */
    CUresult (CUDAAPI *cuGraphExecGetConditionalNodeSchedulingHandle)(CUgraphExec graph, CUgraphNode sourceConditionalNode, cuuint64_t *handle_out);

    /**
     * \brief Gets a handle to the body graph of a conditional node
     *
     * Gets a handle to the graph constituting the body of a conditional node. This
     * call does not clone the graph. Changes to the graph will be reflected in
     * the node, and the node retains ownership of the graph.
     *
     * \param body_out - Location to store a handle to the body graph
     * \param hNode    - Node to get the body graph for
     *
     * \return
     * ::CUDA_SUCCESS,
     * ::CUDA_ERROR_DEINITIALIZED,
     * ::CUDA_ERROR_NOT_INITIALIZED,
     * ::CUDA_ERROR_INVALID_VALUE,
     * \note_graph_thread_safety
     */
    CUresult (CUDAAPI *cuGraphConditionalNodeGetBody)(CUgraphNode conditionalNode, CUgraph *body_out);

} CUetblCudaGraphs;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
