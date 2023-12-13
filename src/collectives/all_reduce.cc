/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "nccl.h"
#include "param.h"

#include "msccl/msccl_lifecycle.h"
extern int64_t ncclParamResilientEnabled();

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  struct NvtxParamsAllReduce {
    size_t bytes;
    ncclRedOp_t op;
  };
  // Just pass the size of one message and not the total bytes sent/received.
  static constexpr nvtxPayloadSchemaEntry_t AllReduceSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"},
    {0, NVTX_PAYLOAD_ENTRY_NCCL_REDOP, "Reduction operation", nullptr, 0,
      offsetof(NvtxParamsAllReduce, op)}
  };
  NvtxParamsAllReduce payload{count * ncclTypeSize(datatype), op};
  NVTX3_FUNC_WITH_PARAMS(AllReduce, AllReduceSchema, payload)

  ncclResult_t ret;
  if (mscclAvailable() && !mscclIsCaller()) {
    ret = mscclEnqueueCheck(
      sendbuff, nullptr, nullptr, recvbuff, nullptr, nullptr,
      count, datatype, 0, 0, op, mscclFuncAllReduce, comm, stream);
  }
  else{
    struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
      sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
      ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
    ret = ncclEnqueueCheck(&info);
  }
  if (ncclParamResilientEnabled()){
    cudaStreamSynchronize(stream);
  }

  return ret;
}
