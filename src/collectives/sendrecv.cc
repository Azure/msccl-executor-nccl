/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "collectives.h"
#include "argcheck.h" // Need some checks here since we access comm
#include "param.h"

#include "msccl/msccl_lifecycle.h"

extern int64_t ncclParamResilientEnabled();

struct NvtxParamsSendRecv {
    size_t bytes;
    int peer;
};
constexpr const nvtxPayloadSchemaEntry_t SendRecvSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Bytes"},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Peer rank", nullptr, 0, offsetof(NvtxParamsSendRecv, peer)}
};

NCCL_API(ncclResult_t, ncclSend, const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  NvtxParamsSendRecv payload{count * ncclTypeSize(datatype), peer};
  NVTX3_FUNC_WITH_PARAMS(Send, SendRecvSchema, payload)

  ncclResult_t ret;
  if (mscclAvailable() && !mscclIsCaller()) {
    ret = mscclEnqueueCheck(
      sendbuff, nullptr, nullptr, nullptr, nullptr, nullptr,
      count, datatype, 0, peer, ncclSum, mscclFuncSend, comm, stream);
  }
  else{
    struct ncclInfo info = { ncclFuncSend, "Send",
    NULL, (void*)sendbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };
    NCCLCHECK(ncclGroupStart());
    ret = ncclEnqueueCheck(&info);
    NCCLCHECK(ncclGroupEnd());
  }
  if (ncclParamResilientEnabled()){
    cudaStreamSynchronize(stream);
  }

  return ret;
}

NCCL_API(ncclResult_t, ncclRecv, void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  NvtxParamsSendRecv payload{count * ncclTypeSize(datatype), peer};
  NVTX3_FUNC_WITH_PARAMS(Recv, SendRecvSchema, payload)

  ncclResult_t ret;
  if (mscclAvailable() && !mscclIsCaller()) {
    ret = mscclEnqueueCheck(
      nullptr, nullptr, nullptr, recvbuff, nullptr, nullptr,
      count, datatype, 0, peer, ncclSum, mscclFuncRecv, comm, stream);
  }
  else{
    struct ncclInfo info = { ncclFuncRecv, "Recv",
    NULL, recvbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };
    NCCLCHECK(ncclGroupStart());
    ret = ncclEnqueueCheck(&info);
    NCCLCHECK(ncclGroupEnd());
  }
  if (ncclParamResilientEnabled()){
    cudaStreamSynchronize(stream);
  }
  
  return ret;
}
