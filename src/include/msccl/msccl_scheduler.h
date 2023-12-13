/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#ifndef MSCCL_SCHEDULER_H_
#define MSCCL_SCHEDULER_H_

typedef enum { mscclFuncReduce             =  0,
               mscclFuncBroadcast          =  1,
               mscclFuncAllReduce          =  2,
               mscclFuncReduceScatter      =  3,
               mscclFuncAllGather          =  4,
               mscclFuncSend               =  5,
               mscclFuncRecv               =  6,
               mscclFuncGather             =  7,
               mscclFuncScatter            =  8,
               mscclFuncAllToAll           =  9,
               mscclFuncAllToAllv          =  10,
               mscclNumFuncs               =  11 } mscclFunc_t;

struct mscclSchedulerParam {
  const void* sendBuff;
  const size_t* sendCounts;
  const size_t* sDisPls;
  void* recvBuff;
  const size_t* recvCounts;
  const size_t* rDisPls;
  size_t count;
  ncclDataType_t dataType;
  int root;
  int peer;
  ncclRedOp_t op;
  mscclFunc_t func;
  int rank;
  int nRanks;
  bool scheduled;
  mscclAlgoHandle_t handle;
  bool repair;
};

typedef struct {
  // Name of the scheduler (mainly for logs)
  const char* name;
  // Load all algorithms
  ncclResult_t (*init)(struct mscclSchedulerInitParam *initParam);
  // Select an algorithm
  ncclResult_t (*selectAlgo)(struct mscclSchedulerParam* param);
  // Unload all algorithms
  ncclResult_t (*teardown)();
} mscclSchedulerInterface;

struct mscclSchedulerInitParam{
  int rank;
  int nRanks;
  int nNodes;
  void* bootstrap;
  // bootstrap send operation
  ncclResult_t (*send)(void* commState, int peer, int tag, void* data, int size);
  // bootstrap receive operation
  ncclResult_t (*receive)(void* commState, int peer, int tag, void* data, int size);
  // bootstrap allgather operation
  ncclResult_t (*allgather)(void* commState, void* allData, int size);
};

#endif
