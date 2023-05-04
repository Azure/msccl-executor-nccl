#!/bin/sh

echo ""
echo "Starting the msccl tests, which will including below algo"
echo "allreduce_a100_allpairs, allreduce_a100_ring, alltoall_allpairs"
echo "with both nvlink and ib (NCCL_P2P_DISABLE=1, NCCL_SHM_DISABLE=1) and run both graph (-G 1) and non-graph mode in nccl-tests"
echo ""
TEST_TYPE=$1
MSCCL_PATH=$HOME/nccl/build/lib
MSCCL_ALGOS=(allreduce_a100_allpairs allreduce_a100_ring alltoall_allpairs)
# MSCCL_PROTO=(LL128 Simple LL) 
MSCCL_PROTO=(Simple) 
declare MSCCL_DATA_TYPE 
declare MSCCL_OP_TYPE
if [ $TEST_TYPE -eq "all" ]; then
    MSCCL_DATA_TYPE=(int8 uint8 int32 uint32 int64 uint64 half float double bfloat16 fp8_e4m3 fp8_e5m2)
    MSCCL_OP_TYPE=(sum, prod, max, min, avg, mulsum)
else
    MSCCL_DATA_TYPE=(float fp8_e4m3 fp8_e5m2)
    MSCCL_OP_TYPE=(sum)
fi
MSCCL_TOOL=$HOME/msccl-tools/examples/mscclang
MSCCL_ALGO_PATH=$HOME/msccl-algo
MSCCL_ALGO_TEST_PATH=$MSCCL_PATH/msccl-algorithms
NCCL_TESTS_PATH=$HOME/nccl-tests/build
TESTRESULT_HOME=$HOME/msccl-test-results

if [ ! -d "$MSCCL_ALGO_PATH" ]; then
    mkdir $MSCCL_ALGO_PATH
fi

if [ ! -d "$TESTRESULT_HOME" ]; then
    mkdir $TESTRESULT_HOME
fi

if [ ! -d "$MSCCL_ALGO_TEST_PATH" ]; then
    mkdir $MSCCL_ALGO_TEST_PATH
fi


for algo in ${MSCCL_ALGOS[@]}; do
    for proto in ${MSCCL_PROTO[@]}; do
        algo_file=${algo}_8n_$proto.xml
        if [ ! -e $MSCCL_ALGO_PATH/$algo_file ]; then
            echo "The algo file: $algo_file does not exist, generating..."
            msccl_algo_gen="python $MSCCL_TOOL/$algo.py --protocol=$proto 8 2 > $MSCCL_ALGO_PATH/$algo_file"
            if [ $algo = "allreduce_a100_ring" ]; then
                msccl_algo_gen="python $MSCCL_TOOL/$algo.py --protocol=$proto 8 1 2 > $MSCCL_ALGO_PATH/$algo_file"
            fi
            echo $msccl_algo_gen
            eval $msccl_algo_gen
        fi
        echo "Copying the algo file $MSCCL_ALGO_PATH/$algo_file to $MSCCL_ALGO_TEST_PATH/test.xml"
        cp $MSCCL_ALGO_PATH/$algo_file $MSCCL_ALGO_TEST_PATH/test.xml
        echo "Running the $algo with $proto"
        if [ $algo = "allreduce_a100_allpairs" ] || [ $algo = "allreduce_a100_ring" ]; then
            NCCL_TEST_TYPE=all_reduce_perf
        else
            NCCL_TEST_TYPE=alltoall_perf
        fi
        
        for NCCL_P2P_DISABLE in 0 1; do
            for NCCL_SHM_DISABLE in 0 1; do
                for ENABLE_CUDA_GRAPH in 0 1; do
                    for ENABLE_ONE_PROCESS in 0 1; do
                        for DATA_TYPE in ${MSCCL_DATA_TYPE[@]}; do
                            for OP_TYPE in ${MSCCL_OP_TYPE[@]}; do
                                testresult=$TESTRESULT_HOME/${algo}_${proto}_${NCCL_P2P_DISABLE}_${NCCL_SHM_DISABLE}_${ENABLE_CUDA_GRAPH}_${ENABLE_ONE_PROCESS}_${DATA_TYPE}_${OP_TYPE}.txt
                                echo "Running the $algo with $proto with configs: NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE, NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE, ENABLE_CUDA_GRAPH=$ENABLE_CUDA_GRAPH, ENABLE_ONE_PROCESS=$ENABLE_ONE_PROCESS, DATA_TYPE=$DATA_TYPE, OP_TYPE=$OP_TYPE"
                                if [ $ENABLE_ONE_PROCESS -eq 1 ]; then
                                    msccl_test="mpirun --allow-run-as-root -np 1 -x LD_LIBRARY_PATH=$MSCCL_PATH/:$LD_LIBRARY_PATH -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=INIT,ENV -x NCCL_ALGO=MSCCL,RING,TREE  -x NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE -x NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE $NCCL_TESTS_PATH/$NCCL_TEST_TYPE -b 8 -e 8G -d $DATA_TYPE -f 2 -g 8 -c 1 -o $OP_TYPE -n 100 -w 100 -G $ENABLE_CUDA_GRAPH -z 0 > $testresult"
                                else
                                    msccl_test="mpirun --allow-run-as-root -np 8 -x LD_LIBRARY_PATH=$MSCCL_PATH/:$LD_LIBRARY_PATH -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=INIT,ENV -x NCCL_ALGO=MSCCL,RING,TREE  -x NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE -x NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE $NCCL_TESTS_PATH/$NCCL_TEST_TYPE -b 8 -e 8G -d $DATA_TYPE -f 2 -g 1 -c 1 -o $OP_TYPE -n 100 -w 100 -G $ENABLE_CUDA_GRAPH -z 0 > $testresult"
                                fi
                                echo $msccl_test
                                eval $msccl_test
                                wait
                            done
                        done
                    done
                done
            done
        done
    done
done









