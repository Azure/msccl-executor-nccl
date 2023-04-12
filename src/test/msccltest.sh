#!/bin/sh

echo ""
echo "Starting the msccl tests, which will including below algo"
echo "allreduce_a100_allpairs, allreduce_a100_ring, alltoall_allpairs"
echo "with both nvlink and ib (NCCL_P2P_DISABLE=1, NCCL_SHM_DISABLE=1) and run both graph (-G 1) and non-graph mode in nccl-tests"
echo ""

MSCCL_PATH=$HOME/nccl/build/lib
MSCCL_ALGOS=(allreduce_a100_allpairs allreduce_a100_ring alltoall_allpairs)
MSCCL_PROTO=(LL LL128 Simple) 
MSCCL_TOOL=$HOME/msccl-tools/examples/mscclang
MSCCL_ALGO_PATH=$HOME/msccl-algo
NCCL_TESTS_PATH=$HOME/nccl-tests/build
TESTRESULT_HOME=$HOME/msccl-test-results

if [ ! -d "$MSCCL_ALGO_PATH" ]; then
    mkdir $MSCCL_ALGO_PATH
fi

if [ ! -d "$TESTRESULT_HOME" ]; then
    mkdir $TESTRESULT_HOME
fi

for algo in ${MSCCL_ALGOS[@]}; do
    for proto in ${MSCCL_PROTO[@]}; do
        algo_file=${algo}_8n_$proto.xml
        if [ ! -e $MSCCL_ALGO_PATH/$algo_file]; then
            echo "The algo file: $algo_file does not exist, generating..."
            msccl_algo_gen="python $MSCCL_TOOL/$algo.py --protocol=$proto 8 2 > $MSCCL_ALGO_PATH/$algo_file"
            echo $msccl_algo_gen
            eval $msccl_algo_gen
        fi
        echo "Copying the algo file $MSCCL_ALGO_PATH/$algo_file to $MSCCL_PATH/msccl-algorithms/test.xml"
        cp $MSCCL_ALGO_PATH/$algo_file $MSCCL_PATH/msccl-algorithms/test.xml
        echo "Running the $algo with $proto"
        if [ $algo = "allreduce_a100_allpairs" ] || [ $algo = "allreduce_a100_ring" ]; then
            NCCL_TEST_TYPE=all_reduce_perf
        else
            NCCL_TEST_TYPE=alltoall_perf
        fi
        
        for NCCL_P2P_DISABLE in 0 1; do
            for NCCL_SHM_DISABLE in 0 1; do
                testresult=$TESTRESULT_HOME/${algo}_${proto}_${NCCL_P2P_DISABLE}_${NCCL_SHM_DISABLE}.txt
                echo "Running the $algo with $proto and NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE and NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE"
                msccl_test="mpirun --allow-run-as-root -np 8 -x LD_LIBRARY_PATH=$MSCCL_PATH/:$LD_LIBRARY_PATH -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=INIT,ENV -x NCCL_ALGO=MSCCL,RING,TREE  -x NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE -x NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE $NCCL_TESTS_PATH/$NCCL_TEST_TYPE -b 128 -e 32MB -f 2 -g 1 -c 1 -n 100 -w 100 -G 1 -z 0 > $testresult"
                echo $msccl_test
                eval $msccl_test
            done
        done
    done
done









