#!/bin/sh
# This script is used to summarize the test results.
MSCCL_RESULT_DIR=$1
if [ ! -d "$MSCCL_RESULT_DIR" ]; then
    echo "Given Path is empty or is not a directory"
    exit 1
fi

passed_count=0
failed_count=0
for file in `ls $MSCCL_RESULT_DIR`
do
    if grep -q "# Out of bounds values : 0 OK" "$MSCCL_RESULT_DIR"/"$file"; then
        passed_count=$((passed_count + 1))
    else
        failed_count=$((failed_count + 1))
        echo "$(echo $file | sed 's/.txt//'):$(head -n 1 "$MSCCL_RESULT_DIR"/"$file")"
    fi
done
echo "Total number of tests passed: $passed_count"
echo "Total number of tests failed: $failed_count"