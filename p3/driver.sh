#!/usr/bin/env bash

END_NUM=${1:-570}
BEG_NUM=11

echo "Running $END_NUM times (with ${BEG_NUM}..$END_NUM lines of data)"
#
for ((i=$BEG_NUM; i <= $END_NUM; i++))
do
    echo -n "${i}: "
    head -n $i data.csv > data-test.csv
    ./p3.py > /dev/null 2>&1
    if [ $? -ne 0 ];
    then
        echo BAD
        break
    fi
    echo good
done
