#!/bin/bash

if [ $# == 2 ]
then
    ./assignment.exe $1 $2
    ./assignment_advanced.exe $1 $2
else
    echo $1
    echo $2
    echo "Usage: $0 [block size] [number of threads per block]"
fi
