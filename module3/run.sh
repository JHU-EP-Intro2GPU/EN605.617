#!/usr/bin/env bash

set -e

FILENAME=assignment3

nvcc $FILENAME.cu -lcudart -o $FILENAME

if [[ ! -v CI ]]; then
    echo "Example Addition 0"

    ./$FILENAME -t 256 -b 16

    echo "Example Addition 1: Changing number of threads"

    ./$FILENAME -t 512 -b 16

    echo "Example Addition 2: Changing number of threads"

    ./$FILENAME -t 1024 -b 16

    echo "Example Addition 3: Changing block size"

    ./$FILENAME -t 254 -b 32

    echo "Example Addition 4: Changing block size"

    ./$FILENAME -t 256 -b 64
    
    echo "Custom Addition: Allow for user input"
    read -rp "Enter total number of threads:"  THREADS
    read -rp "Enter total threads per block (block size):"  BLOCKS

    ./$FILENAME -t $THREADS -b $BLOCKS

fi 