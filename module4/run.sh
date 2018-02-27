#!/usr/bin/env bash

FILENAME=mem_speed_compare

nvcc $FILENAME.cu -lcudart -o $FILENAME

if [[ ! -v CI ]]; then
    ./$FILENAME -t 1024 -b 16
    ./$FILENAME -t 256 -b 16
fi