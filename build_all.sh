#!/usr/bin/env bash

modules=(
    "module4"
)
if [[ -v CI ]]; then
    for module in modules; do
        pushd module
            ./run.sh
        popd
    done
else
    echo "Exiting. This script is designed for CI only"
    exit 1
fi