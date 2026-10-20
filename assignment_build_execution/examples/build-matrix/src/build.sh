#!/usr/bin/env bash
# Prints each argument it receives. Passing --fail makes it exit 3 so the
# example shows how a failing configuration is reported.
for arg in "$@"; do
    echo "received: ${arg}"
    if [ "$arg" = "--fail" ]; then
        echo "requested failure"
        exit 3
    fi
done
echo "build complete"
