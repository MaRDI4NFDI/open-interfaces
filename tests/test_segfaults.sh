#!/usr/bin/env zsh
# This script runs Python tests multiple times to make sure that they
# run without segmentation faults (segfaults).
# If this script prints 'Run was unsuccessful', then something is wrong
# and must be investigated.


for i in $(seq 1 20); do
    output=$(pytest tests/lang_python/ -v --capture=tee-sys 2>&1 >/dev/null)
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "Run was unsuccessful" >&2
    else
        echo $i $exit_code
    fi
done
