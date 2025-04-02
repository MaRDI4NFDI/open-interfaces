#!/usr/bin/bash
rm -rf dist/mardi_open_interfaces-0.5.0/


if ! python -m build --sdist ; then
    echo -e "\e[01;31mERROR: build failed\e[0m"
    exit 1
fi

tar xzf dist/mardi_open_interfaces-0.5.0.tar.gz -C dist/
open dist/mardi_open_interfaces-0.5.0/
