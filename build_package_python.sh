#!/usr/bin/bash
package="openinterfaces"
version=$(grep 'version =' pyproject.toml | sed 's/version = "//' | sed 's/"//')
echo -e "\e[01;32mBuilding package: $package version $version\e[0m"
rm -r "dist/${package}-${version}/" && echo "deleted old dist/${package}-${version}/"

if ! python -m build --sdist ; then
    echo -e "\e[01;31mERROR: build failed\e[0m"
    exit 1
fi

tar xzf "dist/${package}-${version}.tar.gz" -C dist/
open "dist/${package}-${version}/"
