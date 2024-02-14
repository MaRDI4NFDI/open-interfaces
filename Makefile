UNAME := $(shell uname)
PWD := $(shell pwd)

ifeq ($(UNAME), Darwin)  # macOS
    DSO_EXT := dylib
else                    # Linux
    DSO_EXT := so
endif

export LD_LIBRARY_PATH:=build
export PYTHONPATH:=$(PWD)/oif/interfaces/python:$(PWD)/oif_impl/python:$(PWD)/oif_impl/impl:$(PWD)/oif/lang_python:$(PWD)/src:$(PWD)/build

.PHONY : all
all :
	cmake -S . -B build -DCMAKE_VERBOSE_MAKEFILE:BOOL=FALSE -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
	cmake --build build

.PHONY : test
test : all
	cd build && ctest
	pytest tests/lang_python

.PHONY : release
release :
	rm -rf build
	cmake -S . -B build -DCMAKE_VERBOSE_MAKEFILE:BOOL=FALSE -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
	cmake --build build

.PHONY : docs
docs :
	cd docs && make html

.PHONY : docs-from-scratch
docs-from-scratch:
	cd docs && rm -rf build && make html
