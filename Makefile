UNAME := $(shell uname)
PWD := $(shell pwd)

ifeq ($(UNAME), Darwin)  # macOS
    DSO_EXT := dylib
else                    # Linux
    DSO_EXT := so
endif

export LD_LIBRARY_PATH:=build
export PYTHONPATH:=$(PWD)/oif/interfaces/python:$(PWD)/oif_impl/python:$(PWD)/oif_impl/impl:$(PWD)/oif/lang_python:$(PWD)/src:$(PWD)/build

export OIF_IMPL_ROOT_DIR := $(PWD)

.PHONY : all
all :
	cmake -S . -B build.debug -DCMAKE_VERBOSE_MAKEFILE:BOOL=FALSE -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
	cmake --build build.debug && \
	rm -f build && \
	ln -sv build.debug build

.PHONY : test
test :
	cd build && ctest
	pytest tests/lang_python

.PHONY : pytest-valgrind
pytest-valgrind :
	PYTHONMALLOC=malloc valgrind --show-leak-kinds=definite --log-file=/tmp/valgrind-output \
	python -m pytest -s -vv --valgrind --valgrind-log=/tmp/valgrind-output

.PHONY : release
release :
	cmake -S . -B build.release -DCMAKE_VERBOSE_MAKEFILE:BOOL=FALSE -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
	cmake --build build.release && \
	rm -f build && \
	ln -sv build.release build

.PHONY : docs
docs :
	cd docs && make html

.PHONY : docs-from-scratch
docs-from-scratch:
	cd docs && rm -rf build && make html
