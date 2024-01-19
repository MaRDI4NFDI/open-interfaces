UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)  # macOS
    DSO_EXT := dylib
else                    # Linux
    DSO_EXT := so
endif

export LD_LIBRARY_PATH:=build
export PYTHONPATH:=oif/interfaces/python:oif_impl/python:oif_impl/impl:oif/lang_python:src:"$(PYTHONPATH)"

.PHONY : all
all :
	cmake -S . -B build -DCMAKE_VERBOSE_MAKEFILE:BOOL=FALSE -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
	cmake --build build

.PHONY : test
test : all
	cd build && ./test_qeq && ./test_linsolve && ./test_ivp
	python -m pytest tests/lang_python
