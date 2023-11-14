UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)  # macOS
    DSO_EXT := dylib
else                    # Linux
    DSO_EXT := so
endif

.PHONY : all
all :
	cmake -S . -B build -DCMAKE_VERBOSE_MAKEFILE:BOOL=FALSE -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
	cmake --build build

.PHONY : test
test : all
	cd build && ctest
	pytest tests/lang_python
