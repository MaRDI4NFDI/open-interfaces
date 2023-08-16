UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)  # macOS
    DSO_EXT := dylib
else                    # Linux
    DSO_EXT := so
endif

.PHONY : all
all :
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
	cmake --build build && \
	cp build/liboif_dispatch.$(DSO_EXT) . && \
	cp build/src/oif/backend_c/liboif_backend_c.$(DSO_EXT) .

.PHONY : test
test :
	cd build && ctest
