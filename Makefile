UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)  # macOS
    DSO_EXT := dylib
else                    # Linux
    DSO_EXT := so
endif

.PHONY : all
all :
	cmake -S . -B build -DCMAKE_VERBOSE_MAKEFILE:BOOL=FALSE -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
	cmake --build build && \
	cp build/oif/liboif_dispatch.$(DSO_EXT) . && \
	cp build/oif/lang_c/liboif_c.$(DSO_EXT) .
	cp build/oif_impl/c/liboif_dispatch_c.$(DSO_EXT) .
	cp build/oif_impl/python/liboif_dispatch_python.$(DSO_EXT) .
	cp build/oif_impl/impl/qeq/c_qeq_solver/liboif_qeq_c_qeq_solver.$(DSO_EXT) .
	cp build/oif_impl/impl/linsolve/c_lapack/liboif_linsolve_c_lapack.$(DSO_EXT) .

.PHONY : test
test : all
	pytest tests/lang_python
	cd build && ctest
