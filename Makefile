.PHONY : all
all :
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
	cmake --build build && \
	cp build/liboif_dispatch.so . && \
	cp build/src/oif/backend_c/liboif_backend_c.so .

