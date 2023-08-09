.PHONY : all
all :
	cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
	cmake --build build && \
	cp build/liboif_dispatch.so .

