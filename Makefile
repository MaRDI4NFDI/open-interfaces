.PHONY : all
all :
	cmake -S . -B build && \
	cmake --build build && \
	cp build/liboif_dispatch.so .

