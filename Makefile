.PHONY : all
all :
	cmake -S . -B build.debug \
		-DCMAKE_VERBOSE_MAKEFILE:BOOL=TRUE \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		&& \
	cmake --build build.debug && \
	rm -f build && \
	ln -sv build.debug build

.PHONY : test
test :
	cd build && ctest --output-on-failure
	pytest tests/lang_python

.PHONY : pytest-valgrind
pytest-valgrind :
	PYTHONMALLOC=malloc valgrind --show-leak-kinds=definite --log-file=/tmp/valgrind-output \
	python -m pytest -s -vv --valgrind --valgrind-log=/tmp/valgrind-output

.PHONY : release
release :
	cmake -S . -B build.release \
		-DCMAKE_VERBOSE_MAKEFILE:BOOL=FALSE \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		&& \
	cmake --build build.release && \
	rm -f build && \
	ln -sv build.release build

.PHONY : clean
clean :
	$(RM) -r build build.debug build.release

.PHONY : docs
docs :
	cd docs && doxygen && make html

.PHONY : docs-from-scratch
docs-from-scratch:
	cd docs && rm -rf build && doxygen && make html
