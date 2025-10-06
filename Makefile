package := openinterfaces

## Build in release mode. Default target
.PHONY : all
all :
	cmake -S . -B build.release \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		&& \
	cmake --build build.release && \
	rm -f build && \
	ln -sv build.release build


## Build in release mode (alias for `make all` or simply `make`)
.PHONY : release
release: all

## Show this help message
help:
	@awk ' \
		/^##/ {sub(/^##[ ]?/, "", $$0); doc=$$0; next} \
		/^[a-zA-Z0-9_-]+ ?:/ && doc { \
			sub(/:.*/, "", $$1); \
			printf "\033[1;32m%-24s\033[0m %s\n", $$1, doc; \
			doc=""; \
		} \
	' $(MAKEFILE_LIST)


## Run all tests
.PHONY : test
test :
	@echo "=== C tests ==="
	cd build && ctest --output-on-failure
	@echo "=== Julia tests ==="
	julia tests/lang_julia/runtests.jl
	@echo "=== Python tests ==="
	pytest tests/lang_python


## Build in debug mode (without optimizations)
.PHONY : debug
debug :
	cmake -S . -B build.debug \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		&& \
	cmake --build build.debug && \
	rm -f build && \
	ln -sv build.debug build

## Run Python tests checking for memory leaks using Valgrind
.PHONY : pytest-valgrind
pytest-valgrind :
	PYTHONMALLOC=malloc valgrind --show-leak-kinds=definite --log-file=/tmp/valgrind-output \
	python -m pytest -s -vv --valgrind --valgrind-log=/tmp/valgrind-output

## Build C code with verbose debug information and sanitizers to detect memory errors
.PHONY : debug-verbose-info-and-sanitize-address
debug-verbose-info-and-sanitize-address :
	cmake -S . -B build.debug_verbose_info_and_sanitize_address \
		-G Ninja \
		-DCMAKE_VERBOSE_MAKEFILE:BOOL=TRUE \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DOIF_OPTION_VERBOSE_DEBUG_INFO=ON \
        -DOIF_OPTION_SANITIZE=ON \
		&& \
	cmake --build build.debug_verbose_info_and_sanitize_address && \
	rm -f build && \
	ln -sv build.debug_verbose_info_and_sanitize_address build

## Build with verbose debug information
.PHONY : debug-verbose-info
debug-verbose-info :
	cmake -S . -B build.debug_verbose_info \
		-G Ninja \
		-DCMAKE_VERBOSE_MAKEFILE:BOOL=TRUE \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DOIF_OPTION_VERBOSE_DEBUG_INFO=ON \
		&& \
	cmake --build build.debug_verbose_info && \
	rm -f build && \
	ln -sv build.debug_verbose_info build

## Remove all existing build directories
.PHONY : clean
clean :
	$(RM) -r build build.debug build.debug_verbose_info_and_sanitize_address build.release

## Build docs in the HTML format
.PHONY : docs
docs : | mk-docs-build-dir
	cd docs && doxygen && make html

## Remove docs and build them from scratch
.PHONY : docs-from-scratch
docs-from-scratch:
	cd docs && rm -rf build && mkdir build && doxygen && make html

.PHONY : mk-docs-build-dir
mk-docs-build-dir:
	mkdir -p docs/build

## Build the Python sdist package, unpack it, and open the directory
.PHONY : build-package-python
build-package-python :
	@{ \
	version=$$(grep 'version =' pyproject.toml | sed 's/version = "//' | sed 's/"//'); \
	echo "\033[01;32mBuilding package: $(package) version $${version}\033[0m"; \
	rm -r "dist/$(package)-$${version}/" && echo "deleted old dist/$(package)-$${version}/"; \
	\
	if ! python -m build --sdist ; then \
	    echo -e "\033[01;31mERROR: build failed\033[0m"; \
	    exit 1; \
	fi; \
	tar xzf "dist/$(package)-$${version}.tar.gz" -C dist/; \
	open "dist/$(package)-$${version}/"; \
	}

.PHONY : upload-package-python-test
upload-package-python-test :
	@{ \
	version=$$(grep 'version =' pyproject.toml | sed 's/version = "//' | sed 's/"//'); \
	echo "\033[01;32mUploading package: $(package) version $${version}\033[0m"; \
	if ! python -m twine upload --repository testpypi dist/$(package)-$${version}.tar.gz ; then \
	    echo -e "\033[01;31mERROR: upload failed\033[0m"; \
	    exit 1; \
	fi; \
	}

## Upload build Python package
.PHONY : upload-package-python
upload-package-python :
	@{ \
	version=$$(grep 'version =' pyproject.toml | sed 's/version = "//' | sed 's/"//'); \
	echo "\033[01;32mUploading package: $(package) version $${version}\033[0m"; \
	if ! python -m twine upload dist/$(package)-$${version}.tar.gz ; then \
	    echo -e "\033[01;31mERROR: upload failed\033[0m"; \
	    exit 1; \
	fi; \
	}
