package := openinterfaces

.PHONY : all
all :
	cmake -S . -B build.debug \
		-G Ninja \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		&& \
	cmake --build build.debug && \
	rm -f build && \
	ln -sv build.debug build

.PHONY : test
test :
	@echo "=== C tests ==="
	cd build && ctest --output-on-failure
	@echo "=== Julia tests ==="
	julia tests/lang_julia/runtests.jl
	@echo "=== Python tests ==="
	pytest tests/lang_python

.PHONY : pytest-valgrind
pytest-valgrind :
	PYTHONMALLOC=malloc valgrind --show-leak-kinds=definite --log-file=/tmp/valgrind-output \
	python -m pytest -s -vv --valgrind --valgrind-log=/tmp/valgrind-output

.PHONY : release
release :
	cmake -S . -B build.release \
		-G Ninja \
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
docs : | mk-docs-build-dir
	cd docs && doxygen && make html

.PHONY : docs-from-scratch
docs-from-scratch:
	cd docs && rm -rf build && mkdir build && doxygen && make html

.PHONY : mk-docs-build-dir
mk-docs-build-dir:
	mkdir -p docs/build

# Build the Python sdist package, unpack it, and open the directory.
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
