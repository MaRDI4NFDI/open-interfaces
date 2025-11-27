# How to release new version of Open Interfaces

- [ ] Bump version in the following files:
  * `CMakeLists.txt`
  * `pyproject.toml`
  * `version.py`
  * `conf.py`
  * `Project.toml`
  * `Manifest.toml`
  * `CITATION.cff`
- [ ] Run `pre-commit run --all-files`
- [ ] Make a pull request and make sure it passes all the checks
- [ ] Merge the pull request to `main`
- [ ] Go to the GitHub page of the repository and press "Releases"
- [ ] Press "Draft a new release"
- [ ] Create a new tag "vX.Y.Z" (e.g. "v0.1.0") for the `main` branch
- [ ] Press publish release
- [ ] In local working copy of the repository, switch to `main`:
  ```shell
  git checkout main
  ```
- [ ] Fetch changes from the remote repository (from your fork):
  ```shell
  git fetch origin
  ```
- [ ] Fetch tags from the upstream repository:
  ```shell
  git fetch --tags upstream
  ```
- [ ] Make sure that `main` is on the tagged commit:
  `git log` must report that HEAD, `main`, `origin/main`, and `upstream/main`
  are all the same;
  besides, it must have the tag from the above
- [ ] Build the Python package
  ``` shell
  make build-python-package
  ```
- [ ] Check that the package is build correctly by uploading to TestPyPI:
  ```shell
  make upload-package-python-test
  ```
- [ ] Install in a separate virtual environment:
  ```shell
  python -m pip install \
         --index-url https://test.pypi.org/simple/ \
         --extra-index-url https://pypi.org/simple/ \
         openinterfaces
  ```
- [ ] Check that the package is installed correctly and functional in IPython:
  ```ipython
  from openinterfaces.examples import call_ivp_from_python
  call_ivp_from_python.main()
  ```
- [ ] Upload the package to PyPI:
  ```shell
  make upload-package-python
  ```
- [ ] Drink a coffee/tee/a glass of water/a glass of wine 🍻.
