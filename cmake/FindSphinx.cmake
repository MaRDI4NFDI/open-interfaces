include(FindPackageHandleStandardArgs)

find_program(
  SPHINX_EXECUTABLE
  NAMES sphinx-build sphinx-build.exe
  HINTS ${OIF_VENV_PATH}/bin)
mark_as_advanced(SPHINX_EXECUTABLE)

find_package_handle_standard_args(Sphinx DEFAULT_MSG SPHINX_EXECUTABLE)
