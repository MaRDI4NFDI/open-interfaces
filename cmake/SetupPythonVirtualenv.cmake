find_package(Python3 COMPONENTS Interpreter)
set(OIF_VENV_PATH ${CMAKE_CURRENT_BINARY_DIR}/venv)
execute_process(COMMAND "${Python3_EXECUTABLE}" -m venv ${OIF_VENV_PATH})

# Here is the trick update the environment with VIRTUAL_ENV variable (mimic the
# activate script)
set(ENV{VIRTUAL_ENV} "${CMAKE_CURRENT_BINARY_DIR}/venv")
# change the context of the search
set(Python3_FIND_VIRTUALENV FIRST)
# unset Python3_EXECUTABLE because it is also an input variable (see
# documentation, Artifacts Specification section)
unset(Python3_EXECUTABLE)
# Launch a new search
find_package(Python3 COMPONENTS Interpreter Development)

execute_process(
  COMMAND
    "${Python3_EXECUTABLE}" -m pip install -r
    ${PROJECT_SOURCE_DIR}/requirements.txt --cache-dir
    ${CMAKE_CURRENT_BINARY_DIR}/pip_cache)
set(OIF_PYTEST ${OIF_VENV_PATH}/bin/pytest)
