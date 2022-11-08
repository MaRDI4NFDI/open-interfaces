oif_add_test(
  NAME
  pytest
  COMMAND
  ${OIF_PYTEST}
  -c
  ${PROJECT_SOURCE_DIR}/setup.cfg
  ${PROJECT_SOURCE_DIR}/tests/python)

add_test(
  NAME requirements
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${OIF_VENV_PATH}/bin/check_reqs
          ${PROJECT_SOURCE_DIR}/requirements.txt)
