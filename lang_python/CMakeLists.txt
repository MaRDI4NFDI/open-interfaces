find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_custom_target(
  run_lang_python
  ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/oif_python.py
  ${CMAKE_CURRENT_BINARY_DIR}/../connector/liboif_connector.so
  DEPENDS oif_connector oif_julia)
