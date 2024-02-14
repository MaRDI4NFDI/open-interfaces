find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)

add_library(oif_python_conversion MODULE _conversion.c)
target_include_directories(oif_python_conversion
                           PRIVATE ${CMAKE_SOURCE_DIR}/oif/include)
target_link_libraries(oif_python_conversion PRIVATE Python::Python)
target_link_libraries(oif_python_conversion PRIVATE Python::NumPy)

# Parameter `SUFFIX` adds extension suffix, for example,
# `.cpython-312-x86_64-linux-gnu.so` that matches the suffix used by Python's
# `setuptools`.
set_target_properties(
  oif_python_conversion
  PROPERTIES PREFIX ""
             OUTPUT_NAME "_conversion"
             SUFFIX ".${Python_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}")
