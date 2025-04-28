find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFI REQUIRED IMPORTED_TARGET libffi)

add_library(oif_python_conversion MODULE _conversion.c)
target_include_directories(
  oif_python_conversion
  PRIVATE ${CMAKE_SOURCE_DIR}/include
  PRIVATE ${FFI_INCLUDE_DIRS})
target_link_libraries(oif_python_conversion PRIVATE Python::Python)
target_link_libraries(oif_python_conversion PRIVATE Python::NumPy)
target_link_libraries(oif_python_conversion PRIVATE PkgConfig::FFI)
install(TARGETS oif_python_conversion LIBRARY DESTINATION ${Python_SITEARCH})

# Parameter `SUFFIX` adds extension suffix, for example,
# `.cpython-312-x86_64-linux-gnu.so` that matches the suffix used by Python's
# `setuptools`.
set_target_properties(
  oif_python_conversion
  PROPERTIES PREFIX ""
             OUTPUT_NAME "_conversion"
             SUFFIX ".${Python_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}")
