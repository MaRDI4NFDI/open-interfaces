find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)

add_library(oif_dispatch_python SHARED dispatch_python.c)
target_include_directories(oif_dispatch_python
                           PRIVATE ${CMAKE_SOURCE_DIR}/oif/include)
target_link_libraries(oif_dispatch_python PUBLIC Python::Python)
target_link_libraries(oif_dispatch_python PUBLIC Python::NumPy)
target_link_libraries(oif_dispatch_python PRIVATE oif_c)

add_library(oif_dispatch_python_callback MODULE _callback.c)
target_include_directories(oif_dispatch_python_callback
                           PRIVATE ${CMAKE_SOURCE_DIR}/oif/include)
target_link_libraries(oif_dispatch_python_callback PRIVATE Python::Python)
target_link_libraries(oif_dispatch_python_callback PRIVATE Python::NumPy)
target_link_libraries(oif_dispatch_python_callback PRIVATE ffi)

# Parameter `SUFFIX` adds extension suffix, for example,
# `.cpython-312-x86_64-linux-gnu.so` that matches the suffix used by Python's
# `setuptools`.
set_target_properties(
  oif_dispatch_python_callback
  PROPERTIES PREFIX ""
             OUTPUT_NAME "_callback"
             SUFFIX ".${Python_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}")
