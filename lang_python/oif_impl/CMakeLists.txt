find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFI REQUIRED IMPORTED_TARGET libffi)

add_library(oif_bridge_python SHARED bridge_python.c)
target_include_directories(oif_bridge_python
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(oif_bridge_python PRIVATE ${FFI_INCLUDE_DIRS})
target_link_libraries(oif_bridge_python PUBLIC Python::Python)
target_link_libraries(oif_bridge_python PUBLIC Python::NumPy)
target_link_libraries(oif_bridge_python PRIVATE oif_common_data_structures)
target_link_libraries(oif_bridge_python PRIVATE oif_common_util)
install(TARGETS oif_bridge_python DESTINATION lib)

add_library(oif_bridge_python_callback MODULE _callback.c)
target_include_directories(
  oif_bridge_python_callback
  PRIVATE ${CMAKE_SOURCE_DIR}/include
  PRIVATE ${FFI_INCLUDE_DIRS})
target_link_libraries(oif_bridge_python_callback PRIVATE Python::Python)
target_link_libraries(oif_bridge_python_callback PRIVATE Python::NumPy)
target_link_libraries(oif_bridge_python_callback PRIVATE PkgConfig::FFI)
if(${SKBUILD})
  install(TARGETS oif_bridge_python_callback
          DESTINATION ${SKBUILD_PROJECT_NAME})
endif()

# Parameter `SUFFIX` adds extension suffix, for example,
# `.cpython-312-x86_64-linux-gnu.so` that matches the suffix used by Python's
# `setuptools`.
set_target_properties(
  oif_bridge_python_callback
  PROPERTIES PREFIX ""
             OUTPUT_NAME "_callback"
             SUFFIX ".${Python_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}")

add_subdirectory(openinterfaces)
