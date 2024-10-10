# CMakeLists.txt for oif/lang_python directory.
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)

add_library(oif_lang_python_convert MODULE _convert.c)
target_include_directories(oif_lang_python_convert
                           PRIVATE ${CMAKE_SOURCE_DIR}/oif/include)
target_link_libraries(oif_lang_python_convert PRIVATE Python::Python)
target_link_libraries(oif_lang_python_convert PRIVATE Python::NumPy)

add_subdirectory(oif)
