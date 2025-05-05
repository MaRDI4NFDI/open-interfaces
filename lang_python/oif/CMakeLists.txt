# CMakeLists.txt for oif/lang_python directory.
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)

add_subdirectory(openinterfaces)
