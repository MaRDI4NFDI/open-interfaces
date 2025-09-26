# CMakeLists.txt for oif/lang_python directory.
find_package(Python COMPONENTS Interpreter Development NumPy)

if(NOT Python_FOUND)
  message(STATUS "Python not found")
  return()
endif()

add_subdirectory(oif)
add_subdirectory(oif_impl)
