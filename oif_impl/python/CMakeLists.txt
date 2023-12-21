find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)

add_library(oif_dispatch_python SHARED dispatch_python.c)
target_include_directories(oif_dispatch_python
                           PRIVATE ${CMAKE_SOURCE_DIR}/oif/include)
target_link_libraries(oif_dispatch_python PUBLIC Python::Python)
target_link_libraries(oif_dispatch_python PUBLIC Python::NumPy)
target_link_libraries(oif_dispatch_python PRIVATE oif_c)
