find_package(GTest CONFIG)

if(NOT GTest_FOUND)
  message("GTest not found. Skipping C tests.")
  return()
endif()

include(GoogleTest)

add_subdirectory(lang_c)
