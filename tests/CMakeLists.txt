# When using scikit-build-core for building Python package, we do not need C
# tests and GTest.
if(${SKBUILD})
  return()
endif()

find_package(GTest CONFIG)

if(NOT GTest_FOUND)
  message("GTest not found. Fetching GTest from GitHub")

  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  )
  # For Windows: Prevent overriding the parent project's compiler/linker
  # settings
  set(gtest_force_shared_crt
      ON
      CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

add_subdirectory(lang_c)
