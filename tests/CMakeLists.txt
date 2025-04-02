if(IS_DIRECTORY lang_c)
  find_package(GTest CONFIG REQUIRED)
  include(GoogleTest)

  add_subdirectory(lang_c)
endif()
