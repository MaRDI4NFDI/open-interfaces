enable_testing()

configure_file(${PROJECT_SOURCE_DIR}/oif_env_pytest.in
               ${CMAKE_CURRENT_BINARY_DIR}/oif_pytest.env)
file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/oif_pytest.env OIF_STRING_ENV
     REGEX "^[a-zA-Z]")

function(oif_add_test)
  include(CMakeParseArguments)
  set(OPTIONS)
  set(SINGLE_VALUE_ARGS NAME)
  set(MULTI_VALUE_ARGS COMMAND)
  cmake_parse_arguments(OIF_ADD_TEST "${OPTIONS}" "${SINGLE_VALUE_ARGS}"
                        "${MULTI_VALUE_ARGS}" ${ARGN})

  if(OIF_ADD_TEST_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR
        "Unrecognized arguments for oif_add_test: '${OIF_ADD_TEST_UNPARSED_ARGUMENTS}'"
    )
  endif()

  add_test(NAME "${OIF_ADD_TEST_NAME}" COMMAND ${OIF_ADD_TEST_COMMAND})
  set_property(TEST "${OIF_ADD_TEST_NAME}" PROPERTY ENVIRONMENT
                                                    "${OIF_STRING_ENV}")
  set_tests_properties("${OIF_ADD_TEST_NAME}" PROPERTIES DEPENDS build_all_lang)
endfunction()

add_test(NAME build_all_lang
         COMMAND ${CMAKE_COMMAND} --build ${PROJECT_BINARY_DIR} --target
                 all_lang --config $<CONFIGURATION>)

add_subdirectory(python)
add_subdirectory(cpp)
add_subdirectory(c)
add_subdirectory(julia)
add_subdirectory(r)