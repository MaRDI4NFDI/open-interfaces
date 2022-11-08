oif_add_test(NAME test_r COMMAND ${R_SCRIPT} --vanilla
             ${CMAKE_CURRENT_SOURCE_DIR}/test_all.r)

set_tests_properties(test_r PROPERTIES DEPENDS build_all_lang)
