oif_add_test(
  NAME
  test_julia
  COMMAND
  ${Julia_EXECUTABLE}
  --project
  ${PROJECT_SOURCE_DIR}/lang_julia/package
  ${CMAKE_CURRENT_SOURCE_DIR}/test_all.jl)

set_tests_properties(test_julia PROPERTIES DEPENDS build_all_lang)
