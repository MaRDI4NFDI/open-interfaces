find_package(SUNDIALS)

if(NOT SUNDIALS_FOUND)
  message("SUNDIALS not found. Skipping `sundials_cvode` implementation.")
  return()
endif()

add_library(oif_ivp_sundials_cvode SHARED sundials_cvode.c)
target_link_libraries(oif_ivp_sundials_cvode PRIVATE m)
target_include_directories(oif_ivp_sundials_cvode
    PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(oif_ivp_sundials_cvode
    PRIVATE ${PROJECT_SOURCE_DIR}/lang_c/oif_impl/include)
target_link_libraries(oif_ivp_sundials_cvode PRIVATE oif_common_data_structures)
target_link_libraries(oif_ivp_sundials_cvode PRIVATE oif_common_util)
target_link_libraries(oif_ivp_sundials_cvode PRIVATE SUNDIALS::cvode)
