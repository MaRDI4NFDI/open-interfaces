if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ivp/CMakeLists.txt)
  add_subdirectory(ivp)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/linsolve/CMakeLists.txt)
  add_subdirectory(linsolve)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/qeq/CMakeLists.txt)
  add_subdirectory(qeq)
endif()

file(
  GLOB_RECURSE OIF_IMPL_CONF_FILES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  *.conf)

foreach(file IN LISTS OIF_IMPL_CONF_FILES)
  # Get the directory of the current file
  get_filename_component(dir ${file} DIRECTORY)

  # Create the corresponding directory structure in the destination
  file(MAKE_DIRECTORY openinterfaces/data/${dir})

  install(FILES ${file} DESTINATION openinterfaces/data/${dir})
endforeach()
