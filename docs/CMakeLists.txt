find_package(Sphinx)
find_package(Doxygen)

if(NOT (Sphinx_FOUND AND Doxygen_FOUND))
  message(STATUS "Cannot build documentation without Sphinx and Doxygen found.")
  add_custom_target(
    docs COMMAND echo
                 "doxygen or sphinx is missing, cannot build documentation.")
  return()
endif()

set(DOCS_OUTPUT_ROOT_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(DOXYGEN_XML_DIR ${DOCS_OUTPUT_ROOT_DIR}/doxygen_xml)
set(DOXYGEN_XML ${DOXYGEN_XML_DIR}/index.xml)
set(SPHINX_SOURCE_DIR_ORIGIN ${PROJECT_SOURCE_DIR}/docs/sphinx_src)
set(SPHINX_SOURCE_DIR ${DOCS_OUTPUT_ROOT_DIR}/sphinx_src)
set(SPHINX_HTML_DIR ${DOCS_OUTPUT_ROOT_DIR}/sphinx_html)
set(SPHINX_HTML ${SPHINX_HTML_DIR}/index.html)
set(SPHINX_SOURCE_INDEX ${SPHINX_SOURCE_DIR}/index.md)
file(GLOB_RECURSE DOXYGEN_INPUTS ${PROJECT_SOURCE_DIR}/*.c
     ${PROJECT_SOURCE_DIR}/*.cc ${PROJECT_SOURCE_DIR}/*.h
     ${PROJECT_SOURCE_DIR}/*.hh)
configure_file(conf.py.in conf.py @ONLY)
configure_file(Doxyfile.in Doxyfile @ONLY)
add_custom_target(docs)

add_custom_command(
  OUTPUT ${DOXYGEN_XML}
  DEPENDS Doxyfile ${DOXYGEN_INPUTS}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${DOXYGEN}
  COMMENT "Creating doxygen xml.")
add_custom_target(doxygen ALL DEPENDS ${DOXYGEN_XML})

add_custom_command(
  OUTPUT ${SPHINX_SOURCE_INDEX}
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${SPHINX_SOURCE_DIR}"
  COMMAND "${CMAKE_COMMAND}" -E copy_directory "${SPHINX_SOURCE_DIR_ORIGIN}"
          "${SPHINX_SOURCE_DIR}/")

add_custom_command(
  OUTPUT ${SPHINX_HTML}
  DEPENDS conf.py ${SPHINX_SOURCE_INDEX} doxygen
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${SPHINX_EXECUTABLE} -b html -v -c ${CMAKE_CURRENT_BINARY_DIR}
          ${SPHINX_SOURCE_DIR} ${SPHINX_HTML_DIR}
  COMMENT "Creating sphinx documentation.")

add_custom_target(sphinx DEPENDS ${SPHINX_HTML})
add_dependencies(docs sphinx)
