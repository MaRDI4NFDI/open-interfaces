include(ExternalProject)

FetchContent_Declare(
  extern_flexiblas
  GIT_REPOSITORY
    "https://gitlab.mpi-magdeburg.mpg.de/software/flexiblas-release"
  GIT_TAG "v3.2.0"
  SOURCE_DIR ${PROJECT_BINARY_DIR}/flexiblas_src BINARY_DIR
  ${PROJECT_BINARY_DIR}/flexiblas_build)

FetchContent_MakeAvailable(extern_flexiblas)
