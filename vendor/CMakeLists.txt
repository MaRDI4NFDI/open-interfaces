include(FetchContent)
FetchContent_Declare(
  hashmap
  GIT_REPOSITORY https://github.com/DavidLeeds/hashmap.git
  GIT_TAG v2.0.3 # Nov 6, 2023
)

FetchContent_MakeAvailable(hashmap)

FetchContent_Declare(
  cwpack
  GIT_REPOSITORY https://github.com/clwi/CWPack
  GIT_TAG master)
FetchContent_MakeAvailable(cwpack)
# To add `cwpack`, use the variable ${cwpack_SOURCE_DIR} to refer to its files.

set_property(GLOBAL PROPERTY cwpack_SOURCE_DIR "${cwpack_SOURCE_DIR}")
# To use in another CMakeLists.txt: get_property(cwpack_SOURCE_DIR GLOBAL
# PROPERTY cwpack_SOURCE_DIR)
