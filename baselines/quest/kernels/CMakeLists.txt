cmake_minimum_required(VERSION 3.26.4)
set(CMAKE_C_COMPILER "/usr/bin/gcc-11")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-11")
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# ------------- configure rapids-cmake --------------#
include(${CMAKE_SOURCE_DIR}/cmake/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

project(bsk LANGUAGES CUDA CXX)

# ------------- configure raft -----------------#
rapids_cpm_init()
include(${CMAKE_SOURCE_DIR}/cmake/get_raft.cmake)

# ------------- add gtest -----------------#
add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty/googletest)

# ------------- compile test_batch_decode -----------------#
add_executable(test_batch_decode ${CMAKE_SOURCE_DIR}/src/test/test_batch_decode.cu)
target_include_directories(test_batch_decode PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_batch_decode PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
target_include_directories(test_batch_decode PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/flashinfer/include)
target_include_directories(test_batch_decode PRIVATE ${CMAKE_SOURCE_DIR}/src/include)
target_link_libraries(test_batch_decode PRIVATE gtest gtest_main)

# ------------- compile test_page -----------------#
add_executable(test_page ${CMAKE_SOURCE_DIR}/src/test/test_page.cu)
target_include_directories(test_page PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_page PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
target_include_directories(test_page PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/flashinfer/include)
target_include_directories(test_page PRIVATE ${CMAKE_SOURCE_DIR}/src/include)
target_link_libraries(test_page PRIVATE gtest gtest_main)

# ------------- compile test_max_possible -----------------#
add_executable(test_max_possible ${CMAKE_SOURCE_DIR}/src/test/test_max_possible.cu)
target_include_directories(test_max_possible PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_max_possible PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
target_include_directories(test_max_possible PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/flashinfer/include)
target_include_directories(test_max_possible PRIVATE ${CMAKE_SOURCE_DIR}/src/include)
target_link_libraries(test_max_possible PRIVATE gtest gtest_main)

# ------------- compile test_prefill -----------------#
add_executable(test_prefill ${CMAKE_SOURCE_DIR}/src/test/test_prefill.cu)
target_include_directories(test_prefill PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_prefill PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
target_include_directories(test_prefill PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/flashinfer/include)
target_include_directories(test_prefill PRIVATE ${CMAKE_SOURCE_DIR}/src/include)
target_link_libraries(test_prefill PRIVATE gtest gtest_main)
