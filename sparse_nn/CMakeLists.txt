cmake_minimum_required(VERSION 3.18)

project(SparseNN CXX)

option(BUILD_PYBIND "Use BUILD_PYBIND" true) # true by default, by default, build pybind11 library, if false, builds executable
# to use the flag: cmake .. -DMY_FLAG,
if ( BUILD_PYBIND )
    add_subdirectory("./pybind11")
endif ()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -fopenmp")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()




### Library ###
link_directories("/opt/homebrew/opt/libomp/lib")
include_directories("/opt/homebrew/opt/libomp/include")

include_directories("./include")


if ( BUILD_PYBIND )
    pybind11_add_module(${PROJECT_NAME} main_pybind.cxx)
else ()
    add_executable(${PROJECT_NAME} main.cxx)
endif ()

target_link_libraries(${PROJECT_NAME} PRIVATE omp )
  
### Tests ###
add_subdirectory("./tests")
enable_testing()

add_test(NAME test_utils COMMAND $<TARGET_FILE:test_utils> WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(test_array PUBLIC omp)
add_test(NAME test_array COMMAND $<TARGET_FILE:test_array> WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(test_array_max PUBLIC omp)
add_test(NAME test_array_max COMMAND $<TARGET_FILE:test_array_max> WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(test_sparse_array PUBLIC omp)
add_test(NAME test_sparse_array COMMAND $<TARGET_FILE:test_sparse_array> WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(test_softmax PUBLIC omp)
add_test(NAME test_softmax COMMAND $<TARGET_FILE:test_softmax> WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(test_logsoftmax PUBLIC omp)
add_test(NAME test_logsoftmax COMMAND $<TARGET_FILE:test_logsoftmax> WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(test_relu PUBLIC omp)
add_test(NAME test_relu COMMAND $<TARGET_FILE:test_relu> WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_test(NAME test_flatten COMMAND $<TARGET_FILE:test_flatten> WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_test(NAME test_max COMMAND $<TARGET_FILE:test_max> WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(test_linear PUBLIC omp)
add_test(NAME test_linear COMMAND $<TARGET_FILE:test_linear> WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(test_convolution PUBLIC omp)
add_test(NAME test_convolution COMMAND $<TARGET_FILE:test_convolution> WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(test_max_3d_pooling PUBLIC omp)
add_test(NAME test_max_3d_pooling COMMAND $<TARGET_FILE:test_max_3d_pooling> WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(test_max_2d_pooling PUBLIC omp)
add_test(NAME test_max_2d_pooling COMMAND $<TARGET_FILE:test_max_2d_pooling> WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(test_sparse_linear PUBLIC omp)
add_test(NAME test_sparse_linear COMMAND $<TARGET_FILE:test_sparse_linear> WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(test_sparse_convolution PUBLIC omp)
add_test(NAME test_sparse_convolution COMMAND $<TARGET_FILE:test_sparse_convolution> WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(test_neural_network PUBLIC omp)
add_test(NAME test_neural_network COMMAND $<TARGET_FILE:test_neural_network> WORKING_DIRECTORY ${CMAKE_BINARY_DIR})