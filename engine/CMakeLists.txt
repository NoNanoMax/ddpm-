cmake_minimum_required(VERSION 3.20)
project(ddpm_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

file(GLOB_RECURSE DDPM_CORE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/ops/*.cpp)
list(FILTER DDPM_CORE_SRCS EXCLUDE REGEX "main\\.cpp$")

add_library(ddpm_core STATIC
    src/model.cpp
    ${DDPM_CORE_SRCS}
)
target_include_directories(ddpm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(ddpm-engine src/main.cpp)
target_link_libraries(ddpm-engine PRIVATE ddpm_core)

enable_testing()
add_executable(test_ops tests/test_ops.cpp)
target_link_libraries(test_ops PRIVATE ddpm_core)
add_test(NAME ops COMMAND test_ops)
