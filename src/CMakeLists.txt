cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME dolfin-scaling-test)
project(${PROJECT_NAME})

# Use C++11
set(CMAKE_CXX_STANDARD 11)

# Require C++11
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# add HPCReport lib
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(hpcreport)
add_library(hpcreportlib hpcreport/HPCReport.cpp)
target_link_libraries(hpcreportlib ${Boost_LIBRARIES})

# Get DOLFIN configuration data (DOLFINConfig.cmake must be in
# DOLFIN_CMAKE_CONFIG_PATH)
find_package(DOLFIN REQUIRED)
include(${DOLFIN_USE_FILE})

set(CMAKE_BUILD_TYPE "Release")

# Executable
add_executable(${PROJECT_NAME} main.cpp)

# Target libraries
target_link_libraries(${PROJECT_NAME} hpcreportlib dolfin)
