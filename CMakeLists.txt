cmake_minimum_required(VERSION 3.16)
project(CylinderTag VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# Find pybind11 (optional for Python bindings)
find_package(pybind11 QUIET)

# Core library with shared implementation
add_library(cylindertag_core
    CylinderTag.cpp
    corner_detector.cpp
    pose_estimation.cpp
)
target_include_directories(cylindertag_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/header
)
target_link_libraries(cylindertag_core
    PUBLIC
        ${OpenCV_LIBS}
        ${CERES_LIBRARIES}
)
set_target_properties(cylindertag_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# CLI executable
add_executable(${PROJECT_NAME} main.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/header)
target_link_libraries(${PROJECT_NAME} PRIVATE cylindertag_core)

# Optional Python module
if(pybind11_FOUND)
    pybind11_add_module(cylindertag_cpp bindings/cylindertag_py.cpp)
    target_include_directories(cylindertag_cpp PRIVATE ${PROJECT_SOURCE_DIR}/header)
    target_link_libraries(cylindertag_cpp PRIVATE cylindertag_core)
endif()
