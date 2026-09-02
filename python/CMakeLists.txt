cmake_minimum_required(VERSION 3.25.2)
set(CMAKE_CXX_STANDARD 20)
project(pyfideslib LANGUAGES CXX)

# Python bindings for the fideslib API (CPU/OpenFHE or CUDA per context, see pyfideslib/__init__.py).
# Build:  cmake -S python -B build-py && cmake --build build-py
#         export PYTHONPATH=$PWD/python
find_package(fideslib REQUIRED CONFIG)
set(CMAKE_CXX_COMPILER ${FIDESLIB_CXX_COMPILER})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FIDESLIB_CXX_FLAGS}")

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG QUIET)
if (NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(pybind11 GIT_REPOSITORY https://github.com/pybind/pybind11.git GIT_TAG v2.13.6)
    FetchContent_MakeAvailable(pybind11)
endif ()

pybind11_add_module(_core src/bindings.cpp)
target_link_libraries(_core PRIVATE fideslib::fideslib)
set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/pyfideslib)
