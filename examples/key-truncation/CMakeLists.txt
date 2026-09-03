cmake_minimum_required (VERSION 3.25.2)

# C++ Config

set (CMAKE_CXX_STANDARD 20)

# Project

project (key-truncation LANGUAGES CXX)

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Packages

find_package (fideslib REQUIRED CONFIG)

# FIDESlib configuration.

if (fideslib_FOUND)
    message (STATUS "FOUND PACKAGE FIDESlib")
    message (STATUS "FIDESlib Version: ${fideslib_VERSION}")
    message (STATUS "FIDESlib header files location: ${FIDESLIB_INCLUDE_PATH}")
    message (STATUS "FIDESlib lib files location: ${FIDESLIB_LIBRARIES_PATH}")
    message (STATUS "FIDESlib binary files location: ${FIDESLIB_BINARY_PATH}")
    message (STATUS "FIDESlib C++ compiler: ${FIDESLIB_CXX_COMPILER}")
else ()
    message (FATAL_ERROR "PACKAGE FIDESlib NOT FOUND")
endif ()

set (CMAKE_CXX_COMPILER ${FIDESLIB_CXX_COMPILER})
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FIDESLIB_CXX_FLAGS}")

# Executable definition.

set (SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# One executable per source file: each is a self-contained main().
foreach (EXAMPLE key-truncation key-grow-repro)
    string (REPLACE "-" "_" EXAMPLE_SOURCE ${EXAMPLE})
    set (EXAMPLE_SOURCE ${SOURCE_DIR}/${EXAMPLE_SOURCE}.cpp)
    set_source_files_properties(${EXAMPLE_SOURCE} PROPERTIES LANGUAGE CXX)
    add_executable (${EXAMPLE} ${EXAMPLE_SOURCE})
    target_link_libraries (${EXAMPLE} PRIVATE fideslib::fideslib)
endforeach ()