cmake_minimum_required(VERSION 3.1)

project(SAI2-MASTER)

# CMake settings
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


######################
# External libraries #
######################

set(SAI2_EXTERNAL_DIR ${PROJECT_SOURCE_DIR}/lib)

# Eigen
set(EIGEN3_INCLUDE_DIR ${SAI2_EXTERNAL_DIR}/eigen)

# RBDL
# TODO use FindRBDL.cmake in rbdl/examples
set(RBDL_DIR ${SAI2_EXTERNAL_DIR}/rbdl)
file(GLOB RBDL_PATHS ${RBDL_DIR}/build* ${RBDL_DIR}/build*/addons/urdfreader)
find_library(RBDL_LIBRARY rbdl REQUIRED PATHS ${RBDL_PATHS})
find_library(RBDL_URDFReader_LIBRARY rbdl_urdfreader REQUIRED PATHS ${RBDL_PATHS})
find_path(RBDL_URDFReader_INCLUDE_DIR rbdl/addons/urdfreader/urdfreader.h HINTS ${SAI2_EXTERNAL_DIR})
file(GLOB RBDL_INCLUDE_DIR ${RBDL_DIR}/include ${RBDL_DIR}/build*/include)
set(RBDL_FOUND true)

# Chai3d
# TODO: Find better place to put this?
find_package(CHAI3D PATHS ${SAI2_EXTERNAL_DIR}/chai3d.git/build)
find_library(DRD_LIBRARY drd PATHS ${CHAI3D_LIBRARY_DIRS})
find_library(CHAI3D_LIBRARY chai3d paths ${CHAI3D_DIR})
list(REMOVE_ITEM CHAI3D_LIBRARIES drd)
list(REMOVE_ITEM CHAI3D_LIBRARIES chai3d)
set(CHAI3D_LIBRARIES ${CHAI3D_LIBRARY} ${DRD_LIBRARY} ${CHAI3D_LIBRARIES})

# SAI2 Common
set(BUILD_EXAMPLES OFF CACHE BOOL "")
add_subdirectory(${SAI2_EXTERNAL_DIR}/sai2-simulation.git)
add_subdirectory(${SAI2_EXTERNAL_DIR}/sai2-common.git)

# TODO: Find local way to do this
find_package(SAI2-COMMON PATHS ${SAI2_EXTERNAL_DIR}/sai2-common.git/build)
include_directories(${SAI2-COMMON_INCLUDE_DIRS})


###################
# Project Sources #
###################

# Set include include directories
include_directories(${SAI2_EXTERNAL_DIR})
include_directories(${PROJECT_SOURCE_DIR}/src)

# Set common libraries to be linked
set(SAI2_LIBRARIES
	${SAI2-COMMON_LIBRARY}
)

# Add specific apps
add_subdirectory(src/demo_project)
add_subdirectory(src/simulation)
add_subdirectory(src/visualization)
add_subdirectory(src/gym_sai2)
# add_subdirectory(python)
