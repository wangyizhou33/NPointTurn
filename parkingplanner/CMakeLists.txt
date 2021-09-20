# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
set(CODEOWNERS @yizhouw)

project(dw_parkingplanner C CXX)

#-------------------------------------------------------------------------------
# Project files
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Enable static analysis
#-------------------------------------------------------------------------------
sdk_enable_static_analysis()

set(PUBLIC_HEADERS
    ParkingPlanner.h
)

set(SOURCES
    ParkingPlanner2.hpp
    ParkingPlanner2.cpp
    ParkingPlanner2.cu
    Coord3d.hpp
    ParkingPlanner_interface.cpp
)

set(PRIVATE_LIBRARIES
    dw_core
    dw_worldmodel
    dw_math
    dw_experimental_behaviorplanner
    dw_experimental_pathunstructured
)

sdk_add_module_library("${PROJECT_NAME}"
                       "${PUBLIC_HEADERS}"
                       "${SOURCES}"
                       "${PUBLIC_C_LIBRARIES}"
                       "${PUBLIC_CXX_LIBRARIES}"
                       "${PRIVATE_LIBRARIES}")

target_compile_options(${PROJECT_NAME}_object PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                 -use_fast_math
                 >)
#-------------------------------------------------------------------------------
# Enable auto formatting
#-------------------------------------------------------------------------------
sdk_enable_auto_formatting("${CMAKE_CURRENT_SOURCE_DIR}")
