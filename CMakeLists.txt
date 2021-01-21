
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(NPointTurn LANGUAGES CXX CUDA)

find_package(GTest REQUIRED)
enable_testing()

FIND_PACKAGE(CUDA REQUIRED)
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

#### build library ####
add_library(paper STATIC
    src/Paper.cu
)

target_compile_features(paper PUBLIC cxx_std_14)

set_target_properties(paper
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(Solver STATIC
    # util code
    src/Types.hpp
    src/Helper.hpp
    src/Polygon.hpp
    src/Polygon.cpp
    src/Freespace.hpp
    src/Freespace.cu
    src/Obstacle.hpp

    # solver code
    src/Solver.hpp
    src/Solver.cpp
    src/GenericCostSolver.hpp
    src/GenericCostSolver.cu
    src/MinimumTurnSolver.hpp
    src/MinimumTurnSolver.cu

    # visualization code
    src/SavePPM.hpp
    src/SimpleDisp.hpp
    src/Visualization.hpp
    src/Visualization.cpp
)

target_compile_features(Solver PUBLIC cxx_std_14)

set_target_properties(Solver
                     PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#### build test ####
# to run test
# >$ dkb make test
add_executable(Test
    tests/tests.cu
    tests/VectorTests.cpp
    tests/SolverTests.cpp
    tests/PolygonTests.cpp
    tests/FreespaceTests.cpp
    tests/SOLTests.cu
    tests/main.cpp
)

target_link_libraries(Test
    gtest
    paper
    Solver
    ${CUDA_LIBRARIES}
)

set_property(TARGET Test 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

gtest_add_tests(Test "" AUTO)

#### build executable ####
add_executable(main
    src/main.cu
)

target_link_libraries(main
    paper
    ${CUDA_LIBRARIES}
)

set_property(TARGET main 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
