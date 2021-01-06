#include "gtest/gtest.h"
#include "../src/Solver.hpp"

TEST(SolverTests, dimension)
{
    EXPECT_EQ(128u, Solver::POS_DIM);
    EXPECT_EQ(512u, Solver::HDG_DIM);
}
