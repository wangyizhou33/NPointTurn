#include "gtest/gtest.h"
#include "../src/Paper.hpp"
#include <array>

TEST(PaperTests, BitVectorRead)
{
    std::array<uint32_t, 10> cells{};

    EXPECT_EQ(0, bitVectorRead(&cells.front(), 10));

    // put an on bit in the 31st bit
    cells[0] = 1u << 31;
    for (uint32_t i = 0; i < 32; ++i)
    {
        EXPECT_EQ(1 << (31 - i), bitVectorRead(&cells.front(), i));
    }
    for (uint32_t i = 32; i < 64; ++i)
    {
        EXPECT_EQ(0, bitVectorRead(&cells.front(), i));
    }

    // put an on bit in the 32nd bit
    cells[1] = 1 << 0;
    EXPECT_EQ(3, bitVectorRead(&cells.front(), 31));
}

TEST(PaperTests, BitVectorWrite)
{
    std::array<uint32_t, 10> cells{};

    // a random val
    uint32_t val = 1 << 15;

    // what's written should be exactly what's read
    for (uint32_t i = 0; i < 32 * 9; ++i)
    {
        cells.fill(0u); // reset

        bitVectorWrite(&cells.front(), val, i);
        EXPECT_EQ(val, bitVectorRead(&cells.front(), i));
    }

    // write should not update bits outside the range
    cells.fill(0u);
    cells[0] = 1 << 0;
    cells[1] = 1 << 1;

    // write 32 off bits starting from the 1st bit
    bitVectorWrite(&cells.front(), 0, 1);

    // expect the cells unchanged
    EXPECT_EQ(1 << 0, cells[0]);
    EXPECT_EQ(1 << 1, cells[1]);
}

TEST(PaperTests, RaceConditionGPU)
{
    constexpr size_t N = 4;

    // 4 cells of off bits
    uint32_t *cell, *dev_cell;

    cell = (uint32_t*)malloc(N * sizeof(uint32_t));
    for (uint32_t i = 0; i < N; ++i)
    {
        cell[i] = 0u;
    }

    HANDLE_ERROR(cudaMalloc((void**)&dev_cell, N * sizeof(uint32_t)));
    HANDLE_ERROR(cudaMemcpy(dev_cell, cell, N * sizeof(uint32_t),
                            cudaMemcpyHostToDevice));

    uint32_t offset = 15u; // consistent with writeOnes
    writeOnes<<<1, N - 1>>>(dev_cell, offset);

    HANDLE_ERROR(cudaMemcpy(cell, dev_cell, N * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost));

    EXPECT_EQ(4294934528, bitVectorRead(cell, 0)); // 2^32 - 2^offset
    EXPECT_EQ(4294967295, bitVectorRead(cell, offset + 32 * 0));
    EXPECT_EQ(4294967295, bitVectorRead(cell, offset + 32 * 1));
    EXPECT_EQ(4294967295, bitVectorRead(cell, offset + 32 * 2));
    EXPECT_EQ(4294967295, bitVectorRead(cell, offset + 32 * 2));
    EXPECT_EQ(32767, bitVectorRead(cell, N - 1)); // 2^offset - 1

    // delete
    HANDLE_ERROR(cudaFree(dev_cell));
    free(cell);
}

TEST(PaperTests, RaceConditionCPU)
{
    constexpr size_t N = 4;

    // 4 cells of off bits
    uint32_t* cell;

    cell = (uint32_t*)malloc(N * sizeof(uint32_t));
    for (uint32_t i = 0; i < N; ++i)
    {
        cell[i] = 0u;
    }

    uint32_t offset = 15u; // consistent with writeOnes

    bitVectorWrite(cell, 4294967295, offset + 32 * 0);
    bitVectorWrite(cell, 4294967295, offset + 32 * 1);
    bitVectorWrite(cell, 4294967295, offset + 32 * 2);

    EXPECT_EQ(4294967295, bitVectorRead(cell, offset + 32 * 0));
    EXPECT_EQ(4294967295, bitVectorRead(cell, offset + 32 * 1));
    EXPECT_EQ(4294967295, bitVectorRead(cell, offset + 32 * 2));

    // delete
    free(cell);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}