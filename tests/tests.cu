#include "gtest/gtest.h"
#include "../src/Paper.hpp"
#include <array>
#include <string>

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
    constexpr size_t N = 32 * 32;

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
    writeOnes<<<1, N>>>(dev_cell, offset);
    HANDLE_ERROR(cudaDeviceSynchronize());

    HANDLE_ERROR(cudaMemcpy(cell, dev_cell, N * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost));

    EXPECT_EQ(4294934528, bitVectorRead(cell, 0)); // 2^32 - 2^offset
    for (uint32_t i = 0; i + 1 < N; ++i)
    {
        uint32_t expect = 4294967295;
        uint32_t actual = bitVectorRead(cell, offset + 32 * i);

        if (expect != actual)
        {
            std::cerr << i << " " << expect << " " << actual << std::endl;
            EXPECT_TRUE(false);
        }
        EXPECT_EQ(4294967295, bitVectorRead(cell, offset + 32 * i));
    }
    EXPECT_EQ(32767, bitVectorRead(cell, 32 * (N - 1))); // 2^offset - 1

    // delete
    HANDLE_ERROR(cudaFree(dev_cell));
    free(cell);
}

TEST(PaperTests, TurnCoord1)
{
    // assert every theta slice is a pure translation
    // i.e. turnCoord(x, y, theta + 1) -> turnCoord(x, y, theta) is the same as
    // turnCoord(x+1, y, theta + 1) -> turnCoord(x+1, y, theta)
    // same thing for y

    uint32_t x = X_DIM / 2;
    uint32_t y = Y_DIM / 2;

    for (uint32_t theta = 0; theta < THETA_DIM; ++theta)
    {
        uint32_t a1 = turnCoord(x, y, theta, X_DIM, Y_DIM, POS_RES, HDG_RES, TURN_R);
        uint32_t a2 = turnCoord(x, y, theta + 1, X_DIM, Y_DIM, POS_RES, HDG_RES, TURN_R);
        uint32_t a  = a2 - a1;

        // use this to build the bit offset table
        std::cout << a << "u, " << std::endl; // the last one is garbage

        for (uint32_t i = 0u; i <= 5u; ++i)
        {
            uint32_t b1 = turnCoord(x + i, y, theta, X_DIM, Y_DIM, POS_RES, HDG_RES, TURN_R);
            uint32_t b2 = turnCoord(x + i, y, theta + 1, X_DIM, Y_DIM, POS_RES, HDG_RES, TURN_R);
            uint32_t b  = b2 - b1;

            EXPECT_EQ(a, b);
        }

        for (uint32_t i = 0u; i <= 5u; ++i)
        {
            uint32_t b1 = turnCoord(x, y + i, theta, X_DIM, Y_DIM, POS_RES, HDG_RES, TURN_R);
            uint32_t b2 = turnCoord(x, y + i, theta + 1, X_DIM, Y_DIM, POS_RES, HDG_RES, TURN_R);
            uint32_t b  = b2 - b1;

            EXPECT_EQ(a, b);
        }
    }
}

TEST(PaperTests, TurnCoord2)
{
    // assert the circular shape of the turn coord
    float32_t x{0.f};
    float32_t y{0.f};
    float32_t tol = 1e-4f;

    int32_t xprev{0};
    int32_t yprev{0};

    std::string left       = "R = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);\n";
    std::string right      = "R = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);\n";
    std::string up         = "i = ((i & 0xFFFFFE00) | ((i + 4) & 511));\n";
    std::string down       = "i = ((i & 0xFFFFFE00) | ((i - 4) & 511));\n";
    std::string shiftTheta = "i += 512u;\n";
    std::string read       = "R &= Fb[i];\nR |= RbI[i];\n";
    std::string write      = "RbO[i] = R;\n";

    auto comment = [](uint32_t i) {
        return "// theta : " + std::to_string(i) + "\n";
    };

    std::string code{};
    bool genCode{true};

    if (genCode)
    {
        code += "<<<<<< start of auto codegen\n";
        code += "uint32_t R = 0u;\n\n";
        code += comment(0u);
        code += read;
        code += write;
        code += shiftTheta;
        code += "\n";
    }

    for (uint32_t i = 1; i < THETA_DIM; ++i)
    {
        float32_t xout{};
        float32_t yout{};
        float32_t theta{static_cast<float32_t>(i) * HDG_RES};

        // center of the trajectory is (0, -TURN_R)
        turnCoord(xout, yout, x, y, theta, -TURN_R);
        EXPECT_NEAR(TURN_R * TURN_R, xout * xout + (yout + TURN_R) * (yout + TURN_R), tol);

        // center of the trajectory is (0, TURN_R)
        turnCoord(xout, yout, x, y, theta, TURN_R);
        EXPECT_NEAR(TURN_R * TURN_R, xout * xout + (yout - TURN_R) * (yout - TURN_R), tol);

        int32_t xround = static_cast<int32_t>(floor((xout / POS_RES) + 0.5f));
        int32_t yround = static_cast<int32_t>(floor((yout / POS_RES) + 0.5f));

        int32_t xdiff = xround - xprev;
        int32_t ydiff = yround - yprev;

        // std::cerr << "theta " << i
        //           << " xdiff " << xdiff
        //           << " ydiff " << ydiff << std::endl;

        xprev = xround;
        yprev = yround;

        if (genCode)
        {
            // read
            code += comment(i);
            code += read;

            if (xdiff == 1)
                code += left;
            else if (xdiff == -1)
                code += right;

            if (ydiff == 1)
                code += up;
            else if (ydiff == -1)
                code += down;

            code += write;
            code += shiftTheta;
            code += "\n";
        }
    }

    if (genCode)
    {
        code += ">>>>>> end of auto codegen\n";
        std::cout << code;
    }
}

TEST(PaperTests, temp)
{
    // std::array<uint32_t, 10> cells{};
    // cells[0] = 4294934528;
    // cells[1] = 4294967295;
    // cells[2] = 4294967295;

    // std::cerr << bitVectorRead(&cells.front(), 0) << std::endl;
    // std::cerr << bitVectorRead(&cells.front(), 1) << std::endl;
    // std::cerr << bitVectorRead(&cells.front(), 2) << std::endl;

    for (uint32_t cr = 0; cr < 32; ++cr)
    {
        std::cout << cr << " " << ~((1u << cr) - 1u) << std::endl;
    }
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

TEST(PaperTests, CountBits)
{
    uint32_t n = (1u << 1) + (1u << 10) + (1u << 15) + (1u << 30);

    EXPECT_EQ(4u, countBits(n));

    n = 4294967295;
    EXPECT_EQ(32u, countBits(n));
}

TEST(PaperTests, Reachability)
{
    uint32_t *dev_reach0, *dev_reach1;
    uint32_t *reach0, *reach1;
    uint32_t* dev_fb;

    HANDLE_ERROR(cudaMalloc((void**)&dev_reach0, SIZE));
    HANDLE_ERROR(cudaMalloc((void**)&dev_reach1, SIZE));

    HANDLE_ERROR(cudaMemset((void*)dev_reach0, 0, SIZE));
    HANDLE_ERROR(cudaMemset((void*)dev_reach1, 0, SIZE));

    reach0 = (uint32_t*)malloc(SIZE);
    reach1 = (uint32_t*)malloc(SIZE);

    memset((void*)reach0, 0, SIZE);
    memset((void*)reach1, 0, SIZE);

    HANDLE_ERROR(cudaMalloc((void**)&dev_fb, SIZE));
    HANDLE_ERROR(cudaMemset((void*)dev_fb, 2147483647, SIZE)); // set all ones

    // set reach0
    uint32_t middle = turnCoord(X_DIM / 2, Y_DIM / 2, 0,
                                X_DIM, Y_DIM, POS_RES, HDG_RES, TURN_R);

    bitVectorWrite(reach0, 4294967295u, middle);

    // std::cerr << reach0[4 * 64] << " " << reach0[4 * 64 + 1] << " " << reach0[4 * 64 + 2] << " " << reach0[4 * 6 + 3] << std::endl;

    HANDLE_ERROR(cudaMemcpy(dev_reach0, reach0, SIZE,
                            cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();

    TIME_PRINT("sweep ",
               newSweepTurn(dev_reach1,
                            dev_fb,
                            dev_reach0,
                            TURN_R,
                            nullptr);
               cudaDeviceSynchronize(););

    HANDLE_ERROR(cudaGetLastError());

    HANDLE_ERROR(cudaMemcpy(reach1, dev_reach1, SIZE,
                            cudaMemcpyDeviceToHost));

    // std::cerr << reach1[4 * 64] << " " << reach1[4 * 64 + 1] << " " << reach1[4 * 64 + 2] << " " << reach1[4 * 64 + 3] << std::endl;
    // std::cerr << reach1[4 * 63] << " " << reach1[4 * 63 + 1] << " " << reach1[4 * 63 + 2] << " " << reach1[4 * 63 + 3] << std::endl;

    // assert each theta slice has 32 ON-bits
    for (uint32_t theta = 0; theta < THETA_DIM; ++theta)
    {
        uint32_t startIndex = X_DIM * Y_DIM * theta;
        uint32_t endIndex   = X_DIM * Y_DIM * (theta + 1);

        uint32_t reachableBitCount = 0u;

        for (uint32_t coordIndex = startIndex / 32; coordIndex < endIndex / 32; coordIndex++)
        {
            reachableBitCount += countBits(reach1[coordIndex]);
        }
        if (reachableBitCount != 32)
            std::cerr << " theta " << theta
                      << " reachable bits " << reachableBitCount << std::endl;
    }

    // delete
    HANDLE_ERROR(cudaFree(dev_reach0));
    HANDLE_ERROR(cudaFree(dev_reach1));
    HANDLE_ERROR(cudaFree(dev_fb));

    free(reach0);
    free(reach1);
}

TEST(PaperTests, Obstacle)
{
    uint32_t *dev_reach0, *dev_reach1;
    uint32_t *reach0, *reach1;
    uint32_t* dev_fb;

    HANDLE_ERROR(cudaMalloc((void**)&dev_reach0, SIZE));
    HANDLE_ERROR(cudaMalloc((void**)&dev_reach1, SIZE));

    HANDLE_ERROR(cudaMemset((void*)dev_reach0, 0, SIZE));
    HANDLE_ERROR(cudaMemset((void*)dev_reach1, 0, SIZE));

    reach0 = (uint32_t*)malloc(SIZE);
    reach1 = (uint32_t*)malloc(SIZE);

    memset((void*)reach0, 0, SIZE);
    memset((void*)reach1, 0, SIZE);

    HANDLE_ERROR(cudaMalloc((void**)&dev_fb, SIZE));
    HANDLE_ERROR(cudaMemset((void*)dev_fb, 2147483647, SIZE / 2)); // set ones for half of the theta slices

    // set reach0
    uint32_t middle = turnCoord(X_DIM / 2, Y_DIM / 2, 0,
                                X_DIM, Y_DIM, POS_RES, HDG_RES, TURN_R);

    bitVectorWrite(reach0, 4294967295, middle);

    HANDLE_ERROR(cudaMemcpy(dev_reach0, reach0, SIZE,
                            cudaMemcpyHostToDevice));

    TIME_PRINT("sweep ",
               bitSweepTurn(dev_reach1,
                            dev_fb,
                            dev_reach0,
                            TURN_R,
                            nullptr);
               cudaDeviceSynchronize(););

    HANDLE_ERROR(cudaGetLastError());

    HANDLE_ERROR(cudaMemcpy(reach1, dev_reach1, SIZE,
                            cudaMemcpyDeviceToHost));

    // assert bits are ON at theta = 179
    uint32_t theta      = THETA_DIM / 2u - 1u;
    uint32_t startIndex = X_DIM * Y_DIM * theta;
    uint32_t endIndex   = X_DIM * Y_DIM * (theta + 1);

    uint32_t reachableBitCount = 0u;
    for (uint32_t coordIndex = startIndex / 32; coordIndex < endIndex / 32; coordIndex++)
    {
        reachableBitCount += countBits(reach1[coordIndex]);
    }
    EXPECT_EQ(32, reachableBitCount);

    // assert bits are OFF at theta = 180
    theta      = THETA_DIM / 2u;
    startIndex = endIndex;
    endIndex   = X_DIM * Y_DIM * (theta + 1);

    reachableBitCount = 0u;
    for (uint32_t coordIndex = startIndex / 32; coordIndex < endIndex / 32; coordIndex++)
    {
        reachableBitCount += countBits(reach1[coordIndex]);
    }
    EXPECT_EQ(0, reachableBitCount);

    // delete
    HANDLE_ERROR(cudaFree(dev_reach0));
    HANDLE_ERROR(cudaFree(dev_reach1));
    HANDLE_ERROR(cudaFree(dev_fb));

    free(reach0);
    free(reach1);
}

TEST(PaperTests, shuffle)
{
    constexpr size_t N = 8;

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

    shuffle<<<1, N>>>(dev_cell);
    HANDLE_ERROR(cudaDeviceSynchronize());

    HANDLE_ERROR(cudaMemcpy(cell, dev_cell, N * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost));

    for (uint32_t i = 1u; i < N; ++i)
    {
        EXPECT_EQ(1 << (i - 1u), cell[i]);
    }
    EXPECT_EQ((1 << 7), cell[0]);
}

TEST(PaperTests, SOL)
{
    uint32_t *dev_reach0, *dev_reach1;
    uint32_t *reach0, *reach1;

    size_t SIZE1 = SIZE;

    HANDLE_ERROR(cudaMalloc((void**)&dev_reach0, SIZE1));
    HANDLE_ERROR(cudaMemset((void*)dev_reach0, 0, SIZE1));
    HANDLE_ERROR(cudaMalloc((void**)&dev_reach1, SIZE1));
    HANDLE_ERROR(cudaMemset((void*)dev_reach1, 0, SIZE1));

    reach0 = (uint32_t*)malloc(SIZE1);
    reach1 = (uint32_t*)malloc(SIZE1);
    memset((void*)reach0, 0, SIZE1);
    memset((void*)reach1, 0, SIZE1);

    std::cout << "mem size in bytes: " << SIZE1 << std::endl;

    TIME_PRINT("copy h2d: ",
               HANDLE_ERROR(cudaMemcpy(dev_reach0, reach0, SIZE1,
                                       cudaMemcpyHostToDevice));
               HANDLE_ERROR(cudaDeviceSynchronize()););

    TIME_PRINT("copy d2d: ",
               HANDLE_ERROR(cudaMemcpy(dev_reach1, dev_reach0, SIZE1,
                                       cudaMemcpyDeviceToDevice));
               HANDLE_ERROR(cudaDeviceSynchronize()););

    auto copyKernel = [&]() {
        uint32_t N          = SIZE1 / 4u;
        uint32_t BLOCK_SIZE = 512;
        copy<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(dev_reach1, dev_reach0, N);
        cudaDeviceSynchronize();
        // HANDLE_ERROR(cudaGetLastError());
    };

    TIME_PRINT("kernel copy d2d: ", copyKernel());

    HANDLE_ERROR(cudaFree(dev_reach0));
    HANDLE_ERROR(cudaFree(dev_reach1));

    free(reach0);
    free(reach1);
}

TEST(PaperTests, SOL1)
{
    uint32_t *dev_reach0, *dev_reach1;
    uint32_t *reach0, *reach1;

    size_t SIZE1 = SIZE;

    HANDLE_ERROR(cudaMalloc((void**)&dev_reach0, SIZE1));
    HANDLE_ERROR(cudaMemset((void*)dev_reach0, 0, SIZE1));
    HANDLE_ERROR(cudaMalloc((void**)&dev_reach1, SIZE1));
    HANDLE_ERROR(cudaMemset((void*)dev_reach1, 0, SIZE1));

    reach0 = (uint32_t*)malloc(SIZE1);
    reach1 = (uint32_t*)malloc(SIZE1);
    memset((void*)reach0, 0, SIZE1);
    memset((void*)reach1, 0, SIZE1);

    std::cout << "mem size in bytes: " << SIZE1 << std::endl;

    cudaEvent_t startEvent, stopEvent;
    float ms{};

    HANDLE_ERROR(cudaEventCreate(&startEvent));
    HANDLE_ERROR(cudaEventCreate(&stopEvent));

    HANDLE_ERROR(cudaEventRecord(startEvent, 0));
    HANDLE_ERROR(cudaMemcpy(dev_reach0, reach0, SIZE1,
                            cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaEventRecord(stopEvent, 0));
    HANDLE_ERROR(cudaEventSynchronize(stopEvent));

    HANDLE_ERROR(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    std::cout << "copy h2d: " << ms << " ms" << std::endl;

    HANDLE_ERROR(cudaEventRecord(startEvent, 0));
    HANDLE_ERROR(cudaMemcpy(dev_reach1, dev_reach0, SIZE1,
                            cudaMemcpyDeviceToDevice));
    HANDLE_ERROR(cudaEventRecord(stopEvent, 0));
    HANDLE_ERROR(cudaEventSynchronize(stopEvent));
    HANDLE_ERROR(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    std::cout << "copy d2d: " << ms << " ms" << std::endl;

    auto copyKernel = [&]() {
        uint32_t N          = SIZE1 / 4;
        uint32_t BLOCK_SIZE = 512;
        copy<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE / 512, BLOCK_SIZE>>>(dev_reach1, dev_reach0, N);
        cudaDeviceSynchronize();
        // HANDLE_ERROR(cudaGetLastError());
    };

    HANDLE_ERROR(cudaEventRecord(startEvent, 0));
    copyKernel();
    HANDLE_ERROR(cudaEventRecord(stopEvent, 0));
    HANDLE_ERROR(cudaEventSynchronize(stopEvent));
    HANDLE_ERROR(cudaEventElapsedTime(&ms, startEvent, stopEvent));

    std::cout << "kernel copy d2d: " << ms << " ms" << std::endl;

    HANDLE_ERROR(cudaEventDestroy(startEvent));
    HANDLE_ERROR(cudaEventDestroy(stopEvent));

    free(reach0);
    free(reach1);
}

TEST(PaperTests, GoalTest)
{
    uint32_t* reach0 = (uint32_t*)malloc(SIZE);

    memset((void*)reach0, 0, SIZE);

    uint32_t middle = turnCoord(X_DIM / 2, Y_DIM / 2, 0,
                                X_DIM, Y_DIM, POS_RES, HDG_RES, TURN_R);

    bitVectorWrite(reach0, 3, middle);

    EXPECT_FALSE(testGoal(reach0, middle - 1u));
    EXPECT_TRUE(testGoal(reach0, middle));
    EXPECT_TRUE(testGoal(reach0, middle + 1u));
    EXPECT_FALSE(testGoal(reach0, middle + 2u));

    free(reach0);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}