#include "gtest/gtest.h"
#include "../src/Helper.hpp"
#include "../src/GenericCostSolver.hpp"
#include "../src/MinimumTurnSolver.hpp"
#include <bitset>

__global__ void naiveCopy(uint32_t* dst, const uint32_t* src, uint32_t N)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // while loop implementation
    while (i < N)
    {
        dst[i] = src[i];
        i += gridDim.x * blockDim.x;
    }
}

__global__ void naiveTranspose(uint32_t* dst, uint32_t* src, const uint32_t nx, const uint32_t ny)
{
    // matrix coordinate (ix,iy)
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    // transpose with boundary test
    if (ix < nx && iy < ny)
    {
        dst[ix * ny + iy] = src[iy * nx + ix];
    }
}

TEST(SOLTests, copyBenchmark)
{
    uint32_t *src, *dst;

    // 16777216 bytes (16.78 MB)
    // 0.076665 ms
    // 218.87 GB/s
    size_t byteSize = GenericCostSolver::getByteSize();

    // 1048576 bytes (1.05 MB)
    // 0.017904 ms
    // 58.65 GB/s
    // size_t byteSize = MinimumTurnSolver::getByteSize();

    size_t root = std::sqrt(byteSize);

    HANDLE_ERROR(cudaMalloc((void**)&src, byteSize));
    HANDLE_ERROR(cudaMemset((void*)src, 1, byteSize));
    HANDLE_ERROR(cudaMalloc((void**)&dst, byteSize));
    HANDLE_ERROR(cudaMemset((void*)dst, 0, byteSize));

    std::cout << "Copying " << byteSize << " bytes" << std::endl;
    TIME_PRINT("copy d2d: ",
               HANDLE_ERROR(cudaMemcpy(dst, src, byteSize,
                                       cudaMemcpyDeviceToDevice));
               HANDLE_ERROR(cudaDeviceSynchronize()););

    auto naiveCopyKernel = [&]() {
        uint32_t N          = byteSize / sizeof(uint32_t);
        uint32_t BLOCK_SIZE = 512u;
        naiveCopy<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(dst, src, N);
        cudaDeviceSynchronize();
    };

    // the time is very close to cudaMemcpy
    TIME_PRINT("naive copy d2d: ", naiveCopyKernel());

    HANDLE_ERROR(cudaFree(src));
    HANDLE_ERROR(cudaFree(dst));
}

TEST(SOLTests, transposeBenchmark)
{
    uint32_t *src, *dst;

    // 16777216 bytes (16.78 MB)
    // 0.076665 ms
    // 218.87 GB/s
    // size_t byteSize = GenericCostSolver::getByteSize();

    // 1048576 bytes (1.05 MB)
    size_t byteSize = MinimumTurnSolver::getByteSize();

    HANDLE_ERROR(cudaMalloc((void**)&src, byteSize));
    HANDLE_ERROR(cudaMemset((void*)src, 1, byteSize));
    HANDLE_ERROR(cudaMalloc((void**)&dst, byteSize));
    HANDLE_ERROR(cudaMemset((void*)dst, 0, byteSize));

    auto naiveTransposeKernel = [&]() {
        uint32_t N  = byteSize / sizeof(uint32_t);
        uint32_t nx = sqrt(N);
        uint32_t ny = sqrt(N);

        dim3 block(32, 32);
        dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
        // dim3 grid2  ((nx + block.x * 2 - 1) / (block.x * 2),
        //              (ny + block.y - 1) / block.y);

        std::cerr << N << " " << grid.x << " " << grid.y << std::endl;
        naiveTranspose<<<grid, block>>>(dst, src, nx, ny);
        cudaDeviceSynchronize();
    };

    // 1.79ms vs 0.076665 ms in copy
    // 0.094218 ms vs 0.017904 ms in copy
    TIME_PRINT("naive transpose d2d: ", naiveTransposeKernel());

    HANDLE_ERROR(cudaGetLastError());

    HANDLE_ERROR(cudaFree(src));
    HANDLE_ERROR(cudaFree(dst));
}

// reference https://stackoverflow.com/questions/41778362/how-to-efficiently-transpose-a-2d-bit-matrix
void transpose64(uint64_t a[64])
{
    int j, k;
    uint64_t m, t;

    for (j = 32, m = 0x00000000FFFFFFFF; j; j >>= 1, m ^= m << j)
    {
        for (k = 0; k < 64; k = ((k + j) + 1) & ~j)
        {
            t = (a[k] ^ (a[k + j] >> j)) & m;
            a[k] ^= t;
            a[k + j] ^= (t << j);
        }
    }
}

uint64_t logo[] = {
    0b0000000000000000000000000000000000000000000100000000000000000000,
    0b0000000000000000000000000000000000000000011100000000000000000000,
    0b0000000000000000000000000000000000000000111110000000000000000000,
    0b0000000000000000000000000000000000000001111111000000000000000000,
    0b0000000000000000000000000000000000000000111111100000000000000000,
    0b0000000000000000000000000000000000000000111111100000000000000000,
    0b0000000000000000000000000000000000000000011111110000000000000000,
    0b0000000000000000000000000000000000000000001111111000000000000000,
    0b0000000000000000000000000000000000000000001111111100000000000000,
    0b0000000000000000000000000000000010000000000111111100000000000000,
    0b0000000000000000000000000000000011100000000011111110000000000000,
    0b0000000000000000000000000000000111110000000001111111000000000000,
    0b0000000000000000000000000000001111111000000001111111100000000000,
    0b0000000000000000000000000000011111111100000000111111100000000000,
    0b0000000000000000000000000000001111111110000000011111110000000000,
    0b0000000000000000000000000000000011111111100000001111111000000000,
    0b0000000000000000000000000000000001111111110000001111111100000000,
    0b0000000000000000000000000000000000111111111000000111111100000000,
    0b0000000000000000000000000000000000011111111100000011111110000000,
    0b0000000000000000000000000000000000001111111110000001111111000000,
    0b0000000000000000000000000000000000000011111111100001111111100000,
    0b0000000000000000000000001100000000000001111111110000111111100000,
    0b0000000000000000000000001111000000000000111111111000011111110000,
    0b0000000000000000000000011111110000000000011111111100001111100000,
    0b0000000000000000000000011111111100000000001111111110001111000000,
    0b0000000000000000000000111111111111000000000011111111100110000000,
    0b0000000000000000000000011111111111110000000001111111110000000000,
    0b0000000000000000000000000111111111111100000000111111111000000000,
    0b0000000000000000000000000001111111111111100000011111110000000000,
    0b0000000000000000000000000000011111111111111000001111100000000000,
    0b0000000000000000000000000000000111111111111110000011000000000000,
    0b0000000000000000000000000000000001111111111111100000000000000000,
    0b0000000000000000000000000000000000001111111111111000000000000000,
    0b0000000000000000000000000000000000000011111111111100000000000000,
    0b0000000000000000000111000000000000000000111111111100000000000000,
    0b0000000000000000000111111110000000000000001111111000000000000000,
    0b0000000000000000000111111111111100000000000011111000000000000000,
    0b0000000000000000000111111111111111110000000000110000000000000000,
    0b0000000000000000001111111111111111111111100000000000000000000000,
    0b0000000000000000001111111111111111111111111111000000000000000000,
    0b0000000000000000000000011111111111111111111111100000000000000000,
    0b0000001111110000000000000001111111111111111111100000111111000000,
    0b0000001111110000000000000000000011111111111111100000111111000000,
    0b0000001111110000000000000000000000000111111111100000111111000000,
    0b0000001111110000000000000000000000000000001111000000111111000000,
    0b0000001111110000000000000000000000000000000000000000111111000000,
    0b0000001111110000000000000000000000000000000000000000111111000000,
    0b0000001111110000001111111111111111111111111111000000111111000000,
    0b0000001111110000001111111111111111111111111111000000111111000000,
    0b0000001111110000001111111111111111111111111111000000111111000000,
    0b0000001111110000001111111111111111111111111111000000111111000000,
    0b0000001111110000001111111111111111111111111111000000111111000000,
    0b0000001111110000001111111111111111111111111111000000111111000000,
    0b0000001111110000000000000000000000000000000000000000111111000000,
    0b0000001111110000000000000000000000000000000000000000111111000000,
    0b0000001111110000000000000000000000000000000000000000111111000000,
    0b0000001111110000000000000000000000000000000000000000111111000000,
    0b0000001111110000000000000000000000000000000000000000111111000000,
    0b0000001111111111111111111111111111111111111111111111111111000000,
    0b0000001111111111111111111111111111111111111111111111111111000000,
    0b0000001111111111111111111111111111111111111111111111111111000000,
    0b0000001111111111111111111111111111111111111111111111111111000000,
    0b0000001111111111111111111111111111111111111111111111111111000000,
    0b0000001111111111111111111111111111111111111111111111111111000000,
};

void printbits(uint64_t a[64])
{
    int i, j;
    for (i = 0; i < 64; i++)
    {
        std::cout << std::bitset<64>(a[i]) << std::endl;
    }
}

TEST(SOLTests, bitTranspose)
{
    std::cout << "Before: " << "\n\n";
    printbits(logo);
    std::cout << "\n\n";

    transpose64(logo);

    std::cout << "After: " << "\n\n";
    printbits(logo);
}
