#ifndef HELPER_HPP_
#define HELPER_HPP_

#include <cstdlib>
#include <stdio.h>
#include <cstdint>
#include <cuda_runtime_api.h> // cudaError_t
#include <chrono>

#include "Types.hpp"
#include "Vector.hpp"

struct Dimension
{
    uint32_t row{128u};    // elements in row
    uint32_t col{128u};    // elements in col
    uint32_t height{512u}; // elements in height

    float32_t posRes{64.f / 128.f};
    float32_t hdgRes{2.f * M_PI / 512.f};
};

__device__ __host__ __inline__ Vector2f toCartesian(float32_t i, float32_t j, float32_t row, float32_t col, float32_t res)
{
    return Vector2f{
        (i - row / 2) * res,
        (j - col / 2) * res};
};

__device__ __host__ __inline__ Vector2ui toIndex(float32_t x, float32_t y, uint32_t row, uint32_t col, float32_t res)
{
    return Vector2ui{
        static_cast<uint32_t>(x / res + static_cast<float32_t>(row) / 2.f),
        static_cast<uint32_t>(y / res + static_cast<float32_t>(col) / 2.f)};
    // Note: static_cast<uint32_t>(y / res) + col / 2u shows a wrong result in device
};

__device__ __host__ __inline__ uint32_t index(uint32_t i, uint32_t j, uint32_t k,
                                              uint32_t row, uint32_t col, uint32_t height)
{
    return k * row * col + j * row + i;
};

__device__ __host__ __inline__ bool isInBoundary(float32_t x, float32_t y, float32_t range)
{
    return x < range && x > -range && y < range && y > -range;
}

static void HandleError(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#define HANDLE_NULL(a)                                                           \
    {                                                                            \
        if (a == NULL)                                                           \
        {                                                                        \
            printf("Host memory failed in %s at line %d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }

constexpr bool DEBUG_PRINT_RESULTS = true;
#ifndef PRINT_TIMING
#define TIME_PRINT(name, a)                                                                                                    \
    if (DEBUG_PRINT_RESULTS)                                                                                                   \
    {                                                                                                                          \
        auto start = std::chrono::high_resolution_clock::now();                                                                \
        a;                                                                                                                     \
        auto elapsed = std::chrono::high_resolution_clock::now() - start;                                                      \
        std::cout << name                                                                                                      \
                  << ": "                                                                                                      \
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count() / 1000000.0f << " ms" << std::endl; \
    }                                                                                                                          \
    else                                                                                                                       \
    {                                                                                                                          \
        a;                                                                                                                     \
    }
#endif

#endif // HELPER_HPP_