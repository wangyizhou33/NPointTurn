#ifndef HELPER_HPP_
#define HELPER_HPP_

#include <cstdlib>
#include <stdio.h>
#include <cstdint>
#include <cuda_runtime_api.h> // cudaError_t
#include <chrono>

#include "Types.hpp"

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