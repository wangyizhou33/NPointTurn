#include "gtest/gtest.h"
#include "../src/Freespace.hpp"
#include "../src/Visualization.hpp"

/**
 * Testing strategy:
 * - assert the CPU / GPU implementations create the same freespace volume
 * - assert that either volume (because they are the same) make the same free or not determine
 *   as the polygonal intersection check
 * 
 */

// mock a rectangular obstacle
Obstacle createObstacle()
{
    Obstacle obs{};
    obs.pos = {5.0f, 0.0f};
    obs.hdg = 0.0f;
    obs.boundaryPoints.push_back({3.0, 1.0});
    obs.boundaryPoints.push_back({7.0, 1.0});
    obs.boundaryPoints.push_back({7.0, -1.0});
    obs.boundaryPoints.push_back({3.0, -1.0});
    obs.boundaryPoints.push_back({3.0, 1.0});

    return obs;
}

TEST(FreespaceTests, temp)
{
    Freespace freespace{};
    Visualization vis{};
    vis.setFreespace(freespace.get());

    std::vector<Obstacle> vec{};
    vec.push_back(createObstacle());

    // 1430.49 ms
    // TIME_PRINT("CPU", freespace.computeFreespaceCPU(vec));

    // 4.61719 ms
    TIME_PRINT("GPU", freespace.computeFreespaceGPU(vec));

    vis.draw();
}

TEST(FreespaceTests, CPUvsGPU)
{
    Freespace freespace{};

    std::vector<Obstacle> vec{};
    vec.push_back(createObstacle());

    Dimension dim{};
    size_t size = dim.row * dim.col * dim.height;

    using value_type = Freespace::value_type;
    std::unique_ptr<value_type[]> cpuResult{};
    cpuResult = std::make_unique<value_type[]>(size);

    freespace.computeFreespaceCPU(vec);
    std::copy_n(freespace.get(), size, cpuResult.get());

    freespace.computeFreespaceGPU(vec);
    const value_type* gpuResult = freespace.get();

    // check element-wise equality
    uint32_t cnt{0u};
    for (uint32_t i = 0u; i < size; ++i)
    {
        // failure case
        if (cpuResult[i] != gpuResult[i])
        {
            cnt++;
            std::cerr << "Failure index: " << i
                      << " cpu " << cpuResult[i]
                      << " vs gpu " << gpuResult[i]
                      << std::endl;
            EXPECT_TRUE(false);
        }
    }
    std::cerr << "Total mismatch cnt: " << cnt << std::endl;
}
