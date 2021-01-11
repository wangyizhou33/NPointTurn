#include "gtest/gtest.h"
#include "../src/Freespace.hpp"
#include "../src/Visualization.hpp"

TEST(FreespaceTests, temp)
{
    ASSERT_TRUE(true);

    Freespace freespace{};
    Visualization vis{};
    vis.setFreespace(freespace.get());

    std::vector<Obstacle> vec{};
    // mock a rectangular obstacle
    {
        Obstacle obs{};
        obs.pos = {5.0f, 0.0f};
        obs.hdg = 0.0f;
        obs.boundaryPoints.push_back({3.0, 1.0});
        obs.boundaryPoints.push_back({7.0, 1.0});
        obs.boundaryPoints.push_back({7.0, -1.0});
        obs.boundaryPoints.push_back({3.0, -1.0});
        obs.boundaryPoints.push_back({3.0, 1.0});

        vec.push_back(obs);
    }
    freespace.computeFreespace(vec);

    vis.draw();
}
