#include "gtest/gtest.h"
#include "../src/Vector.hpp"
#include <cstdlib> // rand

TEST(VectorTests, rotation)
{

    Vector2f v1{1.0f, 1.0f};
    Vector2f v2 = v1.rotate(M_PI / 2.0f);

    Vector2i v3{1, 1};
    Vector2i v4 = v3.shearRotate(M_PI / 2.0f);

    EXPECT_FLOAT_EQ(v4.x, v2.x);
    EXPECT_FLOAT_EQ(v4.y, v2.y);

    v2 = v1.rotate(-M_PI / 2.0f);
    v4 = v3.shearRotate(-M_PI / 2.f);

    EXPECT_FLOAT_EQ(v4.x, v2.x);
    EXPECT_FLOAT_EQ(v4.y, v2.y);
}

TEST(VectorTests, correspondence)
{
    for (int32_t x = -50; x <= 50; ++x)
    {
        for (int32_t y = -50; y <= 50; ++y)
        {
            int r           = rand() % 10;                              // [0, 9]
            float32_t angle = (static_cast<float32_t>(r) - 4.5f) / 9.f; // [-1.f, 1.f]
            angle *= 3.f * M_PI / 4.0f;

            Vector2i vec{x, y};
            auto vec1 = vec.shearRotate(angle);
            auto vec2 = vec1.shearRotate(-angle);

            EXPECT_EQ(vec.x, vec2.x);
            EXPECT_EQ(vec.y, vec2.y);
        }
    }
}
