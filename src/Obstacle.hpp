#ifndef OBSTACLE_HPP_
#define OBSTACLE_HPP_

#include <vector>
#include "Vector.hpp"
#include "Types.hpp"

struct Obstacle
{
    Vector2f pos{0.0f, 0.0f}; // position [metter]
    float32_t hdg{};          // heading [radian]

    std::vector<Vector2f> boundaryPoints{};

}; // struct Obstacle

#endif // OBSTACLE_HPP_