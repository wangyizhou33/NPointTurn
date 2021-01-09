#include "gtest/gtest.h"
#include "../src/Polygon.hpp"

constexpr float32_t TOL = 1e-6f;

TEST(ConvexPolygonTests, CreatePolygon)
{
    std::vector<Vector2f> vertices{};
    // construct polygon with less than 3 vertices
    vertices.push_back(Vector2f(1.0f, 0.0f));
    vertices.push_back(Vector2f(0.0f, 1.0f));
    ASSERT_ANY_THROW(ConvexPolygon p1(vertices));
    // construct a convex polygon
    vertices.push_back(Vector2f(0.0f, 0.0f));
    ASSERT_NO_THROW(ConvexPolygon p2(vertices));

    // constuct a convex polygon with repeated vertices
    vertices.push_back(Vector2f(0.0f, 0.0f));
    ASSERT_ANY_THROW(ConvexPolygon p3(vertices));
    // construct a concave polygon
    vertices.back() = Vector2f(0.25f, 0.25f);
    ASSERT_NO_THROW(ConvexPolygon p4(vertices)); // TODO(yizhouw) should throw after impl
}

TEST(ConvexPolygonTests, CreateRectangle)
{
    // create a rectangle at centered at (0,0) with hdg = 45deg, width = 2.0m. depth 4.0
    Vector2f center{0.0f, 0.0f};
    float32_t hdg   = static_cast<float32_t>(M_PI_4);
    float32_t width = 2.0f;
    float32_t depth = 4.0f;

    ConvexPolygon polygon = ConvexPolygon::createRectangle(center, hdg, width, depth);
    // check center
    ASSERT_NEAR(polygon.getCentroid()(0), center.x, TOL);
    ASSERT_NEAR(polygon.getCentroid()(1), center.y, TOL);

    ASSERT_EQ(polygon.getVertices().size(), 4u);
    // check vertices to center distance
    for (const Vector2f& v : polygon.getVertices())
    {
        ASSERT_NEAR(v.norm(), sqrtf(5.0f), TOL);
    }

    ASSERT_EQ(polygon.getAxes().size(), 4u);
    // check axes are orthogonal to either (1,1) or (-1,1)
    Vector2f xAxis{1.0f, 1.0f};
    Vector2f yAxis{-1.0f, 1.0f};
    for (const Vector2f& v : polygon.getAxes())
    {
        ASSERT_NEAR(v.dot(xAxis) * v.dot(yAxis), 0.0f, TOL); // one of the multiplier must be 0
    }
}

TEST(ConvexPolygonTests, ProjectVerticesOnAxis)
{
    // create a rectangle at centered at (0,0) with hdg = 45deg, width = 2.0m. depth 4.0
    Vector2f center{0.0f, 0.0f};
    float32_t hdg   = static_cast<float32_t>(M_PI_4);
    float32_t width = 2.0f;
    float32_t depth = 4.0f;

    ConvexPolygon polygon = ConvexPolygon::createRectangle(center, hdg, width, depth);

    // try project onto the principal axes (1,1) , (-1 , 1)
    Vector2f axis1(1.0f, 1.0f);
    Vector2f proj1 = polygon.projectVerticesOnAxis(axis1);

    ASSERT_NEAR(proj1(0), -depth / 2.0f, TOL);
    ASSERT_NEAR(proj1(1), depth / 2.0f, TOL);

    Vector2f axis2(-1.0f, 1.0f);
    Vector2f proj2 = polygon.projectVerticesOnAxis(axis2);

    ASSERT_NEAR(proj2(0), -width / 2.0f, TOL);
    ASSERT_NEAR(proj2(1), width / 2.0f, TOL);
}

TEST(ConvexPolygonTests, ProjectionIntersection)
{
    Vector2f l1(-2.0f, 1.0f);
    Vector2f l2(1.0f, 2.0f);

    ASSERT_FALSE(ConvexPolygon::doProjectionIntersect(l1, l2));

    // move l2 left end left a little bit
    l2(0) = 0.9f;
    ASSERT_TRUE(ConvexPolygon::doProjectionIntersect(l1, l2));
}

TEST(ConvexPolygonTests, PolygonIntersection)
{
    // create a rectangle at centered at (0,0) with hdg = 45deg, width = 2.0m. depth 4.0
    Vector2f center{0.0f, 0.0f};
    float32_t hdg            = static_cast<float32_t>(M_PI_4);
    float32_t width          = 2.0f;
    float32_t depth          = 4.0f;
    ConvexPolygon rectangle1 = ConvexPolygon::createRectangle(center, hdg, width, depth);

    // create a triangle with vertices (1,0) (0,1) (-1 0)
    std::vector<Vector2f> v{Vector2f(1.0f, 0.0f),
                            Vector2f(0.0f, 1.0f),
                            Vector2f(-1.0f, 0.0f)};
    ConvexPolygon triangle(v);

    ASSERT_TRUE(rectangle1.intersect(triangle));

    // shift the center of the rectangle sqrt(2) + 1.0 toward left
    // now the two shapes should be touching edges, but not overlapping
    center.x                 = -sqrtf(2.0f) - 1.0f;
    ConvexPolygon rectangle2 = ConvexPolygon::createRectangle(center, hdg, width, depth);

    ASSERT_FALSE(rectangle2.intersect(triangle));

    // shift the center of the rectangle1 sqrt(2) + 1.0 up
    center.x                 = 0.0f;
    center.y                 = sqrtf(2.0f) + 1.0f;
    ConvexPolygon rectangle3 = ConvexPolygon::createRectangle(center, hdg, width, depth);

    ASSERT_FALSE(rectangle3.intersect(triangle));
}