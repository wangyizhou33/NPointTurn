#ifndef POLYGON_HPP_
#define POLYGON_HPP_

#include "Vector.hpp"
#include <vector>

#define FRIEND_TEST(test_case_name, test_name) \
    friend class test_case_name##_##test_name##_Test

class ConvexPolygon
{
    FRIEND_TEST(ConvexPolygonTests, ProjectVerticesOnAxis);
    FRIEND_TEST(ConvexPolygonTests, ProjectionIntersection);

public:
    ConvexPolygon() = delete;

    explicit ConvexPolygon(const std::vector<Vector2f>& vertices);

    ~ConvexPolygon() = default;

    inline bool intersect(const ConvexPolygon& p) const
    {
        return doConvexPolygonIntersect(*this, p);
    }

    /// getter of vertices
    inline const std::vector<Vector2f>& getVertices() const
    {
        return m_vertices;
    }

    /// getter of axes (unit vectors)
    inline const std::vector<Vector2f>& getAxes() const
    {
        return m_axes;
    }

    /// getter of centroid
    Vector2f getCentroid() const;

    static ConvexPolygon createRectangle(const Vector2f& center,
                                         float32_t heading,
                                         float32_t width,
                                         float32_t depth);

private:
    /// compute axes, that are perpendicular to edges, as unit vectors.
    void computeAxes();

    /// project vertices of this polygon onto given axis
    /// return [min, max] of the projection
    Vector2f projectVerticesOnAxis(const Vector2f& axis) const;

    /// project intersection checker
    /// two inputs are projections on the same axis
    /// @return true if there is a intersection, false if no intersection or just touching on a single point
    static bool doProjectionIntersect(const Vector2f& l1, const Vector2f& l2);

    /// polygon intersection checker
    /// use "separating axis theorem"
    /// ref http://www.dyn4j.org/2010/01/sat/
    /// @return true if there is a intersection, false if no intersection or just touching on a single point
    static bool doConvexPolygonIntersect(const ConvexPolygon& p1, const ConvexPolygon& p2);

    // vertices
    std::vector<Vector2f> m_vertices{};

    // axes, unit vectors, perpendicular to edges
    std::vector<Vector2f> m_axes{};
}; // class ConvexPolygon

#endif // POLYGON_HPP_