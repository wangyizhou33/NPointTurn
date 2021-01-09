#include "Polygon.hpp"
#include <limits>
#include <cmath>

ConvexPolygon::ConvexPolygon(const std::vector<Vector2f>& vertices)
{
    if (vertices.size() < 3)
    {
        throw "Fail to create ConvexPolygon with less than 3 vertices.";
    }

    for (Vector2f v : vertices)
    {
        m_vertices.push_back(v);
    }
    // TODO(yizhouw) convexity check

    computeAxes();
}

Vector2f ConvexPolygon::getCentroid() const
{
    auto center = Vector2f::Zero();

    for (size_t i = 0; i < m_vertices.size(); ++i)
    {
        center += m_vertices.at(i);
    }
    center /= static_cast<float32_t>(m_vertices.size());

    return center;
}

ConvexPolygon ConvexPolygon::createRectangle(const Vector2f& center,
                                             float32_t heading,
                                             float32_t width,
                                             float32_t depth)
{
    float32_t halfDepth = depth / 2.F;
    float32_t halfWidth = width / 2.F;

    std::vector<Vector2f> vertices{};

    float32_t sinHeading = std::sin(heading);
    float32_t cosHeading = std::cos(heading);
    // the order is important here
    // as edges are computed assuming vertices are physically adjacent
    vertices.emplace_back(center.x + cosHeading * halfDepth - sinHeading * halfWidth,
                          center.y + sinHeading * halfDepth + cosHeading * halfWidth);

    vertices.emplace_back(center.x + cosHeading * -halfDepth - sinHeading * halfWidth,
                          center.y + sinHeading * -halfDepth + cosHeading * halfWidth);

    vertices.emplace_back(center.x + cosHeading * -halfDepth - sinHeading * -halfWidth,
                          center.y + sinHeading * -halfDepth + cosHeading * -halfWidth);

    vertices.emplace_back(center.x + cosHeading * halfDepth - sinHeading * -halfWidth,
                          center.y + sinHeading * halfDepth + cosHeading * -halfWidth);

    return ConvexPolygon(vertices);
}

void ConvexPolygon::computeAxes()
{
    m_axes.clear();

    for (size_t i = 0; i < m_vertices.size(); ++i)
    {
        // calculate edge for each adjacent points pair
        const Vector2f& p1 = m_vertices.at((i + 1) % m_vertices.size()); // % wraps around index
        const Vector2f& p2 = m_vertices.at(i % m_vertices.size());

        // compute unit vector along the edge
        Vector2f edge = p2 - p1;
        if (edge.isZero())
        {
            throw "ConvexPolygon::computeAxes: \
                  Cannot create edge with the identical vertices.";
        }
        edge.normalize();

        // compute axis as unit vector, perpendicular to each edge
        m_axes.emplace_back(-edge(1), edge(0));
    }
}

Vector2f ConvexPolygon::projectVerticesOnAxis(const Vector2f& axis) const
{
    float32_t min = std::numeric_limits<float32_t>::max();
    float32_t max = std::numeric_limits<float32_t>::lowest();

    // normalize input
    Vector2f axisNormalized = axis / axis.norm();

    for (const auto& v : m_vertices)
    {
        float32_t proj = v.dot(axisNormalized);

        min = (proj < min) ? proj : min;
        max = (proj > max) ? proj : max;
    }

    return Vector2f(min, max); // 0th elem is min, 1st elem is max
}

bool ConvexPolygon::doProjectionIntersect(const Vector2f& l1, const Vector2f& l2)
{
    bool result = true;

    // assert input is valid, i.e. 0th elem is smaller than 1st elem
    if ((l1(0) > l1(1) + std::numeric_limits<float32_t>::epsilon()) ||
        (l2(0) > l2(1) + std::numeric_limits<float32_t>::epsilon()))
    {
        throw "ConvexPolygon::doProjectionIntersect: \
               0th elem greater than 1st elem in projection vector.";
    }

    // if there is a gap, then no overlap
    // l1.max <= l2.min or l2.max <= l1.min
    if ((l1(1) < l2(0) + std::numeric_limits<float32_t>::epsilon()) ||
        (l2(1) < l1(0) + std::numeric_limits<float32_t>::epsilon()))
    {
        result = false;
    }

    return result;
}

bool ConvexPolygon::doConvexPolygonIntersect(const ConvexPolygon& p1, const ConvexPolygon& p2)
{
    bool result = true;

    // check overlaping of projections onto axes
    for (const Vector2f& v : p1.getAxes())
    {
        // project vertices of the two polygons to the same axis
        Vector2f p1Proj = p1.projectVerticesOnAxis(v);
        Vector2f p2Proj = p2.projectVerticesOnAxis(v);

        //check for overlap. If no overlap, then no intersection between polygons so we return
        if (!doProjectionIntersect(p1Proj, p2Proj))
        {
            result = false;
            break;
        }
    }

    // same thing for p2 axes, but skip if we have determined intersection already
    if (result == true)
    {
        for (const Vector2f& v : p2.getAxes())
        {
            // project vertices of the two polygons to the same axis
            Vector2f p1Proj = p1.projectVerticesOnAxis(v);
            Vector2f p2Proj = p2.projectVerticesOnAxis(v);

            //check for overlap. If no overlap, then no intersection between polygons so we return
            if (!doProjectionIntersect(p1Proj, p2Proj))
            {
                result = false;
                break;
            }
        }
    }

    return result;
}
