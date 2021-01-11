#include "Freespace.hpp"
#include <algorithm>
#include <iostream>

Freespace::Freespace()
{
    m_size = m_dim.row * m_dim.col * m_dim.height;
    m_mem  = std::make_unique<uint32_t[]>(m_size);
    std::fill_n(m_mem.get(), m_size, 0u);
}

void Freespace::computeFreespace(const std::vector<Obstacle>& vec)
{
    auto occupy = [this](const Vector2f& v) {
        Vector2ui ind = toIndex(v.x, v.y, m_dim.row, m_dim.col, m_dim.posRes);

        m_mem[index(ind.x, ind.y, 0u, m_dim.row, m_dim.col, m_dim.height)] = 1u;
    };

    // fill the 0-th slice of the volume
    for (const Obstacle& obs : vec)
    {
        for (uint32_t i = 0; i + 1 < obs.boundaryPoints.size(); ++i)
        {
            const Vector2f& v0 = obs.boundaryPoints.at(i);
            const Vector2f& v1 = obs.boundaryPoints.at(i + 1);

            Vector2f e  = v1 - v0;
            float32_t l = e.norm();
            e.normalize();

            for (float32_t d = 0.f; d < l; d += m_dim.posRes)
            {
                Vector2f v = v0 + e * d;
                occupy(v);
            }
            occupy(v1);
        }
    }
    // end of process the 0-th slice

    for (uint32_t k = 1u; k < m_dim.height; ++k)
    {
        computeSlice(k);
    }
}

void Freespace::computeSlice(uint32_t k)
{
    for (uint32_t i = 0u; i < m_dim.row; ++i)
    {
        for (uint32_t j = 0u; j < m_dim.col; ++j)
        {
            Vector2f pos    = toCartesian(i, j, m_dim.row, m_dim.col, m_dim.posRes);
            float32_t theta = static_cast<float32_t>(k) * m_dim.hdgRes;
            uint32_t ind    = index(i, j, k, m_dim.row, m_dim.col, m_dim.height);

            Vector2f newPos = pos.rotate(-theta);

            if (isInBoundary(newPos.x, newPos.y, 30.f)) //TODO: fix the hack
            {
                Vector2ui newInd = toIndex(newPos.x, newPos.y, m_dim.row, m_dim.col, m_dim.posRes);
                if (m_mem[index(newInd.x, newInd.y, 0u, m_dim.row, m_dim.col, m_dim.height)] == 1u)
                {
                    m_mem[ind] = 1u;
                }
            }
        }
    }
}
