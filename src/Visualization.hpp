#ifndef VISUALIZATION_HPP_
#define VISUALIZATION_HPP_

#include "SimpleDisp.hpp"
#include "Types.hpp"
#include "Helper.hpp"
#include "Vector.hpp"
#include <iostream>

template <class IT>
inline void drawDiamond(IT* img, int i, int j, int radius, int w, int h, IT fillval)
{
    for (int io = -radius; io <= radius; io++)
    {
        int ir;
        if (io < 0)
            ir = radius + io;
        else
            ir = radius - io;
        int it = io + i;
        for (int jt = j - ir; jt <= j + ir; jt++)
        {
            if (it >= 0 && it < h && jt >= 0 && jt < w)
            {
                img[it * w + jt] = fillval;
            }
        }
    }
}

template <class IT>
inline void drawDiamondColor(IT* img, int i, int j, int radius, int w, int h, IT fv1, IT fv2, IT fv3)
{
    drawDiamond<IT>(img, i, j, radius, w, h, fv1);
    drawDiamond<IT>(img + w * h, i, j, radius, w, h, fv2);
    drawDiamond<IT>(img + 2 * w * h, i, j, radius, w, h, fv3);
}

struct Pixel
{
    int i;
    int j;
};

class Visualization
{
public:
    Visualization();

    void draw();

    void setFreespace(const uint32_t* mem);

private:
    void updateView();

    Pixel toPixel(const Vector2f& in) const;

    // theta = k slice
    void drawFreespace(uint32_t k);

    SimpleDisp m_d{};

    float32_t m_xc{0.0f}; // center x
    float32_t m_yc{0.0f}; // center y
    float32_t m_scale{12.0f};

    Dimension m_dim{};
    const uint32_t* m_freespace{nullptr};

    bool m_skipInput{true};

    uint32_t m_theta{0u};

}; // class Visualization

#endif // VISUALIZATION_HPP_