#ifndef VISUALIZATION_HPP_
#define VISUALIZATION_HPP_

#include "SimpleDisp.hpp"
#include "Types.hpp"
#include <iostream>

void signal_callback_handler(int signum)
{
    std::cout << "Caught signal " << signum << std::endl;
    // Terminate program
    exit(signum);
}

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

class Visualization
{
public:
    Visualization()
    {
        m_d.init();
    };

    void draw()
    {
        int w              = m_d.getw();
        int h              = m_d.geth();
        unsigned char* img = m_d.get();

        // Register signal and signal handler
        // catch ctrl+C
        signal(SIGINT, signal_callback_handler);
        do
        {
            m_d.clear();

            // draw example
            int i1 = h / 2 - (int)((0.0f - m_yc) * m_scale + 0.5f);
            int j1 = w / 2 + (int)((0.0f - m_xc) * m_scale + 0.5f);
            drawDiamondColor<unsigned char>(img, i1, j1, 5, w, h, 255, 255, 255);

            m_d.update();
            updateView();
        } while (m_skipInput);
    }

    void updateView()
    {
        std::cout << "Adjust display {a, s}, {d, f}, {q, w}" << std::endl;
        std::cout << "(S,xc,yc):(" << m_scale << "," << m_xc << "," << m_yc << ")" << std::endl;

        unsigned char cm{};
#ifdef WIN32
        cm = _getch();
#else
        cm = getch();
#endif
        if (cm == 'a')
        {
            m_xc += m_scale;
        }
        else if (cm == 's')
        {
            m_xc -= m_scale;
        }
        else if (cm == 'd')
        {
            m_yc += m_scale;
        }
        else if (cm == 'f')
        {
            m_yc -= m_scale;
        }
        else if (cm == 'q')
        {
            m_scale *= 2.0f;
        }
        else if (cm == 'w')
        {
            m_scale *= 0.5f;
        }
        // else if (cm == 'p')
        // {
        //     break;
        // }
        else if (cm == 'u')
        {
            m_skipInput = false;
        }
    }

private:
    SimpleDisp m_d{};

    float32_t m_xc{0.0f}; // center x
    float32_t m_yc{0.0f}; // center y
    float32_t m_scale{4.0f};

    bool m_skipInput{true};

}; // class Visualization

#endif // VISUALIZATION_HPP_