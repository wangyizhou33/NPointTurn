#include "Visualization.hpp"
#include "Helper.hpp"
#include <csignal>   //SIGINT
#include <algorithm> // min

void signal_callback_handler(int signum)
{
    std::cout << "Caught signal " << signum << std::endl;
    // Terminate program
    exit(signum);
}

Visualization::Visualization()
{
    m_d.init();
}

void Visualization::draw()
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
        // auto pix = toPixel(Vector2f{-8.0f, -8.0f});
        // drawDiamondColor<unsigned char>(img, pix.i, pix.j, 5, w, h, 0, 0, 0);

        // real deal
        drawFreespace(m_theta);

        m_d.update();
        updateView();
    } while (m_skipInput);
}

void Visualization::setFreespace(const uint32_t* mem)
{
    m_freespace = mem;
}

void Visualization::updateView()
{
    std::cout << "Adjust display {a, s}, {d, f}, {q, w}, {z, x}" << std::endl;
    std::cout << "(S,xc,yc):(" << m_scale << "," << m_xc << "," << m_yc << "," << m_theta << ")" << std::endl;

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
    else if (cm == 'u')
    {
        m_skipInput = false;
    }
    else if (cm == 'z')
    {
        m_theta += 10u;
    }
    else if (cm == 'x')
    {
        m_theta = (m_theta < 10u) ? m_theta : m_theta - 10u;
    }
}

Pixel Visualization::toPixel(const Vector2f& in) const
{
    // check in view
    if (in.x > (m_xc + m_d.getw() / 2) * m_scale ||
        in.x < (m_xc - m_d.getw() / 2) * m_scale ||
        in.y > (m_yc + m_d.geth() / 2) * m_scale ||
        in.y < (m_yc - m_d.geth() / 2) * m_scale)
        return {.i = 0, .j = 0};

    int i = m_d.geth() / 2 + (int)((in.y - m_yc) * m_scale + 0.5f);
    int j = m_d.getw() / 2 + (int)((in.x - m_xc) * m_scale + 0.5f);

    return {.i = i, .j = j};
}

void Visualization::drawFreespace(uint32_t k)
{
    if (!m_freespace)
    {
        std::cerr << "ERRRO: Freespace not set" << std::endl;
        return;
    }

    int w              = m_d.getw();
    int h              = m_d.geth();
    unsigned char* img = m_d.get();

    for (uint32_t i = 0u; i < m_dim.row; ++i)
    {
        for (uint32_t j = 0u; j < m_dim.col; ++j)
        {
            Vector2f pos = toCartesian(i, j, m_dim.row, m_dim.col, m_dim.posRes);
            Pixel pix    = toPixel(pos);
            uint32_t ind = index(i, j, k, m_dim.row, m_dim.col, m_dim.height);

            if (m_freespace[ind] == 0u)
            {
                drawDiamondColor<unsigned char>(img, pix.i, pix.j, 1, w, h, 0, 0, 0);
            }
            else
            {
                drawDiamondColor<unsigned char>(img, pix.i, pix.j, 5, w, h, 255, 0, 0);
            }
        }
    }
}
