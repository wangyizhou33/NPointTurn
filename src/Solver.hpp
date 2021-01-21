#ifndef SOLVER_HPP_
#define SOLVER_HPP_

#include "Helper.hpp"

/**
 * Interface class
 */
class Solver
{
public:
    /**** dimensions ****/
    // positional range in [m], i.e. x \n [-POS_RANGE, POS_RANGE]. Same as y
    static constexpr float32_t POS_RANGE = 32.0f;

    // positional index dimension [0, POS_DIM)
    static constexpr size_t POS_DIM = 128u;

    // positional resolution in [m]
    static constexpr float32_t POS_RES = 2.f * POS_RANGE / static_cast<float32_t>(POS_DIM);

    // rotational index dimension [0, HDG_DIM)
    static constexpr size_t HDG_DIM = 512u;

    // rotational resolution in [deg]
    static constexpr float32_t HDG_RES = 360.0f / static_cast<float32_t>(HDG_DIM);

    Solver()
    {
        setUp();
    }

    void setUp(){};
    void TearDown(){};

protected:
    static const Dimension m_dim;

private:
}; // class Solver

#endif // SOLVER_HPP_