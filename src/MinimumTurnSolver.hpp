#ifndef MINIMUMTURNSOLVER_HPP_
#define MINIMUMTURNSOLVER_HPP_

#include "Solver.hpp"

class MinimumTurnSolver : public Solver
{
public:
    static size_t getByteSize()
    {
        return m_dim.row / 32u * m_dim.col * m_dim.height * sizeof(uint32_t);
    }

}; // class MinimumTurnSolver

#endif // MINIMUMTURNSOLVER_HPP_