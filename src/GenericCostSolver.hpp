
#ifndef GENERICCOSTSOLVER_HPP_
#define GENERICCOSTSOLVER_HPP_

#include "Solver.hpp"

class GenericCostSolver : public Solver
{
public:
    static size_t getByteSize()
    {
        return m_dim.row * m_dim.col * m_dim.height * sizeof(uint16_t);
    }

}; // class GenericCostSolver

#endif // GENERICCOSTSOLVER_HPP_