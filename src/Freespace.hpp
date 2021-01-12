#ifndef FREESPACE_HPP_
#define FREESPACE_HPP_

#include "Types.hpp"
#include "Helper.hpp"
#include "Obstacle.hpp"
#include <memory>

class Freespace
{
public:
    Freespace();

    void computeFreespace(const std::vector<Obstacle>& vec);

    const uint32_t* get() const { return m_mem.get(); };

private:
    void computeSlice(uint32_t k);

    void dilateSlice(uint32_t* mem);

    // freespace volume: row x col x height
    uint32_t m_size{};

    Dimension m_dim{};

    std::unique_ptr<uint32_t[]> m_mem{}; // 0 is free

}; // Freespace

#endif // FREESPACE_HPP_