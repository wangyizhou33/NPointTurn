#ifndef FREESPACE_HPP_
#define FREESPACE_HPP_

#include "Types.hpp"
#include "Helper.hpp"
#include "Obstacle.hpp"
#include <memory>

class Freespace
{
public:
    using value_type = uint32_t;
    Freespace();
    ~Freespace();

    void reset();

    /**
     * Calculate the freespace volume from input obstacle vector
     * CPU implementation
     */
    void computeFreespaceCPU(const std::vector<Obstacle>& vec);
    /**
     * Calculate the freespace volume from input obstacle vector
     * GPU implementation
     */
    void computeFreespaceGPU(const std::vector<Obstacle>& vec);

    const value_type* get() const { return m_mem.get(); };

private:
    void compute0Slice(const std::vector<Obstacle>& vec);

    void computeSliceCPU(uint32_t k);
    void dilateSliceCPU(value_type* mem);
    void rotateCPU();

    void computeSliceGPU();
    void dilateSliceGPU();
    void rotateGPU();

    // freespace volume: row x col x height
    uint32_t m_size{};

    Dimension m_dim{};

    std::unique_ptr<value_type[]> m_mem{}; // host memory of freespace volume
                                           // 0 is free
    value_type* m_cuMem{};                 // device memory of freespace volume
    value_type* m_cuMem1{};                // temp memory for the dilation step

    // dilation amount
    int radioR = 1;
    int radioC = 4;

}; // Freespace

#endif // FREESPACE_HPP_