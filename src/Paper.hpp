#ifndef PAPER_HPP_
#define PAPER_HPP_

#include "Helper.hpp"

constexpr uint32_t X_DIM    = 96u;
constexpr uint32_t Y_DIM    = 96u;
constexpr float32_t POS_RES = 0.5f;
constexpr float32_t HDG_RES = 1.0f;
constexpr uint32_t X_CELLS  = X_DIM / 32u;
constexpr uint32_t Y_CELLS  = Y_DIM / 32u;

void bitSweepLeft(uint32_t* RbO,
                  const uint32_t* Fb,
                  const uint32_t* RbI,
                  float32_t turnRadius,
                  cudaStream_t cuStream = nullptr);

#endif // PAPER_HPP_