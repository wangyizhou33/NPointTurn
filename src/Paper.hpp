#ifndef PAPER_HPP_
#define PAPER_HPP_

#include "Helper.hpp"

constexpr uint32_t ITER_CNT   = 10u;
constexpr uint32_t X_DIM      = 32u * 5u;
constexpr uint32_t Y_DIM      = 32u * 5u;
constexpr float32_t POS_RES   = 0.5f;
constexpr float32_t HDG_RES   = 1.0f;
constexpr uint32_t X_CELLS    = X_DIM / 32u;
constexpr uint32_t Y_CELLS    = Y_DIM / 32u;
constexpr uint32_t THETA_STEP = 360u;
constexpr uint32_t GRID_SIZE  = X_DIM * Y_DIM * THETA_STEP;
constexpr float32_t HALF_X    = X_DIM / 2;
constexpr float32_t HALF_Y    = Y_DIM / 2;
constexpr float32_t TURN_R    = 10.0f;
// byte size
constexpr size_t SIZE = GRID_SIZE / 32u * sizeof(uint32_t);

__device__ __host__ uint32_t countBits(uint32_t n);

uint32_t countBitsInVolume(uint32_t* vol);

__global__ void writeOnes(uint32_t* R, uint32_t offset);

__global__ void shuffle(uint32_t* R);

__device__ __host__ uint32_t bitVectorRead(const uint32_t* RbI, uint32_t c);
__device__ __host__ void bitVectorWrite(uint32_t* R, uint32_t val, uint32_t c);

__device__ __host__ float32_t deg2Rad(float32_t deg);

__device__ __host__ uint32_t volCoord(uint32_t x,
                                      uint32_t y,
                                      uint32_t theta,
                                      uint32_t X_DIM,
                                      uint32_t Y_DIM);

__device__ __host__ uint32_t turnCoordLeft(uint32_t x,
                                           uint32_t y,
                                           uint32_t theta,
                                           uint32_t X_DIM,
                                           uint32_t Y_DIM,
                                           float32_t POS_RES,
                                           float32_t HDG_RES,
                                           float32_t turnRadius);

void bitSweepLeft(uint32_t* RbO,
                  const uint32_t* Fb,
                  const uint32_t* RbI,
                  float32_t turnRadius,
                  cudaStream_t cuStream = nullptr);

void setUp(uint32_t** dev_reach, uint32_t** reach);

void tearDown(uint32_t** dev_reach, uint32_t** reach);

#endif // PAPER_HPP_