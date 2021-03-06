#ifndef PAPER_HPP_
#define PAPER_HPP_

#include "Helper.hpp"

constexpr uint32_t ITER_CNT  = 10u;
constexpr uint32_t X_DIM     = 32u * 4u;
constexpr uint32_t Y_DIM     = 32u * 4u;
constexpr uint32_t THETA_DIM = 512u;
constexpr uint32_t GRID_SIZE = X_DIM * Y_DIM * THETA_DIM;

constexpr float32_t POS_RES = 0.5f;
constexpr float32_t HDG_RES = 360.f / static_cast<float32_t>(THETA_DIM);

constexpr float32_t HALF_X = X_DIM / 2;
constexpr float32_t HALF_Y = Y_DIM / 2;

constexpr float32_t TURN_R = 10.0f;

// byte size
constexpr size_t SIZE = GRID_SIZE / 32u * sizeof(uint32_t);

__device__ __host__ uint32_t countBits(uint32_t n);

uint32_t countBitsInVolume(uint32_t* vol);

__global__ void writeOnes(uint32_t* R, uint32_t offset);

__global__ void shuffle(uint32_t* R);

__device__ __host__ __inline__ uint32_t bitVectorRead(const uint32_t* RbI, uint32_t c)
{
    uint32_t cm = (c >> 5);
    uint32_t cr = (c & 31);
    uint32_t Ro = 0;

    if (cr)
        Ro = (RbI[cm] >> cr) | (RbI[cm + 1] << (32 - cr));
    else
        Ro = RbI[cm];
    // printf("%u, %u\n", (RbI[cm] >> cr), (RbI[cm + 1] << (32 - cr)));
    return Ro;
}

__device__ __host__ __inline__ void bitVectorWrite(uint32_t* R, uint32_t val, uint32_t c)
{
    uint32_t cm = (c >> 5);
    uint32_t cr = (c & 31);
    R[cm]       = ((R[cm] & ((1 << cr) - 1)) | (val << cr));
    if (cr)
        R[cm + 1] = ((R[cm + 1] & (~((1 << cr) - 1))) | (val >> (32 - cr)));
    // else
    //     R[cm + 1] = R[cm + 1] & (~((1 << cr) - 1));
}

__device__ __host__ float32_t deg2Rad(float32_t deg);

__device__ __host__ uint32_t volCoord(uint32_t x,
                                      uint32_t y,
                                      uint32_t theta,
                                      uint32_t X_DIM,
                                      uint32_t Y_DIM);

__device__ __host__ void turnCoord(float32_t& xout, // meter
                                   float32_t& yout, // meter
                                   float32_t xin,
                                   float32_t yin,
                                   float32_t theta,       // degree
                                   float32_t turnRadius); // meter

__device__ __host__ uint32_t turnCoord(uint32_t x,
                                       uint32_t y,
                                       uint32_t theta,
                                       uint32_t X_DIM,
                                       uint32_t Y_DIM,
                                       float32_t POS_RES,
                                       float32_t HDG_RES,
                                       float32_t turnRadius);

void bitSweepTurn(uint32_t* RbO,
                  const uint32_t* Fb,
                  const uint32_t* RbI,
                  float32_t turnRadius,
                  cudaStream_t cuStream = nullptr);

void newSweepTurn(uint32_t* RbO,
                  const uint32_t* Fb,
                  const uint32_t* RbI,
                  float32_t turnRadius,
                  cudaStream_t cuStream = nullptr);

bool testGoal(const uint32_t* R, uint32_t c);

void prepareFreespace(uint32_t* Fb,
                      uint32_t X_DIM,
                      uint32_t Y_DIM);

void setUp(uint32_t** dev_reach, uint32_t** reach);

void tearDown(uint32_t** dev_reach, uint32_t** reach);

__global__ void copy(uint32_t* dst, const uint32_t* src, uint32_t N);

__global__ void copySection(uint32_t* dst,
                            const uint32_t* src,
                            uint32_t X_DIM,
                            uint32_t Y_DIM,
                            uint32_t section);

__global__ void sweepSectionFirst(uint32_t* Gf,        // sectional forward reachability volume
                                  uint32_t* Gr,        // sectional reverse reachability volume
                                  uint32_t* Fs,        // sectional freespace volume
                                  const uint32_t* Fb,  // input bit freespace volume
                                  const uint32_t* RbI, // input bit reachability volume
                                  uint32_t X_DIM,
                                  uint32_t Y_DIM,
                                  uint32_t section);

__global__ void sweepSectionMiddle(uint32_t* Gf,       // sectional forward reachability volume
                                   uint32_t* Gr,       // sectional reverse reachability volume
                                   const uint32_t* Fs, // sectional freespace volume
                                   uint32_t X_DIM,
                                   uint32_t Y_DIM,
                                   uint32_t section);

__global__ void sweepSectionLast(uint32_t* RbO,
                                 const uint32_t* Gf, // sectional forward reachability volume
                                 const uint32_t* Gr, // sectional reverse reachability volume
                                 const uint32_t* Fb,
                                 uint32_t X_DIM,
                                 uint32_t Y_DIM,
                                 uint32_t section);

// merge turn volumes into one
__global__ void merge(uint32_t* RbO,
                      const uint32_t* Ri,
                      uint32_t turnSize);

#endif // PAPER_HPP_