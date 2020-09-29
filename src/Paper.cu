#include "Paper.hpp"

__device__ __host__ uint32_t countBits(uint32_t n)
{
    uint32_t count = 0u;
    while (n)
    {
        count += n & 1;
        n >>= 1;
    }
    return count;
}

__global__ void writeOnes(uint32_t* R, uint32_t offset)
{
    uint32_t tid = threadIdx.x;
    uint32_t val = 4294967295;

    if (tid % 2)
        bitVectorWrite(R, val, tid * 32 + offset);

    if (!(tid % 2))
        bitVectorWrite(R, val, tid * 32 + offset);
};

__device__ __host__ uint32_t bitVectorRead(const uint32_t* RbI, uint32_t c)
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

__device__ __host__ void bitVectorWrite(uint32_t* R, uint32_t val, uint32_t c)
{
    uint32_t cm = (c >> 5);
    uint32_t cr = (c & 31);
    R[cm]       = ((R[cm] & ((1 << cr) - 1)) | (val << cr));
    if (cr)
        R[cm + 1] = ((R[cm + 1] & (~((1 << cr) - 1))) | (val >> (32 - cr)));
    else
        R[cm + 1] = R[cm + 1] & (~((1 << cr) - 1));
}

__device__ __host__ float32_t deg2Rad(float32_t deg)
{
    return 0.0174533f * deg;
};

__device__ __host__ uint32_t volCoord(uint32_t x,
                                      uint32_t y,
                                      uint32_t theta,
                                      uint32_t X_DIM,
                                      uint32_t Y_DIM)
{
    return (x + X_DIM * (y + Y_DIM * theta));
}

__device__ __host__ uint32_t turnCoordLeft(uint32_t x,
                                           uint32_t y,
                                           uint32_t theta,
                                           uint32_t X_DIM,
                                           uint32_t Y_DIM,
                                           float32_t POS_RES,
                                           float32_t HDG_RES,
                                           float32_t turnRadius)
{
    float32_t actualX = static_cast<float32_t>(x - HALF_X) * POS_RES + turnRadius * sin(deg2Rad(static_cast<float32_t>(theta * HDG_RES)));
    float32_t actualY = static_cast<float32_t>(y - HALF_Y) * POS_RES + turnRadius * (1.0f - cos(deg2Rad(static_cast<float32_t>(theta * HDG_RES))));
    float32_t roundX  = floor((actualX / POS_RES) + 0.5f);
    float32_t roundY  = floor((actualY / POS_RES) + 0.5f);

    uint32_t newIndexX = static_cast<uint32_t>(roundX + HALF_X);
    uint32_t newIndexY = static_cast<uint32_t>(roundY + HALF_Y);

    // printf("%f, %f, %f, %f, %f, %f, %u, %u \n", actualX, actualY, roundX, roundY, HALF_X, HALF_Y, newIndexX, newIndexY);

    return volCoord(newIndexX, newIndexY, theta, X_DIM, Y_DIM);
}

__global__ void _bitSweepLeft(uint32_t* RbO,
                              const uint32_t* Fb,
                              const uint32_t* RbI,
                              uint32_t X_DIM,
                              uint32_t Y_DIM,
                              float32_t POS_RES,
                              float32_t HDG_RES,
                              float32_t turnRadius)
{
    uint32_t tid = threadIdx.x; // [0, 5)
    uint32_t x   = tid * 32;
    uint32_t y   = blockIdx.x; // [0, 160)

    uint32_t cid = y * blockDim.x + tid;
    // printf("cid %u\n", cid);

    if (tid == 0 || tid + 1 == blockDim.x)
    {
        return;
    }

    uint32_t R = 0;
    for (uint32_t theta = 0; theta < 360; theta++)
    {
        uint32_t c = turnCoordLeft(x, y, theta, X_DIM, Y_DIM, POS_RES, HDG_RES, turnRadius);

        uint32_t F1 = bitVectorRead(Fb, c);
        uint32_t R1 = bitVectorRead(RbI, c);

        R &= F1;
        R |= R1;

        if (countBits(R))
        {
            printf("%u %u %u %u %u %u\n", theta, cid, R, x, y, c);
        }

        if (cid == 400)
            printf("%u %u %u %u %u %u\n", theta, cid, R, x, y, c);

        if (cid & 1)
        {
            bitVectorWrite(RbO, R, c);
            // if (R)
            //     printf("odd: %u\n", bitVectorRead(RbO, c));
        }

        if (!(cid & 1))
        {
            bitVectorWrite(RbO, R, c);
            // if (R)
            //     printf("even: %u\n", bitVectorRead(RbO, c));
        }
        // if (R)
        //     printf("after: %u\n", bitVectorRead(RbO, c));
    }
}

void bitSweepLeft(uint32_t* RbO,
                  const uint32_t* Fb,
                  const uint32_t* RbI,
                  float32_t turnRadius,
                  cudaStream_t cuStream)
{
    _bitSweepLeft<<<Y_DIM, X_DIM / 32, 0, cuStream>>>(RbO, Fb, RbI, X_DIM, Y_DIM, POS_RES, HDG_RES, turnRadius);
}