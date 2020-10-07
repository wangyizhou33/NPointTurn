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

uint32_t countBitsInVolume(uint32_t* vol)
{
    uint32_t reachableBitCount = 0u;

    for (uint32_t theta = 0; theta < 360; ++theta)
    {
        uint32_t startIndex = X_DIM * Y_DIM * theta;
        uint32_t endIndex   = X_DIM * Y_DIM * (theta + 1);

        for (uint32_t coordIndex = startIndex / 32; coordIndex < endIndex / 32; coordIndex++)
        {
            reachableBitCount += countBits(vol[coordIndex]);
        }
    }

    return reachableBitCount;
}

__global__ void writeOnes(uint32_t* R, uint32_t offset)
{
    uint32_t tid    = threadIdx.x;
    uint32_t val    = (tid + 1 != blockDim.x) ? 4294967295 : 0;
    uint32_t laneid = tid % 32;

    // option1: no concurrent write for adjacent threads
    if (tid % 2)
        bitVectorWrite(R, val, tid * 32 + offset);
    __syncthreads();
    if (!(tid % 2))
        bitVectorWrite(R, val, tid * 32 + offset);

    // option2: send data up in a "lane"
    // uint32_t remainder  = val << offset;
    // uint32_t receiveVal = 0;
    // uint32_t sendVal    = val >> (32 - offset);

    // // __shfl_up(xx,xxx,1,xx)
    // receiveVal = __shfl_sync(0xffffffff, sendVal, laneid - 1, 32);
    // if (receiveVal == 0)
    //     printf("tid: %u, laneid: %u, remainder: %u, sendVal: %u, receiveVal: %u \n", tid, laneid, remainder, sendVal, receiveVal);

    // bitVectorWrite(R, remainder + receiveVal, tid * 32);

    // option3: naive concurrent write
    // bitVectorWrite(R, val, tid * 32 + offset);
};

__global__ void shuffle(uint32_t* R)
{
    uint32_t tid = threadIdx.x;

    uint32_t send = (1 << tid);

    uint32_t receive = __shfl_sync(0xffffffff, send, tid - 1, 8);

    // equivalently
    // uint32_t receive = __shfl_down_sync(0xffffffff, send, 1, 8);
    // @note: mask tends to do nothing

    printf("tid: %u, send: %u, receive: %u\n", tid, send, receive);

    R[tid] = receive;
}

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
    // else
    //     R[cm + 1] = R[cm + 1] & (~((1 << cr) - 1));
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

    if (tid == 0 ||
        tid + 1 == blockDim.x ||
        y > Y_DIM - 32 ||
        y < 32)
    {
        return;
    }

    uint32_t R = 0;
    for (uint32_t theta = 0; theta < 360; theta++)
    {
        uint32_t c  = turnCoordLeft(x, y, theta, X_DIM, Y_DIM, POS_RES, HDG_RES, turnRadius);
        uint32_t F1 = bitVectorRead(Fb, c);
        uint32_t R1 = bitVectorRead(RbI, c);

        R &= F1;
        R |= R1;

        // option1: perf is about 0.1 ms slower
        // than concurrent write
        if (cid & 1)
        {
            bitVectorWrite(RbO, R, c);
        }

        if (!(cid & 1))
        {
            bitVectorWrite(RbO, R, c);
        }

        // option2: __shfl
        // uint32_t cm = (c >> 5); // always the starting bit of a uint32_t memory
        // uint32_t cr = (c & 31); // shift cr bits

        // uint32_t remainder  = (R << cr);                                    // part of R that should be written in this RbO[cm]
        // uint32_t sendVal    = (cr != 0) ? R >> (32 - cr) : 0;               // part of R that should be written in this RbO[cm + 1]
        // uint32_t receiveVal = __shfl_sync(0xffffffff, sendVal, tid - 1, 8); // receive from the thread tid-1

        // bitVectorWrite(RbO, remainder + receiveVal, cm * 32);
        // bitVectorWrite(RbO, R, c);
    }
}

void bitSweepLeft(uint32_t* RbO,
                  const uint32_t* Fb,
                  const uint32_t* RbI,
                  float32_t turnRadius,
                  cudaStream_t cuStream)
{
    constexpr uint32_t ROWS_PER_BLOCK = 1u; // 1, 2, 4, 8, 16, 32
    _bitSweepLeft<<<Y_DIM / ROWS_PER_BLOCK, ROWS_PER_BLOCK * X_DIM / 32, 0, cuStream>>>(RbO, Fb, RbI, X_DIM, Y_DIM, POS_RES, HDG_RES, turnRadius);
}