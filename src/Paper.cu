#include "Paper.hpp"

__device__ __constant__ uint32_t dev_bitOffsets[512] =
    {
        0u,
        16384u,
        16384u,
        16385u,
        16384u,
        16384u,
        16384u,
        16385u,
        16384u,
        16384u,
        16384u,
        16385u,
        16384u,
        16384u,
        16384u,
        16385u,
        16384u,
        16384u,
        16384u,
        16513u,
        16384u,
        16384u,
        16384u,
        16385u,
        16384u,
        16384u,
        16384u,
        16385u,
        16384u,
        16384u,
        16384u,
        16384u,
        16513u,
        16384u,
        16384u,
        16384u,
        16385u,
        16384u,
        16384u,
        16384u,
        16384u,
        16385u,
        16512u,
        16384u,
        16384u,
        16384u,
        16385u,
        16384u,
        16384u,
        16512u,
        16385u,
        16384u,
        16384u,
        16384u,
        16384u,
        16384u,
        16513u,
        16384u,
        16384u,
        16384u,
        16384u,
        16385u,
        16512u,
        16384u,
        16384u,
        16384u,
        16384u,
        16385u,
        16512u,
        16384u,
        16384u,
        16384u,
        16384u,
        16513u,
        16384u,
        16384u,
        16384u,
        16384u,
        16384u,
        16512u,
        16385u,
        16384u,
        16384u,
        16512u,
        16384u,
        16384u,
        16384u,
        16385u,
        16512u,
        16384u,
        16384u,
        16384u,
        16384u,
        16512u,
        16384u,
        16384u,
        16384u,
        16513u,
        16384u,
        16384u,
        16384u,
        16384u,
        16512u,
        16384u,
        16384u,
        16384u,
        16512u,
        16384u,
        16384u,
        16384u,
        16513u,
        16384u,
        16384u,
        16384u,
        16512u,
        16384u,
        16384u,
        16384u,
        16512u,
        16384u,
        16384u,
        16384u,
        16512u,
        16384u,
        16384u,
        16384u,
        16512u,
        16384u,
        16384u,
        16384u,
        16384u,
        16512u,
        16384u,
        16384u,
        16384u,
        16512u,
        16384u,
        16384u,
        16384u,
        16512u,
        16384u,
        16384u,
        16384u,
        16512u,
        16384u,
        16384u,
        16384u,
        16511u,
        16384u,
        16384u,
        16384u,
        16512u,
        16384u,
        16384u,
        16384u,
        16512u,
        16384u,
        16384u,
        16384u,
        16384u,
        16511u,
        16384u,
        16384u,
        16384u,
        16512u,
        16384u,
        16384u,
        16384u,
        16384u,
        16512u,
        16383u,
        16384u,
        16384u,
        16384u,
        16512u,
        16384u,
        16384u,
        16383u,
        16512u,
        16384u,
        16384u,
        16384u,
        16384u,
        16384u,
        16511u,
        16384u,
        16384u,
        16384u,
        16384u,
        16512u,
        16383u,
        16384u,
        16384u,
        16384u,
        16384u,
        16512u,
        16383u,
        16384u,
        16384u,
        16384u,
        16384u,
        16511u,
        16384u,
        16384u,
        16384u,
        16384u,
        16384u,
        16383u,
        16512u,
        16384u,
        16384u,
        16383u,
        16384u,
        16384u,
        16384u,
        16512u,
        16383u,
        16384u,
        16384u,
        16384u,
        16384u,
        16383u,
        16384u,
        16384u,
        16384u,
        16511u,
        16384u,
        16384u,
        16384u,
        16384u,
        16383u,
        16384u,
        16384u,
        16384u,
        16383u,
        16384u,
        16384u,
        16384u,
        16511u,
        16384u,
        16384u,
        16384u,
        16383u,
        16384u,
        16384u,
        16384u,
        16383u,
        16384u,
        16384u,
        16384u,
        16383u,
        16384u,
        16384u,
        16384u,
        16383u,
        16384u,
        16384u,
        16384u,
        16384u,
        16383u,
        16384u,
        16384u,
        16384u,
        16383u,
        16384u,
        16384u,
        16384u,
        16383u,
        16384u,
        16384u,
        16384u,
        16383u,
        16384u,
        16384u,
        16384u,
        16255u,
        16384u,
        16384u,
        16384u,
        16383u,
        16384u,
        16384u,
        16384u,
        16383u,
        16384u,
        16384u,
        16384u,
        16384u,
        16255u,
        16384u,
        16384u,
        16384u,
        16383u,
        16384u,
        16384u,
        16384u,
        16384u,
        16383u,
        16256u,
        16384u,
        16384u,
        16384u,
        16383u,
        16384u,
        16384u,
        16256u,
        16383u,
        16384u,
        16384u,
        16384u,
        16384u,
        16384u,
        16255u,
        16384u,
        16384u,
        16384u,
        16384u,
        16383u,
        16256u,
        16384u,
        16384u,
        16384u,
        16384u,
        16383u,
        16256u,
        16384u,
        16384u,
        16384u,
        16384u,
        16255u,
        16384u,
        16384u,
        16384u,
        16384u,
        16384u,
        16256u,
        16383u,
        16384u,
        16384u,
        16256u,
        16384u,
        16384u,
        16384u,
        16383u,
        16256u,
        16384u,
        16384u,
        16384u,
        16384u,
        16256u,
        16384u,
        16384u,
        16384u,
        16255u,
        16384u,
        16384u,
        16384u,
        16384u,
        16256u,
        16384u,
        16384u,
        16384u,
        16256u,
        16384u,
        16384u,
        16384u,
        16255u,
        16384u,
        16384u,
        16384u,
        16256u,
        16384u,
        16384u,
        16384u,
        16256u,
        16384u,
        16384u,
        16384u,
        16256u,
        16384u,
        16384u,
        16384u,
        16256u,
        16384u,
        16384u,
        16384u,
        16384u,
        16256u,
        16384u,
        16384u,
        16384u,
        16256u,
        16384u,
        16384u,
        16384u,
        16256u,
        16384u,
        16384u,
        16384u,
        16256u,
        16384u,
        16384u,
        16384u,
        16257u,
        16384u,
        16384u,
        16384u,
        16256u,
        16384u,
        16384u,
        16384u,
        16256u,
        16384u,
        16384u,
        16384u,
        16384u,
        16257u,
        16384u,
        16384u,
        16384u,
        16256u,
        16384u,
        16384u,
        16384u,
        16384u,
        16256u,
        16385u,
        16384u,
        16384u,
        16384u,
        16256u,
        16384u,
        16384u,
        16385u,
        16256u,
        16384u,
        16384u,
        16384u,
        16384u,
        16384u,
        16257u,
        16384u,
        16384u,
        16384u,
        16384u,
        16256u,
        16385u,
        16384u,
        16384u,
        16384u,
        16384u,
        16256u,
        16385u,
        16384u,
        16384u,
        16384u,
        16384u,
        16257u,
        16384u,
        16384u,
        16384u,
        16384u,
        16384u,
        16385u,
        16256u,
        16384u,
        16384u,
        16385u,
        16384u,
        16384u,
        16384u,
        16256u,
        16385u,
        16384u,
        16384u,
        16384u,
        16384u,
        16385u,
        16384u,
        16384u,
        16384u,
        16257u,
        16384u,
        16384u,
        16384u,
        16384u,
        16385u,
        16384u,
        16384u,
        16384u,
        16385u,
        16384u,
        16384u,
        16384u,
        16257u,
        16384u,
        16384u,
        16384u,
        16385u,
        16384u,
        16384u,
        16384u,
        16385u,
        16384u,
        16384u,
        16384u,
        16385u,
        16384u,
        16384u,
        16384u,
        16385u,
        16384u};

__device__ __host__ uint32_t
countBits(uint32_t n)
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

    for (uint32_t theta = 0; theta < THETA_DIM; ++theta)
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
    // uint32_t receive = __shfl_up_sync(0xffffffff, send, 1, 8);
    // @note: mask tends to do nothing

    printf("tid: %u, send: %u, receive: %u\n", tid, send, receive);

    R[tid] = receive;
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

__device__ __host__ void turnCoord(float32_t& xout,
                                   float32_t& yout,
                                   float32_t xin,
                                   float32_t yin,
                                   float32_t theta,
                                   float32_t turnRadius)
{
    xout = xin + turnRadius * sin(deg2Rad(theta));
    yout = yin + turnRadius * (1.f - cos(deg2Rad(theta)));
}

__device__ __host__ uint32_t turnCoord(uint32_t x,
                                       uint32_t y,
                                       uint32_t theta,
                                       uint32_t X_DIM,
                                       uint32_t Y_DIM,
                                       float32_t POS_RES,
                                       float32_t HDG_RES,
                                       float32_t turnRadius)
{
    float32_t actualX{};
    float32_t actualY{};

    turnCoord(actualX, actualY,
              static_cast<float32_t>(x - HALF_X) * POS_RES,
              static_cast<float32_t>(y - HALF_Y) * POS_RES,
              static_cast<float32_t>(theta * HDG_RES),
              turnRadius);

    float32_t roundX = floor((actualX / POS_RES) + 0.5f);
    float32_t roundY = floor((actualY / POS_RES) + 0.5f);

    if (roundX >= HALF_X)
    {
        roundX -= 2 * HALF_X;
    }
    else if (roundX < -HALF_X)
    {
        roundX += 2 * HALF_X;
    }

    if (roundY >= HALF_Y)
    {
        roundY -= 2 * HALF_Y;
    }
    else if (roundY < -HALF_Y)
    {
        roundY += 2 * HALF_Y;
    }

    uint32_t newIndexX = static_cast<uint32_t>(roundX + HALF_X);
    uint32_t newIndexY = static_cast<uint32_t>(roundY + HALF_Y);

    // printf("%f, %f, %f, %f, %f, %f, %u, %u \n", actualX, actualY, roundX, roundY, HALF_X, HALF_Y, newIndexX, newIndexY);

    return volCoord(newIndexX, newIndexY, theta, X_DIM, Y_DIM);
}

__global__ void _bitSweepTurn(uint32_t* RbO,
                              const uint32_t* Fb,
                              const uint32_t* RbI,
                              uint32_t X_DIM,
                              uint32_t Y_DIM,
                              float32_t POS_RES,
                              float32_t HDG_RES,
                              float32_t turnRadius)
{
    uint32_t cellsPerRow   = X_DIM / 32u;
    uint32_t cellsPerBlock = blockDim.x;
    uint32_t rowsPerBlock  = cellsPerBlock / cellsPerRow;
    uint32_t tid           = threadIdx.x;                                   // [0, 4 * rowsPerBlock)
    uint32_t x             = tid % cellsPerRow * 32u;                       // bit offset
    uint32_t y             = blockIdx.x * rowsPerBlock + tid / cellsPerRow; // [0, 128)
    uint32_t cid           = y * X_DIM + tid % cellsPerRow;

    // print to check
    // printf("tid %u, x %u, y %u, \n", tid, x, y);

    // padding not necessary for shfl strategy
    // if (tid == 0 ||
    //     tid + 1 == blockDim.x ||
    //     y > Y_DIM - 32 ||
    //     y < 32)
    // {
    //     return;
    // }

    uint32_t R = 0;
    uint32_t c = volCoord(x, y, 0u, X_DIM, Y_DIM);

#pragma unroll
    for (uint32_t theta = 0; theta < THETA_DIM; theta++)
    {
        c += dev_bitOffsets[theta];
        // uint32_t c = turnCoord(x, y, theta, X_DIM, Y_DIM, POS_RES, HDG_RES, turnRadius);

        uint32_t F1 = bitVectorRead(Fb, c);
        uint32_t R1 = bitVectorRead(RbI, c);

        R &= F1;
        R |= R1;

        // option2: __shfl
        uint32_t cm = (c >> 5); // always the starting bit of a uint32_t memory
        uint32_t cr = (c & 31); // shift cr bits

        uint32_t remainder  = (R << cr);                                    // part of R that should be written in this RbO[cm]
        uint32_t sendVal    = (cr != 0) ? R >> (32 - cr) : 0;               // part of R that should be written in this RbO[cm + 1]
        uint32_t receiveVal = __shfl_sync(0xffffffff, sendVal, tid - 1, 4); // receive from the thread tid-1

        bitVectorWrite(RbO, remainder + receiveVal, cm * 32);
    }
}

void bitSweepTurn(uint32_t* RbO,
                  const uint32_t* Fb,
                  const uint32_t* RbI,
                  float32_t turnRadius,
                  cudaStream_t cuStream)
{
    constexpr uint32_t ROWS_PER_BLOCK = 32u; // must be power of 2, 1, 2, 4, 8, 16, 32, 64, 128
    _bitSweepTurn<<<Y_DIM / ROWS_PER_BLOCK, ROWS_PER_BLOCK * X_DIM / 32, 0, cuStream>>>(RbO,
                                                                                        Fb,
                                                                                        RbI,
                                                                                        X_DIM,
                                                                                        Y_DIM,
                                                                                        POS_RES,
                                                                                        HDG_RES,
                                                                                        turnRadius);
}

__global__ void _newSweepTurn(uint32_t* RbO,
                              const uint32_t* Fb,
                              const uint32_t* RbI,
                              uint32_t X_DIM,
                              uint32_t Y_DIM,
                              float32_t POS_RES,
                              float32_t HDG_RES,
                              float32_t turnRadius)
{
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

    // 0.11 ms
    // uint32_t R = 0u;

    // #pragma unroll
    //     for (uint32_t theta = 0; theta < THETA_DIM; theta++)
    //     {
    //         R &= Fb[i];
    //         R |= RbI[i];
    //         RbO[i] = R;

    //         // 0.10 ms
    //         // RbO[i] = RbI[i];
    //         i += 512u;
    //     }

    // SOL: 0.012 ms
    // RbO[i] = RbI[i];

    // R &= Fb[i];
    // R |= RbI[i];
    // move left
    // R = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);

    // move right
    // R = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4); //Move Right

    // move up
    // i  = ((i & 0xFFFFFE00) | ((i + 4) & 511));

    // move down
    // i = ((i & 0xFFFFFE00) | ((i - 4) & 511));

    // RbO[i] = R;
    uint32_t R = 0u;

    // theta : 0
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 1
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 2
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 3
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 4
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 5
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 6
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 7
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 8
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 9
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 10
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 11
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 12
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 13
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 14
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 15
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 16
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 17
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 18
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 19
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 20
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 21
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 22
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 23
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 24
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 25
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 26
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 27
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 28
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 29
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 30
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 31
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 32
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 33
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 34
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 35
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 36
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 37
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 38
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 39
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 40
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 41
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 42
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 43
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 44
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 45
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 46
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 47
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 48
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 49
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 50
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 51
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 52
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 53
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 54
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 55
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 56
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 57
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 58
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 59
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 60
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 61
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 62
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 63
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 64
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 65
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 66
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 67
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 68
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 69
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 70
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 71
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 72
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 73
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 74
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 75
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 76
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 77
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 78
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 79
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 80
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 81
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 82
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 83
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 84
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 85
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 86
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 87
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 88
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 89
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 90
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 91
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 92
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 93
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 94
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 95
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 96
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 97
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 98
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 99
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 100
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 101
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 102
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 103
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 104
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 105
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 106
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 107
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 108
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 109
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 110
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 111
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 112
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 113
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 114
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 115
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 116
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 117
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 118
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 119
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 120
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 121
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 122
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 123
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 124
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 125
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 126
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 127
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 128
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 129
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 130
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 131
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 132
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 133
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 134
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 135
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 136
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 137
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 138
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 139
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 140
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 141
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 142
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 143
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 144
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 145
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 146
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 147
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 148
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 149
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 150
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 151
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 152
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 153
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 154
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 155
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 156
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 157
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 158
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 159
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 160
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 161
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 162
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 163
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 164
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 165
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 166
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 167
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 168
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 169
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 170
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 171
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 172
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 173
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 174
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 175
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 176
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 177
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 178
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 179
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 180
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 181
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 182
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 183
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 184
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 185
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 186
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 187
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 188
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 189
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 190
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 191
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 192
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 193
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 194
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 195
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 196
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 197
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 198
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 199
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 200
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 201
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 202
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 203
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 204
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 205
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 206
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 207
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 208
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 209
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 210
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 211
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 212
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 213
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 214
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 215
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 216
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 217
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 218
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 219
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 220
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 221
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 222
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 223
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 224
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 225
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 226
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 227
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 228
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 229
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 230
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 231
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 232
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 233
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 234
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 235
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 236
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 237
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 238
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    i      = ((i & 0xFFFFFE00) | ((i + 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 239
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 240
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 241
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 242
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 243
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 244
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 245
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 246
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 247
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 248
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 249
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 250
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 251
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 252
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 253
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 254
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 255
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 256
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 257
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 258
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 259
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 260
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 261
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 262
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 263
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 264
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 265
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 266
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 267
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 268
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 269
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 270
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 271
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 272
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 273
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 274
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 275
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 276
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 277
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 278
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 279
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 280
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 281
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 282
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 283
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 284
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 285
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 286
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 287
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 288
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 289
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 290
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 291
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 292
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 293
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 294
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 295
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 296
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 297
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 298
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 299
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 300
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 301
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 302
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 303
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 304
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 305
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 306
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 307
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 308
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 309
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 310
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 311
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 312
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 313
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 314
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 315
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 316
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 317
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 318
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 319
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 320
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 321
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 322
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 323
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 324
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 325
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 326
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 327
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 328
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 329
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 330
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 331
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 332
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 333
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 334
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 335
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 336
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 337
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 338
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 339
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 340
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 341
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 342
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 343
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 344
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 345
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 346
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 347
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 348
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 349
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 350
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 351
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 352
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 353
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 354
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 355
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 356
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 357
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 358
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 359
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 360
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 361
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 362
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 363
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 364
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 365
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 366
    R &= Fb[i];
    R |= RbI[i];
    R      = (R << 1) | __shfl_sync(0xFFFFFFFF, R >> 31, threadIdx.x - 1, 4);
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 367
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 368
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 369
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 370
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 371
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 372
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 373
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 374
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 375
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 376
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 377
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 378
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 379
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 380
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 381
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 382
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 383
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 384
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 385
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 386
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 387
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 388
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 389
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 390
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 391
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 392
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 393
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 394
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 395
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 396
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 397
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 398
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 399
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 400
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 401
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 402
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 403
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 404
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 405
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 406
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 407
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 408
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 409
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 410
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 411
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 412
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 413
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 414
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 415
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 416
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 417
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 418
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 419
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 420
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 421
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 422
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 423
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 424
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 425
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 426
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 427
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 428
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 429
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 430
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 431
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 432
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 433
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 434
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 435
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 436
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 437
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 438
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 439
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 440
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 441
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 442
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 443
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 444
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 445
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 446
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 447
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 448
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 449
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 450
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 451
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 452
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 453
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 454
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 455
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 456
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 457
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 458
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 459
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 460
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 461
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 462
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 463
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 464
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 465
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 466
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 467
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 468
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 469
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 470
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 471
    R &= Fb[i];
    R |= RbI[i];
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 472
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 473
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 474
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 475
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 476
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 477
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 478
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 479
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 480
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 481
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 482
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 483
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 484
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 485
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 486
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 487
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 488
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 489
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 490
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 491
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 492
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 493
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 494
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    i      = ((i & 0xFFFFFE00) | ((i - 4) & 511));
    RbO[i] = R;
    i += 512u;

    // theta : 495
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 496
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 497
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 498
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 499
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 500
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 501
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 502
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 503
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 504
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 505
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 506
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 507
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 508
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 509
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;

    // theta : 510
    R &= Fb[i];
    R |= RbI[i];
    R      = (R >> 1) | __shfl_sync(0xFFFFFFFF, R << 31, threadIdx.x + 1, 4);
    RbO[i] = R;
    i += 512u;

    // theta : 511
    R &= Fb[i];
    R |= RbI[i];
    RbO[i] = R;
    i += 512u;
}

void newSweepTurn(uint32_t* RbO,
                  const uint32_t* Fb,
                  const uint32_t* RbI,
                  float32_t turnRadius,
                  cudaStream_t cuStream)
{
    constexpr uint32_t ROWS_PER_BLOCK = 64u; // must be power of 2, 1, 2, 4, 8, 16, 32, 64, 128
    _newSweepTurn<<<Y_DIM / ROWS_PER_BLOCK, ROWS_PER_BLOCK * X_DIM / 32, 0, cuStream>>>(RbO,
                                                                                        Fb,
                                                                                        RbI,
                                                                                        X_DIM,
                                                                                        Y_DIM,
                                                                                        POS_RES,
                                                                                        HDG_RES,
                                                                                        turnRadius);
}

bool testGoal(const uint32_t* R, uint32_t c)
{
    uint32_t r = bitVectorRead(R, c);

    return (r & 1u);
}

void prepareFreespace(uint32_t* Fb,
                      uint32_t X_DIM,
                      uint32_t Y_DIM)
{
    uint32_t SIZE = X_DIM * Y_DIM * THETA_DIM / 32u * sizeof(uint32_t);
    memset((void*)Fb, 2147483647, SIZE);

    for (uint32_t theta = 0u; theta < THETA_DIM; ++theta)
    {
        auto occupyBit = [theta, X_DIM, Y_DIM, Fb](uint32_t x, uint32_t y) {
            // bit offset
            uint32_t b = volCoord(x, y, theta, X_DIM, Y_DIM);
            bitVectorWrite(Fb, 4294967294u, b);
        };
        auto occupyCell = [theta, X_DIM, Y_DIM, Fb](uint32_t x, uint32_t y) {
            // bit offset
            uint32_t b = volCoord(x, y, theta, X_DIM, Y_DIM);
            bitVectorWrite(Fb, 0u, b);
        };

        for (uint32_t y = 0; y < Y_DIM; ++y)
        {
            occupyBit(0, y);
            occupyBit(X_DIM - 1, y);
        }

        for (uint32_t x = 0; x < X_DIM; x += 32u)
        {
            occupyCell(x, 0u);
            occupyCell(x, Y_DIM - 1u);
        }
    }
}

__global__ void copy(uint32_t* dst, const uint32_t* src, uint32_t N)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // while loop implementation
    while (i < N)
    {
        dst[i] = src[i];
        i += gridDim.x * blockDim.x;
    }

    // for loop implementation
    // #pragma unroll
    //     for (uint32_t j = 0; j < 512u; ++j)
    //     {
    //         dst[i + j * gridDim.x * blockDim.x] = src[i + j * gridDim.x * blockDim.x];
    //     }
}