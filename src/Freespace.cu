#include "Freespace.hpp"
#include <algorithm>
#include <iostream>

Freespace::Freespace()
{
    m_size = m_dim.row * m_dim.col * m_dim.height;
    m_mem  = std::make_unique<value_type[]>(m_size);

    HANDLE_ERROR(cudaMalloc((void**)&m_cuMem, m_size * sizeof(value_type)));
    HANDLE_ERROR(cudaMalloc((void**)&m_cuMem1, m_size * sizeof(value_type)));

    reset();
}

Freespace::~Freespace()
{
    if (m_cuMem)
    {
        HANDLE_ERROR(cudaFree(m_cuMem));
    }
    if (m_cuMem1)
    {
        HANDLE_ERROR(cudaFree(m_cuMem1));
    }
}

void Freespace::reset()
{
    if (m_mem)
    {
        std::fill_n(m_mem.get(), m_size, 0u);
    }
    if (m_cuMem)
    {
        HANDLE_ERROR(cudaMemset((void*)m_cuMem, 0, m_size * sizeof(value_type)));
    }
    if (m_cuMem1)
    {
        HANDLE_ERROR(cudaMemset((void*)m_cuMem1, 0, m_size * sizeof(value_type)));
    }
}

void Freespace::computeFreespaceCPU(const std::vector<Obstacle>& vec)
{
    // compute "occupancy" in the 0-th slice
    compute0Slice(vec);

    for (uint32_t k = 1u; k < m_dim.height; ++k)
    {
        computeSliceCPU(k);
        dilateSliceCPU(m_mem.get() + k * m_dim.row * m_dim.col);
    }

    // dilate the 0-th slice
    dilateSliceCPU(m_mem.get());

    // rotate all slices, except the 0-th
    rotateCPU();
}

void Freespace::computeFreespaceGPU(const std::vector<Obstacle>& vec)
{
    // compute "occupancy" in the 0-th slice
    compute0Slice(vec);

    // copy the result to device
    HANDLE_ERROR(cudaMemcpy(m_cuMem,
                            m_mem.get(),
                            sizeof(value_type) * m_dim.row * m_dim.col, // just 0-th single slice
                            cudaMemcpyHostToDevice));

    // compute "occupancy" of the other slices based on the 0-th
    computeSliceGPU();

    // perform dilation
    dilateSliceGPU();

    // perform rotation
    rotateGPU();

    cudaDeviceSynchronize();

    // copy the result back to host
    HANDLE_ERROR(cudaMemcpy(m_mem.get(),
                            m_cuMem,
                            sizeof(value_type) * m_size, // whole volume
                            cudaMemcpyDeviceToHost));
}

void Freespace::compute0Slice(const std::vector<Obstacle>& vec)
{
    auto occupy = [this](const Vector2f& v) {
        Vector2ui ind = toIndex(v.x, v.y, m_dim.row, m_dim.col, m_dim.posRes);

        m_mem[index(ind.x, ind.y, 0u, m_dim.row, m_dim.col)] = 1u;
    };

    // clear
    std::fill_n(m_mem.get(), m_dim.row * m_dim.col, 0u);

    // fill the 0-th slice of the volume
    for (const Obstacle& obs : vec)
    {
        for (uint32_t i = 0; i + 1 < obs.boundaryPoints.size(); ++i)
        {
            const Vector2f& v0 = obs.boundaryPoints.at(i);
            const Vector2f& v1 = obs.boundaryPoints.at(i + 1);

            Vector2f e  = v1 - v0;
            float32_t l = e.norm();
            e.normalize();

            for (float32_t d = 0.f; d < l; d += m_dim.posRes)
            {
                Vector2f v = v0 + e * d;
                occupy(v);
            }
            occupy(v1);
        }
    }
}

void Freespace::computeSliceCPU(uint32_t k)
{
    for (uint32_t i = 0u; i < m_dim.row; ++i)
    {
        for (uint32_t j = 0u; j < m_dim.col; ++j)
        {
            Vector2f pos    = toCartesian(i, j, m_dim.row, m_dim.col, m_dim.posRes);
            float32_t theta = static_cast<float32_t>(k) * m_dim.hdgRes;
            uint32_t ind    = index(i, j, k, m_dim.row, m_dim.col);

            Vector2f newPos = pos.rotate(-theta);

            if (isInBoundary(newPos.x, newPos.y, 30.f)) //TODO: fix the hack
            {
                Vector2ui newInd = toIndex(newPos.x, newPos.y, m_dim.row, m_dim.col, m_dim.posRes);
                if (m_mem[index(newInd.x, newInd.y, 0u, m_dim.row, m_dim.col)] == 1u)
                {
                    m_mem[ind] = 1u;
                }
            }
        }
    }
}

#define imax(a, b) (a > b) ? a : b;
#define imin(a, b) (a < b) ? a : b;

void Freespace::dilateSliceCPU(value_type* mem)
{
    int height = m_dim.row;
    int width  = m_dim.col;

    value_type* tmp = new value_type[width * height];
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int start_j      = imax(0, j - radioC);
            int end_j        = imin(width - 1, j + radioC);
            value_type value = (value_type)0;
            for (int jj = start_j; jj <= end_j; jj++)
            {
                value = imax(mem[i * width + jj], value);
            }
            tmp[i * width + j] = value;
        }
    }
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int start_i      = imax(0, i - radioR);
            int end_i        = imin(height - 1, i + radioR);
            value_type value = (value_type)0;
            for (int ii = start_i; ii <= end_i; ii++)
            {
                value = imax(tmp[ii * width + j], value);
            }
            if (value == 1u)
            {
                if (mem[i * width + j] == 0u)
                {
                    mem[i * width + j] = 2u; // differentiate actual and dilation
                }
            }
            // mem[i * width + j] = value;
        }
    }
    delete[](tmp);
}

void Freespace::rotateCPU()
{
    // temp memory
    value_type* tmp = new value_type[m_dim.row + m_dim.col];

    auto rotate = [this](value_type* dst, value_type* src, float32_t theta) {
        for (uint32_t i = 0u; i < m_dim.row; ++i)
        {
            for (uint32_t j = 0u; j < m_dim.col; ++j)
            {
                // std::cerr << i << " " << j << std::endl;
                Vector2f pos = toCartesian(i, j, m_dim.row, m_dim.col, m_dim.posRes);
                uint32_t ind = index(i, j, 0, m_dim.row, m_dim.col);

                Vector2f newPos = pos.rotate(theta);

                if (isInBoundary(newPos.x, newPos.y, 30.f)) //TODO: fix the hack
                {
                    Vector2ui newInd = toIndex(newPos.x, newPos.y, m_dim.row, m_dim.col, m_dim.posRes);
                    dst[ind]         = src[index(newInd.x, newInd.y, 0u, m_dim.row, m_dim.col)];
                }
            }
        }
        return;
    };

    for (uint32_t k = 1u; k < m_dim.height; ++k)
    {
        // std::cerr << "k " << k << std::endl;
        std::copy(&m_mem[k * m_dim.row * m_dim.col],
                  &m_mem[(k + 1) * m_dim.row * m_dim.col],
                  tmp);
        // perhaps wes want to clear the dst memory here.

        rotate(&m_mem[k * m_dim.row * m_dim.col],
               tmp,
               static_cast<float32_t>(k) * m_dim.hdgRes);
    }

    delete[](tmp);
}

__global__ void _computeSliceGPU(Freespace::value_type* mem,
                                 uint32_t row,
                                 uint32_t col,
                                 float32_t posRes,
                                 float32_t hdgRes)
{
    uint32_t k      = blockIdx.x;
    uint32_t height = gridDim.x;
    uint32_t tid    = threadIdx.x;

    // 0-th slice is already done
    if (k == 0)
    {
        return;
    }

    while (tid < row * col)
    {
        uint32_t j = tid / row;
        uint32_t i = tid % row;

        if (j > col)
            return;

        Vector2f pos    = toCartesian((float32_t)i, (float32_t)j, (float32_t)row, (float32_t)col, posRes);
        float32_t theta = static_cast<float32_t>(k) * hdgRes;
        uint32_t ind    = index(i, j, k, row, col);

        Vector2f newPos = pos.rotate(-theta);

        if (isInBoundary(newPos.x, newPos.y, 30.f)) //TODO: fix the hack
        {
            Vector2ui newInd = toIndex(newPos.x, newPos.y, row, col, posRes);
            if (mem[index(newInd.x, newInd.y, 0u, row, col)] == (Freespace::value_type)1)
            {
                mem[ind] = (Freespace::value_type)1;
            }
        }

        tid += blockDim.x; // 32 times
    }
}

void Freespace::computeSliceGPU()
{
    dim3 grid  = {m_dim.height};
    dim3 block = {1024}; // each thread needs to handle 128 * 128 / 1024 = 16 cells

    // TODO(yizhouw): use non-default cudastream
    _computeSliceGPU<<<grid, block>>>(m_cuMem,
                                      m_dim.row,
                                      m_dim.col,
                                      m_dim.posRes,
                                      m_dim.hdgRes);
    HANDLE_ERROR(cudaGetLastError());
}

__global__ void _dilateKernel(Freespace::value_type* dst,
                              Freespace::value_type* src,
                              int row,
                              int col,
                              int radioR,
                              int radioC)
{
    int k   = blockIdx.x;
    int tid = threadIdx.x;

    while (tid < row * col)
    {
        int y = tid / row;
        int x = tid % row;

        int start_i                 = max(y - radioR, 0);
        int end_i                   = min(row - 1, y + radioR);
        int start_j                 = max(x - radioC, 0);
        int end_j                   = min(col - 1, x + radioC);
        Freespace::value_type value = (Freespace::value_type)0;
        for (int i = start_i; i <= end_i; i++)
        {
            for (int j = start_j; j <= end_j; j++)
            {
                value = max(value, src[i * col + j + k * row * col]);
            }
        }

        if (value == (Freespace::value_type)1)
        {
            Freespace::value_type srcVal = src[y * col + x + k * row * col];
            if (srcVal == (Freespace::value_type)0)
            {
                dst[y * col + x + k * row * col] = (Freespace::value_type)2;
            }
            else
            {
                dst[y * col + x + k * row * col] = (Freespace::value_type)1;
            }
        }

        tid += blockDim.x; // 32 times
    }
}

void Freespace::dilateSliceGPU()
{
    dim3 grid  = {m_dim.height};
    dim3 block = {1024};

    _dilateKernel<<<grid, block>>>(m_cuMem1,
                                   m_cuMem,
                                   m_dim.row,
                                   m_dim.col,
                                   radioR,
                                   radioC);

    HANDLE_ERROR(cudaGetLastError());
}

__global__ void _rotateKernel(Freespace::value_type* dst,
                              Freespace::value_type* src,
                              uint32_t row,
                              uint32_t col,
                              float32_t posRes,
                              float32_t hdgRes)
{
    uint32_t k      = blockIdx.x;
    uint32_t height = gridDim.x;
    uint32_t tid    = threadIdx.x;

    while (tid < row * col)
    {
        uint32_t j = tid / row;
        uint32_t i = tid % row;

        if (j > col)
            return;

        Vector2f pos    = toCartesian((float32_t)i, (float32_t)j, (float32_t)row, (float32_t)col, posRes);
        float32_t theta = static_cast<float32_t>(k) * hdgRes;
        uint32_t ind    = index(i, j, k, row, col);

        Vector2f newPos = pos.rotate(theta);

        if (isInBoundary(newPos.x, newPos.y, 30.f)) //TODO: fix the hack
        {
            Vector2ui newInd = toIndex(newPos.x, newPos.y, row, col, posRes);
            dst[ind]         = src[index(newInd.x, newInd.y, k, row, col)];
        }

        tid += blockDim.x; // 32 times
    }
}

void Freespace::rotateGPU()
{
    dim3 grid  = {m_dim.height};
    dim3 block = {1024};

    _rotateKernel<<<grid, block>>>(m_cuMem,
                                   m_cuMem1,
                                   m_dim.row,
                                   m_dim.col,
                                   m_dim.posRes,
                                   m_dim.hdgRes);

    HANDLE_ERROR(cudaGetLastError());
}