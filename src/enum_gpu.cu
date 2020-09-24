#include "Helper.hpp"

#define N 10

__global__ void add(int* a, int* b, int* c)
{
    int tid = blockIdx.x; // this thread handles the data at its thread id
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

int main(void)
{
    cudaDeviceProp prop;

    int count;
    HANDLE_ERROR(cudaGetDeviceCount(&count));
    for (int i = 0; i < count; i++)
    {
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
        printf("   --- General Information for device %d ---\n", i);
        printf("Name:  %s\n", prop.name);
        printf("Compute capability:  %d.%d\n", prop.major, prop.minor);
        printf("Clock rate:  %d\n", prop.clockRate);
        printf("Device copy overlap:  ");
        if (prop.deviceOverlap)
            printf("Enabled\n");
        else
            printf("Disabled\n");
        printf("Kernel execution timeout :  ");
        if (prop.kernelExecTimeoutEnabled)
            printf("Enabled\n");
        else
            printf("Disabled\n");

        printf("   --- Memory Information for device %d ---\n", i);
        printf("Total global mem:  %ld\n", prop.totalGlobalMem);
        printf("Total constant Mem:  %ld\n", prop.totalConstMem);
        printf("Max mem pitch:  %ld\n", prop.memPitch);
        printf("Texture Alignment:  %ld\n", prop.textureAlignment);

        printf("   --- MP Information for device %d ---\n", i);
        printf("Multiprocessor count:  %d\n", prop.multiProcessorCount);
        printf("Shared mem per mp:  %ld\n", prop.sharedMemPerBlock);
        printf("Registers per mp:  %d\n", prop.regsPerBlock);
        printf("Threads in warp:  %d\n", prop.warpSize);
        printf("Max threads per block:  %d\n", prop.maxThreadsPerBlock);
        printf("Max thread dimensions:  (%d, %d, %d)\n", prop.maxThreadsDim[0],
               prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions:  (%d, %d, %d)\n", prop.maxGridSize[0],
               prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\n");
    }

    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    // allocate the memory on the GPU
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

    // fill the arrays 'a' and 'b' on the CPU
    for (int i = 0; i < N; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }

    // copy the arrays 'a' and 'b' to the GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

    add<<<N, 1>>>(dev_a, dev_b, dev_c);

    // copy the array 'c' back from the GPU to the CPU
    HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

    // display the results
    for (int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // free the memory allocated on the GPU
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));
}
