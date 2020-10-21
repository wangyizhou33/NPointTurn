#include "Paper.hpp"
#include <iostream>

int main(void)
{
    uint32_t* dev_reach[ITER_CNT];
    uint32_t* reach[ITER_CNT];
    uint32_t* dev_fb;
    uint32_t* fb;

    // setup
    for (uint32_t iter = 0; iter < ITER_CNT; ++iter)
    {
        HANDLE_ERROR(cudaMalloc((void**)&dev_reach[iter], SIZE));
        HANDLE_ERROR(cudaMemset((void*)dev_reach[iter], 0, SIZE));

        reach[iter] = (uint32_t*)malloc(SIZE);
        memset((void*)reach[iter], 0, SIZE);
    }

    fb = (uint32_t*)malloc(SIZE);
    HANDLE_ERROR(cudaMalloc((void**)&dev_fb, SIZE));

    prepareFreespace(fb, X_DIM, Y_DIM);
    HANDLE_ERROR(cudaMemcpy(dev_fb, fb, SIZE,
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaDeviceSynchronize());

    // set start
    uint32_t origin = turnCoord(X_DIM / 2, Y_DIM / 2, 0,
                                X_DIM, Y_DIM, POS_RES, HDG_RES, TURN_R);
    bitVectorWrite(reach[0], 1, origin);

    // host to device
    std::cout << "mem size in bytes: " << SIZE << std::endl;

    TIME_PRINT("copy h2d: ",
               HANDLE_ERROR(cudaMemcpy(dev_reach[0], reach[0], SIZE,
                                       cudaMemcpyHostToDevice));
               HANDLE_ERROR(cudaDeviceSynchronize()));

    TIME_PRINT("search: ",
               for (uint32_t iter = 0; iter + 2 < ITER_CNT; iter += 2) {
                   bitSweepTurn(dev_reach[iter + 1],
                                dev_fb,
                                dev_reach[iter],
                                TURN_R,
                                nullptr);

                   bitSweepTurn(dev_reach[iter + 2],
                                dev_fb,
                                dev_reach[iter + 1],
                                -TURN_R,
                                nullptr);
               } HANDLE_ERROR(cudaDeviceSynchronize()););
    TIME_PRINT("copy d2h",
               for (uint32_t iter = 0; iter + 1 < ITER_CNT; iter++) {
                   // device to host
                   HANDLE_ERROR(cudaMemcpy(reach[iter], dev_reach[iter], SIZE,
                                           cudaMemcpyDeviceToHost));

                   //    std::cout << "reachable bits "
                   //              << iter
                   //              << " "
                   //              << countBitsInVolume(reach[iter])
                   //              << std::endl;
               });

    // teardown
    for (uint32_t iter = 0; iter < ITER_CNT; ++iter)
    {
        HANDLE_ERROR(cudaFree(dev_reach[iter]));
        free(reach[iter]);
    }

    HANDLE_ERROR(cudaFree(dev_fb));

    return 0;
}
