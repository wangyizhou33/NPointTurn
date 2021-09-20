/////////////////////////////////////////////////////////////////////////////////////////
// This code contains NVIDIA Confidential Information and is disclosed
// under the Mutual Non-Disclosure Agreement.
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//
// NVIDIA Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA Corporation. No third party distribution is allowed unless
// expressly authorized by NVIDIA.  Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA Corporation.
//
// Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#include <dw/experimental/parkingplanner/ParkingPlanner2.hpp>
#include <cuda_runtime.h>
#include <cuda.h>
#include <dw/cuda/CudaHelper.hpp> // memcpyAsync
#include <iostream>
#include <dw/math/MathUtils.hpp> // rad2Deg

namespace dw
{
namespace planner
{

const Coord3d ParkingPlanner2::INVALID_POSE = Coord3d{-1, -1, -1};

ParkingPlanner2::ParkingPlanner2(const dwParkingPlannerParams* params, core::Context* context)
    : Object(context)
{
    m_start       = {0, 0, 0};
    m_destination = INVALID_POSE; //Destination is initialized with an invalid pose. So path cannot be planned unless destination is set externally.

    m_grid    = makeUniqueSpan<GridCell>(getGridSize());
    m_gridNew = makeUniqueSpan<uint8_t>(getGridSize());  // analysis
    m_grid16  = makeUniqueSpan<uint16_t>(getGridSize()); // analysis
    m_grid32  = makeUniqueSpan<uint32_t>(getGridSize()); // analysis

    m_reach0 = makeUniqueSpan<uint32_t>(X_CELLS * Y_CELLS * THETA_STEP);
    m_reach1 = makeUniqueSpan<uint32_t>(X_CELLS * Y_CELLS * THETA_STEP);
    m_free   = makeUniqueSpan<uint32_t>(X_CELLS * Y_CELLS * THETA_STEP);

    m_cuGrid       = makeUniqueDeviceSpan<GridCell>(getGridSize());
    m_cuGridWarped = makeUniqueDeviceSpan<GridCell>(getGridSize());
    m_cuGridNew    = makeUniqueDeviceSpan<uint8_t>(getGridSize());  //analysis
    m_cuGrid16     = makeUniqueDeviceSpan<uint16_t>(getGridSize()); //analysis
    m_cuGrid32     = makeUniqueDeviceSpan<uint32_t>(getGridSize()); //analysis

    m_cuReach0 = makeUniqueDeviceSpan<uint32_t>(X_CELLS * Y_CELLS * THETA_STEP);
    m_cuReach1 = makeUniqueDeviceSpan<uint32_t>(X_CELLS * Y_CELLS * THETA_STEP);
    m_cuFree   = makeUniqueDeviceSpan<uint32_t>(X_CELLS * Y_CELLS * THETA_STEP);

    //Cost based
    m_gridMain  = makeUniqueSpan<uint16_t>(getGridSize3());
    m_gridObs   = makeUniqueSpan<uint16_t>(getGridSize3());
    m_gridTurns = makeUniqueSpan<uint16_t>(TURN_TYPES_NUM * getGridSize3());

    m_cuGridMain  = makeUniqueDeviceSpan<uint16_t>(getGridSize3());
    m_cuGridObs   = makeUniqueDeviceSpan<uint16_t>(getGridSize3());
    m_cuGridTurns = makeUniqueDeviceSpan<uint16_t>(TURN_TYPES_NUM * getGridSize3());
    //64
    m_cuGridTurns64 = makeUniqueDeviceSpan<uint64_t>(TURN_TYPES_NUM * getGridSize3() / 4);
    m_cuGridObs64   = makeUniqueDeviceSpan<uint64_t>(getGridSize3() / 4);

    //Copy 32 test
    m_cuOut32 = makeUniqueDeviceSpan<uint32_t>(64 * 128 * 512);
    m_cuIn32  = makeUniqueDeviceSpan<uint32_t>(64 * 128 * 512);

    //Copy 64 test
    m_cuOut64 = makeUniqueDeviceSpan<uint64_t>(32 * 128 * 512);
    m_cuIn64  = makeUniqueDeviceSpan<uint64_t>(32 * 128 * 512);

    m_cuObstacles = makeUniqueDeviceSpan<dwObstacle>(MAX_OBSTABLE_NUM);

    m_startTurnType = 8;

    if (params)
    {
        m_parkingPlannerParams.max_turns    = params->max_turns;
        m_parkingPlannerParams.turnRadius_m = params->turnRadius_m;
    }

    TIME_PRINT("constructor:setStartGPU", setStartGPU();)
}

void ParkingPlanner2::setObstacle(const Coord3d& obs)
{
    ParkingPlanner2::getCell(m_grid.get(), obs.x, obs.y, obs.hdg).obstacle = true;
}
void ParkingPlanner2::setObstacle(int32_t x, int32_t y, int32_t theta)
{
    Coord3d obs = {x, y, theta};
    this->setObstacle(obs);
}

void ParkingPlanner2::setSpecialObstacle(ObstacleName obs)
{
    switch (obs)
    {
    case NONE:
        break;
    case WALL:
        setWall();
        setDestination({30, 0, 0});
        break;
    case THREE_WALL_MAZE:
        setThreeWallMaze();
        setDestination({40, 40, 0});
        break;
    case BAY:
        setBay();
        setDestination({10, 10, 0});
        break;
    case HURDLE:
        setHurdle();
        setDestination({20, 0, 180});
        break;
    case PARALLEL:
        setParallel();
        setDestination({17, 0, 0});
        break;
    default:
        break;
    }
}

void ParkingPlanner2::setObstacles(span<const dwObstacle> obstacles)
{
    if (obstacles.size() > MAX_OBSTABLE_NUM)
    {
        throw Exception(DW_INTERNAL_ERROR,
                        "ParkingPlanner2: input obstacle number exceeds capacity.");
    }

    cuda::memcpyAsync(m_cuObstacles.get().data(),
                      obstacles.data(),
                      obstacles.size(),
                      cudaStream_t{});

    m_obstacleCount = obstacles.size();
}

void ParkingPlanner2::setTargetPose(const dwUnstructuredTarget& t)
{
    Coord3d targetPos = Coord3d{Vector3f{t.position.x,
                                         t.position.y,
                                         math::rad2Deg(t.heading)},
                                POS_RES,
                                HDG_RES};
    if (!withinPlanSpace(targetPos))
    {
        LOGSTREAM_WARN(this) << __func__
                             << ": Target pose not in planning space."
                             << Logger::State::endl;
    }
    setDestination(targetPos);
}

void ParkingPlanner2::setStart(const Coord3d& start)
{
    m_start = start;
    ParkingPlanner2::getCell(m_grid.get(), start.x, start.y, start.hdg).reachable = true;
    ParkingPlanner2::getCell(m_grid.get(), start.x, start.y, start.hdg).prevPose  = INVALID_POSE;
}
void ParkingPlanner2::setStart(int32_t x, int32_t y, int32_t theta)
{
    Coord3d start = {x, y, theta};
    setStart(start);
}

void ParkingPlanner2::setResetStart(const Coord3d& start)
{
    m_start = start;
}
void ParkingPlanner2::setResetStart(int32_t x, int32_t y, int32_t theta)
{
    Coord3d start = {x, y, theta};
    m_start = start;
}


void ParkingPlanner2::setDestination(const Coord3d& dest)
{
    m_destination = dest;
}
void ParkingPlanner2::setDestination(int32_t x, int32_t y, int32_t theta)
{
    Coord3d dest = {x, y, theta};
    setDestination(dest);
}

void ParkingPlanner2::reset()
{
    TIME_PRINT("reset:memset:cuGrid", DW_CHECK_CUDA_ERROR(cudaMemset(m_cuGrid.get().get().data(), 0, getGridSize() * sizeof(GridCell)));)
    TIME_PRINT("reset:memset:cuGridWarped", DW_CHECK_CUDA_ERROR(cudaMemset(m_cuGridWarped.get().get().data(), 0, getGridSize() * sizeof(GridCell)));)

    // reset the start and destination
    setEmptyStart();
    setDestination({});
    m_vertices.clear();
    m_path.clear();
    m_pathDrivingDirs.clear();
    m_segmentDirs.clear();
    m_maneuverList.clear();
    TIME_PRINT("reset:setStartGPU", setStartGPU();)
}

void ParkingPlanner2::reset2()
{

    // reset the start and destination
    initializeTrCost();
    setTransitionCost();
    initializeGrid();
    initObstacle();
    m_vertices.clear();
    m_path.clear();
    m_turns.clear();
}
void ParkingPlanner2::process()
{
    prepareOccupancyGrid();

    uint32_t turnCount{0};
    while (!isDestinationReachedGPU() &&
           !isMaxTurnsReached(turnCount++))
    {
        processOneTurn();
    }

    if (isDestinationReached())
    {
        buildPath();
    }
}

void ParkingPlanner2::process2()
{
    TIME_PRINT("process2:prepOccupancyGrid", prepareOccupancyGrid();)

    uint8_t turnCount{0};
    while (!isDestinationReachedGPU() &&
           !isMaxTurnsReached(turnCount++))
    {
        printf("\nturnCount:%hhu\n", turnCount);
        TIME_PRINT("ProcessOneTurn2", processOneTurn2(turnCount);)
    }

    if (isDestinationReached())
    {
        TIME_PRINT("buildPath2", buildPath2();)
    }
}

void ParkingPlanner2::timeGPU()
{
    TIME_PRINT("memset:m_cuGridNew", DW_CHECK_CUDA_ERROR(cudaMemset(m_cuGridNew.get().get().data(), static_cast<uint8_t>(0), getGridSize() * sizeof(uint8_t)));)
    TIME_PRINT("memset:m_gridNew", memset(m_gridNew.get().data(), static_cast<uint8_t>(0), getGridSize() * sizeof(uint8_t));)

    TIME_PRINT("H2D", copyGridNewHostToDevice();)

    TIME_PRINT("\n\n\nsweepAdd1", sweepAdd1(m_cuGridNew.get().get().data());)
    TIME_PRINT("sweepAdd2", sweepAdd2(m_cuGridNew.get().get().data());)

    //TIME_PRINT("memset:m_cuGridNew", DW_CHECK_CUDA_ERROR(cudaMemset(m_cuGridNew.get().get().data(), 0, getGridSize() * sizeof(uint8_t)));)
    TIME_PRINT("sweepAddLeft", sweepAddLeft(m_cuGridNew.get().get().data(), POS_RES, HDG_RES, 5.0f);)

    //setting Start at (0,0,0)
    TIME_PRINT("memset:m_cuGridNew", DW_CHECK_CUDA_ERROR(cudaMemset(m_cuGridNew.get().get().data(), static_cast<uint8_t>(0), getGridSize() * sizeof(uint8_t)));)
    m_gridNew[(Y_SIZE >> 2) + (X_SIZE >> 2) * Y_SIZE] = (1 << 7);
    DW_CHECK_CUDA_ERROR(cudaMemcpyAsync(m_cuGridNew.get().data().get() + (Y_SIZE >> 2) + (X_SIZE >> 2) * Y_SIZE,
                                        &m_gridNew[(Y_SIZE >> 2) + (X_SIZE >> 2) * Y_SIZE],
                                        sizeof(uint8_t),
                                        cudaMemcpyHostToDevice,
                                        m_cuStream));
    TIME_PRINT("iter 1, kernelLeft", kernelLeft(m_cuGridNew.get().get().data(), POS_RES, HDG_RES, 5.0f, 1);)
    TIME_PRINT("iter 2, kernelRight", kernelRight(m_cuGridNew.get().get().data(), POS_RES, HDG_RES, 5.0f, 2);)
    TIME_PRINT("iter 3, kerneLeft", kernelLeft(m_cuGridNew.get().get().data(), POS_RES, HDG_RES, 5.0f, 3);)
    TIME_PRINT("D2H", copyGridNewDeviceToHost();)
    printf("dest 0,0,180 cell_value: %hhu\n", m_gridNew[(Y_SIZE >> 2) + (((X_SIZE >> 2) + 180 * X_SIZE) * Y_SIZE)]);

    TIME_PRINT("memset:m_cuGrid16", DW_CHECK_CUDA_ERROR(cudaMemset(m_cuGrid16.get().get().data(), static_cast<uint16_t>(1), getGridSize() * sizeof(uint16_t)));)
    TIME_PRINT("memset:m_grid16", memset(m_grid16.get().data(), static_cast<uint16_t>(1), getGridSize() * sizeof(uint16_t));)

    TIME_PRINT("H2D 16", copyGrid16HostToDevice();)

    TIME_PRINT("\n\n\nsweepAdd16", sweepAdd16(m_cuGrid16.get().get().data());)

    //TIME_PRINT("memset:m_cuGridNew", DW_CHECK_CUDA_ERROR(cudaMemset(m_cuGridNew.get().get().data(), 0, getGridSize() * sizeof(uint8_t)));)
    TIME_PRINT("sweepAddLeft16", sweepAddLeft16(m_cuGrid16.get().get().data(), POS_RES, HDG_RES, 5.0f);)
    TIME_PRINT("D2H 16", copyGrid16DeviceToHost();)

    TIME_PRINT("\n\nmemset:m_grid32", memset(m_grid32.get().data(), static_cast<uint32_t>(1), getGridSize() * sizeof(uint32_t));)
    TIME_PRINT("H2D 32", copyGrid32HostToDevice();)
    TIME_PRINT("D2H 32", copyGrid32DeviceToHost();)

    //New bit kernel implementation
    memset(m_reach0.get().data(), static_cast<uint32_t>(0), X_CELLS * Y_CELLS * THETA_STEP * sizeof(uint32_t));
    memset(m_free.get().data(), static_cast<uint32_t>(~(0u)), X_CELLS * Y_CELLS * THETA_STEP * sizeof(uint32_t));
    printf("%u\n", m_free[100]);
    DW_CHECK_CUDA_ERROR(cudaMemset(m_cuReach1.get().get().data(), static_cast<uint32_t>(0), X_CELLS * Y_CELLS * THETA_STEP * sizeof(uint32_t)));
    uint32_t index         = (X_DIM >> 1) + X_DIM * (Y_DIM >> 1);
    uint32_t bitpos        = (index & 31);
    m_reach0[(index >> 5)] = (1 << bitpos);
    printf("%u, index: %u, cm: %u, bitpos: %u\n\n\n", m_reach0[(index >> 5)], index, index >> 5, bitpos);
    TIME_PRINT("H2D Reach 0", cuda::memcpyAsync(m_cuReach0.get().data(),
                                                m_reach0.get().data(),
                                                m_cuReach0.get().size(),
                                                m_cuStream);)
    TIME_PRINT("H2D Reach 0", cuda::memcpyAsync(m_cuReach0.get().data(),
                                                m_reach0.get().data(),
                                                m_cuReach0.get().size(),
                                                m_cuStream);)
    TIME_PRINT("H2D Reach 0", cuda::memcpyAsync(m_cuReach0.get().data(),
                                                m_reach0.get().data(),
                                                m_cuReach0.get().size(),
                                                m_cuStream);)
    TIME_PRINT("H2D Reach 0", cuda::memcpyAsync(m_cuReach0.get().data(),
                                                m_reach0.get().data(),
                                                m_cuReach0.get().size(),
                                                m_cuStream);)
    TIME_PRINT("H2D Reach 0", cuda::memcpyAsync(m_cuReach0.get().data(),
                                                m_reach0.get().data(),
                                                m_cuReach0.get().size(),
                                                m_cuStream);)
    TIME_PRINT("H2D Free", cuda::memcpyAsync(m_cuFree.get().data(),
                                             m_free.get().data(),
                                             m_cuFree.get().size(),
                                             m_cuStream);)
    TIME_PRINT("H2D Free", cuda::memcpyAsync(m_cuFree.get().data(),
                                             m_free.get().data(),
                                             m_cuFree.get().size(),
                                             m_cuStream);)
    TIME_PRINT("H2D Free", cuda::memcpyAsync(m_cuFree.get().data(),
                                             m_free.get().data(),
                                             m_cuFree.get().size(),
                                             m_cuStream);)
    TIME_PRINT("H2D Free", cuda::memcpyAsync(m_cuFree.get().data(),
                                             m_free.get().data(),
                                             m_cuFree.get().size(),
                                             m_cuStream);)
    TIME_PRINT("H2D Free", cuda::memcpyAsync(m_cuFree.get().data(),
                                             m_free.get().data(),
                                             m_cuFree.get().size(),
                                             m_cuStream);)

    TIME_PRINT("bitSweepLeft", bitSweepLeft(m_cuReach1.get().get().data(), m_cuFree.get().get().data(), m_cuReach0.get().get().data(), 5.0f);)

    TIME_PRINT("D2H Reach 1", cuda::memcpyAsync(m_reach1.get().data(),
                                                m_cuReach1.get().data(),
                                                m_cuReach1.get().size(),
                                                m_cuStream);)
    TIME_PRINT("D2H Reach 1", cuda::memcpyAsync(m_reach1.get().data(),
                                                m_cuReach1.get().data(),
                                                m_cuReach1.get().size(),
                                                m_cuStream);)
    TIME_PRINT("D2H Reach 1", cuda::memcpyAsync(m_reach1.get().data(),
                                                m_cuReach1.get().data(),
                                                m_cuReach1.get().size(),
                                                m_cuStream);)
    TIME_PRINT("D2H Reach 1", cuda::memcpyAsync(m_reach1.get().data(),
                                                m_cuReach1.get().data(),
                                                m_cuReach1.get().size(),
                                                m_cuStream);)

    //checking (10,10,90)
    index  = (10 + (X_DIM >> 1)) + X_DIM * (10 + (Y_DIM >> 1) + Y_DIM * (90));
    bitpos = (index & 31);
    printf("\n\n\n(10,10,90): (index>>5):%d\n bitpos = %d,\n value in 32 bit cell = %d\n", (index >> 5), bitpos, m_reach1[(index >> 5)]);

    //checking (0,20,180)
    index  = (0 + (X_DIM >> 1)) + X_DIM * (20 + (Y_DIM >> 1) + Y_DIM * (180));
    bitpos = (index & 31);
    printf("\n(0,20,180): (index>>5):%d\n bitpos = %d,\n value in 32 bit cell = %d\n", (index >> 5), bitpos, m_reach1[(index >> 5)]);

    //checking (-10,10,270)
    index  = (-10 + (X_DIM >> 1)) + X_DIM * (10 + (Y_DIM >> 1) + Y_DIM * (270));
    bitpos = (index & 31);
    printf("\n(-10,10,270): (index>>5):%d\n bitpos = %d,\n value in 32 bit cell = %d\n", (index >> 5), bitpos, m_reach1[(index >> 5)]);

    //New bit kernel CPU
    memset(m_reach0.get().data(), static_cast<uint32_t>(0), X_CELLS * Y_CELLS * THETA_STEP * sizeof(uint32_t));
    memset(m_free.get().data(), static_cast<uint32_t>(~(0u)), X_CELLS * Y_CELLS * THETA_STEP * sizeof(uint32_t));
    printf("%u\n", m_free[100]);
    DW_CHECK_CUDA_ERROR(cudaMemset(m_cuReach1.get().get().data(), static_cast<uint32_t>(0), X_CELLS * Y_CELLS * THETA_STEP * sizeof(uint32_t)));
    index                  = (X_DIM >> 1) + X_DIM * (Y_DIM >> 1);
    bitpos                 = (index & 31);
    m_reach0[(index >> 5)] = (1 << bitpos);
    printf("%u, index: %u, cm: %u, bitpos: %u\n", m_reach0[(index >> 5)], index, index >> 5, bitpos);
    /*TIME_PRINT("H2D Reach 0", cuda::memcpyAsync(m_cuReach0.get().data(),
                                        m_reach0.get().data(),
                                        m_cuReach0.get().size(),
                                        m_cuStream);)
    TIME_PRINT("H2D Free", cuda::memcpyAsync(m_cuFree.get().data(),
                                        m_free.get().data(),
                                        m_cuFree.get().size(),
                                        m_cuStream);)*/

    TIME_PRINT("bitSweepLeftCPU", bitSweepLeftCPU(m_reach1.get().data(), m_free.get().data(), m_reach0.get().data(), 5.0f);)

    /*TIME_PRINT("D2H Reach 1", cuda::memcpyAsync(m_reach1.get().data(),
                                        m_cuReach1.get().data(),
                                        m_cuReach1.get().size(),
                                        m_cuStream);)*/

    //checking (10,10,90)
    index  = (10 + (X_DIM >> 1)) + X_DIM * (10 + (Y_DIM >> 1) + Y_DIM * (90));
    bitpos = (index & 31);
    printf("\n(10,10,90): (index>>5):%d\n bitpos = %d,\n value in 32 bit cell = %d\n", (index >> 5), bitpos, m_reach1[(index >> 5)]);

    //checking (0,20,180)
    index  = (0 + (X_DIM >> 1)) + X_DIM * (20 + (Y_DIM >> 1) + Y_DIM * (180));
    bitpos = (index & 31);
    printf("\n(0,20,180): (index>>5):%d\n bitpos = %d,\n value in 32 bit cell = %d\n", (index >> 5), bitpos, m_reach1[(index >> 5)]);

    //checking (-10,10,270)
    index  = (-10 + (X_DIM >> 1)) + X_DIM * (10 + (Y_DIM >> 1) + Y_DIM * (270));
    bitpos = (index & 31);
    printf("\n(-10,10,270): (index>>5):%d\n bitpos = %d,\n value in 32 bit cell = %d\n", (index >> 5), bitpos, m_reach1[(index >> 5)]);

    // *** Our own MemCpy iterations
    uint32_t *source32, *dest32;
    cudaMalloc(&source32, static_cast<unsigned long>(sizeof(uint32_t) * 4 * 128 * 360));
    cudaMalloc(&dest32, static_cast<unsigned long>(sizeof(uint32_t) * 4 * 128 * 360));
    DW_CHECK_CUDA_ERROR(cudaMemset(source32, static_cast<uint32_t>(2), static_cast<unsigned long>(sizeof(uint32_t) * 4 * 128 * 360)));
    DW_CHECK_CUDA_ERROR(cudaMemset(dest32, 0, static_cast<unsigned long>(sizeof(uint32_t) * 4 * 128 * 360)));
    TIME_PRINT("OurMemCopy32:", ourMemCpy32(dest32, source32, 4 * 128);)
    cudaFree(source32);
    cudaFree(dest32);

    // *** Our own MemCpy iterations direct copy
    //uint32_t *source32, *dest32;
    cudaMalloc(&source32, static_cast<unsigned long>(sizeof(uint32_t) * 4 * 128 * 360));
    cudaMalloc(&dest32, static_cast<unsigned long>(sizeof(uint32_t) * 4 * 128 * 360));
    DW_CHECK_CUDA_ERROR(cudaMemset(source32, static_cast<uint32_t>(2), static_cast<unsigned long>(sizeof(uint32_t) * 4 * 128 * 360)));
    DW_CHECK_CUDA_ERROR(cudaMemset(dest32, 0, static_cast<unsigned long>(sizeof(uint32_t) * 4 * 128 * 360)));
    TIME_PRINT("OurMemCopy32Direct:", ourMemCpy32Direct(dest32, source32, 4 * 128 * 360);)
    cudaFree(source32);
    cudaFree(dest32);

    // ***temporary test code
    /*int32_t *sourceMem, *destMem;
    cudaMalloc(&sourceMem, static_cast<unsigned long>(sizeof(int32_t) * 4*128*360));
    cudaMalloc(&destMem, static_cast<unsigned long>(sizeof(int32_t) * 4*128*360));
    DW_CHECK_CUDA_ERROR(cudaMemset(sourceMem, static_cast<int32_t>(~0), static_cast<unsigned long>(sizeof(int32_t) * 4*128*360)));
    DW_CHECK_CUDA_ERROR(cudaMemset(destMem, static_cast<int32_t>(0), static_cast<unsigned long>(sizeof(int32_t) * 4*128*360)));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaMemcpyAsync(sourceMem,
                    destMem,
                    static_cast<unsigned long>(sizeof(int32_t) * 4*128*360),
                    cudaMemcpyDeviceToDevice,
                    m_cuStream);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float32_t milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("DeviceToDevice copy time: %f\n", milliseconds);
    cudaFree(sourceMem);
    cudaFree(destMem);
    // ***
    printf("size: %d", X_LENGTH);
*/
}

void ParkingPlanner2::timeSOL()
{
    TIME_PRINT("OurSectionCopy(4)", sections16());
    TIME_PRINT("OurDirectCopy16", directCopy16());
    DW_CHECK_CUDA_ERROR(cudaMemset(m_cuOut32.get().get().data(), static_cast<uint32_t>(0), static_cast<unsigned long>(sizeof(int32_t) * 64 * 128 * 512)));
    DW_CHECK_CUDA_ERROR(cudaMemset(m_cuIn32.get().get().data(), static_cast<uint32_t>(~0), static_cast<unsigned long>(sizeof(int32_t) * 64 * 128 * 512)));
    TIME_PRINT("OurDirectCopy32", directCopy32());
    TIME_PRINT("OurDirectCopyUnroll32", directCopyUnroll32());
    DW_CHECK_CUDA_ERROR(cudaMemset(m_cuOut64.get().get().data(), static_cast<uint64_t>(0), static_cast<unsigned long>(sizeof(int64_t) * 32 * 128 * 512)));
    DW_CHECK_CUDA_ERROR(cudaMemset(m_cuIn64.get().get().data(), static_cast<uint64_t>(4294967295u), static_cast<unsigned long>(sizeof(int64_t) * 32 * 128 * 512)));
    TIME_PRINT("OurDirectCopy64", directCopy64());

    // ***temporary test code
    int16_t *sourceMem, *destMem;
    cudaMalloc(&sourceMem, static_cast<unsigned long>(sizeof(int16_t) * 128 * 128 * 512));
    cudaMalloc(&destMem, static_cast<unsigned long>(sizeof(int16_t) * 128 * 128 * 512));
    DW_CHECK_CUDA_ERROR(cudaMemset(sourceMem, static_cast<uint16_t>(~0), static_cast<unsigned long>(sizeof(int16_t) * 128 * 128 * 512)));
    DW_CHECK_CUDA_ERROR(cudaMemset(destMem, static_cast<uint16_t>(0), static_cast<unsigned long>(sizeof(int16_t) * 128 * 128 * 512)));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaMemcpyAsync(sourceMem,
                    destMem,
                    static_cast<unsigned long>(sizeof(int16_t) * 128 * 128 * 512),
                    cudaMemcpyDeviceToDevice,
                    m_cuStream);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float32_t milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("DeviceToDevice copy time: %f\n", milliseconds);
    cudaFree(sourceMem);
    cudaFree(destMem);
    // ***
}

/*void ParkingPlanner2::initializeTrCost()
{
    uint16_t p = 1000u;
    for(int32_t i = 0; i < TURN_TYPES_NUM; i++)
    {
        for(int32_t j = 0; j < TURN_TYPES_NUM; j++)
        {
            if(i == j)
            {
                m_trCost[i + TURN_TYPES_NUM*j] = 0;
            }
            else
            {
                m_trCost[i + TURN_TYPES_NUM*j] = p;
            }
            
        }
    }
}*/

void ParkingPlanner2::initializeTrCost()
{
    float32_t p    = 180.0f / M_PI; //parameters that can be varied to tune
    float32_t q    = 1000.0f;
    float32_t r[8] = {-10.0f, -20.0f, 20.0f, 10.0f, -10.0f, -20.0f, 20.0f, 10.0f};

    for (int32_t i = 0; i < TURN_TYPES_NUM; i++)
    {
        for (int32_t j = i; j < TURN_TYPES_NUM; j++)
        {
            if (i == j)
            {
                m_trCost[i + TURN_TYPES_NUM * j] = 0;
            }
            else if ((i < 4 && j < 4) || (i >= 4 && j >= 4))
            {
                m_trCost[i + TURN_TYPES_NUM * j] = static_cast<uint16_t>(p * WHEELBASE3 * STEER_RATIO3 * abs((1 / r[i]) - (1 / r[j])));
            }
            else
            {
                m_trCost[i + TURN_TYPES_NUM * j] = static_cast<uint16_t>(p * WHEELBASE3 * STEER_RATIO3 * abs((1 / r[i]) - (1 / r[j])) + q);
            }
            m_trCost[j + TURN_TYPES_NUM * i] = m_trCost[i + TURN_TYPES_NUM * j];
        }
        m_trCost[i + TURN_TYPES_NUM * TURN_TYPES_NUM] = 0;
    }

}

void ParkingPlanner2::computeIndexIncrement()
{
    Coord3d coord10{};
    Coord3d coord10p{};
    Coord3d coord20{};
    Coord3d coord20p{};

    for (int32_t theta = 1; theta < THETA_DIM3; theta++)
    {
        coord10  = ParkingPlanner2::turnIndexPlain(0, 0, theta, true, 10.0f);
        coord10p = ParkingPlanner2::turnIndexPlain(0, 0, theta - 1, true, 10.0f);

        m_turnIncrementR10[theta - 1] = (((coord10.y - coord10p.y + 1) << 2) | (coord10.x - coord10p.x + 1));

        coord20  = ParkingPlanner2::turnIndexPlain(0, 0, theta, true, 20.0f);
        coord20p = ParkingPlanner2::turnIndexPlain(0, 0, theta - 1, true, 20.0f);

        m_turnIncrementR20[theta - 1] = (((coord20.y - coord20p.y + 1) << 2) | (coord20.x - coord20p.x + 1));
        /*printf("%d : R10: %d, %d, %hhd \t R20: %d, %d, %hhd\n", theta, (coord10.x - coord10p.x + 1), (coord10.y - coord10p.y + 1), m_turnIncrementR10[theta - 1],
               (coord20.x - coord20p.x + 1), (coord20.y - coord20p.y + 1), m_turnIncrementR20[theta - 1]);*/
    }
    coord10                 = ParkingPlanner2::turnIndexPlain(0, 0, 0, true, 10.0f);
    coord10p                = ParkingPlanner2::turnIndexPlain(0, 0, 511, true, 10.0f);
    m_turnIncrementR10[511] = (((coord10.y - coord10p.y + 1) << 2) | (coord10.x - coord10p.x + 1)); // 511 -> 0

    coord20                 = ParkingPlanner2::turnIndexPlain(0, 0, 0, true, 20.0f);
    coord20p                = ParkingPlanner2::turnIndexPlain(0, 0, 511, true, 20.0f);
    m_turnIncrementR20[511] = (((coord20.y - coord20p.y + 1) << 2) | (coord20.x - coord20p.x + 1));
    /*printf("%d : R10: %d, %d, %hhd \t R20: %d, %d, %hhd\n", 511, (coord10.x - coord10p.x + 1), (coord10.y - coord10p.y + 1), m_turnIncrementR10[511],
           (coord20.x - coord20p.x + 1), (coord20.y - coord20p.y + 1), m_turnIncrementR20[511]);*/
}

void ParkingPlanner2::computeIndexIncrement4()
{
    for (int32_t theta = 0; theta < THETA_DIM3; theta += 4)
    {
        Coord3d coord10  = ParkingPlanner2::turnIndexPlain(0, 0, theta + 1, true, 10.0f);
        Coord3d coord10p = ParkingPlanner2::turnIndexPlain(0, 0, theta, true, 10.0f);

        m_turnIncrementR10Four[theta] = (((coord10.y - coord10p.y + 1) << 2) | (coord10.x - coord10p.x + 1));

        coord10  = ParkingPlanner2::turnIndexPlain(0, 0, theta + 2, true, 10.0f);
        coord10p = ParkingPlanner2::turnIndexPlain(0, 0, theta, true, 10.0f);

        m_turnIncrementR10Four[theta + 1] = (((coord10.y - coord10p.y + 1) << 2) | (coord10.x - coord10p.x + 1));

        coord10  = ParkingPlanner2::turnIndexPlain(0, 0, theta + 3, true, 10.0f);
        coord10p = ParkingPlanner2::turnIndexPlain(0, 0, theta, true, 10.0f);

        m_turnIncrementR10Four[theta + 2] = (((coord10.y - coord10p.y + 1) << 2) | (coord10.x - coord10p.x + 1));

        coord10  = ParkingPlanner2::turnIndexPlain(0, 0, theta + 4, true, 10.0f);
        coord10p = ParkingPlanner2::turnIndexPlain(0, 0, theta, true, 10.0f);

        m_turnIncrementR10Four[theta + 3] = (((coord10.y - coord10p.y + 1) << 2) | (coord10.x - coord10p.x + 1));
        //printf("%d : R10: %d, %d, %hhd \t R20: %d, %d, %hhd\n",theta, (coord10.x - coord10p.x + 1),(coord10.y - coord10p.y + 1),m_turnIncrementR10[theta-1],
        //(coord20.x - coord20p.x + 1),(coord20.y - coord20p.y + 1),m_turnIncrementR20[theta-1]);
    }
}

void ParkingPlanner2::initializeGrid()
{
    int32_t startIndex = volIndex(m_start.x, m_start.y, m_start.hdg);
    /*std::fill_n(m_gridMain.get().data(),getGridSize3(),MAX_COST);
    cuda::memcpyAsync(m_cuGridMain.get().data(), m_gridMain.get().data(), m_cuGridMain.get().size(), m_cuStream);
    cudaDeviceSynchronize();
    cudaMemcpyAsync(m_cuGridMain.get().get().data(),m_cuGridR10LeftForward.get().get().data(), getGridSize3(),cudaMemcpyDeviceToDevice,m_cuStream);
    cudaDeviceSynchronize();*/
    DW_CHECK_CUDA_ERROR(cudaMemset(m_cuGridTurns.get().get().data(), MAX_COST, (TURN_TYPES_NUM * getGridSize3() * sizeof(uint16_t))));

    DW_CHECK_CUDA_ERROR(cudaMemset(m_cuGridTurns.get().get().data() + startIndex, m_trCost[0 + TURN_TYPES_NUM*m_startTurnType], sizeof(uint16_t)));
    DW_CHECK_CUDA_ERROR(cudaMemset(m_cuGridTurns.get().get().data() + 1 * getGridSize3() + startIndex, m_trCost[1 + TURN_TYPES_NUM*m_startTurnType], 
                                    sizeof(uint16_t)));
    DW_CHECK_CUDA_ERROR(cudaMemset(m_cuGridTurns.get().get().data() + 2 * getGridSize3() + startIndex, m_trCost[2 + TURN_TYPES_NUM*m_startTurnType], 
                                    sizeof(uint16_t)));
    DW_CHECK_CUDA_ERROR(cudaMemset(m_cuGridTurns.get().get().data() + 3 * getGridSize3() + startIndex, m_trCost[3 + TURN_TYPES_NUM*m_startTurnType], 
                                    sizeof(uint16_t)));
    DW_CHECK_CUDA_ERROR(cudaMemset(m_cuGridTurns.get().get().data() + 4 * getGridSize3() + startIndex, m_trCost[4 + TURN_TYPES_NUM*m_startTurnType], 
                                    sizeof(uint16_t)));
    DW_CHECK_CUDA_ERROR(cudaMemset(m_cuGridTurns.get().get().data() + 5 * getGridSize3() + startIndex, m_trCost[5 + TURN_TYPES_NUM*m_startTurnType], 
                                    sizeof(uint16_t)));
    DW_CHECK_CUDA_ERROR(cudaMemset(m_cuGridTurns.get().get().data() + 6 * getGridSize3() + startIndex, m_trCost[6 + TURN_TYPES_NUM*m_startTurnType], 
                                    sizeof(uint16_t)));
    DW_CHECK_CUDA_ERROR(cudaMemset(m_cuGridTurns.get().get().data() + 7 * getGridSize3() + startIndex, m_trCost[7 + TURN_TYPES_NUM*m_startTurnType], 
                                    sizeof(uint16_t)));
}

void ParkingPlanner2::initializeGrid32()
{
    int32_t startIndex = 32 + 64 * 64;
    DW_CHECK_CUDA_ERROR(cudaMemset(m_cuOut32.get().get().data(), (64250u), (64 * 128 * 512 * sizeof(uint32_t))));
    DW_CHECK_CUDA_ERROR(cudaMemset(m_cuOut32.get().get().data() + startIndex, 0, (sizeof(uint32_t))));
    DW_CHECK_CUDA_ERROR(cudaMemset(m_cuIn32.get().get().data(), 0, (64 * 128 * 512 * sizeof(uint32_t))));
}

void ParkingPlanner2::initializeGrid64()
{
    int32_t startIndex = 16 + 32 * 64;
    DW_CHECK_CUDA_ERROR(cudaMemset(m_cuOut64.get().get().data(), (64250u), (32 * 128 * 512 * sizeof(uint64_t))));
    DW_CHECK_CUDA_ERROR(cudaMemset(m_cuOut64.get().get().data() + startIndex, 0, (sizeof(uint64_t))));
    DW_CHECK_CUDA_ERROR(cudaMemset(m_cuIn64.get().get().data(), 0, (32 * 128 * 512 * sizeof(uint32_t))));
}
void ParkingPlanner2::printTrCost()
{
    for (int32_t i = 0; i <= TURN_TYPES_NUM; i++)
    {
        for (int32_t j = 0; j < TURN_TYPES_NUM; j++)
        {
            printf("[%d + size*%d]: %hu \t", i, j, m_trCost[j + TURN_TYPES_NUM * i]);
        }
        printf("\n");
    }
}

void ParkingPlanner2::intermediateTestKernel()
{
    TIME_PRINT("initializeGrid", initializeGrid();)
    TIME_PRINT("initObstacle", initObstacle();)
    TIME_PRINT("computeIndexIncrement", computeIndexIncrement();)
    //TIME_PRINT("computeIndexIncrement4",computeIndexIncrement4();)
    TIME_PRINT("setTurnIncrement", setTurnIncrement();)
    TIME_PRINT("sweep Up16", costUp16(m_cuStream);)
    // TIME_PRINT("loopUnroll sweep Up16",loopUnroll16(m_cuStream);)
}

void ParkingPlanner2::testSweep32()
{
    TIME_PRINT("\n\ncomputeIndexIncrement", computeIndexIncrement();)
    TIME_PRINT("setTurnIncrement", setTurnIncrement();)
    TIME_PRINT("initializeGrid32", initializeGrid32();)
    TIME_PRINT("costSweep32", costSweep32();)
    TIME_PRINT("copy32To16", copy32To16();)
    copyDeviceToHostMain();
}

void ParkingPlanner2::testSweep64()
{
    TIME_PRINT("\n\ncomputeIndexIncrement", computeIndexIncrement();)
    TIME_PRINT("setTurnIncrement", setTurnIncrement();)
    TIME_PRINT("initializeGrid", initializeGrid();)
    TIME_PRINT("initObstacle", initObstacle();)
    TIME_PRINT("costSweep64", costSweep64();)
    TIME_PRINT("minCost for visualization", minCost();)
    //TIME_PRINT("copy64To16", copy64To16();)
    copyDeviceToHostMain();
    copyDeviceToHostObstacle();
}

void ParkingPlanner2::processSweepStep()
{
    /*TIME_PRINT("\tsweep R10RightForward",costSweepR10RightForward(m_cuStream);)
    TIME_PRINT("\tsweep R20RightForward",costSweepR20RightForward(m_cuStream);)
    TIME_PRINT("\tsweep R20LeftForward",costSweepR20LeftForward(m_cuStream);)
    TIME_PRINT("\tsweep R20LeftForward",costSweepR10LeftForward(m_cuStream);)
    TIME_PRINT("\tsweep R10RightReverse",costSweepR10RightReverse(m_cuStream);)
    TIME_PRINT("\tsweep R20RightReverse",costSweepR20RightReverse(m_cuStream);)
    TIME_PRINT("\tsweep R20LeftReverse",costSweepR20LeftReverse(m_cuStream);)
    TIME_PRINT("\tsweep R10LeftReverse",costSweepR10LeftReverse(m_cuStream);)*/
    TIME_PRINT("\n\n\tsweep R10RightForward", costSweepB(m_cuGridTurns.get().get().data(), m_cuGridObs.get().get().data(), false, 10.0f, m_cuStream);)
    TIME_PRINT("\tsweep R20RightForward", costSweepB(m_cuGridTurns.get().get().data() + 1 * getGridSize3(), m_cuGridObs.get().get().data(), false, 20.0f, m_cuStream);)
    TIME_PRINT("\tsweep R20LeftForward", costSweepA(m_cuGridTurns.get().get().data() + 2 * getGridSize3(), m_cuGridObs.get().get().data(), true, 20.0f, m_cuStream);)
    TIME_PRINT("\tsweep R10LeftForward", costSweepA(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data(), true, 10.0f, m_cuStream);)
    //TIME_PRINT("sweep Up16",costUp16(m_cuStream);)
    TIME_PRINT("\tsweep R10RightReverse", costSweepA(m_cuGridTurns.get().get().data() + 4 * getGridSize3(), m_cuGridObs.get().get().data(), false, 10.0f, m_cuStream);)
    TIME_PRINT("\tsweep R20RightReverse", costSweepA(m_cuGridTurns.get().get().data() + 5 * getGridSize3(), m_cuGridObs.get().get().data(), false, 20.0f, m_cuStream);)
    TIME_PRINT("\tsweep R20LeftReverse", costSweepB(m_cuGridTurns.get().get().data() + 6 * getGridSize3(), m_cuGridObs.get().get().data(), true, 20.0f, m_cuStream);)
    TIME_PRINT("\tsweep R10LeftReverse", costSweepB(m_cuGridTurns.get().get().data() + 7 * getGridSize3(), m_cuGridObs.get().get().data(), true, 10.0f, m_cuStream);)
}

void ParkingPlanner2::processCostStep()
{
    TIME_PRINT("\nFinal Sweep*8", processSweepStep();)
    TIME_PRINT("minCost for visualization", minCost();)
    copyDeviceToHostMain();
    copyDeviceToHostObstacle();
}

void ParkingPlanner2::copyDeviceToHostMain()
{
    TIME_PRINT("D2H Min: ", cuda::memcpyAsync(m_gridMain.get().data(),
                                              m_cuGridMain.get().data(),
                                              m_cuGridMain.get().size(),
                                              m_cuStream);)
}

void ParkingPlanner2::copyDeviceToHostGrid()
{
    TIME_PRINT("D2H Cost Volumes", cuda::memcpyAsync(m_gridTurns.get().data(), m_cuGridTurns.get().data(), m_cuGridTurns.get().size(), m_cuStream);)
}

void ParkingPlanner2::copyDeviceToHostObstacle()
{
    TIME_PRINT("D2H Obstacle", cuda::memcpyAsync(m_gridObs.get().data(), m_cuGridObs.get().data(), m_cuGridObs.get().size(), m_cuStream);)
}

void ParkingPlanner2::processCost()
{
    //setDestination(-51, 44, 0);
    TIME_PRINT("\n\n\ncomputeIndexIncrement", computeIndexIncrement();)
    //TIME_PRINT("computeIndexIncrement4",computeIndexIncrement4();)
    TIME_PRINT("setTurnIncrement", setTurnIncrement();)

    TIME_PRINT("initializeTrCost", initializeTrCost();)

    TIME_PRINT("setTransitionCost", setTransitionCost();)
    TIME_PRINT("initializeGrid", initializeGrid();)
    TIME_PRINT("initObstacle", initObstacle();)

    for (int32_t i = 0; i < MAX_TURNS; i++)
    {
        //TIME_PRINT("Sweep*8",processSweepStep();)
        //TIME_PRINT("Sweep*8 all",costSweepAll());
        TIME_PRINT("Sweep64*8", costSweepAll64();)
        TIME_PRINT("costTransition", costTransition();)
    }

    //TIME_PRINT("\nFinal Sweep*8",processSweepStep();)
    //TIME_PRINT("Sweep*8 all",costSweepAll());
    TIME_PRINT("Sweep64*8", costSweepAll64());

    if (destinationCheck())
    {
        TIME_PRINT("minCost for visualization", minCost();)
        copyDeviceToHostMain();
        int32_t destIndex = volIndex(m_destination.x, m_destination.y, m_destination.hdg);
        printf("Dest cost: %hu\n", m_gridMain[destIndex]);
        copyDeviceToHostObstacle();
        copyDeviceToHostGrid();
        if (backTraceCost())
        {
            //printBackTrace();
            //printFullPath();
            //printf("cost calculated: %hu\n", pathToCost(0,static_cast<int32_t>(m_path.size())));
        }
    }
    else
    {
        TIME_PRINT("minCost for visualization", minCost();)
        copyDeviceToHostMain();
        copyDeviceToHostObstacle();
        copyDeviceToHostGrid();
        printf("destination not reached.\n");
    }

}

void ParkingPlanner2::printBackTrace()
{
    printf("\n\nPath segments: %lu \n", m_turns.size());
    auto itt = m_turns.begin();
    auto itv = m_vertices.begin();
    while (itt != m_turns.end() && itv != m_vertices.end())
    {
        Vector3f point = *itv;
        printf("%f, %f, %f : %d\n", point.x(), point.y(), point.z(), *itt);
        itt++;
        itv++;
    }
}

void ParkingPlanner2::printFullPath()
{
    printf("\n\nPath size: %lu \n", m_turnEachStep.size());
    auto itt = m_turnEachStep.begin();
    auto itv = m_path.begin();
    auto itpt = m_pathTheta.begin();
    while (itt != m_turnEachStep.end() && itv != m_path.end())
    {
        Vector3f point = *itv;
        printf("%f, %f, %f : %d: %d\n", point.x(), point.y(), point.z(),*itpt, *itt);
        itt++;
        itv++;
        itpt++;
    }
}

int32_t ParkingPlanner2::turnNextIndex(Coord3d coord, int32_t turnType)
{
    switch (turnType)
    {
    case 0:
    {
        return nextIndex(coord.x, coord.y, coord.hdg, 10.0f, false, false);
    }
    case 1:
    {
        return nextIndex(coord.x, coord.y, coord.hdg, 20.0f, false, false);
    }
    case 2:
    {
        return nextIndex(coord.x, coord.y, coord.hdg, 20.0f, true, false);
    }
    case 3:
    {
        return nextIndex(coord.x, coord.y, coord.hdg, 10.0f, true, false);
    }
    case 4:
    {
        return nextIndex(coord.x, coord.y, coord.hdg, 10.0f, false, true);
    }
    case 5:
    {
        return nextIndex(coord.x, coord.y, coord.hdg, 20.0f, false, true);
    }
    case 6:
    {
        return nextIndex(coord.x, coord.y, coord.hdg, 20.0f, true, true);
    }
    case 7:
    {
        return nextIndex(coord.x, coord.y, coord.hdg, 10.0f, true, true);
    }
    default:
        return -1;
    }
}

uint16_t ParkingPlanner2::turnStepCost(int32_t turnType)
{
    switch (turnType)
    {
    case 0:
    {
        return static_cast<uint16_t>(10.0f / MIN_TURNRADIUS3);
    }
    case 1:
    {
        return static_cast<uint16_t>(20.0f / MIN_TURNRADIUS3);
    }
    case 2:
    {
        return static_cast<uint16_t>(20.0f / MIN_TURNRADIUS3);
    }
    case 3:
    {
        return static_cast<uint16_t>(10.0f / MIN_TURNRADIUS3);
    }
    case 4:
    {
        return static_cast<uint16_t>(10.0f / MIN_TURNRADIUS3);
    }
    case 5:
    {
        return static_cast<uint16_t>(20.0f / MIN_TURNRADIUS3);
    }
    case 6:
    {
        return static_cast<uint16_t>(20.0f / MIN_TURNRADIUS3);
    }
    case 7:
    {
        return static_cast<uint16_t>(10.0f / MIN_TURNRADIUS3);
    }
    default:
        return 0;
    }
}

bool ParkingPlanner2::backTraceCost()
{
    m_path.clear();
    m_vertices.clear();
    m_turns.clear();
    m_turnEachStep.clear();
    m_pathTheta.clear();

    //find minCost of Destination and corresponding volume
    Coord3d current                = m_destination;
    int32_t index                  = volIndex(current.x, current.y, current.hdg);
    uint16_t costs[TURN_TYPES_NUM] = {m_gridTurns[index],
                                      m_gridTurns[1 * getGridSize3() + index],
                                      m_gridTurns[2 * getGridSize3() + index],
                                      m_gridTurns[3 * getGridSize3() + index],
                                      m_gridTurns[4 * getGridSize3() + index],
                                      m_gridTurns[5 * getGridSize3() + index],
                                      m_gridTurns[6 * getGridSize3() + index],
                                      m_gridTurns[7 * getGridSize3() + index]};

    //printf("Destination cost: %hu, %hu, %hu, %hu, %hu, %hu, %hu, %hu\n", costs[0], costs[1], costs[2], costs[3], costs[4], costs[5], costs[6], costs[7]);

    int32_t turnType = static_cast<int32_t>(std::min_element(costs, costs + TURN_TYPES_NUM) - costs);

    m_path.push_back(current.getPose(POS_RES3, HDG_RES_DEG3));
    m_pathTheta.push_back(current.hdg);
    m_vertices.push_back(current.getPose(POS_RES3, HDG_RES_DEG3));
    m_turns.push_back(turnType);
    m_turnEachStep.push_back(turnType);
    //printf("current: %d, %d, %d, turnType: %d, cost:%hu\n",current.x, current.y, current.hdg, turnType, m_gridTurns[index + turnType*getGridSize3()]);

    //Loop moving back step by step until reaching start
    while (index != volIndex(m_start.x, m_start.y, m_start.hdg) && !m_turns.full())
    {
        if ((turnNextIndex(current, turnType) > 0) &&
            (m_gridTurns[index + turnType * getGridSize3()] >=
             m_gridTurns[turnNextIndex(current, turnType) + turnType * getGridSize3()] + turnStepCost(turnType)))
        {
            index   = turnNextIndex(current, turnType);
            current = index2Coord(index, DIM3);
            m_path.push_back(current.getPose(POS_RES3, HDG_RES_DEG3));
            m_turnEachStep.push_back(turnType);
            m_pathTheta.push_back(current.hdg);

            //printf("current: %d, %d, %d, turnType: %d, cost:%hu\n", current.x, current.y, current.hdg, turnType, m_gridTurns[index + turnType * getGridSize3()]);
        }
        else
        {
            int32_t prevTurnType = turnType;
            for (int32_t i = 0; i < TURN_TYPES_NUM; ++i)
            {
                if (i == turnType)
                {
                    costs[i] = MAX_COST;
                }
                else
                {
                    costs[i] = MAX_COST;
                    if (turnNextIndex(current, i) > 0)
                    {
                        if (m_gridTurns[index + turnType * getGridSize3()] >=
                            m_gridTurns[turnNextIndex(current, i) + i * getGridSize3()] + m_trCost[turnType + TURN_TYPES_NUM * i] + turnStepCost(i))
                        {
                            costs[i] = m_gridTurns[turnNextIndex(current, i) + i * getGridSize3()];
                        }
                    }
                }
            }
            turnType = static_cast<int32_t>(std::min_element(costs, costs + TURN_TYPES_NUM) - costs);
            if (costs[turnType] == MAX_COST)
            {
                printf("Backtrace failed at %d,%d,%d : %hu. turnType:%d\n", current.x, current.y, current.hdg, m_gridTurns[index + prevTurnType * getGridSize3()], prevTurnType);
                for (int32_t i = 0; i < TURN_TYPES_NUM; i++)
                {
                    Coord3d coord = index2Coord(turnNextIndex(current, i), DIM3);
                    uint16_t cost = m_gridTurns[turnNextIndex(current, i) + i * getGridSize3()];
                    printf("turn %d, Next Coord: %d,%d,%d, cost:%hu: %hu, %hu, %hu, %hu, %hu, %hu, %hu, %hu\n", i, coord.x, coord.y, coord.hdg, cost,
                           m_gridTurns[turnNextIndex(current, i) + 0 * getGridSize3()], m_gridTurns[turnNextIndex(current, i) + 1 * getGridSize3()],
                           m_gridTurns[turnNextIndex(current, i) + 2 * getGridSize3()], m_gridTurns[turnNextIndex(current, i) + 3 * getGridSize3()],
                           m_gridTurns[turnNextIndex(current, i) + 4 * getGridSize3()], m_gridTurns[turnNextIndex(current, i) + 5 * getGridSize3()],
                           m_gridTurns[turnNextIndex(current, i) + 6 * getGridSize3()], m_gridTurns[turnNextIndex(current, i) + 7 * getGridSize3()]);
                }
                printf("%d, %d, %d: %hu, %hu, %hu, %hu, %hu, %hu, %hu, %hu", current.x, current.y, current.hdg, m_gridTurns[index],
                       m_gridTurns[index + 1 * getGridSize3()], m_gridTurns[index + 2 * getGridSize3()], m_gridTurns[index + 3 * getGridSize3()],
                       m_gridTurns[index + 4 * getGridSize3()], m_gridTurns[index + 5 * getGridSize3()], m_gridTurns[index + 6 * getGridSize3()],
                       m_gridTurns[index + 7 * getGridSize3()]);
                return false;
            }
            index   = turnNextIndex(current, turnType);
            current = index2Coord(index, DIM3);
            m_turns.push_back(turnType);
            m_turnEachStep.push_back(turnType);
            m_vertices.push_back(current.getPose(POS_RES3, HDG_RES_DEG3));
            m_path.push_back(current.getPose(POS_RES3, HDG_RES_DEG3));
            m_pathTheta.push_back(current.hdg);

            //printf("current: %d, %d, %d, turnType: %d\n", current.x, current.y, current.hdg, turnType);
        }
    }
    if (index == volIndex(m_start.x, m_start.y, m_start.hdg) && m_vertices.available() >= 1)
    {
        std::reverse(m_vertices.begin(), m_vertices.end());
        std::reverse(m_path.begin(), m_path.end());
        //printf("PathLength: %lu\n", m_path.size());
        std::reverse(m_turns.begin(), m_turns.end());
        std::reverse(m_turnEachStep.begin(), m_turnEachStep.end());
        std::reverse(m_pathTheta.begin(), m_pathTheta.end());
        return true;
    }
    else
    {
        printf("Backtrace exit because path full. %d\n", m_turns.full());
        return false;
    }
}

uint16_t ParkingPlanner2::pathToCost(uint32_t startIndex, uint32_t endIndex)
{
    int32_t angleCost{};
    uint16_t trCost{};
    if(startIndex > m_path.size() || endIndex < startIndex || endIndex > m_path.size())
    {
        printf("Invalid Indices\n");
        return 0u;
    }

    for(uint32_t i = startIndex; i < endIndex-1;i++)
    {
        if(m_turnEachStep[i] == 0 || m_turnEachStep[i] == 3 || m_turnEachStep[i] == 4 || m_turnEachStep[i] == 7 )
            angleCost += 1;
        else
            angleCost += 2;
        if(m_turnEachStep[i] != m_turnEachStep[i+1])
            trCost += m_trCost[m_turnEachStep[i] + TURN_TYPES_NUM*m_turnEachStep[i+1]];
    }
    return trCost + static_cast<uint16_t>(angleCost);
}

void ParkingPlanner2::processNew()
{
    TIME_PRINT("process2:prepOccupancyGrid", prepareOccupancyGrid();)

    uint8_t turnCount{0};
    while (!isDestinationReachedGPU() &&
           !isMaxTurnsReached(turnCount++))
    {
        printf("\nturnCount:%hhu\n", turnCount);
        TIME_PRINT("ProcessOneTurnNew", processOneTurnNew(turnCount);)
    }

    if (isDestinationReached())
    {
        TIME_PRINT("buildPath2", buildPath2();)
    }
}

void ParkingPlanner2::processOneTurn()
{
    processStraight();
    processLeft();
    processRight();
}

void ParkingPlanner2::processOneTurn2(uint8_t iter)
{
    TIME_PRINT("processStraight2", processStraight2(iter);)
    TIME_PRINT("processLeft2", processLeft2(iter);)
    TIME_PRINT("processRight2", processRight2(iter);)
}

void ParkingPlanner2::processOneTurnNew(uint8_t iter)
{
    TIME_PRINT("processStraightNew", processStraightNew(iter);)
    TIME_PRINT("processLeft2", processLeft2(iter);)
    TIME_PRINT("processRight2", processRight2(iter);)
}

void ParkingPlanner2::processStraight()
{
    setUnwarpedPose();

    warpStraight(m_cuGridWarped.get().get().data(),
                 m_cuGrid.get().get().data(),
                 POS_RES,
                 HDG_RES,
                 m_cuStream);
    sweepStraight(m_cuGridWarped.get().get().data(),
                  m_cuStream);
    warpStraight(m_cuGrid.get().get().data(),
                 m_cuGridWarped.get().get().data(),
                 POS_RES,
                 HDG_RES,
                 m_cuStream,
                 true);
}
void ParkingPlanner2::processLeft()
{
    setUnwarpedPose();

    warpLeft(m_cuGridWarped.get().get().data(),
             m_cuGrid.get().get().data(),
             POS_RES,
             HDG_RES,
             m_parkingPlannerParams.turnRadius_m,
             m_cuStream);
    sweepArc(m_cuGridWarped.get().get().data(),
             ManeuverType::LEFT,
             m_cuStream);
    warpLeft(m_cuGrid.get().get().data(),
             m_cuGridWarped.get().get().data(),
             POS_RES,
             HDG_RES,
             m_parkingPlannerParams.turnRadius_m,
             m_cuStream,
             true);
}
void ParkingPlanner2::processRight()
{
    setUnwarpedPose();

    warpRight(m_cuGridWarped.get().get().data(),
              m_cuGrid.get().get().data(),
              POS_RES,
              HDG_RES,
              m_parkingPlannerParams.turnRadius_m,
              m_cuStream);
    sweepArc(m_cuGridWarped.get().get().data(),
             ManeuverType::RIGHT,
             m_cuStream);
    warpRight(m_cuGrid.get().get().data(),
              m_cuGridWarped.get().get().data(),
              POS_RES,
              HDG_RES,
              m_parkingPlannerParams.turnRadius_m,
              m_cuStream,
              true);
}

void ParkingPlanner2::processStraight2(uint8_t iter)
{

    TIME_PRINT("\nwarpStraight", warpStraight(m_cuGridWarped.get().get().data(),
                                              m_cuGrid.get().get().data(),
                                              POS_RES,
                                              HDG_RES,
                                              m_cuStream);)
    TIME_PRINT("sweepStraight2", sweepStraight2(m_cuGridWarped.get().get().data(),
                                                iter,
                                                m_cuStream);)
    TIME_PRINT("unwarpStraight", warpStraight(m_cuGrid.get().get().data(),
                                              m_cuGridWarped.get().get().data(),
                                              POS_RES,
                                              HDG_RES,
                                              m_cuStream,
                                              true);)
}

void ParkingPlanner2::processStraightNew(uint8_t iter)
{

    TIME_PRINT("\nwarpStraightNew", warpStraightNew(m_cuGridWarped.get().get().data(),
                                                    m_cuGrid.get().get().data(),
                                                    POS_RES,
                                                    HDG_RES,
                                                    m_cuStream);)
    TIME_PRINT("sweepStraightNew", sweepStraightNew(m_cuGridWarped.get().get().data(),
                                                    iter,
                                                    m_cuStream);)
    TIME_PRINT("unwarpStraightNew", warpStraightNew(m_cuGrid.get().get().data(),
                                                    m_cuGridWarped.get().get().data(),
                                                    POS_RES,
                                                    HDG_RES,
                                                    m_cuStream,
                                                    true);)
}

void ParkingPlanner2::processLeft2(uint8_t iter)
{

    TIME_PRINT("\nwarpLeft", warpLeft(m_cuGridWarped.get().get().data(),
                                      m_cuGrid.get().get().data(),
                                      POS_RES,
                                      HDG_RES,
                                      m_parkingPlannerParams.turnRadius_m,
                                      m_cuStream);)
    TIME_PRINT("sweepArc2", sweepArc2(m_cuGridWarped.get().get().data(),
                                      ManeuverType::LEFT,
                                      iter,
                                      m_cuStream);)
    TIME_PRINT("unwarpLeft", warpLeft(m_cuGrid.get().get().data(),
                                      m_cuGridWarped.get().get().data(),
                                      POS_RES,
                                      HDG_RES,
                                      m_parkingPlannerParams.turnRadius_m,
                                      m_cuStream,
                                      true);)
}
void ParkingPlanner2::processRight2(uint8_t iter)
{

    TIME_PRINT("\nwarpRight", warpRight(m_cuGridWarped.get().get().data(),
                                        m_cuGrid.get().get().data(),
                                        POS_RES,
                                        HDG_RES,
                                        m_parkingPlannerParams.turnRadius_m,
                                        m_cuStream);)
    TIME_PRINT("sweepArc2", sweepArc2(m_cuGridWarped.get().get().data(),
                                      ManeuverType::RIGHT,
                                      iter,
                                      m_cuStream);)
    TIME_PRINT("unwarpRight", warpRight(m_cuGrid.get().get().data(),
                                        m_cuGridWarped.get().get().data(),
                                        POS_RES,
                                        HDG_RES,
                                        m_parkingPlannerParams.turnRadius_m,
                                        m_cuStream,
                                        true);)
}

bool ParkingPlanner2::backtrace()
{
    bool found{};
    m_vertices.clear();
    m_maneuverList.clear();
    m_segmentDirs.clear();
    if (isDestinationReached())
    {
        Coord3d current = m_destination;
        bool brk{false};
        auto loopbody = [&]() -> bool {
            m_vertices.push_back_maybe(current.getPose(POS_RES, HDG_RES));
            const GridCell& cell = getCell(m_grid.get(), current);
            m_maneuverList.push_back(cell.maneuver);
            m_segmentDirs.push_back(cell.reverse);

            if (!cell.reachable)
            {
                return true;
            }
            printf("Current:%d, %d, %d\n", current.x, current.y, current.hdg);
            current = cell.prevPose;
            return false;
        };
        brk = loopbody();
        while (current != m_start && m_vertices.available() > 1 && !brk) // need at least two spots
        {
            brk = loopbody();
            if (brk)
            {
                break;
            }
        }

        if (current == m_start && !m_vertices.full())
        {
            found = true;
            // push back start
            m_vertices.push_back(current.getPose(POS_RES, HDG_RES));
            std::reverse(m_vertices.begin(), m_vertices.end()); //reorders the path from start to destination
            std::reverse(m_maneuverList.begin(), m_maneuverList.end());
            std::reverse(m_segmentDirs.begin(), m_segmentDirs.end());
        }
        else
        {
            found = false;
            m_vertices.clear();
            m_maneuverList.clear();
            m_segmentDirs.clear();
        }
    }
    else
    {
        found = false;
    }

    return found;
}

bool ParkingPlanner2::backtrace2()
{
    bool found{};
    m_vertices.clear();
    m_maneuverList.clear();
    m_segmentDirs.clear();
    if (isDestinationReached())
    {
        Coord3d current  = m_destination;
        uint8_t itersNum = getCell(m_grid.get(), current).iterCount;
        bool alive       = true;
        for (int16_t iter = static_cast<int16_t>(itersNum); iter > 0; iter--)
        {
            if (!alive)
            {
                printf("Breaking because alive failed at iter:%d\n", iter);
                break;
            }
            alive = false;
            printf("\n\ncurrent: %d,%d,%d\n", current.x, current.y, current.hdg);
            const GridCell& cell = getCell(m_grid.get(), current);
            printf("src: %d", cell.src);
            m_vertices.push_back_maybe(current.getPose(POS_RES, HDG_RES));
            m_maneuverList.push_back(cell.maneuver);
            m_segmentDirs.push_back(cell.reverse);

            switch (cell.maneuver)
            {
            case ManeuverType::STRAIGHT:
            {
                printf("Straight\n");
                Coord3d warped = unwarpStraight(current, getPosRes(), getHdgRes());
                Coord3d search = warped;
                if (!cell.reverse)
                {
                    printf("Forward\n");
                    for (int16_t i = warped.x - 1; i >= -(getSize() << 2); i--)
                    {
                        search.x         = i;
                        Coord3d unwarped = warpStraight(search, getPosRes(), getHdgRes());
                        if (!withinBounds(unwarped))
                        {
                            continue;
                        }
                        GridCell& tempCell = getCell(m_grid.get(), unwarped);
                        if (cell.src == unwarped.x)
                        {
                            printf("Checking: %d,%d,%d    iter:%hhu\n", unwarped.x, unwarped.y, unwarped.hdg, tempCell.iterCount);
                        }
                        if (tempCell.iterCount == iter - 1)
                        {
                            current = unwarped;
                            alive   = true;
                            break;
                        }
                    }
                }
                else
                {
                    printf("Reverse\n");
                    for (int16_t i = warped.x + 1; i < (getSize() << 2); i++)
                    {
                        search.x         = i;
                        Coord3d unwarped = warpStraight(search, getPosRes(), getHdgRes());
                        if (!withinBounds(unwarped))
                        {
                            continue;
                        }
                        GridCell& tempCell = getCell(m_grid.get(), unwarped);
                        if (cell.src == unwarped.x)
                        {
                            printf("Checking: %d,%d,%d    iter:%hhu\n", unwarped.x, unwarped.y, unwarped.hdg, tempCell.iterCount);
                        }
                        if (tempCell.iterCount == iter - 1)
                        {
                            current = unwarped;
                            alive   = true;
                            break;
                        }
                    }
                }
                break;
            }
            case ManeuverType::LEFT:
            {
                printf("Left\n");
                Coord3d warped = unwarpLeft(current, getPosRes(), getHdgRes(), m_parkingPlannerParams.turnRadius_m);
                Coord3d search = warped;
                if (!cell.reverse)
                {
                    printf("forward\n");
                    for (int16_t i = warped.hdg - 1; i >= 0; i--)
                    {
                        search.hdg         = i;
                        Coord3d unwarped   = warpLeft(search, getPosRes(), getHdgRes(), m_parkingPlannerParams.turnRadius_m);
                        GridCell& tempCell = getCell(m_grid.get(), unwarped);
                        if (cell.src == unwarped.hdg)
                        {
                            printf("Checking: %d,%d,%d    iter:%hhu\n", unwarped.x, unwarped.y, unwarped.hdg, tempCell.iterCount);
                        }
                        if (tempCell.iterCount == iter - 1)
                        {
                            current = unwarped;
                            alive   = true;
                            break;
                        }
                    }
                    if (!alive)
                    {
                        printf("continues\n");
                        for (int16_t i = getThetaStep() - 1; i > warped.hdg; i--)
                        {
                            search.hdg         = i;
                            Coord3d unwarped   = warpLeft(search, getPosRes(), getHdgRes(), m_parkingPlannerParams.turnRadius_m);
                            GridCell& tempCell = getCell(m_grid.get(), unwarped);
                            if (cell.src == unwarped.hdg)
                            {
                                printf("Checking: %d,%d,%d    iter:%hhu\n", unwarped.x, unwarped.y, unwarped.hdg, tempCell.iterCount);
                            }
                            if (tempCell.iterCount == iter - 1)
                            {
                                current = unwarped;
                                alive   = true;
                                break;
                            }
                        }
                    }
                }
                else
                {
                    printf("reverse\n");
                    for (uint16_t i = warped.hdg + 1; i < getThetaStep(); i++)
                    {
                        search.hdg         = i;
                        Coord3d unwarped   = warpLeft(search, getPosRes(), getHdgRes(), m_parkingPlannerParams.turnRadius_m);
                        GridCell& tempCell = getCell(m_grid.get(), unwarped);
                        if (cell.src == unwarped.hdg)
                        {
                            printf("Checking: %d,%d,%d    iter:%hhu\n", unwarped.x, unwarped.y, unwarped.hdg, tempCell.iterCount);
                        }
                        if (tempCell.iterCount == iter - 1)
                        {
                            current = unwarped;
                            alive   = true;
                            break;
                        }
                    }
                    if (!alive)
                    {
                        printf("continuing\n");
                        for (int16_t i = 0; i < warped.hdg; i++)
                        {
                            search.hdg         = i;
                            Coord3d unwarped   = warpLeft(search, getPosRes(), getHdgRes(), m_parkingPlannerParams.turnRadius_m);
                            GridCell& tempCell = getCell(m_grid.get(), unwarped);
                            if (cell.src == unwarped.hdg)
                            {
                                printf("Checking: %d,%d,%d    iter:%hhu\n", unwarped.x, unwarped.y, unwarped.hdg, tempCell.iterCount);
                            }
                            if (tempCell.iterCount == iter - 1)
                            {
                                current = unwarped;
                                alive   = true;
                                break;
                            }
                        }
                    }
                }
                break;
            }
            case ManeuverType::RIGHT:
            {
                printf("right\n");
                Coord3d warped = unwarpRight(current, getPosRes(), getHdgRes(), m_parkingPlannerParams.turnRadius_m);
                Coord3d search = warped;
                if (!cell.reverse)
                {
                    printf("forward\n");
                    for (uint16_t i = warped.hdg + 1; i < getThetaStep(); i++)
                    {
                        search.hdg         = i;
                        Coord3d unwarped   = warpRight(search, getPosRes(), getHdgRes(), m_parkingPlannerParams.turnRadius_m);
                        GridCell& tempCell = getCell(m_grid.get(), unwarped);
                        if (cell.src == unwarped.hdg)
                        {
                            printf("Checking: %d,%d,%d    iter:%hhu\n", unwarped.x, unwarped.y, unwarped.hdg, tempCell.iterCount);
                        }
                        if (tempCell.iterCount == iter - 1)
                        {
                            current = unwarped;
                            alive   = true;
                            break;
                        }
                    }
                    if (!alive)
                    {
                        printf("Continuing\n");
                        for (int16_t i = 0; i < warped.hdg; i++)
                        {
                            search.hdg         = i;
                            Coord3d unwarped   = warpRight(search, getPosRes(), getHdgRes(), m_parkingPlannerParams.turnRadius_m);
                            GridCell& tempCell = getCell(m_grid.get(), unwarped);
                            if (cell.src == unwarped.hdg)
                            {
                                printf("Checking: %d,%d,%d    iter:%hhu\n", unwarped.x, unwarped.y, unwarped.hdg, tempCell.iterCount);
                            }
                            if (tempCell.iterCount == iter - 1)
                            {
                                current = unwarped;
                                alive   = true;
                                break;
                            }
                        }
                    }
                }
                else
                {
                    printf("reverse\n");
                    for (int16_t i = warped.hdg - 1; i >= 0; i--)
                    {
                        search.hdg         = i;
                        Coord3d unwarped   = warpRight(search, getPosRes(), getHdgRes(), m_parkingPlannerParams.turnRadius_m);
                        GridCell& tempCell = getCell(m_grid.get(), unwarped);
                        if (cell.src == unwarped.hdg)
                        {
                            printf("Checking: %d,%d,%d    iter:%hhu\n", unwarped.x, unwarped.y, unwarped.hdg, tempCell.iterCount);
                        }
                        if (tempCell.iterCount == iter - 1)
                        {
                            current = unwarped;
                            alive   = true;
                            break;
                        }
                    }
                    if (!alive)
                    {
                        printf("Continuing\n");
                        for (int16_t i = getThetaStep() - 1; i > warped.hdg; i--)
                        {
                            search.hdg         = i;
                            Coord3d unwarped   = warpRight(search, getPosRes(), getHdgRes(), m_parkingPlannerParams.turnRadius_m);
                            GridCell& tempCell = getCell(m_grid.get(), unwarped);
                            if (cell.src == unwarped.hdg)
                            {
                                printf("Checking: %d,%d,%d    iter:%hhu\n", unwarped.x, unwarped.y, unwarped.hdg, tempCell.iterCount);
                            }
                            if (tempCell.iterCount == iter - 1)
                            {
                                current = unwarped;
                                alive   = true;
                                break;
                            }
                        }
                    }
                }
                break;
            }
            default:
                break;
            }
        }
        if (current == m_start && !m_vertices.full())
        {
            found = true;
            // push back start
            m_vertices.push_back(current.getPose(POS_RES, HDG_RES));
            std::reverse(m_vertices.begin(), m_vertices.end()); //reorders the path from start to destination
            std::reverse(m_maneuverList.begin(), m_maneuverList.end());
            std::reverse(m_segmentDirs.begin(), m_segmentDirs.end());
        }
        else
        {
            printf("Destination reached but path not traced.\n");
            found = false;
            m_vertices.clear();
            m_maneuverList.clear();
            m_segmentDirs.clear();
        }
    }
    else
    {
        printf("Destination not reached.\n");
        found = false;
    }

    return found;
}

void ParkingPlanner2::buildPath()
{
    m_path.clear();
    m_pathDrivingDirs.clear();
    if (backtrace()) //if backtrace evaluates to true m_vertices, m_maneuverList and m_segmentDirs cannot be empty.
    {
        VectorFixed<Vector3f>::iterator itPath         = m_vertices.begin();
        VectorFixed<ManeuverType>::iterator itManeuver = m_maneuverList.begin();
        VectorFixed<bool>::iterator itDir              = m_segmentDirs.begin();
        while (itPath + 1 != m_vertices.end() && itManeuver != m_maneuverList.end())
        {
            //generating waypoints between every two points based on maneuver
            switch (*itManeuver)
            {
            case ManeuverType::STRAIGHT:
                generateStraight(*itPath, *(itPath + 1), *itDir);
                break;
            case ManeuverType::LEFT:
                generateArc(*itPath,
                            *(itPath + 1),
                            *itDir,
                            true,
                            static_cast<Vector3f (*)(const Vector3f&, float32_t)>(warpLeft),
                            static_cast<Vector3f (*)(const Vector3f&, float32_t)>(unwarpLeft));
                break;
            case ManeuverType::RIGHT:
                generateArc(*itPath,
                            *(itPath + 1),
                            *itDir,
                            false,
                            static_cast<Vector3f (*)(const Vector3f&, float32_t)>(warpRight),
                            static_cast<Vector3f (*)(const Vector3f&, float32_t)>(unwarpRight));
                break;
            default:
                break;
            }
            itPath++;
            itManeuver++;
            itDir++;
        }
        //Adding the endpoint
        Vector3f endTail              = *itPath;
        dwPathPlannerDrivingState dir = *(itDir - 1) ? DW_PATH_BACKWARD : DW_PATH_FORWARD;
        m_path.push_back(endTail);
        m_pathDrivingDirs.push_back(dir);
    }
}
void ParkingPlanner2::buildPath2()
{
    m_path.clear();
    m_pathDrivingDirs.clear();
    if (backtrace2()) //if backtrace evaluates to true m_vertices, m_maneuverList and m_segmentDirs cannot be empty.
    {
        VectorFixed<Vector3f>::iterator itPath         = m_vertices.begin();
        VectorFixed<ManeuverType>::iterator itManeuver = m_maneuverList.begin();
        VectorFixed<bool>::iterator itDir              = m_segmentDirs.begin();
        while (itPath + 1 != m_vertices.end() && itManeuver != m_maneuverList.end())
        {
            //generating waypoints between every two points based on maneuver
            switch (*itManeuver)
            {
            case ManeuverType::STRAIGHT:
                generateStraight(*itPath, *(itPath + 1), *itDir);
                break;
            case ManeuverType::LEFT:
                generateArc(*itPath,
                            *(itPath + 1),
                            *itDir,
                            true,
                            static_cast<Vector3f (*)(const Vector3f&, float32_t)>(warpLeft),
                            static_cast<Vector3f (*)(const Vector3f&, float32_t)>(unwarpLeft));
                break;
            case ManeuverType::RIGHT:
                generateArc(*itPath,
                            *(itPath + 1),
                            *itDir,
                            false,
                            static_cast<Vector3f (*)(const Vector3f&, float32_t)>(warpRight),
                            static_cast<Vector3f (*)(const Vector3f&, float32_t)>(unwarpRight));
                break;
            default:
                break;
            }
            itPath++;
            itManeuver++;
            itDir++;
        }
        //Adding the endpoint
        Vector3f endTail              = *itPath;
        dwPathPlannerDrivingState dir = *(itDir - 1) ? DW_PATH_BACKWARD : DW_PATH_FORWARD;
        m_path.push_back(endTail);
        m_pathDrivingDirs.push_back(dir);
    }
}

void ParkingPlanner2::generateStraight(const Vector3f& head, const Vector3f& tail, const bool reverse)
{
    float32_t stepLen             = 0.2f;
    dwPathPlannerDrivingState dir = reverse ? DW_PATH_BACKWARD : DW_PATH_FORWARD;

    m_path.push_back(head);
    m_pathDrivingDirs.push_back(dir);
    float32_t stepsfloat{};
    int32_t stepsint{};
    Vector2f head2Tail{};
    head2Tail.x() = tail.x() - head.x();
    head2Tail.y() = tail.y() - head.y();
    stepsfloat    = floor(head2Tail.norm() / stepLen); //number of waypoints calculated using steplen and vector length
    stepsint      = static_cast<int>(stepsfloat);
    for (int32_t i = 1; i < stepsint; i++)
    {
        Vector3f temp;
        temp.x() = head.x() + static_cast<float32_t>(i) * head2Tail.x() / stepsfloat;
        temp.y() = head.y() + static_cast<float32_t>(i) * head2Tail.y() / stepsfloat;
        temp.z() = tail.z();
        m_path.push_back(temp);
        m_pathDrivingDirs.push_back(dir);
    }
}
void ParkingPlanner2::generateArc(const Vector3f& head,
                                  const Vector3f& tail,
                                  const bool reverse,
                                  const bool left,
                                  std::function<Vector3f(const Vector3f&, float32_t)> warpFunc,
                                  std::function<Vector3f(const Vector3f&, float32_t)> unwarpFunc)
{
    float32_t degStep             = 1.0f; //Minimum possible step size is 1. For smaller sizes the idea has to changed.
    dwPathPlannerDrivingState dir = reverse ? DW_PATH_BACKWARD : DW_PATH_FORWARD;
    Vector3f warped               = unwarpFunc(head, m_parkingPlannerParams.turnRadius_m); //actually warps the pose

    if ((!reverse && left) || (reverse && !left)) //left: theta increase while moving forward
    {                                             //right: theta decrease while moving forward
        if (head.z() < tail.z())
        {
            while (warped.z() < tail.z())
            {
                Vector3f temp = warpFunc(warped, m_parkingPlannerParams.turnRadius_m);
                m_path.push_back(temp);
                m_pathDrivingDirs.push_back(dir);
                warped.z() = warped.z() + degStep;
            }
        }
        else // i.e. head.z() > tail.z(). We have to cross 360 and maintain circularity
        {
            while (warped.z() < 360.0f)
            {
                Vector3f temp = warpFunc(warped, m_parkingPlannerParams.turnRadius_m);
                m_path.push_back(temp);
                m_pathDrivingDirs.push_back(dir);
                warped.z() = warped.z() + degStep;
            }

            warped.z() = 0.0f;

            while (warped.z() < tail.z())
            {
                Vector3f temp = warpFunc(warped, m_parkingPlannerParams.turnRadius_m);
                m_path.push_back(temp);
                m_pathDrivingDirs.push_back(dir);
                warped.z() = warped.z() + degStep;
            }
        }
    }
    else
    {
        if (head.z() > tail.z())
        {
            while (warped.z() > tail.z())
            {
                Vector3f temp = warpFunc(warped, m_parkingPlannerParams.turnRadius_m);
                m_path.push_back(temp);
                m_pathDrivingDirs.push_back(dir);
                warped.z() = warped.z() - degStep;
            }
        }
        else // i.e. head.z() < tail.z(). We have to cross 0 and maintain circularity
        {
            while (warped.z() > 0.0f)
            {
                Vector3f temp = warpFunc(warped, m_parkingPlannerParams.turnRadius_m);
                m_path.push_back(temp);
                m_pathDrivingDirs.push_back(dir);
                warped.z() = warped.z() - degStep;
            }

            warped.z() = 359.0f;

            while (warped.z() > tail.z())
            {
                Vector3f temp = warpFunc(warped, m_parkingPlannerParams.turnRadius_m);
                m_path.push_back(temp);
                m_pathDrivingDirs.push_back(dir);
                warped.z() = warped.z() - degStep;
            }
        }
    }
}

core::span<const Vector3f> ParkingPlanner2::getPathSegment() const
{
    uint32_t lastIdx = 0;

    if (m_pathDrivingDirs.empty())
    {
        return make_span(m_path.data(), 0); //returning empty span
    }
    for (auto it = m_pathDrivingDirs.begin(); (it + 1) != m_pathDrivingDirs.end(); ++it)
    {
        lastIdx++;
        if (*it != *(it + 1))
        {
            break;
        }
    }
    return make_span(m_path.data(), lastIdx + 1);
}

void ParkingPlanner2::copyGridHostToDevice()
{
    TIME_PRINT("H2D", cuda::memcpyAsync(m_cuGrid.get().data(),
                                        m_grid.get().data(),
                                        m_cuGrid.get().size(),
                                        m_cuStream);)
}

void ParkingPlanner2::copyGridNewHostToDevice()
{
    TIME_PRINT("H2D", cuda::memcpyAsync(m_cuGridNew.get().data(),
                                        m_gridNew.get().data(),
                                        m_cuGridNew.get().size(),
                                        m_cuStream);)
}

void ParkingPlanner2::copyGrid16HostToDevice()
{
    TIME_PRINT("H2D", cuda::memcpyAsync(m_cuGrid16.get().data(),
                                        m_grid16.get().data(),
                                        m_cuGrid16.get().size(),
                                        m_cuStream);)
}

void ParkingPlanner2::copyGrid32HostToDevice()
{
    TIME_PRINT("H2D", cuda::memcpyAsync(m_cuGrid32.get().data(),
                                        m_grid32.get().data(),
                                        m_cuGrid32.get().size(),
                                        m_cuStream);)
}

void ParkingPlanner2::copyGridDeviceToHost()
{
    TIME_PRINT("\nD2H", cuda::memcpyAsync(m_grid.get().data(),
                                          m_cuGrid.get().data(),
                                          m_cuGrid.get().size(),
                                          m_cuStream);)
}
void ParkingPlanner2::copyGridNewDeviceToHost()
{
    TIME_PRINT("D2H", cuda::memcpyAsync(m_gridNew.get().data(),
                                        m_cuGridNew.get().data(),
                                        m_cuGridNew.get().size(),
                                        m_cuStream);)
}

void ParkingPlanner2::copyGrid16DeviceToHost()
{
    TIME_PRINT("D2H", cuda::memcpyAsync(m_grid16.get().data(),
                                        m_cuGrid16.get().data(),
                                        m_cuGrid16.get().size(),
                                        m_cuStream);)
}

void ParkingPlanner2::copyGrid32DeviceToHost()
{
    TIME_PRINT("D2H", cuda::memcpyAsync(m_grid32.get().data(),
                                        m_cuGrid32.get().data(),
                                        m_cuGrid32.get().size(),
                                        m_cuStream);)
}
//TODO(yizhouw) This is not a very efficient way to implement this. Optimize in the next MR.
bool ParkingPlanner2::isDestinationReachedGPU()
{
    copyGridDeviceToHost();
    return isDestinationReached();
}

void ParkingPlanner2::setStartGPU()
{
    GridCell origin{};
    origin.reachable = true;
    origin.prevPose  = INVALID_POSE;

    DW_CHECK_CUDA_ERROR(cudaMemcpyAsync(m_cuGrid.get().data().get() + getCell(0, 0, 0),
                                        &origin,
                                        sizeof(GridCell),
                                        cudaMemcpyHostToDevice,
                                        m_cuStream));
}

} // namespace planner
} // namespace dw
