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

#include <gtest/gtest.h>
#include <dw/experimental/parkingplanner/ParkingPlanner2.hpp>
#include <dw/experimental/parkingplanner/ParkingPlanner.h>

#ifndef PRINT_TIMING
#define PRINT_TIMING

constexpr bool DEBUG_PRINT_RESULTS = true;
// clang-format off
#define TIME_PRINT(name, a) \
    if (DEBUG_PRINT_RESULTS) { \
        auto start = std::chrono::high_resolution_clock::now(); \
        a; \
        auto elapsed = std::chrono::high_resolution_clock::now() - start; \
        std::cout << name \
                  << ": " \
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count() \
                     / 1000000.0f << " ms" << std::endl; \
    } else { \
        a; \
    }
// clang-format on
#endif

namespace dw
{
namespace planner
{

TEST(ParkingPlanner2Test, CAPIThreePointTurn_L0)
{
    dwParkingPlannerHandle_t parkingPlannerHandle;

    dwParkingPlannerParams params{};
    dwParkingPlanner_initDefaultParams(&params);

    dwContextHandle_t contextHandle{DW_NULL_HANDLE};
    dwVersion version;
    dwGetVersion(&version);
    dwContextParameters sdkParams = {};
    dwInitialize(&contextHandle, version, &sdkParams);

    dwParkingPlanner_initialize(&params, contextHandle, &parkingPlannerHandle);

    dwUnstructuredTarget tarDest{};
    tarDest.position.x = 0.0f;
    tarDest.position.y = 0.0f;
    tarDest.heading    = math::deg2Rad(180.0f);
    dwParkingPlanner_setTarget(tarDest, parkingPlannerHandle);

    dwParkingPlanner_computePath(parkingPlannerHandle);

    std::unique_ptr<dwParkingState[]> path(new dwParkingState[5000]);
    std::unique_ptr<dwPathPlannerDrivingState[]> dir(new dwPathPlannerDrivingState[5000]);
    size_t size{};
    size_t capacity{5000};
    dwParkingPlanner_getPathDetails(path.get(), dir.get(), &size, capacity, parkingPlannerHandle);

    //EXPECT_EQ(size,231);
    bool xcomp = (abs(path[230].x - 0.0f) < 0.00001f);
    bool ycomp = (abs(path[230].y - 0.0f) < 0.00001f);
    bool zcomp = (abs(path[230].heading - 180.0f) < 0.00001f);
    EXPECT_TRUE(xcomp && ycomp && zcomp);

    xcomp = (abs(path[138].x + 5.0f) < 0.00001f);
    ycomp = (abs(path[138].y - 5.0f) < 0.00001f);
    zcomp = (abs(path[138].heading - 88.0f) < 0.00001f);
    EXPECT_TRUE(xcomp && ycomp && zcomp);

    xcomp = (abs(path[50].x + 10.0f) < 0.00001f);
    ycomp = (abs(path[50].y - 0.0f) < 0.00001f);
    zcomp = (abs(path[50].heading - 0.0f) < 0.00001f);
    EXPECT_TRUE(xcomp && ycomp && zcomp);

    dwParkingPlanner_release(parkingPlannerHandle);
}

TEST(ParkingPlanner2Test, CAPIObstacles_L0)
{
    dwParkingPlannerHandle_t parkingPlannerHandle;

    dwParkingPlannerParams params{};
    dwParkingPlanner_initDefaultParams(&params);

    dwContextHandle_t contextHandle{DW_NULL_HANDLE};
    dwVersion version;
    dwGetVersion(&version);
    dwContextParameters sdkParams = {};
    dwInitialize(&contextHandle, version, &sdkParams);

    dwParkingPlanner_initialize(&params, contextHandle, &parkingPlannerHandle);

    dwUnstructuredTarget tarDest{};
    tarDest.position.x = 8.0f;
    tarDest.position.y = 0.0f;
    tarDest.heading    = math::deg2Rad(0.0f);
    dwParkingPlanner_setTarget(tarDest, parkingPlannerHandle);

    StaticVectorFixed<dwObstacle, 2> obstacles{};
    dwObstacle obs;
    obs.position.x = 4.0f;
    obs.position.y = 0.0f;
    obstacles.push_back(obs);

    obs.position.x = 4.0f;
    obs.position.y = 5.0f;
    obstacles.push_back(obs);
    dwParkingPlanner_setObstacles(&obstacles[0], obstacles.size(), parkingPlannerHandle);

    dwParkingPlanner_computePath(parkingPlannerHandle);

    std::unique_ptr<dwParkingState[]> path(new dwParkingState[5000]);
    std::unique_ptr<dwPathPlannerDrivingState[]> dir(new dwPathPlannerDrivingState[5000]);
    size_t size{};
    size_t capacity{5000};
    dwParkingPlanner_getPathDetails(path.get(), dir.get(), &size, capacity, parkingPlannerHandle);

    //EXPECT_EQ(size,418);
    bool xcomp = (abs(path[417].x - 8.0f) < 0.00001f);
    bool ycomp = (abs(path[417].y - 0.0f) < 0.00001f);
    bool zcomp = (abs(path[417].heading - 0.0f) < 0.00001f);
    EXPECT_TRUE(xcomp && ycomp && zcomp);

    xcomp = (abs(path[238].x - 8.0f) < 0.00001f);
    ycomp = (abs(path[238].y - 10.0f) < 0.00001f);
    zcomp = (abs(path[238].heading - 179.0f) < 0.00001f);
    EXPECT_TRUE(xcomp && ycomp && zcomp);

    xcomp = (abs(path[196].x + 0.5f) < 0.00001f);
    ycomp = (abs(path[196].y - 10.0f) < 0.00001f);
    zcomp = (abs(path[196].heading - 179.0f) < 0.00001f);
    EXPECT_TRUE(xcomp && ycomp && zcomp);

    xcomp = (abs(path[179].x + 2.0f) < 0.00001f);
    ycomp = (abs(path[179].y - 10.0f) < 0.00001f);
    zcomp = (abs(path[179].heading - 162.0f) < 0.00001f);
    EXPECT_TRUE(xcomp && ycomp && zcomp);

    xcomp = (abs(path[17].x + 3.5f) < 0.00001f);
    ycomp = (abs(path[17].y - 0.0f) < 0.00001f);
    zcomp = (abs(path[17].heading - 0.0f) < 0.00001f);
    EXPECT_TRUE(xcomp && ycomp && zcomp);

    dwParkingPlanner_release(parkingPlannerHandle);
}

TEST(ParkingPlanner2Test, CAPIBay_L0)
{
    dwParkingPlannerHandle_t parkingPlannerHandle;

    dwParkingPlannerParams params{};
    dwParkingPlanner_initDefaultParams(&params);

    dwContextHandle_t contextHandle{DW_NULL_HANDLE};
    dwVersion version;
    dwGetVersion(&version);
    dwContextParameters sdkParams = {};
    dwInitialize(&contextHandle, version, &sdkParams);

    dwParkingPlanner_initialize(&params, contextHandle, &parkingPlannerHandle);

    dwUnstructuredTarget tarDest{};
    tarDest.position.x = 5.0f;
    tarDest.position.y = 5.0f;
    tarDest.heading    = math::deg2Rad(0.0f);
    dwParkingPlanner_setTarget(tarDest, parkingPlannerHandle);

    StaticVectorFixed<dwObstacle, 7> obstacles{};
    dwObstacle obs;
    obs.position.x = 5.0f;
    obs.position.y = 1.0f;
    obstacles.push_back(obs);

    obs.position.x = 1.0f;
    obs.position.y = 5.0f;
    obstacles.push_back(obs);

    obs.position.x = 5.0f;
    obs.position.y = 9.0f;
    obstacles.push_back(obs);

    obs.position.x = 3.0f;
    obs.position.y = 3.0f;
    obstacles.push_back(obs);

    obs.position.x = 3.0f;
    obs.position.y = 7.0f;
    obstacles.push_back(obs);

    obs.position.x = 7.0f;
    obs.position.y = 7.5f;
    obstacles.push_back(obs);

    obs.position.x = 7.0f;
    obs.position.y = 2.5f;
    obstacles.push_back(obs);
    dwParkingPlanner_setObstacles(&obstacles[0], obstacles.size(), parkingPlannerHandle);

    dwParkingPlanner_computePath(parkingPlannerHandle);

    std::unique_ptr<dwParkingState[]> path(new dwParkingState[5000]);
    std::unique_ptr<dwPathPlannerDrivingState[]> dir(new dwPathPlannerDrivingState[5000]);
    size_t size{};
    size_t capacity{5000};
    dwParkingPlanner_getPathDetails(path.get(), dir.get(), &size, capacity, parkingPlannerHandle);

    bool xcomp = (abs(path[748].x - 5.0f) < 0.00001f);
    bool ycomp = (abs(path[748].y - 5.0f) < 0.00001f);
    bool zcomp = (abs(path[748].heading - 0.0f) < 0.00001f);
    EXPECT_TRUE(xcomp && ycomp && zcomp);

    xcomp = (abs(path[717].x - 7.5f) < 0.00001f);
    ycomp = (abs(path[717].y - 4.5f) < 0.00001f);
    zcomp = (abs(path[717].heading - 329.0f) < 0.00001f);
    EXPECT_TRUE(xcomp && ycomp && zcomp);

    xcomp = (abs(path[501].x - 9.5f) < 0.00001f);
    ycomp = (abs(path[501].y - 14.0f) < 0.00001f);
    zcomp = (abs(path[501].heading - 186.0f) < 0.00001f);
    EXPECT_TRUE(xcomp && ycomp && zcomp);

    xcomp = (abs(path[318].x + 27.0f) < 0.00001f);
    ycomp = (abs(path[318].y - 10.0f) < 0.00001f);
    zcomp = (abs(path[318].heading - 186.0f) < 0.00001f);
    EXPECT_TRUE(xcomp && ycomp && zcomp);

    xcomp = (abs(path[132].x + 26.5f) < 0.00001f);
    ycomp = (abs(path[132].y - 0.0f) < 0.00001f);
    zcomp = (abs(path[132].heading - 0.0f) < 0.00001f);
    EXPECT_TRUE(xcomp && ycomp && zcomp);

    dwParkingPlanner_release(parkingPlannerHandle);
}

TEST(ParkingPlanner2Test, CAPIHurdle_L0)
{
    dwParkingPlannerHandle_t parkingPlannerHandle;

    dwParkingPlannerParams params{};
    dwParkingPlanner_initDefaultParams(&params);

    dwContextHandle_t contextHandle{DW_NULL_HANDLE};
    dwVersion version;
    dwGetVersion(&version);
    dwContextParameters sdkParams = {};
    dwInitialize(&contextHandle, version, &sdkParams);

    dwParkingPlanner_initialize(&params, contextHandle, &parkingPlannerHandle);

    dwUnstructuredTarget tarDest{};
    tarDest.position.x = 10.0f;
    tarDest.position.y = 0.0f;
    tarDest.heading    = math::deg2Rad(180.0f);
    dwParkingPlanner_setTarget(tarDest, parkingPlannerHandle);

    StaticVectorFixed<dwObstacle, 3> obstacles{};
    dwObstacle obs;
    obs.position.x = 5.0f;
    obs.position.y = -3.5f;
    obstacles.push_back(obs);

    obs.position.x = 5.0f;
    obs.position.y = 0.0f;
    obstacles.push_back(obs);

    obs.position.x = 5.0f;
    obs.position.y = 3.5f;
    obstacles.push_back(obs);

    dwParkingPlanner_setObstacles(&obstacles[0], obstacles.size(), parkingPlannerHandle);

    dwParkingPlanner_computePath(parkingPlannerHandle);

    std::unique_ptr<dwParkingState[]> path(new dwParkingState[5000]);
    std::unique_ptr<dwPathPlannerDrivingState[]> dir(new dwPathPlannerDrivingState[5000]);
    size_t size{};
    size_t capacity{5000};
    dwParkingPlanner_getPathDetails(path.get(), dir.get(), &size, capacity, parkingPlannerHandle);
    for (size_t i = 0; i < size; ++i)
    {
        printf("%lu: Path point: %f, %f, %f  \tDirection:%d\n", i, path[i].x, path[i].y, path[i].heading, dir[i]);
    }

    bool xcomp = (abs(path[379].x - 10.0f) < 0.00001f);
    bool ycomp = (abs(path[379].y - 0.0f) < 0.00001f);
    bool zcomp = (abs(path[379].heading - 180.0f) < 0.00001f);
    EXPECT_TRUE(xcomp && ycomp && zcomp);

    xcomp = (abs(path[367].x - 7.5f) < 0.00001f);
    ycomp = (abs(path[367].y - 0.0f) < 0.00001f);
    zcomp = (abs(path[367].heading - 180.0f) < 0.00001f);
    EXPECT_TRUE(xcomp && ycomp && zcomp);

    xcomp = (abs(path[100].x - 2.5f) < 0.00001f);
    ycomp = (abs(path[100].y - 5.0f) < 0.00001f);
    zcomp = (abs(path[100].heading - 88.0f) < 0.00001f);
    EXPECT_TRUE(xcomp && ycomp && zcomp);

    xcomp = (abs(path[12].x + 2.5f) < 0.00001f);
    ycomp = (abs(path[12].y - 0.0f) < 0.00001f);
    zcomp = (abs(path[12].heading - 0.0f) < 0.00001f);
    EXPECT_TRUE(xcomp && ycomp && zcomp);

    dwParkingPlanner_release(parkingPlannerHandle);
}

TEST(ParkingPlanner2Test, CAPIParallel_L0)
{
    dwParkingPlannerHandle_t parkingPlannerHandle;

    dwParkingPlannerParams params{};
    dwParkingPlanner_initDefaultParams(&params);

    dwContextHandle_t contextHandle{DW_NULL_HANDLE};
    dwVersion version;
    dwGetVersion(&version);
    dwContextParameters sdkParams = {};
    dwInitialize(&contextHandle, version, &sdkParams);

    dwParkingPlanner_initialize(&params, contextHandle, &parkingPlannerHandle);

    dwUnstructuredTarget tarDest{};
    tarDest.position.x = 8.5f;
    tarDest.position.y = 0.0f;
    tarDest.heading    = math::deg2Rad(0.0f);
    dwParkingPlanner_setTarget(tarDest, parkingPlannerHandle);

    StaticVectorFixed<dwObstacle, 6> obstacles{};
    dwObstacle obs;
    obs.position.x = 5.0f;
    obs.position.y = -3.5f;
    obstacles.push_back(obs);

    obs.position.x = 5.0f;
    obs.position.y = 0.0f;
    obstacles.push_back(obs);

    obs.position.x = 5.0f;
    obs.position.y = 3.5f;
    obstacles.push_back(obs);

    obs.position.x = 12.5f;
    obs.position.y = -3.5f;
    obstacles.push_back(obs);

    obs.position.x = 12.5f;
    obs.position.y = 0.0f;
    obstacles.push_back(obs);

    obs.position.x = 12.5f;
    obs.position.y = 3.5f;
    obstacles.push_back(obs);
    dwParkingPlanner_setObstacles(&obstacles[0], obstacles.size(), parkingPlannerHandle);

    dwParkingPlanner_computePath(parkingPlannerHandle);

    std::unique_ptr<dwParkingState[]> path(new dwParkingState[5000]);
    std::unique_ptr<dwPathPlannerDrivingState[]> dir(new dwPathPlannerDrivingState[5000]);
    size_t size{};
    size_t capacity{5000};
    dwParkingPlanner_getPathDetails(path.get(), dir.get(), &size, capacity, parkingPlannerHandle);

    bool xcomp = (abs(path[362].x - 8.5f) < 0.00001f);
    bool ycomp = (abs(path[362].y - 0.0f) < 0.00001f);
    bool zcomp = (abs(path[362].heading - 0.0f) < 0.00001f);
    EXPECT_TRUE(xcomp && ycomp && zcomp);

    xcomp = (abs(path[342].x - 7.0f) < 0.00001f);
    ycomp = (abs(path[342].y - 0.5f) < 0.00001f);
    zcomp = (abs(path[342].heading - 340.0f) < 0.00001f);
    EXPECT_TRUE(xcomp && ycomp && zcomp);

    xcomp = (abs(path[29].x - 3.0f) < 0.00001f);
    ycomp = (abs(path[29].y - 0.5f) < 0.00001f);
    zcomp = (abs(path[29].heading - 27.0f) < 0.00001f);
    EXPECT_TRUE(xcomp && ycomp && zcomp);

    xcomp = (abs(path[2].x - 0.5f) < 0.00001f);
    ycomp = (abs(path[2].y - 0.0f) < 0.00001f);
    zcomp = (abs(path[2].heading - 0.0f) < 0.00001f);
    EXPECT_TRUE(xcomp && ycomp && zcomp);

    dwParkingPlanner_release(parkingPlannerHandle);
}

TEST(ParkingPlanner2Test, NextIndex_L0)
{
    int32_t x      = 0;
    int32_t y      = 0;
    Coord3d coord2 = {0, 0, 0};
    for (int32_t theta = 1; theta < 182; theta++) //After 181 the turn lies outside the boundary
    {
        int32_t i1     = ParkingPlanner2::turnIndex(x, y, theta, false, 10.0f);
        Coord3d coord1 = ParkingPlanner2::index2Coord(i1, ParkingPlanner2::DIM3);
        int32_t i2     = ParkingPlanner2::nextIndex(coord2.x, coord2.y, coord2.hdg, 10.0f, false, false);
        coord2         = ParkingPlanner2::index2Coord(i2, ParkingPlanner2::DIM3);
        EXPECT_EQ(i1, i2);
        printf("turn: %d,%d,%d\t step: %d,%d,%d\n", coord1.x, coord1.y, coord1.hdg, coord2.x, coord2.y, coord2.hdg);
    }
}

TEST(ParkingPlanner2Test, CheckIncrement_L0)
{
    int32_t x = 0;
    int32_t y = 0;
    int32_t i = 511 * 16384 + 64 * 128 + ((64 + 1) & 127);
    ParkingPlanner2 pp;
    pp.computeIndexIncrement();
    Coord3d coord  = ParkingPlanner2::index2Coord(i, 128);
    Coord3d coordA = ParkingPlanner2::index2Coord(ParkingPlanner2::turnIndex(x, y, 511, false, 20.0f), 128);
    printf("Coord: %d,%d,%d, actual: %d,%d,%d\n\n", coord.x, coord.y, coord.hdg, coordA.x, coordA.y, coordA.hdg);

    for (int32_t theta = 510; theta >= 0; theta--)
    {

        Coord3d coord1L = ParkingPlanner2::turnIndexPlain(x, y, theta, true, 20.0f);

        Coord3d coord2L = ParkingPlanner2::turnIndexPlain(x, y, theta - 1, true, 20.0f);

        Coord3d coord1R = ParkingPlanner2::turnIndexPlain(x, y, theta, false, 20.0f);

        Coord3d coord2R = ParkingPlanner2::turnIndexPlain(x, y, theta - 1, false, 20.0f);

        printf("theta: %d \tleft diff: %d,%d \t right diff:%d,%d\n", theta, coord1L.x - coord2L.x, coord1L.y - coord2L.y, coord1R.x - coord2R.x, coord1R.y - coord2R.y);
        EXPECT_EQ(coord1L.x - coord2L.x, coord2R.x - coord1R.x);
        EXPECT_EQ(coord1L.y - coord2L.y, coord2R.y - coord1R.y);
        i = (((i - 16384) & 0x7FC000) | ((i + 128 * (((pp.m_turnIncrementR20[theta] & 12) >> 2) - 1)) & 16256) | ((i + ((pp.m_turnIncrementR20[theta] & 3) - 1)) & 127));
        printf("index:%d 1st seg: %d, 2nd seg:%d, 3rd seg: %d\n", i, ((i + 16384) & 0x7FC000), ((i + 128 * (((pp.m_turnIncrementR20[theta] & 12) >> 2) - 1)) & 16256), ((i + ((pp.m_turnIncrementR20[theta] & 3) - 1)) & 127));
        coord  = ParkingPlanner2::index2Coord(i, 128);
        coordA = ParkingPlanner2::index2Coord(ParkingPlanner2::turnIndex(x, y, theta, false, 20.0f), 128);
        printf("Coord: %d,%d,%d, actual: %d,%d,%d\n\n", coord.x, coord.y, coord.hdg, coordA.x, coordA.y, coordA.hdg);
    }
}
//Test to demostrate warp-unwarp error
/*TEST(ParkingPlanner2Test, WarpUnwarp_L0)
{
    int32_t countOne[360]   = {0};
    int32_t countTwo[360]   = {0};
    int32_t countThree[360] = {0};
    int32_t total[360]      = {0};
    for (int32_t theta = 0; theta < static_cast<int32_t>(ParkingPlanner2::getThetaStep()); theta++)
    {

        for (int32_t x = -1 * (ParkingPlanner2::X_LENGTH >> 1); x <= (ParkingPlanner2::X_LENGTH >> 1); x++)
        {
            for (int32_t y = -1 * (ParkingPlanner2::Y_LENGTH >> 1); y <= (ParkingPlanner2::Y_LENGTH >> 1); y++)
            {
                total[theta]++;
                Coord3d original = {x, y, theta};

                Coord3d straight = ParkingPlanner2::warpStraight(ParkingPlanner2::unwarpStraight(original,
                                                                                                 ParkingPlanner2::getPosRes(), ParkingPlanner2::getHdgRes()),
                                                                 ParkingPlanner2::getPosRes(), ParkingPlanner2::getHdgRes());
                if (original != straight)
                {
                    countOne[theta]++;
                    printf("1 cell error: original: %d, %d, %d,   straight:%d,%d,%d, diff:%d,%d\n",
                           x, y, theta, straight.x, straight.y, straight.hdg,
                           x - straight.x, y - straight.y);
                    Coord3d straight2 = ParkingPlanner2::warpStraight(ParkingPlanner2::unwarpStraight(straight,
                                                                                                      ParkingPlanner2::getPosRes(), ParkingPlanner2::getHdgRes()),
                                                                      ParkingPlanner2::getPosRes(), ParkingPlanner2::getHdgRes());
                    if (original != straight2 && straight != straight2)
                    {
                        countTwo[theta]++;
                        printf("2 cell error: original: %d, %d, %d,   straight:%d,%d,%d,  straight2:%d,%d,%d,  diff-o-s2:%d,%d\n",
                               x, y, theta, straight.x, straight.y, straight.hdg, straight2.x, straight2.y, straight2.hdg,
                               x - straight2.x, y - straight2.y);

                        Coord3d straight3 = ParkingPlanner2::warpStraight(ParkingPlanner2::unwarpStraight(straight2,
                                                                                                          ParkingPlanner2::getPosRes(), ParkingPlanner2::getHdgRes()),
                                                                          ParkingPlanner2::getPosRes(), ParkingPlanner2::getHdgRes());

                        if (straight2 != straight3)
                        {
                            countThree[theta]++;
                            printf("3 cell error: original: %d, %d, %d,   straight:%d,%d,%d, straight2:%d,%d,%d,  straight3:%d,%d,%d, diff-o-s3:%d,%d\n",
                                   x, y, theta, straight.x, straight.y, straight.hdg, straight2.x, straight2.y, straight2.hdg,
                                   straight3.x, straight3.y, straight3.hdg,
                                   x - straight3.x, y - straight3.y);
                        }
                    }
                }
            }
        }
    }
    for (int32_t theta = 0; theta < static_cast<int32_t>(ParkingPlanner2::getThetaStep()); theta++)
    {
        printf("Theta: %d  Total:%d   1 cell errors:%d   2 cell errors:%d   3 cell errors:%d\n", theta, total[theta], countOne[theta], countTwo[theta], countThree[theta]);
    }
}

TEST(ParkingPlanner2Test, MiniWarpUnwarp_L0)
{
    int32_t countOne[360]   = {0};
    int32_t countTwo[360]   = {0};
    int32_t countThree[360] = {0};
    int32_t total[360]      = {0};
    for (int32_t theta = 0; theta < static_cast<int32_t>(ParkingPlanner2::getThetaStep()); theta++)
    {

        for (int32_t x = -5; x <= 5; x++)
        {
            for (int32_t y = -5; y <= 5; y++)
            {
                total[theta]++;
                Coord3d original = {x, y, theta};

                Coord3d straight = ParkingPlanner2::warpStraight(ParkingPlanner2::unwarpStraight(original,
                                                                                                 ParkingPlanner2::getPosRes(), ParkingPlanner2::getHdgRes()),
                                                                 ParkingPlanner2::getPosRes(), ParkingPlanner2::getHdgRes());
                if (original != straight)
                {
                    countOne[theta]++;
                    printf("1 cell error: original: %d, %d, %d,   straight:%d,%d,%d, diff:%d,%d\n",
                           x, y, theta, straight.x, straight.y, straight.hdg,
                           x - straight.x, y - straight.y);
                    Coord3d straight2 = ParkingPlanner2::warpStraight(ParkingPlanner2::unwarpStraight(straight,
                                                                                                      ParkingPlanner2::getPosRes(), ParkingPlanner2::getHdgRes()),
                                                                      ParkingPlanner2::getPosRes(), ParkingPlanner2::getHdgRes());
                    if (original != straight2 && straight != straight2)
                    {
                        countTwo[theta]++;
                        printf("2 cell error: original: %d, %d, %d,   straight:%d,%d,%d,  straight2:%d,%d,%d,  diff-o-s2:%d,%d\n",
                               x, y, theta, straight.x, straight.y, straight.hdg, straight2.x, straight2.y, straight2.hdg,
                               x - straight2.x, y - straight2.y);

                        Coord3d straight3 = ParkingPlanner2::warpStraight(ParkingPlanner2::unwarpStraight(straight2,
                                                                                                          ParkingPlanner2::getPosRes(), ParkingPlanner2::getHdgRes()),
                                                                          ParkingPlanner2::getPosRes(), ParkingPlanner2::getHdgRes());

                        if (straight2 != straight3)
                        {
                            countThree[theta]++;
                            printf("3 cell error: original: %d, %d, %d,   straight:%d,%d,%d, straight2:%d,%d,%d,  straight3:%d,%d,%d, diff-o-s3:%d,%d\n",
                                   x, y, theta, straight.x, straight.y, straight.hdg, straight2.x, straight2.y, straight2.hdg,
                                   straight3.x, straight3.y, straight3.hdg,
                                   x - straight3.x, y - straight3.y);
                        }
                    }
                }
            }
        }
    }
    for (int32_t theta = 0; theta < static_cast<int32_t>(ParkingPlanner2::getThetaStep()); theta++)
    {
        printf("Theta: %d  Total:%d   1 cell errors:%d   2 cell errors:%d   3 cell errors:%d\n", theta, total[theta], countOne[theta], countTwo[theta], countThree[theta]);
    }
}*/

} // planner
} // dw
