
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
#include <dw/core/Matrix.hpp>
#include <dw/math/Polygon.hpp>
#include <dw/math/ConvexPolygon.hpp>

namespace dw
{
namespace planner
{

CUDA_BOTH int32_t ParkingPlanner2::getCell(int32_t x, int32_t y, int32_t theta)
{
    int32_t translatedX = x + (X_SIZE >> 1); //translating from [-halfsize,halfsize] to [0,size]
    int32_t translatedY = y + (Y_SIZE >> 1);

    return ((THETA_STEP * X_SIZE) * translatedY) + (THETA_STEP * translatedX) + theta;
}

CUDA_BOTH GridCell& ParkingPlanner2::getCell(core::span<GridCell> table, int32_t x, int32_t y, int32_t theta)
{
    GridCell& cellRef = table[getCell(x, y, theta)];
    return cellRef;
}

CUDA_BOTH GridCell& ParkingPlanner2::getCell(core::span<GridCell> table, const Coord3d& pos)
{
    return ParkingPlanner2::getCell(table, pos.x, pos.y, pos.hdg);
}

// dst (1, 0, 90) -> src (0, 1, 90)
CUDA_BOTH Coord3d ParkingPlanner2::warpStraight(const Coord3d& dst, float32_t posRes, float32_t hdgRes)
{
    Vector3f dstPose = dst.getPose(posRes, hdgRes);

    return Coord3d{warpStraight(dstPose), posRes, hdgRes};
}

CUDA_BOTH Vector3f ParkingPlanner2::warpStraight(const Vector3f& dstPose)
{
    Vector3f srcPose = {};

    srcPose.x() = static_cast<float32_t>(dstPose.x() * cos(deg2Rad(dstPose.z())) - dstPose.y() * sin(deg2Rad(dstPose.z())));
    srcPose.y() = static_cast<float32_t>(dstPose.y() * cos(deg2Rad(dstPose.z())) + dstPose.x() * sin(deg2Rad(dstPose.z())));
    srcPose.z() = dstPose.z();

    return srcPose;
}

// dst (0, 1, 90) -> src (1, 0, 90)
// the inverse function of warpStraight
CUDA_BOTH Coord3d ParkingPlanner2::unwarpStraight(const Coord3d& dst, float32_t posRes, float32_t hdgRes)
{
    Vector3f dstPose = dst.getPose(posRes, hdgRes);

    return Coord3d{unwarpStraight(dstPose), posRes, hdgRes};
}

CUDA_BOTH Vector3f ParkingPlanner2::unwarpStraight(const Vector3f& dstPose)
{
    Vector3f srcPose = {};

    srcPose.x() = static_cast<float32_t>(dstPose.x() * cos(deg2Rad(dstPose.z())) + dstPose.y() * sin(deg2Rad(dstPose.z())));
    srcPose.y() = static_cast<float32_t>(dstPose.y() * cos(deg2Rad(dstPose.z())) - dstPose.x() * sin(deg2Rad(dstPose.z())));
    srcPose.z() = dstPose.z();

    return srcPose;
}

CUDA_BOTH Coord3d ParkingPlanner2::warpLeft(const Coord3d& dst, float32_t posRes, float32_t hdgRes, float32_t turnRadius_m)
{
    Vector3f dstPose = dst.getPose(posRes, hdgRes);

    return Coord3d{warpLeft(dstPose, turnRadius_m), posRes, hdgRes};
}

CUDA_BOTH Vector3f ParkingPlanner2::warpLeft(const Vector3f& dstPose, float32_t turnRadius_m)
{
    Vector3f srcPose = {};

    float32_t s = 1.0f;

    srcPose.x() = static_cast<float32_t>(dstPose.x() + s * turnRadius_m * sin(deg2Rad(dstPose.z())));
    srcPose.y() = static_cast<float32_t>(dstPose.y() + s * turnRadius_m * (1.0f - cos(deg2Rad(dstPose.z()))));
    srcPose.z() = dstPose.z();

    return srcPose;
}

CUDA_BOTH Coord3d ParkingPlanner2::unwarpLeft(const Coord3d& dst, float32_t posRes, float32_t hdgRes, float32_t turnRadius_m)
{
    Vector3f dstPose = dst.getPose(posRes, hdgRes);

    return Coord3d{unwarpLeft(dstPose, turnRadius_m), posRes, hdgRes};
}

CUDA_BOTH Vector3f ParkingPlanner2::unwarpLeft(const Vector3f& dstPose, float32_t turnRadius_m)
{
    Vector3f srcPose = {};

    float32_t s = 1.0f;

    srcPose.x() = static_cast<float32_t>(dstPose.x() - s * turnRadius_m * sin(deg2Rad(dstPose.z())));
    srcPose.y() = static_cast<float32_t>(dstPose.y() - s * turnRadius_m * (1.0f - cos(deg2Rad(dstPose.z()))));
    srcPose.z() = dstPose.z();

    return srcPose;
}

CUDA_BOTH Coord3d ParkingPlanner2::warpRight(const Coord3d& dst, float32_t posRes, float32_t hdgRes, float32_t turnRadius_m)
{
    Vector3f dstPose = dst.getPose(posRes, hdgRes);

    return Coord3d{warpRight(dstPose, turnRadius_m), posRes, hdgRes};
}

CUDA_BOTH Vector3f ParkingPlanner2::warpRight(const Vector3f& dstPose, float32_t turnRadius_m)
{
    Vector3f srcPose = {};

    float32_t s = -1.0f;

    srcPose.x() = static_cast<float32_t>(dstPose.x() + s * turnRadius_m * sin(deg2Rad(dstPose.z())));
    srcPose.y() = static_cast<float32_t>(dstPose.y() + s * turnRadius_m * (1.0f - cos(deg2Rad(dstPose.z()))));
    srcPose.z() = dstPose.z();

    return srcPose;
}

CUDA_BOTH Coord3d ParkingPlanner2::unwarpRight(const Coord3d& dst, float32_t posRes, float32_t hdgRes, float32_t turnRadius_m)
{
    Vector3f dstPose = dst.getPose(posRes, hdgRes);

    return Coord3d{unwarpRight(dstPose, turnRadius_m), posRes, hdgRes};
}

CUDA_BOTH Vector3f ParkingPlanner2::unwarpRight(const Vector3f& dstPose, float32_t turnRadius_m)
{
    Vector3f srcPose = {};

    float32_t s = -1.0f;

    srcPose.x() = static_cast<float32_t>(dstPose.x() - s * turnRadius_m * sin(deg2Rad(dstPose.z())));
    srcPose.y() = static_cast<float32_t>(dstPose.y() - s * turnRadius_m * (1.0f - cos(deg2Rad(dstPose.z()))));
    srcPose.z() = dstPose.z();

    return srcPose;
}

__global__ void _setThreeWallMaze(GridCell* table)
{
    int32_t x   = blockIdx.x - (ParkingPlanner2::getSize() >> 1); //transforming indices from [0,size] range to [-size/2, size/2] range
    int32_t y   = blockIdx.y - (ParkingPlanner2::getSize() >> 1);
    int32_t hdg = threadIdx.x;

    if (y > 5 && y < 11 && x != 0 && x != -1 && x != 1)
    {
        Coord3d pose   = {x, y, hdg};
        GridCell& cell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), pose);
        cell.obstacle  = true;
    }

    else if (y > 15 && y < 21 && x != 50 && x != 51 && x != 49)
    {
        Coord3d pose   = {x, y, hdg};
        GridCell& cell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), pose);
        cell.obstacle  = true;
    }
    else if (y > 25 && y < 31 && x != -50 && x != -51 && x != -49)
    {
        Coord3d pose   = {x, y, hdg};
        GridCell& cell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), pose);
        cell.obstacle  = true;
    }
    else
    {
        return;
    }
}

void ParkingPlanner2::setThreeWallMaze(cudaStream_t cuStream)
{
    dim3 GridSize(X_SIZE, Y_SIZE);

    _setThreeWallMaze<<<GridSize, THETA_STEP, 0, cuStream>>>(m_cuGrid.get().get().data());
    cudaStreamSynchronize(cuStream);
}

__global__ void _setWall(GridCell* table)
{
    int32_t x   = blockIdx.x - (ParkingPlanner2::getSize() >> 1); //transforming indices from [0,size] range to [-size/2, size/2] range
    int32_t y   = blockIdx.y - (ParkingPlanner2::getSize() >> 1);
    int32_t hdg = threadIdx.x;

    if (x > 15 && x < 25)
    {
        Coord3d pose   = {x, y, hdg};
        GridCell& cell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), pose);
        cell.obstacle  = true;
    }

    else
    {
        return;
    }
}

void ParkingPlanner2::setWall(cudaStream_t cuStream)
{
    dim3 GridSize(X_SIZE, Y_SIZE);

    _setWall<<<GridSize, THETA_STEP, 0, cuStream>>>(m_cuGrid.get().get().data());
    cudaStreamSynchronize(cuStream);
}

__global__ void _setBay(GridCell* table)
{
    int32_t x   = blockIdx.x - (ParkingPlanner2::getSize() >> 1); //transforming indices from [0,size] range to [-size/2, size/2] range
    int32_t y   = blockIdx.y - (ParkingPlanner2::getSize() >> 1);
    int32_t hdg = threadIdx.x;

    if (y >= 10 && x <= 10 && x - y + 10 > 0 && x - y + 5 < 0)
    {
        Coord3d pose   = {x, y, hdg};
        GridCell& cell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), pose);
        cell.obstacle  = true;
    }
    if (y > 11 && x >= 10 && x + y - 25 > 0 && x + y - 30 < 0)
    {
        Coord3d pose   = {x, y, hdg};
        GridCell& cell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), pose);
        cell.obstacle  = true;
    }
    if (y < 9 && x >= 10 && x - y - 10 < 0 && x - y - 5 > 0)
    {
        Coord3d pose   = {x, y, hdg};
        GridCell& cell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), pose);
        cell.obstacle  = true;
    }
    if (x <= 10 && y <= 10 && x + y - 10 > 0 && x + y - 15 < 0)
    {
        Coord3d pose   = {x, y, hdg};
        GridCell& cell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), pose);
        cell.obstacle  = true;
    }
}

void ParkingPlanner2::setBay(cudaStream_t cuStream)
{
    dim3 GridSize(X_SIZE, Y_SIZE);
    _setBay<<<GridSize, THETA_STEP, 0, cuStream>>>(m_cuGrid.get().get().data());
    cudaStreamSynchronize(cuStream);
}

__global__ void _setHurdle(GridCell* table)
{
    int32_t x   = blockIdx.x - (ParkingPlanner2::getSize() >> 1); //transforming indices from [0,size] range to [-size/2, size/2] range
    int32_t y   = blockIdx.y - (ParkingPlanner2::getSize() >> 1);
    int32_t hdg = threadIdx.x;

    if (x > 5 && x < 15 && y <= 10 && y >= -10)
    {
        Coord3d pose   = {x, y, hdg};
        GridCell& cell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), pose);
        cell.obstacle  = true;
    }
}

void ParkingPlanner2::setHurdle(cudaStream_t cuStream)
{
    dim3 GridSize(X_SIZE, Y_SIZE);
    _setHurdle<<<GridSize, THETA_STEP, 0, cuStream>>>(m_cuGrid.get().get().data());
    cudaStreamSynchronize(cuStream);
}

__global__ void _setParallel(GridCell* table)
{
    int32_t x   = blockIdx.x - (ParkingPlanner2::getSize() >> 1); //transforming indices from [0,size] range to [-size/2, size/2] range
    int32_t y   = blockIdx.y - (ParkingPlanner2::getSize() >> 1);
    int32_t hdg = threadIdx.x;

    if (x > 5 && x < 15 && y <= 10 && y >= -10)
    {
        Coord3d pose   = {x, y, hdg};
        GridCell& cell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), pose);
        cell.obstacle  = true;
    }
    if (x > 20 && x < 30 && y <= 10 && y >= -10)
    {
        Coord3d pose   = {x, y, hdg};
        GridCell& cell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), pose);
        cell.obstacle  = true;
    }
}

void ParkingPlanner2::setParallel(cudaStream_t cuStream)
{
    dim3 GridSize(X_SIZE, Y_SIZE);
    _setParallel<<<GridSize, THETA_STEP, 0, cuStream>>>(m_cuGrid.get().get().data());
    cudaStreamSynchronize(cuStream);
}

__global__ void _setUnwarpedPose(GridCell* table)
{
    int32_t x   = blockIdx.x - (ParkingPlanner2::getSize() >> 1); //transforming indices from [0,size] range to [-size/2, size/2] range
    int32_t y   = blockIdx.y - (ParkingPlanner2::getSize() >> 1);
    int32_t hdg = threadIdx.x;

    Coord3d original = {x, y, hdg};

    GridCell& cell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), original);

    cell.unwarpedPose = original;
}

void ParkingPlanner2::setUnwarpedPose(cudaStream_t cuStream)
{
    dim3 GridSize(X_SIZE, Y_SIZE);

    _setUnwarpedPose<<<GridSize, THETA_STEP, 0, cuStream>>>(m_cuGrid.get().get().data());
    cudaStreamSynchronize(cuStream);
}

__global__ void _sweepStraight(GridCell* table)
{
    //Indices accessing cells in the warped table
    int32_t x     = blockIdx.x - (ParkingPlanner2::getSize() >> 1); //transforming indices from [0,size] range to [-size/2, size/2] range
    int32_t y     = blockIdx.y - (ParkingPlanner2::getSize() >> 1);
    int32_t theta = threadIdx.x;

    //Search starts from all cells with -X_SIZE/2. Other threads are terminated.
    if (x != (-1 * (ParkingPlanner2::getSize() >> 1)))
        return;

    for (int32_t i = -1 * (ParkingPlanner2::getSize() >> 1); i < (ParkingPlanner2::getSize() >> 1); i++) //Searching loop. Searches for a reachable cell
    {
        Coord3d searcher = {i, y, theta};
        int32_t i_bef;
        GridCell& searchCell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), searcher);
        if (searchCell.obstacle) //Searcher skips through the obstacle
        {
            continue;
        }
        if (searchCell.reachable) //If a reachable cell is found sweep starts
        {
            i_bef            = i;                       //Stores the index where sweep starts
            Coord3d searchP  = searchCell.unwarpedPose; //Details of cell where sweeps starts
            Coord3d searchPP = searchCell.prevPose;
            for (int32_t j = i + 1; j < (ParkingPlanner2::getSize() >> 1); j++) //Forward sweep: Makes all the forward accessible cells reachable
            {
                Coord3d frontGoer   = {j, y, theta};
                GridCell& frontCell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), frontGoer);
                i                   = j; //Moves searcher forward as we sweep to avoid searching again
                if (frontCell.obstacle)  //Stops sweeping and returns to search if it hits obstacle
                {
                    break;
                }
                if (frontCell.reachable) //skips through already reachable cells (to avoid meddling with already stored path and maneuver details)
                {
                    continue;
                }
                frontCell.reachable = true;                                                     //makes reachable
                frontCell.prevPose  = (searchP != frontCell.unwarpedPose) ? searchP : searchPP; //stores unwarped pose of start of sweep. //Conditional statement to avoid path getting stuck at same point.
                frontCell.maneuver  = ManeuverType::STRAIGHT;                                   //Stores primitive
                frontCell.reverse   = false;                                                    //car not reverse
            }
            searchP  = searchCell.unwarpedPose; //Details of cell where sweep starts
            searchPP = searchCell.prevPose;
            for (int32_t k = i_bef - 1; k >= -1 * (ParkingPlanner2::getSize() >> 1); k--) //backward sweep: Sweep backwards from a reachable cell
            {
                Coord3d backGoer   = {k, y, theta};
                GridCell& backCell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), backGoer);
                if (backCell.obstacle)
                {
                    break;
                }
                if (backCell.reachable)
                {
                    continue;
                }
                backCell.reachable = true;
                backCell.prevPose  = (searchP != backCell.unwarpedPose) ? searchP : searchPP;
                backCell.maneuver  = ManeuverType::STRAIGHT;
                backCell.reverse   = true;
            }
        }
    }
}

__global__ void _sweepArc(GridCell* table, ManeuverType maneuver)
{
    if (maneuver != ManeuverType::LEFT && maneuver != ManeuverType::RIGHT)
    {
        return;
    }
    //Indices accessing cells in the warped table
    int32_t x     = blockIdx.x - (ParkingPlanner2::getSize() >> 1); //transforming indices from [0,size] range to [-size/2, size/2] range
    int32_t y     = blockIdx.y - (ParkingPlanner2::getSize() >> 1);
    int32_t theta = threadIdx.x;

    //Search starts from theta = 0; Other threads terminated;
    if (theta != 0)
        return;

    bool endsweep = false; //becomes true if circle completes
    int32_t i_rev = -1;    // marks the end of reverse sweep

    for (int32_t i = 0; i < ParkingPlanner2::getThetaStep(); i++) //Searching loop. Searches for a reachable cell
    {
        if (endsweep || i == i_rev) //End search; If circle completed or search reaches as far as reverse sweep's end
        {
            break;
        }
        Coord3d searcher = {x, y, i};
        int32_t i_bef, i_aft;
        bool circularity     = false; //true if front sweep hits 360 or back sweep hits 0.
        GridCell& searchCell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), searcher);
        if (searchCell.obstacle)
        {
            continue;
        }
        if (searchCell.reachable)
        {
            i_bef            = i;                       //Stores the index where sweep starts
            Coord3d searchP  = searchCell.unwarpedPose; //Details of cell where sweep starts
            Coord3d searchPP = searchCell.prevPose;
            for (int32_t j = i + 1; j < ParkingPlanner2::getThetaStep(); j++) //Forward sweep: moves onward from start of sweep
            {
                Coord3d frontGoer   = {x, y, j};
                GridCell& frontCell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), frontGoer);
                i                   = j; //search moved to end of forward sweep
                i_aft               = i; //End of forward sweep stored to set a limit to reverse sweep
                if (frontCell.obstacle)  //Stops sweeping and returns to search if it hits obstacle
                {
                    break;
                }
                if (frontCell.reachable)
                {
                    continue;
                }
                frontCell.reachable = true;
                frontCell.prevPose  = (searchP != frontCell.unwarpedPose) ? searchP : searchPP;
                frontCell.maneuver  = maneuver;
                frontCell.reverse   = (maneuver != ManeuverType::LEFT);

                if (j == ParkingPlanner2::getThetaStep() - 1)
                {
                    endsweep    = true; //search can be terminated after this if cells where sweep till 360
                    circularity = true; //359 -> circularity -> 0 -> 1 -> 2 ...
                }
                if (circularity)
                {
                    for (int32_t k = 0; k < i_bef; k++) //sweep continues until search cell
                    {
                        Coord3d frontGoer   = {x, y, k};
                        GridCell& frontCell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), frontGoer);
                        if (frontCell.obstacle)
                        {
                            break;
                        }
                        if (frontCell.reachable)
                        {
                            continue;
                        }
                        frontCell.reachable = true;
                        frontCell.prevPose  = (searchP != frontCell.unwarpedPose) ? searchP : searchPP;
                        frontCell.maneuver  = maneuver;
                        frontCell.reverse   = (maneuver != ManeuverType::LEFT);
                    }
                }
            }
            searchP     = searchCell.unwarpedPose; //Details of cell where sweep starts
            searchPP    = searchCell.prevPose;
            circularity = false;                 //circularity reset
            for (int32_t k = i_bef; k >= 0; k--) //starts at search cell and sweeps backwards
            {
                Coord3d backGoer   = {x, y, k};
                GridCell& backCell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), backGoer);
                if (backCell.obstacle)
                {
                    break;
                }
                if (backCell.reachable)
                {
                    continue;
                }
                backCell.reachable = true;
                backCell.prevPose  = (searchP != backCell.unwarpedPose) ? searchP : searchPP;
                backCell.maneuver  = maneuver;
                backCell.reverse   = (maneuver == ManeuverType::LEFT);

                if (k == 0) //0 -> circularity -> 359 ->358 ...
                {
                    circularity = true;
                }
            }
            if (circularity)
            {
                for (int32_t k = ParkingPlanner2::getThetaStep() - 1; k > i_aft; k--) //sweep continues until end of forward sweep
                {
                    Coord3d backGoer   = {x, y, k};
                    GridCell& backCell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), backGoer);
                    i_rev              = k;
                    if (backCell.obstacle)
                    {
                        break;
                    }
                    if (backCell.reachable)
                    {
                        continue;
                    }
                    backCell.reachable = true;
                    backCell.prevPose  = (searchP != backCell.unwarpedPose) ? searchP : searchPP;
                    backCell.maneuver  = maneuver;
                    backCell.reverse   = (maneuver == ManeuverType::LEFT);

                    if (k == i_aft + 1) //Sweep completes circle. Search terminated.
                    {
                        endsweep = true;
                    }
                }
            }
        }
    }
}

void ParkingPlanner2::sweepArc(GridCell* table, ManeuverType maneuver, cudaStream_t cuStream)
{
    dim3 GridSize(X_SIZE, Y_SIZE);

    _sweepArc<<<GridSize, THETA_STEP, 0, cuStream>>>(table, maneuver);
    cudaStreamSynchronize(cuStream);
}

void ParkingPlanner2::sweepStraight(GridCell* table, cudaStream_t cuStream)
{
    dim3 GridSize(X_SIZE, Y_SIZE);

    _sweepStraight<<<GridSize, THETA_STEP, 0, cuStream>>>(table);
    cudaStreamSynchronize(cuStream);
}

__global__ void _sweepStraight2(GridCell* table, uint8_t iter)
{
    //Indices accessing cells in the warped table
    int32_t x     = -(ParkingPlanner2::getSize() >> 1); //transforming indices from [0,size] range to [-size/2, size/2] range
    int32_t y     = blockIdx.x - (ParkingPlanner2::getSize() >> 1);
    int32_t theta = threadIdx.x;

    for (int32_t i = -1 * (ParkingPlanner2::getSize() >> 1); i < (ParkingPlanner2::getSize() >> 1); i++) //Searching loop. Searches for a reachable cell
    {
        Coord3d searcher = {i, y, theta};
        int32_t srcX;
        GridCell& searchCell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), searcher);
        if (searchCell.obstacle) //Searcher skips through the obstacle
        {
            continue;
        }
        if (searchCell.reachable && searchCell.iterCount == (iter - 1)) //If a reachable cell is found sweep starts
        {
            srcX = i;                                                           //Stores the index where sweep starts
            for (int32_t j = i + 1; j < (ParkingPlanner2::getSize() >> 1); j++) //Forward sweep: Makes all the forward accessible cells reachable
            {
                Coord3d frontGoer   = {j, y, theta};
                GridCell& frontCell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), frontGoer);
                i                   = j; //Moves searcher forward as we sweep to avoid searching again
                if (frontCell.obstacle)  //Stops sweeping and returns to search if it hits obstacle
                {
                    break;
                }
                if (frontCell.reachable) //skips through already reachable cells (to avoid meddling with already stored path and maneuver details)
                {
                    continue;
                }
                frontCell.reachable = true; //makes reachable
                frontCell.iterCount = iter;
                frontCell.src       = srcX;
                frontCell.maneuver  = ManeuverType::STRAIGHT; //Stores primitive
                frontCell.reverse   = false;                  //car not reverse
            }

            for (int32_t k = srcX - 1; k >= -1 * (ParkingPlanner2::getSize() >> 1); k--) //backward sweep: Sweep backwards from a reachable cell
            {
                Coord3d backGoer   = {k, y, theta};
                GridCell& backCell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), backGoer);
                if (backCell.obstacle)
                {
                    break;
                }
                if (backCell.reachable)
                {
                    continue;
                }
                backCell.reachable = true;
                backCell.iterCount = iter;
                backCell.src       = srcX;
                backCell.maneuver  = ManeuverType::STRAIGHT;
                backCell.reverse   = true;
            }
        }
    }
}

void ParkingPlanner2::sweepStraight2(GridCell* table, uint8_t iter, cudaStream_t cuStream)
{
    _sweepStraight2<<<Y_SIZE, THETA_STEP, 0, cuStream>>>(table, iter);
    cudaStreamSynchronize(cuStream);
}

__global__ void _sweepStraightNew(GridCell* table, uint8_t iter)
{
    //Indices accessing cells in the warped table
    int32_t x     = -(ParkingPlanner2::getSize() >> 1); //transforming indices from [0,size] range to [-size/2, size/2] range
    int32_t y     = blockIdx.x - (ParkingPlanner2::getSize() >> 1);
    int32_t theta = threadIdx.x;

    for (int32_t i = -1 * (ParkingPlanner2::getSize() >> 1); i < (ParkingPlanner2::getSize() >> 1); i++) //Searching loop. Searches for a reachable cell
    {
        Coord3d searcher = {i, y, theta};
        int32_t srcX;
        GridCell& searchCell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), searcher);
        if (searchCell.obstacle || searchCell.gap) //Searcher skips through the obstacle
        {
            continue;
        }
        if (searchCell.reachable && searchCell.iterCount == (iter - 1)) //If a reachable cell is found sweep starts
        {
            srcX = i;                                                           //Stores the index where sweep starts
            for (int32_t j = i + 1; j < (ParkingPlanner2::getSize() >> 1); j++) //Forward sweep: Makes all the forward accessible cells reachable
            {
                Coord3d frontGoer   = {j, y, theta};
                GridCell& frontCell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), frontGoer);
                i                   = j; //Moves searcher forward as we sweep to avoid searching again
                if (frontCell.obstacle)  //Stops sweeping and returns to search if it hits obstacle
                {
                    break;
                }
                if (frontCell.reachable || frontCell.gap) //skips through already reachable cells (to avoid meddling with already stored path and maneuver details)
                {
                    continue;
                }
                frontCell.reachable = true; //makes reachable
                frontCell.iterCount = iter;
                frontCell.src       = srcX;
                frontCell.maneuver  = ManeuverType::STRAIGHT; //Stores primitive
                frontCell.reverse   = false;                  //car not reverse
            }

            for (int32_t k = srcX - 1; k >= -1 * (ParkingPlanner2::getSize() >> 1); k--) //backward sweep: Sweep backwards from a reachable cell
            {
                Coord3d backGoer   = {k, y, theta};
                GridCell& backCell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), backGoer);
                if (backCell.obstacle)
                {
                    break;
                }
                if (backCell.reachable || backCell.gap)
                {
                    continue;
                }
                backCell.reachable = true;
                backCell.iterCount = iter;
                backCell.src       = srcX;
                backCell.maneuver  = ManeuverType::STRAIGHT;
                backCell.reverse   = true;
            }
        }
    }
}

void ParkingPlanner2::sweepStraightNew(GridCell* table, uint8_t iter, cudaStream_t cuStream)
{
    _sweepStraightNew<<<Y_SIZE, THETA_STEP, 0, cuStream>>>(table, iter);
    cudaStreamSynchronize(cuStream);
}

__global__ void _sweepArc2(GridCell* table, ManeuverType maneuver, uint8_t iter)
{
    if (maneuver != ManeuverType::LEFT && maneuver != ManeuverType::RIGHT)
    {
        return;
    }
    //Indices accessing cells in the warped table
    int32_t x = blockIdx.x - (ParkingPlanner2::getSize() >> 1); //transforming indices from [0,size] range to [-size/2, size/2] range
    int32_t y = threadIdx.x - (ParkingPlanner2::getSize() >> 1);

    bool endsweep = false; //becomes true if circle completes
    int32_t i_rev = -1;    // marks the end of reverse sweep

    for (int32_t i = 0; i < ParkingPlanner2::getThetaStep(); i++) //Searching loop. Searches for a reachable cell
    {
        if (endsweep || i == i_rev) //End search; If circle completed or search reaches as far as reverse sweep's end
        {
            break;
        }
        Coord3d searcher = {x, y, i};
        int32_t srcTheta, i_aft;
        bool circularity     = false; //true if front sweep hits 360 or back sweep hits 0.
        GridCell& searchCell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), searcher);
        if (searchCell.obstacle)
        {
            continue;
        }
        if (searchCell.reachable && searchCell.iterCount == (iter - 1))
        {
            srcTheta = i;                                                     //Stores the index where sweep starts
            for (int32_t j = i + 1; j < ParkingPlanner2::getThetaStep(); j++) //Forward sweep: moves onward from start of sweep
            {
                Coord3d frontGoer   = {x, y, j};
                GridCell& frontCell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), frontGoer);
                i                   = j; //search moved to end of forward sweep
                i_aft               = i; //End of forward sweep stored to set a limit to reverse sweep
                if (frontCell.obstacle)  //Stops sweeping and returns to search if it hits obstacle
                {
                    break;
                }
                if (frontCell.reachable)
                {
                    continue;
                }
                frontCell.reachable = true;
                frontCell.iterCount = iter;
                frontCell.src       = srcTheta;
                frontCell.maneuver  = maneuver;
                frontCell.reverse   = (maneuver != ManeuverType::LEFT);

                if (j == ParkingPlanner2::getThetaStep() - 1)
                {
                    endsweep    = true; //search can be terminated after this if cells where sweep till 360
                    circularity = true; //359 -> circularity -> 0 -> 1 -> 2 ...
                }
                if (circularity)
                {
                    for (int32_t k = 0; k < srcTheta; k++) //sweep continues until search cell
                    {
                        Coord3d frontGoer   = {x, y, k};
                        GridCell& frontCell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), frontGoer);
                        if (frontCell.obstacle)
                        {
                            break;
                        }
                        if (frontCell.reachable)
                        {
                            continue;
                        }
                        frontCell.reachable = true;
                        frontCell.iterCount = iter;
                        frontCell.src       = srcTheta;
                        frontCell.maneuver  = maneuver;
                        frontCell.reverse   = (maneuver != ManeuverType::LEFT);
                    }
                }
            }

            circularity = false;                    //circularity reset
            for (int32_t k = srcTheta; k >= 0; k--) //starts at search cell and sweeps backwards
            {
                Coord3d backGoer   = {x, y, k};
                GridCell& backCell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), backGoer);
                if (backCell.obstacle)
                {
                    break;
                }
                if (backCell.reachable)
                {
                    continue;
                }
                backCell.reachable = true;
                backCell.iterCount = iter;
                backCell.src       = srcTheta;
                backCell.maneuver  = maneuver;
                backCell.reverse   = (maneuver == ManeuverType::LEFT);

                if (k == 0) //0 -> circularity -> 359 ->358 ...
                {
                    circularity = true;
                }
            }
            if (circularity)
            {
                for (int32_t k = ParkingPlanner2::getThetaStep() - 1; k > i_aft; k--) //sweep continues until end of forward sweep
                {
                    Coord3d backGoer   = {x, y, k};
                    GridCell& backCell = ParkingPlanner2::getCell(make_span(table, ParkingPlanner2::getGridSize()), backGoer);
                    i_rev              = k;
                    if (backCell.obstacle)
                    {
                        break;
                    }
                    if (backCell.reachable)
                    {
                        continue;
                    }
                    backCell.reachable = true;
                    backCell.iterCount = iter;
                    backCell.src       = srcTheta;
                    backCell.maneuver  = maneuver;
                    backCell.reverse   = (maneuver == ManeuverType::LEFT);

                    if (k == i_aft + 1) //Sweep completes circle. Search terminated.
                    {
                        endsweep = true;
                    }
                }
            }
        }
    }
}

void ParkingPlanner2::sweepArc2(GridCell* table, ManeuverType maneuver, uint8_t iter, cudaStream_t cuStream)
{
    _sweepArc2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(table, maneuver, iter);
    cudaStreamSynchronize(cuStream);
}

__global__ void _warpStraight(GridCell* out, GridCell* in, float32_t posRes, float32_t hdgRes, bool unwarp)
{
    // these are destination indices
    // used to access "out"
    int32_t x     = blockIdx.x - (ParkingPlanner2::getSize() >> 1); //transforming indices from [0,size] range to [-size/2, size/2] range
    int32_t y     = blockIdx.y - (ParkingPlanner2::getSize() >> 1);
    int32_t theta = threadIdx.x;

    if (!ParkingPlanner2::withinBounds(x, y))
    {
        return;
    }

    Coord3d dstPose{x, y, theta};

    // clear the output
    GridCell& dstCell = ParkingPlanner2::getCell(make_span(out, ParkingPlanner2::getGridSize()), dstPose);

    //not unwarping into already reachable cells
    if (unwarp && dstCell.reachable)
    {
        return;
    }
    dstCell = {};

    // get the corresponding srcPose
    Coord3d srcPose = (!unwarp)
                          ? ParkingPlanner2::warpStraight(dstPose, posRes, hdgRes)
                          : ParkingPlanner2::unwarpStraight(dstPose, posRes, hdgRes);

    // TODO(JK): take care of discretization error
    if (ParkingPlanner2::withinBounds(srcPose))
    {
        const GridCell& srcCell = ParkingPlanner2::getCell(make_span(in, ParkingPlanner2::getGridSize()), srcPose);

        // essentially an image rotation
        dstCell = srcCell;
    }
    //setting the cells outside the plan space as obstacle
    if (!unwarp && !ParkingPlanner2::withinPlanSpace(srcPose))
        dstCell.obstacle = true;
}

__global__ void _warpStraightNew(GridCell* out, GridCell* in, float32_t posRes, float32_t hdgRes, bool unwarp)
{
    // these are destination indices
    // used to access "out"
    int32_t x     = blockIdx.x - (ParkingPlanner2::getSize() >> 1); //transforming indices from [0,size] range to [-size/2, size/2] range
    int32_t y     = blockIdx.y - (ParkingPlanner2::getSize() >> 1);
    int32_t theta = threadIdx.x;

    if (!ParkingPlanner2::withinBounds(x, y, theta))
    {
        return;
    }
    Coord3d truePose{x, y, theta};

    // get the corresponding warpedCell
    Coord3d warpedPose = ParkingPlanner2::unwarpStraight(truePose, posRes, hdgRes);
    if (!ParkingPlanner2::withinBounds(warpedPose))
    {
        return;
    }

    if (!unwarp)
    {
        GridCell& trueCell   = ParkingPlanner2::getCell(make_span(in, ParkingPlanner2::getGridSize()), truePose);
        GridCell& warpedCell = ParkingPlanner2::getCell(make_span(out, ParkingPlanner2::getGridSize()), warpedPose);
        warpedCell           = trueCell;
        warpedCell.gap       = false;
        //setting the cells outside the plan space as obstacle
        if (!ParkingPlanner2::withinPlanSpace(truePose))
            warpedCell.obstacle = true;
    }
    else
    {
        GridCell& trueCell   = ParkingPlanner2::getCell(make_span(out, ParkingPlanner2::getGridSize()), truePose);
        GridCell& warpedCell = ParkingPlanner2::getCell(make_span(in, ParkingPlanner2::getGridSize()), warpedPose);
        if (!trueCell.reachable)
            trueCell = warpedCell;
        trueCell.gap = true;
    }
}

// TODO(yizhouw) code duplication can be much further reduced if
// the warping method "warpXX" "unwarpXX" "warpLeft" "warpRight" is
// pass in the kernel, via function pointer input or template argument
__global__ void _warpLeft(GridCell* out, GridCell* in, float32_t posRes, float32_t hdgRes, float32_t turnRadius_m, bool unwarp)
{
    // these are destination indices
    // used to access "out"
    int32_t x     = blockIdx.x - (ParkingPlanner2::getSize() >> 1); //transforming indices from [0,size] range to [-size/2, size/2] range
    int32_t y     = blockIdx.y - (ParkingPlanner2::getSize() >> 1);
    int32_t theta = threadIdx.x;

    if (!ParkingPlanner2::withinBounds(x, y))
    {
        return;
    }

    Coord3d dstPose{x, y, theta};

    // clear the output
    GridCell& dstCell = ParkingPlanner2::getCell(make_span(out, ParkingPlanner2::getGridSize()), dstPose);

    //not unwarping into already reachable cells
    if (unwarp && dstCell.reachable)
    {
        return;
    }
    dstCell = {};

    // get the corresponding srcPose
    Coord3d srcPose = (!unwarp)
                          ? ParkingPlanner2::warpLeft(dstPose, posRes, hdgRes, turnRadius_m)
                          : ParkingPlanner2::unwarpLeft(dstPose, posRes, hdgRes, turnRadius_m);
    // TODO(JK): take care of discretization error
    if (ParkingPlanner2::withinBounds(srcPose))
    {
        const GridCell& srcCell = ParkingPlanner2::getCell(make_span(in, ParkingPlanner2::getGridSize()), srcPose);

        // essentially an image rotation
        dstCell = srcCell;
    }
    //setting the cells outside the plan space as obstacle
    if (!unwarp && !ParkingPlanner2::withinPlanSpace(srcPose))
        dstCell.obstacle = true;
}

__global__ void _warpRight(GridCell* out, GridCell* in, float32_t posRes, float32_t hdgRes, float32_t turnRadius_m, bool unwarp)
{
    // these are destination indices
    // used to access "out"
    int32_t x     = blockIdx.x - (ParkingPlanner2::getSize() >> 1); //transforming indices from [0,size] range to [-size/2, size/2] range
    int32_t y     = blockIdx.y - (ParkingPlanner2::getSize() >> 1);
    int32_t theta = threadIdx.x;

    if (!ParkingPlanner2::withinBounds(x, y))
    {
        return;
    }

    Coord3d dstPose{x, y, theta};

    // clear the output
    GridCell& dstCell = ParkingPlanner2::getCell(make_span(out, ParkingPlanner2::getGridSize()), dstPose);

    //not unwarping into already reachable cells
    if (unwarp && dstCell.reachable)
    {
        return;
    }
    dstCell = {};

    // get the corresponding srcPose
    Coord3d srcPose = (!unwarp)
                          ? ParkingPlanner2::warpRight(dstPose, posRes, hdgRes, turnRadius_m)
                          : ParkingPlanner2::unwarpRight(dstPose, posRes, hdgRes, turnRadius_m);
    // TODO(JK): take care of discretization error
    if (ParkingPlanner2::withinBounds(srcPose)) //protects cells outside padding getting unwarped
    {
        const GridCell& srcCell = ParkingPlanner2::getCell(make_span(in, ParkingPlanner2::getGridSize()), srcPose);

        // essentially an image rotation
        dstCell = srcCell;
    }
    //setting the cells outside the plan space as obstacle
    if (!unwarp && !ParkingPlanner2::withinPlanSpace(srcPose))
        dstCell.obstacle = true;
}

//***
void ParkingPlanner2::warpStraight(GridCell* out, GridCell* in, float32_t posRes, float32_t hdgRes, cudaStream_t cuStream, bool unwarp)
{
    dim3 GridSize(X_SIZE, Y_SIZE);
    _warpStraight<<<GridSize, THETA_STEP, 0, cuStream>>>(out, in, posRes, hdgRes, unwarp); // one thread per cell
    cudaStreamSynchronize(cuStream);
}

void ParkingPlanner2::warpStraightNew(GridCell* out, GridCell* in, float32_t posRes, float32_t hdgRes, cudaStream_t cuStream, bool unwarp)
{
    dim3 GridSize(X_SIZE, Y_SIZE);
    _warpStraightNew<<<GridSize, THETA_STEP, 0, cuStream>>>(out, in, posRes, hdgRes, unwarp); // one thread per cell
    cudaStreamSynchronize(cuStream);
}

void ParkingPlanner2::warpLeft(GridCell* out, GridCell* in, float32_t posRes, float32_t hdgRes, float32_t turnRadius_m, cudaStream_t cuStream, bool unwarp)
{
    dim3 GridSize(X_SIZE, Y_SIZE);
    _warpLeft<<<GridSize, THETA_STEP, 0, cuStream>>>(out, in, posRes, hdgRes, turnRadius_m, unwarp); // one thread per cell
    cudaStreamSynchronize(cuStream);
}

void ParkingPlanner2::warpRight(GridCell* out, GridCell* in, float32_t posRes, float32_t hdgRes, float32_t turnRadius_m, cudaStream_t cuStream, bool unwarp)
{
    dim3 GridSize(X_SIZE, Y_SIZE);
    _warpRight<<<GridSize, THETA_STEP, 0, cuStream>>>(out, in, posRes, hdgRes, turnRadius_m, unwarp); // one thread per cell
    cudaStreamSynchronize(cuStream);
}

__global__ void _sweepAdd1(uint8_t* cell)
{
    int32_t indexX = blockIdx.x;
    int32_t indexY = threadIdx.x;
    int32_t SIZE   = 169; //=X_SIZE = Y_SIZE

    for (int32_t theta = 0; theta < 360; theta++)
    {
        cell[indexX + (indexY + theta * SIZE) * SIZE] += 1;
    }
}

__global__ void _sweepAdd2(uint8_t* cell)
{
    int32_t indexX = blockIdx.x;
    int32_t indexY = threadIdx.x;
    int32_t SIZE   = 169; //=X_SIZE = Y_SIZE

    for (int32_t theta = 0; theta < 360; theta++)
    {
        cell[indexY + (indexX + theta * SIZE) * SIZE] += 1;
    }
}

__global__ void _sweepAdd16(uint16_t* cell)
{
    int32_t indexX = blockIdx.x;
    int32_t indexY = threadIdx.x;
    int32_t SIZE   = 169; //=X_SIZE = Y_SIZE

    __shared__ uint16_t a[24000];
    a[(blockIdx.x * threadIdx.x) % 24000] += 1;
    for (int32_t theta = 0; theta < 360; theta += 4)
    {
        a[0]                                                = cell[indexY + (indexX + theta * SIZE) * SIZE];
        a[1000]                                             = cell[indexY + (indexX + (theta + 1) * SIZE) * SIZE];
        a[5000]                                             = cell[indexY + (indexX + (theta + 2) * SIZE) * SIZE];
        a[11000]                                            = cell[indexY + (indexX + (theta + 3) * SIZE) * SIZE];
        cell[indexY + (indexX + theta * SIZE) * SIZE]       = a[0] + 1;
        cell[indexY + (indexX + (theta + 1) * SIZE) * SIZE] = a[1000] + 1;
        cell[indexY + (indexX + (theta + 2) * SIZE) * SIZE] = a[5000] + 1;
        cell[indexY + (indexX + (theta + 3) * SIZE) * SIZE] = a[11000] + 1;
    }
}

void ParkingPlanner2::sweepAdd1(uint8_t* cell, cudaStream_t cuStream)
{
    _sweepAdd1<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    cudaStreamSynchronize(cuStream);
}

void ParkingPlanner2::sweepAdd2(uint8_t* cell, cudaStream_t cuStream)
{
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd2<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);

    cudaStreamSynchronize(cuStream);
}

void ParkingPlanner2::sweepAdd16(uint16_t* cell, cudaStream_t cuStream)
{
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    _sweepAdd16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell);
    cudaStreamSynchronize(cuStream);
}

__global__ void _sweepAddLeft(uint8_t* cell, float32_t posRes, float32_t hdgRes, float32_t turnRadius)
{
    int32_t indexX = blockIdx.x;
    int32_t indexY = threadIdx.x;
    int32_t SIZE   = 169; //=X_SIZE = Y_SIZE

    int32_t x = indexX - (169 >> 1);
    int32_t y = indexY - (169 >> 1);

    for (int32_t theta = 0; theta < 360; theta++)
    {
        float32_t rotX    = static_cast<float32_t>(x) * posRes + turnRadius * sin(deg2Rad(static_cast<float32_t>(theta) * hdgRes));
        float32_t rotY    = static_cast<float32_t>(y) * posRes + turnRadius * (1.0f - cos(deg2Rad(static_cast<float32_t>(theta) * hdgRes)));
        int32_t newIndexX = static_cast<int32_t>(floor((rotX / posRes) + 0.5f)) + (SIZE >> 1);
        int32_t newIndexY = static_cast<int32_t>(floor((rotY / posRes) + 0.5f)) + (SIZE >> 1);

        cell[newIndexY + (newIndexX + theta * SIZE) * SIZE] += 1;
    }
}
void ParkingPlanner2::sweepAddLeft(uint8_t* cell, float32_t posRes, float32_t hdgRes, float32_t turnRadius, cudaStream_t cuStream)
{
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);

    cudaStreamSynchronize(cuStream);
}

__global__ void _sweepAddLeft16(uint16_t* cell, float32_t posRes, float32_t hdgRes, float32_t turnRadius)
{
    int32_t indexX = blockIdx.x;
    int32_t indexY = threadIdx.x;
    int32_t SIZE   = 169; //=X_SIZE = Y_SIZE

    int32_t x = indexX - (169 >> 1);
    int32_t y = indexY - (169 >> 1);

    for (int32_t theta = 0; theta < 360; theta++)
    {
        float32_t rotX    = static_cast<float32_t>(x) * posRes + turnRadius * sin(deg2Rad(static_cast<float32_t>(theta) * hdgRes));
        float32_t rotY    = static_cast<float32_t>(y) * posRes + turnRadius * (1.0f - cos(deg2Rad(static_cast<float32_t>(theta) * hdgRes)));
        int32_t newIndexX = static_cast<int32_t>(floor((rotX / posRes) + 0.5f)) + (SIZE >> 1);
        int32_t newIndexY = static_cast<int32_t>(floor((rotY / posRes) + 0.5f)) + (SIZE >> 1);
        if (newIndexX <= SIZE && newIndexY <= SIZE && newIndexX >= 0 && newIndexY >= 0)
            cell[newIndexY + (newIndexX + theta * SIZE) * SIZE] += 1;
    }
}
void ParkingPlanner2::sweepAddLeft16(uint16_t* cell, float32_t posRes, float32_t hdgRes, float32_t turnRadius, cudaStream_t cuStream)
{
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);
    _sweepAddLeft16<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius);

    cudaStreamSynchronize(cuStream);
}

__global__ void _kernelLeft(uint8_t* cell, float32_t posRes, float32_t hdgRes, float32_t turnRadius, uint8_t iter)
{
    int32_t indexX = blockIdx.x;
    int32_t indexY = threadIdx.x;
    int32_t SIZE   = 169; //=X_SIZE = Y_SIZE

    int32_t x = indexX - (SIZE >> 1);
    int32_t y = indexY - (SIZE >> 1);

    uint8_t REACHABLE_BIT = (1 << 7);
    uint8_t OBSTACLE_BIT  = (1 << 6);
    uint8_t MARK_LEFT     = (1 << 4);

    uint8_t cell_value = 0;
    uint8_t sweep      = 0;

    for (int32_t theta = 0; theta < 360; theta++)
    {
        float32_t rotX    = static_cast<float32_t>(x) * posRes + turnRadius * sin(deg2Rad(static_cast<float32_t>(theta) * hdgRes));
        float32_t rotY    = static_cast<float32_t>(y) * posRes + turnRadius * (1.0f - cos(deg2Rad(static_cast<float32_t>(theta) * hdgRes)));
        int32_t newIndexX = static_cast<int32_t>(floor((rotX / posRes) + 0.5f)) + (SIZE >> 1);
        int32_t newIndexY = static_cast<int32_t>(floor((rotY / posRes) + 0.5f)) + (SIZE >> 1);
        if (newIndexX <= SIZE && newIndexY <= SIZE && newIndexX >= 0 && newIndexY >= 0)
        {
            cell_value = cell[newIndexY + (newIndexX + theta * SIZE) * SIZE];
            if (cell_value & OBSTACLE_BIT)
                sweep = 0;
            else if ((cell_value & REACHABLE_BIT) && (cell_value % (1 << 4) == iter - 1))
                sweep = 1;
            else if (!(cell_value & REACHABLE_BIT) && sweep)
                cell[newIndexY + (newIndexX + theta * SIZE) * SIZE] = (REACHABLE_BIT | MARK_LEFT | iter);
        }
    }
}

void ParkingPlanner2::kernelLeft(uint8_t* cell, float32_t posRes, float32_t hdgRes, float32_t turnRadius, uint8_t iter, cudaStream_t cuStream)
{
    _kernelLeft<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius, iter);
    cudaStreamSynchronize(cuStream);
}

__global__ void _kernelRight(uint8_t* cell, float32_t posRes, float32_t hdgRes, float32_t turnRadius, uint8_t iter)
{
    int32_t indexX = blockIdx.x;
    int32_t indexY = threadIdx.x;
    int32_t SIZE   = 169; //=X_SIZE = Y_SIZE

    int32_t x = indexX - (SIZE >> 1);
    int32_t y = indexY - (SIZE >> 1);

    uint8_t REACHABLE_BIT = (1 << 7);
    uint8_t OBSTACLE_BIT  = (1 << 6);
    uint8_t MARK_RIGHT    = (1 << 5);

    uint8_t cell_value = 0;
    uint8_t sweep      = 0;

    for (int32_t theta = 0; theta < 360; theta++)
    {
        float32_t rotX    = static_cast<float32_t>(x) * posRes - turnRadius * sin(deg2Rad(static_cast<float32_t>(theta) * hdgRes));
        float32_t rotY    = static_cast<float32_t>(y) * posRes - turnRadius * (1.0f - cos(deg2Rad(static_cast<float32_t>(theta) * hdgRes)));
        int32_t newIndexX = static_cast<int32_t>(floor((rotX / posRes) + 0.5f)) + (SIZE >> 1);
        int32_t newIndexY = static_cast<int32_t>(floor((rotY / posRes) + 0.5f)) + (SIZE >> 1);
        if (newIndexX <= SIZE && newIndexY <= SIZE && newIndexX >= 0 && newIndexY >= 0)
        {
            cell_value = cell[newIndexY + (newIndexX + theta * SIZE) * SIZE];
            if (cell_value & OBSTACLE_BIT)
                sweep = 0;
            else if ((cell_value & REACHABLE_BIT) && (cell_value % (1 << 4) == iter - 1))
                sweep = 1;
            else if (!(cell_value & REACHABLE_BIT) && sweep)
                cell[newIndexY + (newIndexX + theta * SIZE) * SIZE] = (REACHABLE_BIT | MARK_RIGHT | iter);
        }
    }
}

void ParkingPlanner2::kernelRight(uint8_t* cell, float32_t posRes, float32_t hdgRes, float32_t turnRadius, uint8_t iter, cudaStream_t cuStream)
{
    _kernelRight<<<X_SIZE, Y_SIZE, 0, cuStream>>>(cell, posRes, hdgRes, turnRadius, iter);
    cudaStreamSynchronize(cuStream);
}

CUDA_BOTH_INLINE uint32_t volCoord(uint32_t x, uint32_t y, uint32_t theta, uint32_t X_DIM, uint32_t Y_DIM)
{
    return (x + X_DIM * (y + Y_DIM * theta));
}

CUDA_BOTH inline uint32_t bitVectorRead(const uint32_t* RbI, uint32_t c) //Change pointer to span?
{
    uint32_t cm = (c >> 5);
    uint32_t cr = (c & 31);
    uint32_t Ro = ((RbI[cm] >> cr) | (RbI[cm + 1] << (32 - cr)));
    return Ro;
}

CUDA_BOTH_INLINE void bitVectorWrite(uint32_t* R, uint32_t val, uint32_t c)
{
    uint32_t cm = (c >> 5);
    uint32_t cr = (c & 31);

    R[cm]     = ((R[cm] & ((1 << cr) - 1)) | (val << cr));
    R[cm + 1] = ((R[cm + 1] & (~((1 << cr) - 1))) | (val >> (32 - cr)));
}

CUDA_BOTH_INLINE uint32_t turnIndexLeft(uint32_t x, uint32_t y, uint32_t theta,
                                        uint32_t X_DIM, uint32_t Y_DIM, float32_t POS_RES, float32_t HDG_RES, float32_t turnRadius)
{
    float32_t actualX  = (static_cast<float32_t>(x) - static_cast<float32_t>(X_DIM >> 1)) * POS_RES + turnRadius * sin(deg2Rad(static_cast<float32_t>(theta * HDG_RES)));
    float32_t actualY  = (static_cast<float32_t>(y) - static_cast<float32_t>(Y_DIM >> 1)) * POS_RES + turnRadius * (1.0f - cos(deg2Rad(static_cast<float32_t>(theta * HDG_RES))));
    uint32_t newIndexX = static_cast<uint32_t>(floor((actualX / POS_RES) + 0.5f) + (X_DIM >> 1));
    uint32_t newIndexY = static_cast<uint32_t>(floor((actualY / POS_RES) + 0.5f) + (Y_DIM >> 1));

    return volCoord(newIndexX, newIndexY, theta, X_DIM, Y_DIM);
}

CUDA_BOTH_INLINE uint32_t turnIndexRight(uint32_t x, uint32_t y, uint32_t theta,
                                         uint32_t X_DIM, uint32_t Y_DIM, float32_t POS_RES, float32_t HDG_RES, float32_t turnRadius)
{
    float32_t actualX  = (static_cast<float32_t>(x) - static_cast<float32_t>(X_DIM >> 1)) * POS_RES - turnRadius * sin(deg2Rad(static_cast<float32_t>(theta * HDG_RES)));
    float32_t actualY  = (static_cast<float32_t>(y) - static_cast<float32_t>(Y_DIM >> 1)) * POS_RES - turnRadius * (1.0f - cos(deg2Rad(static_cast<float32_t>(theta * HDG_RES))));
    uint32_t newIndexX = static_cast<uint32_t>(floor((actualX / POS_RES) + 0.5f) + (X_DIM >> 1));
    uint32_t newIndexY = static_cast<uint32_t>(floor((actualY / POS_RES) + 0.5f) + (Y_DIM >> 1));

    return volCoord(newIndexX, newIndexY, theta, X_DIM, Y_DIM);
}

__global__ void _bitSweepLeft(uint32_t* RbO, const uint32_t* Fb, const uint32_t* RbI,
                              uint32_t X_DIM, uint32_t Y_DIM, float32_t POS_RES, float32_t HDG_RES, float32_t turnRadius)
{
    uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t x     = (index % 5) * 32;
    uint32_t y     = ((index - x) / 5);

    uint32_t R = 0;
    for (uint32_t theta = 0; theta < 360; theta++)
    {
        uint32_t c1 = turnIndexLeft(x, y, theta, X_DIM, Y_DIM, POS_RES, HDG_RES, turnRadius);
        uint32_t F1 = bitVectorRead(Fb, c1);
        uint32_t R1 = bitVectorRead(RbI, c1);

        R &= F1;
        R |= R1;

        if (index % 2)
            bitVectorWrite(RbO, R, c1);
        if (!(index % 2))
            bitVectorWrite(RbO, R, c1);
    }
}

void ParkingPlanner2::bitSweepLeft(uint32_t* RbO, const uint32_t* Fb, const uint32_t* RbI, float32_t turnRadius, cudaStream_t cuStream)
{
    dim3 block(32);
    dim3 grid(((X_CELLS * Y_CELLS) + block.x - 1) / block.x);
    _bitSweepLeft<<<grid, block, 0, cuStream>>>(RbO, Fb, RbI, X_DIM, Y_DIM, POS_RES, HDG_RES, turnRadius);

    cudaStreamSynchronize(cuStream);
}

void ParkingPlanner2::bitSweepLeftCPU(uint32_t* RbO, const uint32_t* Fb, const uint32_t* RbI, float32_t turnRadius, cudaStream_t cuStream)
{
    uint32_t x = 0;
    for (uint32_t i = 0; i < X_CELLS; i++)
    {
        x = i * 32;
        for (uint32_t y = 0; y < Y_CELLS; y++)
        {
            uint32_t R = 0;
            for (uint32_t theta = 0; theta < 360; theta++)
            {
                uint32_t c1 = turnIndexLeft(x, y, theta, X_DIM, Y_DIM, POS_RES, HDG_RES, turnRadius);
                uint32_t F1 = bitVectorRead(Fb, c1);
                uint32_t R1 = bitVectorRead(RbI, c1);

                R &= F1;
                R |= R1;
                bitVectorWrite(RbO, R, c1);
            }
        }
    }
}

void ParkingPlanner2::bitSweepRightCPU(uint32_t* RbO, const uint32_t* Fb, const uint32_t* RbI, float32_t turnRadius, cudaStream_t cuStream)
{
    uint32_t x = 0;
    for (uint32_t i = 0; i < X_CELLS; i++)
    {
        x = i * 32;
        for (uint32_t y = 0; y < Y_CELLS; y++)
        {
            uint32_t R = 0;
            for (uint32_t theta = 0; theta < 360; theta++)
            {
                uint32_t c1 = turnIndexRight(x, y, theta, X_DIM, Y_DIM, POS_RES, HDG_RES, turnRadius);
                uint32_t F1 = bitVectorRead(Fb, c1);
                uint32_t R1 = bitVectorRead(RbI, c1);

                R &= F1;
                R |= R1;
                bitVectorWrite(RbO, R, c1);
            }
        }
    }
}

__global__ void _costSweepA(uint16_t* Co, uint16_t* Obs, bool left, float32_t turnRadius, int32_t halfDim, int32_t thetaDim, float32_t posRes)
{

    int32_t y  = blockIdx.x - halfDim;
    int32_t x  = threadIdx.x - halfDim;
    uint16_t c = 64250u;
    //bool check{};
    //Coord3d coord{};

    for (int32_t theta = 0; theta < thetaDim; theta++)
    {
        int32_t i = ParkingPlanner2::turnIndex(x, y, theta, left, turnRadius);

        c = max(c + static_cast<uint16_t>(turnRadius / 10.0f), Obs[i]);
        c = min(c, Co[i]);
        //Co[i] = c;
        //if(i == ParkingPlanner2::volIndex(0,0,256))
        //  check = true;
    }
    for (int32_t theta = 0; theta < thetaDim; theta++)
    {
        int32_t i = ParkingPlanner2::turnIndex(x, y, theta, left, turnRadius);

        c     = max(c + static_cast<uint16_t>(turnRadius / 10.0f), Obs[i]);
        c     = min(c, Co[i]);
        Co[i] = c;
        //coord = ParkingPlanner2::index2Coord(i,128);
        //if(check && left && (turnRadius > 15.0f))
        //printf("A:%f,%d: %d,%d,%d:%hu\n",turnRadius, left, coord.x,coord.y,theta, Co[i]);
    }
}
void ParkingPlanner2::costSweepA(uint16_t* Co, uint16_t* Obs, bool left, float32_t turnRadius, cudaStream_t cuStream)
{
    _costSweepA<<<DIM3, DIM3, 0, cuStream>>>(Co, Obs, left, turnRadius, HALF_DIM3, THETA_DIM3, POS_RES3);

    cudaDeviceSynchronize();
    DW_CHECK_CUDA_ERROR(cudaGetLastError());
    //cudaStreamSynchronize(cuStream);
}

__global__ void _costSweepB(uint16_t* Co, uint16_t* Obs, bool left, float32_t turnRadius, int32_t halfDim, int32_t thetaDim, float32_t posRes)
{
    int32_t y  = blockIdx.x - halfDim;
    int32_t x  = threadIdx.x - halfDim;
    uint16_t c = 64250u;
    //bool check{};
    //Coord3d coord{};

    for (int32_t theta = thetaDim - 1; theta >= 0; theta--)
    {
        int32_t i = ParkingPlanner2::turnIndex(x, y, theta, left, turnRadius);

        c = max(c + static_cast<uint16_t>(turnRadius / 10.0f), Obs[i]);
        c = min(c, Co[i]);
        //Co[i] = c;
        //if(i == ParkingPlanner2::volIndex(0,0,256))
        //check = true;
    }
    for (int32_t theta = thetaDim - 1; theta >= 0; theta--)
    {
        int32_t i = ParkingPlanner2::turnIndex(x, y, theta, left, turnRadius);

        c     = max(c + static_cast<uint16_t>(turnRadius / 10.0f), Obs[i]);
        c     = min(c, Co[i]);
        Co[i] = c;
        //coord = ParkingPlanner2::index2Coord(i,128);
        //if(check && left && (turnRadius > 15.0f))
        //printf("B:%f,%d: %d,%d,%d:%hu\n",turnRadius, left, coord.x,coord.y,theta, Co[i]);
    }
}

void ParkingPlanner2::costSweepB(uint16_t* Co, uint16_t* Obs, bool left, float32_t turnRadius, cudaStream_t cuStream)
{
    _costSweepB<<<DIM3, DIM3, 0, cuStream>>>(Co, Obs, left, turnRadius, HALF_DIM3, THETA_DIM3, POS_RES3);

    //cudaStreamSynchronize(cuStream);
    cudaDeviceSynchronize();
    DW_CHECK_CUDA_ERROR(cudaGetLastError());
}
__constant__ uint8_t turnIncrementR10[ParkingPlanner2::THETA_DIM3];
__constant__ uint8_t turnIncrementR20[ParkingPlanner2::THETA_DIM3];
//__constant__ uint8_t turnIncrementR10Four[ParkingPlanner2::THETA_DIM3];

void ParkingPlanner2::setTurnIncrement()
{
    cudaMemcpyToSymbol(turnIncrementR10, m_turnIncrementR10, static_cast<uint32_t>(THETA_DIM3 * sizeof(uint8_t)));
    cudaMemcpyToSymbol(turnIncrementR20, m_turnIncrementR20, static_cast<uint32_t>(THETA_DIM3 * sizeof(uint8_t)));
    //cudaMemcpyToSymbol(turnIncrementR10Four, m_turnIncrementR10Four, static_cast<uint32_t>(THETA_DIM3*sizeof(uint8_t)));
}

__global__ void _costSweepR10RightForward(uint16_t* Co, uint16_t* Obs)
{
    int32_t i  = threadIdx.x + blockIdx.x * blockDim.x + 511 * 16384;
    i          = (i & 0x7FFF80) | ((i + 1) & 127);
    uint16_t c = 64250u;
    uint8_t inc{};

    for (int32_t theta = 510; theta >= 0; theta--)
    {

        c     = max(c + 1, Obs[i]);
        c     = min(c, Co[i]);
        Co[i] = c;
        inc   = turnIncrementR10[theta];
        i     = (((i - 16384) & 0x7FC000) | ((i + 128 * (((inc & 12) >> 2) - 1)) & 16256) | ((i + ((inc & 3) - 1)) & 127));
    }
    c     = max(c + 1, Obs[i]);
    c     = min(c, Co[i]);
    Co[i] = c;
    i     = threadIdx.x + blockIdx.x * blockDim.x + 511 * 16384;
    i     = (i & 0x7FFF80) | ((i + 1) & 127);
    for (int32_t theta = 510; theta >= 0; theta--)
    {

        c     = max(c + 1, Obs[i]);
        c     = min(c, Co[i]);
        Co[i] = c;
        inc   = turnIncrementR10[theta];
        i     = (((i - 16384) & 0x7FC000) | ((i + 128 * (((inc & 12) >> 2) - 1)) & 16256) | ((i + ((inc & 3) - 1)) & 127));
    }
    c     = max(c + 1, Obs[i]);
    c     = min(c, Co[i]);
    Co[i] = c;
}

void ParkingPlanner2::costSweepR10RightForward(cudaStream_t cuStream)
{
    _costSweepR10RightForward<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data(),
                                                         m_cuGridObs.get().get().data());
    cudaStreamSynchronize(cuStream);
}

__global__ void _costSweepR20RightForward(uint16_t* Co, uint16_t* Obs)
{
    int32_t i  = threadIdx.x + blockIdx.x * blockDim.x + 511 * 16384;
    i          = (i & 0x7FFF80) | ((i + 1) & 127);
    uint16_t c = 64250u;
    uint8_t inc{};

    for (int32_t theta = 510; theta >= 0; theta--)
    {

        c     = max(c + 2, Obs[i]);
        c     = min(c, Co[i]);
        Co[i] = c;
        inc   = turnIncrementR20[theta];
        i     = (((i - 16384) & 0x7FC000) | ((i + 128 * (((inc & 12) >> 2) - 1)) & 16256) | ((i + ((inc & 3) - 1)) & 127));
    }
    c     = max(c + 2, Obs[i]);
    c     = min(c, Co[i]);
    Co[i] = c;
    i     = threadIdx.x + blockIdx.x * blockDim.x + 511 * 16384;
    i     = (i & 0x7FFF80) | ((i + 1) & 127);
    for (int32_t theta = 510; theta >= 0; theta--)
    {

        c     = max(c + 2, Obs[i]);
        c     = min(c, Co[i]);
        Co[i] = c;
        inc   = turnIncrementR20[theta];
        i     = (((i - 16384) & 0x7FC000) | ((i + 128 * (((inc & 12) >> 2) - 1)) & 16256) | ((i + ((inc & 3) - 1)) & 127));
    }
    c     = max(c + 2, Obs[i]);
    c     = min(c, Co[i]);
    Co[i] = c;
}

void ParkingPlanner2::costSweepR20RightForward(cudaStream_t cuStream)
{
    _costSweepR20RightForward<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 1 * getGridSize3(),
                                                         m_cuGridObs.get().get().data());
    cudaStreamSynchronize(cuStream);
}

__global__ void _costSweepR20LeftForward(uint16_t* Co, uint16_t* Obs)
{
    int32_t i  = threadIdx.x + blockIdx.x * blockDim.x;
    uint16_t c = 64250u;
    uint8_t inc{};

    for (int32_t theta = 0; theta < 511; theta++)
    {

        c     = max(c + 2, Obs[i]);
        c     = min(c, Co[i]);
        Co[i] = c;
        inc   = turnIncrementR20[theta];
        i     = (((i + 16384) & 0x7FC000) | ((i + 128 * (((inc & 12) >> 2) - 1)) & 16256) | ((i + ((inc & 3) - 1)) & 127));
    }
    c     = max(c + 2, Obs[i]);
    c     = min(c, Co[i]);
    Co[i] = c;
    i     = threadIdx.x + blockIdx.x * blockDim.x;
    for (int32_t theta = 0; theta < 511; theta++)
    {

        c     = max(c + 2, Obs[i]);
        c     = min(c, Co[i]);
        Co[i] = c;
        inc   = turnIncrementR20[theta];
        i     = (((i + 16384) & 0x7FC000) | ((i + 128 * (((inc & 12) >> 2) - 1)) & 16256) | ((i + ((inc & 3) - 1)) & 127));
    }
    c     = max(c + 2, Obs[i]);
    c     = min(c, Co[i]);
    Co[i] = c;
}

void ParkingPlanner2::costSweepR20LeftForward(cudaStream_t cuStream)
{
    _costSweepR20LeftForward<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 2 * getGridSize3(),
                                                        m_cuGridObs.get().get().data());
    cudaStreamSynchronize(cuStream);
}

__global__ void _costSweepR10LeftForward(uint16_t* Co, uint16_t* Obs)
{
    int32_t i  = threadIdx.x + blockIdx.x * blockDim.x;
    uint16_t c = 64250u;
    uint8_t inc{};

    for (int32_t theta = 0; theta < 511; theta++)
    {

        c     = max(c + 1, Obs[i]);
        c     = min(c, Co[i]);
        Co[i] = c;
        inc   = turnIncrementR10[theta];
        i     = (((i + 16384) & 0x7FC000) | ((i + 128 * (((inc & 12) >> 2) - 1)) & 16256) | ((i + ((inc & 3) - 1)) & 127));
    }
    c     = max(c + 1, Obs[i]);
    c     = min(c, Co[i]);
    Co[i] = c;
    i     = threadIdx.x + blockIdx.x * blockDim.x;
    for (int32_t theta = 0; theta < 511; theta++)
    {

        c     = max(c + 1, Obs[i]);
        c     = min(c, Co[i]);
        Co[i] = c;
        inc   = turnIncrementR10[theta];
        i     = (((i + 16384) & 0x7FC000) | ((i + 128 * (((inc & 12) >> 2) - 1)) & 16256) | ((i + ((inc & 3) - 1)) & 127));
    }
    c     = max(c + 1, Obs[i]);
    c     = min(c, Co[i]);
    Co[i] = c;
}

void ParkingPlanner2::costSweepR10LeftForward(cudaStream_t cuStream)
{
    _costSweepR10LeftForward<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                                        m_cuGridObs.get().get().data());
    cudaStreamSynchronize(cuStream);
}

__global__ void _costSweepR10RightReverse(uint16_t* Co, uint16_t* Obs)
{
    int32_t i  = threadIdx.x + blockIdx.x * blockDim.x;
    uint16_t c = 64250u;
    uint8_t inc{};

    for (int32_t theta = 0; theta < 511; theta++)
    {

        c     = max(c + 1, Obs[i]);
        c     = min(c, Co[i]);
        Co[i] = c;
        inc   = turnIncrementR10[theta];
        i     = (((i + 16384) & 0x7FC000) | ((i - 128 * (((inc & 12) >> 2) - 1)) & 16256) | ((i - ((inc & 3) - 1)) & 127));
    }
    c     = max(c + 1, Obs[i]);
    c     = min(c, Co[i]);
    Co[i] = c;
    i     = threadIdx.x + blockIdx.x * blockDim.x;
    for (int32_t theta = 0; theta < 511; theta++)
    {

        c     = max(c + 1, Obs[i]);
        c     = min(c, Co[i]);
        Co[i] = c;
        inc   = turnIncrementR10[theta];
        i     = (((i + 16384) & 0x7FC000) | ((i - 128 * (((inc & 12) >> 2) - 1)) & 16256) | ((i - ((inc & 3) - 1)) & 127));
    }
    c     = max(c + 1, Obs[i]);
    c     = min(c, Co[i]);
    Co[i] = c;
}

void ParkingPlanner2::costSweepR10RightReverse(cudaStream_t cuStream)
{
    _costSweepR10RightReverse<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 4 * getGridSize3(),
                                                         m_cuGridObs.get().get().data());
    cudaStreamSynchronize(cuStream);
}

__global__ void _costSweepR20RightReverse(uint16_t* Co, uint16_t* Obs)
{
    int32_t i  = threadIdx.x + blockIdx.x * blockDim.x;
    uint16_t c = 64250u;
    uint8_t inc{};

    for (int32_t theta = 0; theta < 511; theta++)
    {

        c     = max(c + 2, Obs[i]);
        c     = min(c, Co[i]);
        Co[i] = c;
        inc   = turnIncrementR20[theta];
        i     = (((i + 16384) & 0x7FC000) | ((i - 128 * (((inc & 12) >> 2) - 1)) & 16256) | ((i - ((inc & 3) - 1)) & 127));
    }
    c     = max(c + 2, Obs[i]);
    c     = min(c, Co[i]);
    Co[i] = c;
    i     = threadIdx.x + blockIdx.x * blockDim.x;
    for (int32_t theta = 0; theta < 511; theta++)
    {

        c     = max(c + 2, Obs[i]);
        c     = min(c, Co[i]);
        Co[i] = c;
        inc   = turnIncrementR20[theta];
        i     = (((i + 16384) & 0x7FC000) | ((i - 128 * (((inc & 12) >> 2) - 1)) & 16256) | ((i - ((inc & 3) - 1)) & 127));
    }
    c     = max(c + 2, Obs[i]);
    c     = min(c, Co[i]);
    Co[i] = c;
}

void ParkingPlanner2::costSweepR20RightReverse(cudaStream_t cuStream)
{
    _costSweepR20RightReverse<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 5 * getGridSize3(),
                                                         m_cuGridObs.get().get().data());
    cudaStreamSynchronize(cuStream);
}

__global__ void _costSweepR20LeftReverse(uint16_t* Co, uint16_t* Obs)
{
    int32_t i  = threadIdx.x + blockIdx.x * blockDim.x + 511 * 16384;
    i          = (i & 0x7FFF80) | ((i + 1) & 127);
    uint16_t c = 64250u;
    uint8_t inc{};

    for (int32_t theta = 510; theta >= 0; theta--)
    {

        c     = max(c + 2, Obs[i]);
        c     = min(c, Co[i]);
        Co[i] = c;
        inc   = turnIncrementR20[theta];
        i     = (((i - 16384) & 0x7FC000) | ((i - 128 * (((inc & 12) >> 2) - 1)) & 16256) | ((i - ((inc & 3) - 1)) & 127));
    }
    c     = max(c + 2, Obs[i]);
    c     = min(c, Co[i]);
    Co[i] = c;
    i     = threadIdx.x + blockIdx.x * blockDim.x + 511 * 16384;
    i     = (i & 0x7FFF80) | ((i + 1) & 127);

    for (int32_t theta = 510; theta >= 0; theta--)
    {

        c     = max(c + 2, Obs[i]);
        c     = min(c, Co[i]);
        Co[i] = c;
        inc   = turnIncrementR20[theta];
        i     = (((i - 16384) & 0x7FC000) | ((i - 128 * (((inc & 12) >> 2) - 1)) & 16256) | ((i - ((inc & 3) - 1)) & 127));
    }
    c     = max(c + 2, Obs[i]);
    c     = min(c, Co[i]);
    Co[i] = c;
}

void ParkingPlanner2::costSweepR20LeftReverse(cudaStream_t cuStream)
{
    _costSweepR20LeftReverse<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 6 * getGridSize3(),
                                                        m_cuGridObs.get().get().data());
    cudaStreamSynchronize(cuStream);
}

__global__ void _costSweepR10LeftReverse(uint16_t* Co, uint16_t* Obs)
{
    int32_t i  = threadIdx.x + blockIdx.x * blockDim.x + 511 * 16384;
    i          = (i & 0x7FFF80) | ((i + 1) & 127);
    uint16_t c = 64250u;
    uint8_t inc{};

    for (int32_t theta = 510; theta >= 0; theta--)
    {

        c     = max(c + 1, Obs[i]);
        c     = min(c, Co[i]);
        Co[i] = c;
        inc   = turnIncrementR10[theta];
        i     = (((i - 16384) & 0x7FC000) | ((i - 128 * (((inc & 12) >> 2) - 1)) & 16256) | ((i - ((inc & 3) - 1)) & 127));
    }
    c     = max(c + 1, Obs[i]);
    c     = min(c, Co[i]);
    Co[i] = c;
    i     = threadIdx.x + blockIdx.x * blockDim.x + 511 * 16384;
    i     = (i & 0x7FFF80) | ((i + 1) & 127);

    for (int32_t theta = 510; theta >= 0; theta--)
    {

        c     = max(c + 1, Obs[i]);
        c     = min(c, Co[i]);
        Co[i] = c;
        inc   = turnIncrementR10[theta];
        i     = (((i - 16384) & 0x7FC000) | ((i - 128 * (((inc & 12) >> 2) - 1)) & 16256) | ((i - ((inc & 3) - 1)) & 127));
    }
    c     = max(c + 1, Obs[i]);
    c     = min(c, Co[i]);
    Co[i] = c;
}

void ParkingPlanner2::costSweepR10LeftReverse(cudaStream_t cuStream)
{
    _costSweepR10LeftReverse<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 7 * getGridSize3(),
                                                        m_cuGridObs.get().get().data());
    cudaStreamSynchronize(cuStream);
}

void ParkingPlanner2::costSweepAll(cudaStream_t cuStream)
{
    _costSweepR10RightForward<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data(),
                                                         m_cuGridObs.get().get().data());
    _costSweepR20RightForward<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 1 * getGridSize3(),
                                                         m_cuGridObs.get().get().data());
    _costSweepR20LeftForward<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 2 * getGridSize3(),
                                                        m_cuGridObs.get().get().data());
    _costSweepR10LeftForward<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                                        m_cuGridObs.get().get().data());
    _costSweepR10RightReverse<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 4 * getGridSize3(),
                                                         m_cuGridObs.get().get().data());
    _costSweepR20RightReverse<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 5 * getGridSize3(),
                                                         m_cuGridObs.get().get().data());
    _costSweepR20LeftReverse<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 6 * getGridSize3(),
                                                        m_cuGridObs.get().get().data());
    _costSweepR10LeftReverse<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 7 * getGridSize3(),
                                                        m_cuGridObs.get().get().data());
    cudaStreamSynchronize(cuStream);
}

__constant__ uint32_t testIncrement[512] = {[0 ... 511] = 16384};
__global__ void _costUp16(uint16_t* Co, uint16_t* Obs)
{
    int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t i2{};
    int32_t i3{};
    int32_t i4{};

    uint16_t c1 = 64250u;
    uint16_t c2 = 64250u;
    uint16_t c3 = 64250u;
    uint16_t c4 = 64250u;

    uint32_t inc = 16384;
    /*uint32_t inc2 = 16384;
    uint32_t inc3 = 16384;
    uint32_t inc4 = 16384;*/
    for (int32_t theta = 0; theta < 512; theta += 4)
    {

        //c = Obs[i]- Co[i];
        /*c1 = max(c1 + 1, Obs[i]);
        c2 = max(c2 + 1, Obs[((i + inc)&0x7FFF80) | ((i+1)&127)]);
        c3 = max(c3 + 1, Obs[((i + 2*inc)&0x7FFF80) | ((i+2)&127)]);
        c4 = max(c4 + 1, Obs[((i + 3*inc)&0x7FFF80) | ((i+3)&127)]);
        
        c1 = min(c1, Co[i]);
        c2 = min(c2, Co[((i + inc)&0x7FFF80) | ((i+1)&127)]);
        c3 = min(c3, Co[((i + 2*inc)&0x7FFF80) | ((i+2)&127)]);
        c4 = min(c4, Co[((i + 3*inc)&0x7FFF80) | ((i+3)&127)]);*/
        c1 = max(c1 + 1, Obs[i]);
        c1 = min(c1, Co[i]);

        inc = turnIncrementR10[theta];
        i2  = (((i + 16384) & 0x7FC000) | ((i + 128 * (((inc & 12) >> 2) - 1)) & 16256) | ((i + ((inc & 3) - 1)) & 127));
        inc = turnIncrementR10[theta + 1];
        i3  = (((i2 + 16384) & 0x7FC000) | ((i2 + 128 * (((inc & 12) >> 2) - 1)) & 16256) | ((i2 + ((inc & 3) - 1)) & 127));
        inc = turnIncrementR10[theta + 2];
        i4  = (((i3 + 16384) & 0x7FC000) | ((i3 + 128 * (((inc & 12) >> 2) - 1)) & 16256) | ((i3 + ((inc & 3) - 1)) & 127));
        c2  = max(c1 + 1, Obs[i2]);
        c2  = min(c2, Co[i2]);

        c3 = max(c2 + 1, Obs[i3]);
        c3 = min(c3, Co[i3]);

        c4 = max(c3 + 1, Obs[i4]);
        c4 = min(c4, Co[i4]);

        Co[i]  = c1;
        Co[i2] = c2;
        Co[i3] = c3;
        Co[i4] = c4;

        inc = turnIncrementR10[theta + 3];
        i   = (((i4 + 16384) & 0x7FC000) | ((i4 + 128 * (((inc & 12) >> 2) - 1)) & 16256) | ((i4 + ((inc & 3) - 1)) & 127));
        c1  = c4;

        //Co[i] = Obs[i];
        /*Co[((i + inc)&0x7FFF80) | ((i+1)&127)] = Obs[((i + inc)&0x7FFF80) | ((i+1)&127)];
        Co[((i + 2*inc)&0x7FFF80) | ((i+2)&127)] = Obs[((i + 2*inc)&0x7FFF80) | ((i+2)&127)];
        Co[((i + 3*inc)&0x7FFF80) | ((i+3)&127)] = Obs[((i + 3*inc)&0x7FFF80) | ((i+3)&127)];*/
        //Co[i+2*inc] = Obs[i+2*inc];
        //Co[i+3*inc] = Obs[i+3*inc];

        //inc = testIncrement[theta];
        //inc = turnIncrementR10[theta];
        //inc = turnIncrementR10[theta+7];
        //i = (((i8+16384)&0x7FC000)|((i8+128*(((inc&12)>>2)-1))&16256) | ((i8+((inc&3)-1))&127));
        //i = i + inc;
        //i = (((i+16384)&0x7FC000)|((i+128*(((inc&12)>>2)-1))&16256) | ((i+((inc&3)-1))&127));
    }

    //c = Obs[i]- Co[i];
    //c = max(c + 1, Obs[i]);
    //c = min(c, Co[i]);

    //Co[i] = c;

    i = threadIdx.x + blockIdx.x * blockDim.x;

    for (int32_t theta = 0; theta < 512; theta += 4)
    {
        //c = Obs[i] - Co[i] ;
        /*c1 = max(c1 + 1, Obs[i]);
        c2 = max(c2 + 1, Obs[((i + inc)&0x7FFF80) | ((i+1)&127)]);
        c3 = max(c3 + 1, Obs[((i + 2*inc)&0x7FFF80) | ((i+2)&127)]);
        c4 = max(c4 + 1, Obs[((i + 3*inc)&0x7FFF80) | ((i+3)&127)]);
        
        c1 = min(c1, Co[i]);
        c2 = min(c2, Co[((i + inc)&0x7FFF80) | ((i+1)&127)]);
        c3 = min(c3, Co[((i + 2*inc)&0x7FFF80) | ((i+2)&127)]);
        c4 = min(c4, Co[((i + 3*inc)&0x7FFF80) | ((i+3)&127)]);*/
        c1 = max(c1 + 1, Obs[i]);
        c1 = min(c1, Co[i]);

        inc = turnIncrementR10[theta];
        i2  = (((i + 16384) & 0x7FC000) | ((i + 128 * (((inc & 12) >> 2) - 1)) & 16256) | ((i + ((inc & 3) - 1)) & 127));
        inc = turnIncrementR10[theta + 1];
        i3  = (((i2 + 16384) & 0x7FC000) | ((i2 + 128 * (((inc & 12) >> 2) - 1)) & 16256) | ((i2 + ((inc & 3) - 1)) & 127));
        inc = turnIncrementR10[theta + 2];
        i4  = (((i3 + 16384) & 0x7FC000) | ((i3 + 128 * (((inc & 12) >> 2) - 1)) & 16256) | ((i3 + ((inc & 3) - 1)) & 127));
        c2  = max(c1 + 1, Obs[i2]);
        c2  = min(c2, Co[i2]);

        c3 = max(c2 + 1, Obs[i3]);
        c3 = min(c3, Co[i3]);

        c4 = max(c3 + 1, Obs[i4]);
        c4 = min(c4, Co[i4]);

        Co[i]  = c1;
        Co[i2] = c2;
        Co[i3] = c3;
        Co[i4] = c4;

        inc = turnIncrementR10[theta + 3];
        i   = (((i4 + 16384) & 0x7FC000) | ((i4 + 128 * (((inc & 12) >> 2) - 1)) & 16256) | ((i4 + ((inc & 3) - 1)) & 127));
        c1  = c4;
        //Co[i] = Obs[i];

        /*Co[i] = c1;
        Co[((i + inc)&0x7FFF80) | ((i+1)&127)] = c2;
        Co[((i + 2*inc)&0x7FFF80) | ((i+2)&127)] = c3;
        Co[((i + 3*inc)&0x7FFF80) | ((i+3)&127)] = c4;*/

        //inc = testIncrement[theta];
        //inc = turnIncrementR10[theta];
        //i = ((i + 4*inc)&0x7FFF80) | ((i+4)&127);
        //i = ((i + inc)&0x7FFF80) | ((i+1)&127);
        //i = i + inc;
        //i = (((i+16384)&0x7FC000)|((i+128*(((inc&12)>>2)-1))&16256) | ((i+((inc&3)-1))&127));
        //inc = turnIncrementR10[theta+7];
        //i = (((i8+16384)&0x7FC000)|((i8+128*(((inc&12)>>2)-1))&16256) | ((i8+((inc&3)-1))&127));
    }
    //c = Obs[i]- Co[i];
    //c = max(c + 1, Obs[i]);
    //c = min(c, Co[i]);

    //Co[i] = Obs[i];
    //Co[i] = c;
}

void ParkingPlanner2::costUp16(cudaStream_t cuStream)
{
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    _costUp16<<<16, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(),
                                         m_cuGridObs.get().get().data());
    cudaStreamSynchronize(cuStream);
}

__global__ void _costSweep32(uint32_t* Co, uint32_t* Obs)
{
    __shared__ uint16_t outRow[128];

    uint32_t localCo{};
    uint32_t localObs{};
    uint16_t first16Co{};
    uint16_t second16Co{};
    uint16_t first16Obs{};
    uint16_t second16Obs{};
    uint16_t c1 = 64250u;
    uint16_t c2 = 64250u;
    int32_t i   = threadIdx.x + blockIdx.x * blockDim.x;

    for (int32_t theta = 0; theta < 512; theta++)
    {
        localCo                                                                   = Co[i];
        localObs                                                                  = Obs[i];
        first16Co                                                                 = localCo & 65535u;
        second16Co                                                                = (localCo >> 16);
        first16Obs                                                                = localObs & 65535u;
        second16Obs                                                               = (localObs >> 16);
        c1                                                                        = max(c1 + 1, first16Obs);
        c1                                                                        = min(c1, first16Co);
        c2                                                                        = max(c2 + 1, second16Obs);
        c2                                                                        = min(c2, second16Co);
        outRow[((threadIdx.x * 2 + (turnIncrementR10[theta] & 3) - 1) & 127)]     = c1;
        outRow[((threadIdx.x * 2 + 1 + (turnIncrementR10[theta] & 3) - 1) & 127)] = c2;
        __syncthreads();
        i = (((i + 8192) & 0x3FE000) | ((i + ((((turnIncrementR10[theta] >> 2) - 1) & 127) << 6)) & 8191));

        c1 = outRow[threadIdx.x * 2];
        c2 = outRow[threadIdx.x * 2 + 1];
        /*if(threadIdx.x ==10 && blockIdx.x == 64)
        {
            printf("%d: %d: %d\n",theta, ((threadIdx.x*2 + (turnIncrementR10[theta]&3)-1)&127), outRow[((threadIdx.x*2 + (turnIncrementR10[theta]&3)-1)&127)] );
            printf("%d: %d: %d\n",theta, ((threadIdx.x*2 + 1 + (turnIncrementR10[theta]&3)-1)&127), outRow[((threadIdx.x*2 + 1 + (turnIncrementR10[theta]&3)-1)&127)] );
            printf("\t%d, %d\n",outRow[threadIdx.x*2], outRow[threadIdx.x*2+1] );
            printf("%d:i:%d, %d, %d, %d\n\n",theta,i,(i&0x3FE000)>>13,(i&8128)>>6,i&63);
        }*/
    }

    i = threadIdx.x + blockIdx.x * blockDim.x;
    for (int32_t theta = 0; theta < 512; theta++)
    {
        localCo                                                                   = Co[i];
        localObs                                                                  = Obs[i];
        first16Co                                                                 = localCo & 65535u;
        second16Co                                                                = (localCo >> 16);
        first16Obs                                                                = localObs & 65535u;
        second16Obs                                                               = (localObs >> 16);
        c1                                                                        = max(c1 + 1, first16Obs);
        c1                                                                        = min(c1, first16Co);
        c2                                                                        = max(c2 + 1, second16Obs);
        c2                                                                        = min(c2, second16Co);
        outRow[((threadIdx.x * 2 + (turnIncrementR10[theta] & 3) - 1) & 127)]     = c1;
        outRow[((threadIdx.x * 2 + 1 + (turnIncrementR10[theta] & 3) - 1) & 127)] = c2;
        __syncthreads();
        Co[i] = (c2 << 16 | c1);
        i     = (((i + 8192) & 0x3FE000) | ((i + ((((turnIncrementR10[theta] >> 2) - 1) & 127) << 6)) & 8191));
        c1    = outRow[threadIdx.x * 2];
        c2    = outRow[threadIdx.x * 2 + 1];
    }
}

void ParkingPlanner2::costSweep32(cudaStream_t cuStream)
{
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _costSweep32<<<128, 64, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    cudaStreamSynchronize(cuStream);
}

/*__global__ void _costSweep64(uint64_t* Co, uint64_t* Obs)
{
    __shared__ uint16_t outRow[256];

    uint64_t localCo{};
    uint64_t localObs{};

    uint16_t c1 = 64250u;
    uint16_t c2 = 64250u;
    uint16_t c3 = 64250u;
    uint16_t c4 = 64250u;
    int32_t i = threadIdx.x + blockIdx.x*blockDim.x;


    for(int32_t theta = 0; theta < 512; theta++)
    {
        localCo = Co[i];
        localObs = Obs[i];

        c1 = max(c1+1,static_cast<uint16_t>(localObs&0xFFFFu));
        c1 = min(c1, static_cast<uint16_t>(localCo&0xFFFFu));
        c2 = max(c2+1,static_cast<uint16_t>(((localObs>>16)&0xFFFFu)));
        c2 = min(c2, static_cast<uint16_t>(((localCo>>16)&0xFFFFu)));
        c3 = max(c3+1,static_cast<uint16_t>(((localObs>>32)&0xFFFFu)));
        c3 = min(c3, static_cast<uint16_t>(((localCo>>32)&0xFFFFu)));
        c4 = max(c4+1,static_cast<uint16_t>((localObs>>48)));
        c4 = min(c4, static_cast<uint16_t>((localCo>>48)));
        outRow[((threadIdx.x>>5)<<7) + ((threadIdx.x*4 + (turnIncrementR10[theta]&3)-1)&127)] = c1;
        outRow[((threadIdx.x>>5)<<7) + ((threadIdx.x*4 + 1 + (turnIncrementR10[theta]&3)-1)&127)] = c2;
        outRow[((threadIdx.x>>5)<<7) + ((threadIdx.x*4 + 2 + (turnIncrementR10[theta]&3)-1)&127)] = c3;
        outRow[((threadIdx.x>>5)<<7) + ((threadIdx.x*4 + 3 + (turnIncrementR10[theta]&3)-1)&127)] = c4;

        i = (((i + 4096)&0x1FF000) | ((i + ((((turnIncrementR10[theta]>>2)-1)&127)<<5))&4095));
        outRow[((threadIdx.x>>5)<<7) + ((threadIdx.x*4 - (turnIncrementR10[theta]&3)+1)&127)] = c1;
        outRow[((threadIdx.x>>5)<<7) + ((threadIdx.x*4 + 1 - (turnIncrementR10[theta]&3)+1)&127)] = c2;
        outRow[((threadIdx.x>>5)<<7) + ((threadIdx.x*4 + 2 - (turnIncrementR10[theta]&3)+1)&127)] = c3;
        outRow[((threadIdx.x>>5)<<7) + ((threadIdx.x*4 + 3 - (turnIncrementR10[theta]&3)+1)&127)] = c4;

        i = (((i + 4096)&0x1FF000) | ((i - (((turnIncrementR10[theta]>>2)-1)<<5))&4064) | (i&31));

        c1 = outRow[threadIdx.x*4];
        c2 = outRow[threadIdx.x*4+1];
        c3 = outRow[threadIdx.x*4+2];
        c4 = outRow[threadIdx.x*4+3];
        if(threadIdx.x ==10 && blockIdx.x == 64)
        {
            printf("%d: %d: %d\n",theta, ((threadIdx.x*2 + (turnIncrementR10[theta]&3)-1)&127), outRow[((threadIdx.x*2 + (turnIncrementR10[theta]&3)-1)&127)] );
            printf("%d: %d: %d\n",theta, ((threadIdx.x*2 + 1 + (turnIncrementR10[theta]&3)-1)&127), outRow[((threadIdx.x*2 + 1 + (turnIncrementR10[theta]&3)-1)&127)] );
            printf("\t%d, %d\n",outRow[threadIdx.x*2], outRow[threadIdx.x*2+1] );
            printf("%d:i:%d, %d, %d, %d\n\n",theta,i,(i&0x3FE000)>>13,(i&8128)>>6,i&63);
        }
    }

    i = threadIdx.x + blockIdx.x*blockDim.x;
    for(int32_t theta = 0; theta < 512; theta++)
    {
        localCo = Co[i];
        localObs = Obs[i];


        c1 = max(c1+1,static_cast<uint16_t>(localObs&0xFFFFu));
        c1 = min(c1, static_cast<uint16_t>(localCo&0xFFFFu));
        c2 = max(c2+1,static_cast<uint16_t>(((localObs>>16)&0xFFFFu)));
        c2 = min(c2, static_cast<uint16_t>(((localCo>>16)&0xFFFFu)));
        c3 = max(c3+1,static_cast<uint16_t>(((localObs>>32)&0xFFFFu)));
        c3 = min(c3, static_cast<uint16_t>(((localCo>>32)&0xFFFFu)));
        c4 = max(c4+1,static_cast<uint16_t>((localObs>>48)));
        c4 = min(c4, static_cast<uint16_t>((localCo>>48)));

        outRow[((threadIdx.x>>5)<<7) + ((threadIdx.x*4 + (turnIncrementR10[theta]&3)-1)&127)] = c1;
        outRow[((threadIdx.x>>5)<<7) + ((threadIdx.x*4 + 1 + (turnIncrementR10[theta]&3)-1)&127)] = c2;
        outRow[((threadIdx.x>>5)<<7) + ((threadIdx.x*4 + 2 + (turnIncrementR10[theta]&3)-1)&127)] = c3;
        outRow[((threadIdx.x>>5)<<7) + ((threadIdx.x*4 + 3 + (turnIncrementR10[theta]&3)-1)&127)] = c4;

        outRow[((threadIdx.x>>5)<<7) + ((threadIdx.x*4 - (turnIncrementR10[theta]&3)+1)&127)] = c1;
        outRow[((threadIdx.x>>5)<<7) + ((threadIdx.x*4 + 1 - (turnIncrementR10[theta]&3)+1)&127)] = c2;
        outRow[((threadIdx.x>>5)<<7) + ((threadIdx.x*4 + 2 - (turnIncrementR10[theta]&3)+1)&127)] = c3;
        outRow[((threadIdx.x>>5)<<7) + ((threadIdx.x*4 + 3 - (turnIncrementR10[theta]&3)+1)&127)] = c4;

        Co[i] = (static_cast<uint64_t>(c4)<<48 | static_cast<uint64_t>(c3)<<32 | static_cast<uint64_t>(c2)<<16 | c1);
        i = (((i + 4096)&0x1FF000) | ((i + ((((turnIncrementR10[theta]>>2)-1)&127)<<5))&4095));
        //i = (((i + 4096)&0x1FF000) | ((i - (((turnIncrementR10[theta]>>2)-1)<<5))&4064) | (i&31));

        c1 = outRow[threadIdx.x*4];
        c2 = outRow[threadIdx.x*4+1];
        c3 = outRow[threadIdx.x*4+2];
        c4 = outRow[threadIdx.x*4+3];
    }
}*/

__global__ void _costSweep64(uint64_t* Co, uint64_t* Obs)
{
    __shared__ uint16_t outRow[256];

    uint64_t localCo{};
    uint64_t localObs{};

    uint16_t c1 = 64250u;
    uint16_t c2 = 64250u;
    uint16_t c3 = 64250u;
    uint16_t c4 = 64250u;
    int32_t i   = threadIdx.x + blockIdx.x * blockDim.x;

    for (int32_t theta = 0; theta < 511; theta++)
    {
        localCo  = Co[i];
        localObs = Obs[i];

        c1                                                                                                    = max(c1 + 1, static_cast<uint16_t>(localObs & 0xFFFFu));
        c1                                                                                                    = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
        c2                                                                                                    = max(c2 + 1, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
        c2                                                                                                    = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
        c3                                                                                                    = max(c3 + 1, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
        c3                                                                                                    = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
        c4                                                                                                    = max(c4 + 1, static_cast<uint16_t>((localObs >> 48)));
        c4                                                                                                    = min(c4, static_cast<uint16_t>((localCo >> 48)));
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 - (turnIncrementR10[theta] & 3) + 1) & 127)]     = c1;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 1 - (turnIncrementR10[theta] & 3) + 1) & 127)] = c2;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 2 - (turnIncrementR10[theta] & 3) + 1) & 127)] = c3;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 3 - (turnIncrementR10[theta] & 3) + 1) & 127)] = c4;

        i = (((i + 4096) & 0x1FF000) | ((i - (((turnIncrementR10[theta] >> 2) - 1) << 5)) & 4064) | (i & 31));

        c1 = outRow[threadIdx.x * 4];
        c2 = outRow[threadIdx.x * 4 + 1];
        c3 = outRow[threadIdx.x * 4 + 2];
        c4 = outRow[threadIdx.x * 4 + 3];
    }
    localCo                                                                                               = Co[i];
    localObs                                                                                              = Obs[i];
    int32_t theta                                                                                         = 511;
    c1                                                                                                    = max(c1 + 1, static_cast<uint16_t>(localObs & 0xFFFFu));
    c1                                                                                                    = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
    c2                                                                                                    = max(c2 + 1, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
    c2                                                                                                    = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
    c3                                                                                                    = max(c3 + 1, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
    c3                                                                                                    = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
    c4                                                                                                    = max(c4 + 1, static_cast<uint16_t>((localObs >> 48)));
    c4                                                                                                    = min(c4, static_cast<uint16_t>((localCo >> 48)));
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 - (turnIncrementR10[theta] & 3) + 1) & 127)]     = c1;
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 1 - (turnIncrementR10[theta] & 3) + 1) & 127)] = c2;
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 2 - (turnIncrementR10[theta] & 3) + 1) & 127)] = c3;
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 3 - (turnIncrementR10[theta] & 3) + 1) & 127)] = c4;

    i = threadIdx.x + blockIdx.x * blockDim.x;

    c1 = outRow[threadIdx.x * 4];
    c2 = outRow[threadIdx.x * 4 + 1];
    c3 = outRow[threadIdx.x * 4 + 2];
    c4 = outRow[threadIdx.x * 4 + 3];

    for (int32_t theta = 0; theta < 511; theta++)
    {
        localCo  = Co[i];
        localObs = Obs[i];

        c1                                                                                                    = max(c1 + 1, static_cast<uint16_t>(localObs & 0xFFFFu));
        c1                                                                                                    = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
        c2                                                                                                    = max(c2 + 1, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
        c2                                                                                                    = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
        c3                                                                                                    = max(c3 + 1, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
        c3                                                                                                    = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
        c4                                                                                                    = max(c4 + 1, static_cast<uint16_t>((localObs >> 48)));
        c4                                                                                                    = min(c4, static_cast<uint16_t>((localCo >> 48)));
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 - (turnIncrementR10[theta] & 3) + 1) & 127)]     = c1;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 1 - (turnIncrementR10[theta] & 3) + 1) & 127)] = c2;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 2 - (turnIncrementR10[theta] & 3) + 1) & 127)] = c3;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 3 - (turnIncrementR10[theta] & 3) + 1) & 127)] = c4;

        Co[i] = (static_cast<uint64_t>(c4) << 48 | static_cast<uint64_t>(c3) << 32 | static_cast<uint64_t>(c2) << 16 | c1);
        i     = (((i + 4096) & 0x1FF000) | ((i - ((((turnIncrementR10[theta] >> 2) - 1) & 127) << 5)) & 4064) | (i & 31));

        c1 = outRow[threadIdx.x * 4];
        c2 = outRow[threadIdx.x * 4 + 1];
        c3 = outRow[threadIdx.x * 4 + 2];
        c4 = outRow[threadIdx.x * 4 + 3];
    }
    localCo  = Co[i];
    localObs = Obs[i];

    c1 = max(c1 + 1, static_cast<uint16_t>(localObs & 0xFFFFu));
    c1 = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
    c2 = max(c2 + 1, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
    c2 = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
    c3 = max(c3 + 1, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
    c3 = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
    c4 = max(c4 + 1, static_cast<uint16_t>((localObs >> 48)));
    c4 = min(c4, static_cast<uint16_t>((localCo >> 48)));

    Co[i] = (static_cast<uint64_t>(c4) << 48 | static_cast<uint64_t>(c3) << 32 | static_cast<uint64_t>(c2) << 16 | c1);
}

void ParkingPlanner2::costSweep64(cudaStream_t cuStream)
{
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweep64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                          reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));

    cudaStreamSynchronize(cuStream);
}

__global__ void _costSweepR10RightForward64(uint64_t* Co, uint64_t* Obs)
{
    __shared__ uint16_t outRow[256];

    uint64_t localCo{};
    uint64_t localObs{};

    uint16_t c1 = 64250u;
    uint16_t c2 = 64250u;
    uint16_t c3 = 64250u;
    uint16_t c4 = 64250u;
    int32_t i   = threadIdx.x + blockIdx.x * blockDim.x + 511 * blockDim.x * gridDim.x;

    for (int32_t theta = 510; theta >= 0; theta--)
    {
        localCo  = Co[i];
        localObs = Obs[i];

        c1                                                                                                    = max(c1 + 1, static_cast<uint16_t>(localObs & 0xFFFFu));
        c1                                                                                                    = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
        c2                                                                                                    = max(c2 + 1, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
        c2                                                                                                    = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
        c3                                                                                                    = max(c3 + 1, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
        c3                                                                                                    = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
        c4                                                                                                    = max(c4 + 1, static_cast<uint16_t>((localObs >> 48)));
        c4                                                                                                    = min(c4, static_cast<uint16_t>((localCo >> 48)));
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + (turnIncrementR10[theta] & 3) - 1) & 127)]     = c1;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 1 + (turnIncrementR10[theta] & 3) - 1) & 127)] = c2;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 2 + (turnIncrementR10[theta] & 3) - 1) & 127)] = c3;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 3 + (turnIncrementR10[theta] & 3) - 1) & 127)] = c4;

        i = (((i - 4096) & 0x1FF000) | ((i + ((((turnIncrementR10[theta] >> 2) - 1) & 127) << 5)) & 4095));

        c1 = outRow[threadIdx.x * 4];
        c2 = outRow[threadIdx.x * 4 + 1];
        c3 = outRow[threadIdx.x * 4 + 2];
        c4 = outRow[threadIdx.x * 4 + 3];
    }
    localCo                                                                                               = Co[i];
    localObs                                                                                              = Obs[i];
    int32_t theta                                                                                         = 511;
    c1                                                                                                    = max(c1 + 1, static_cast<uint16_t>(localObs & 0xFFFFu));
    c1                                                                                                    = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
    c2                                                                                                    = max(c2 + 1, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
    c2                                                                                                    = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
    c3                                                                                                    = max(c3 + 1, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
    c3                                                                                                    = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
    c4                                                                                                    = max(c4 + 1, static_cast<uint16_t>((localObs >> 48)));
    c4                                                                                                    = min(c4, static_cast<uint16_t>((localCo >> 48)));
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + (turnIncrementR10[theta] & 3) - 1) & 127)]     = c1;
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 1 + (turnIncrementR10[theta] & 3) - 1) & 127)] = c2;
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 2 + (turnIncrementR10[theta] & 3) - 1) & 127)] = c3;
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 3 + (turnIncrementR10[theta] & 3) - 1) & 127)] = c4;

    i = threadIdx.x + blockIdx.x * blockDim.x + 511 * blockDim.x * gridDim.x;

    c1 = outRow[threadIdx.x * 4];
    c2 = outRow[threadIdx.x * 4 + 1];
    c3 = outRow[threadIdx.x * 4 + 2];
    c4 = outRow[threadIdx.x * 4 + 3];
    for (int32_t theta = 510; theta >= 0; theta--)
    {
        localCo  = Co[i];
        localObs = Obs[i];

        c1 = max(c1 + 1, static_cast<uint16_t>(localObs & 0xFFFFu));
        c1 = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
        c2 = max(c2 + 1, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
        c2 = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
        c3 = max(c3 + 1, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
        c3 = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
        c4 = max(c4 + 1, static_cast<uint16_t>((localObs >> 48)));
        c4 = min(c4, static_cast<uint16_t>((localCo >> 48)));

        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + (turnIncrementR10[theta] & 3) - 1) & 127)]     = c1;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 1 + (turnIncrementR10[theta] & 3) - 1) & 127)] = c2;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 2 + (turnIncrementR10[theta] & 3) - 1) & 127)] = c3;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 3 + (turnIncrementR10[theta] & 3) - 1) & 127)] = c4;

        Co[i] = (static_cast<uint64_t>(c4) << 48 | static_cast<uint64_t>(c3) << 32 | static_cast<uint64_t>(c2) << 16 | c1);
        i     = (((i - 4096) & 0x1FF000) | ((i + ((((turnIncrementR10[theta] >> 2) - 1) & 127) << 5)) & 4095));

        c1 = outRow[threadIdx.x * 4];
        c2 = outRow[threadIdx.x * 4 + 1];
        c3 = outRow[threadIdx.x * 4 + 2];
        c4 = outRow[threadIdx.x * 4 + 3];
    }
    localCo  = Co[i];
    localObs = Obs[i];

    c1 = max(c1 + 1, static_cast<uint16_t>(localObs & 0xFFFFu));
    c1 = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
    c2 = max(c2 + 1, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
    c2 = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
    c3 = max(c3 + 1, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
    c3 = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
    c4 = max(c4 + 1, static_cast<uint16_t>((localObs >> 48)));
    c4 = min(c4, static_cast<uint16_t>((localCo >> 48)));

    Co[i] = (static_cast<uint64_t>(c4) << 48 | static_cast<uint64_t>(c3) << 32 | static_cast<uint64_t>(c2) << 16 | c1);
}

__global__ void _costSweepR20RightForward64(uint64_t* Co, uint64_t* Obs)
{
    __shared__ uint16_t outRow[256];

    uint64_t localCo{};
    uint64_t localObs{};

    uint16_t c1 = 64250u;
    uint16_t c2 = 64250u;
    uint16_t c3 = 64250u;
    uint16_t c4 = 64250u;
    int32_t i   = threadIdx.x + blockIdx.x * blockDim.x + 511 * 4096;

    for (int32_t theta = 510; theta >= 0; theta--)
    {
        localCo  = Co[i];
        localObs = Obs[i];

        c1                                                                                                    = max(c1 + 2, static_cast<uint16_t>(localObs & 0xFFFFu));
        c1                                                                                                    = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
        c2                                                                                                    = max(c2 + 2, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
        c2                                                                                                    = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
        c3                                                                                                    = max(c3 + 2, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
        c3                                                                                                    = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
        c4                                                                                                    = max(c4 + 2, static_cast<uint16_t>((localObs >> 48)));
        c4                                                                                                    = min(c4, static_cast<uint16_t>((localCo >> 48)));
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + (turnIncrementR20[theta] & 3) - 1) & 127)]     = c1;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 1 + (turnIncrementR20[theta] & 3) - 1) & 127)] = c2;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 2 + (turnIncrementR20[theta] & 3) - 1) & 127)] = c3;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 3 + (turnIncrementR20[theta] & 3) - 1) & 127)] = c4;

        i = (((i - 4096) & 0x1FF000) | ((i + ((((turnIncrementR20[theta] >> 2) - 1) & 127) << 5)) & 4095));

        c1 = outRow[threadIdx.x * 4];
        c2 = outRow[threadIdx.x * 4 + 1];
        c3 = outRow[threadIdx.x * 4 + 2];
        c4 = outRow[threadIdx.x * 4 + 3];
    }
    localCo                                                                                               = Co[i];
    localObs                                                                                              = Obs[i];
    int32_t theta                                                                                         = 511;
    c1                                                                                                    = max(c1 + 2, static_cast<uint16_t>(localObs & 0xFFFFu));
    c1                                                                                                    = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
    c2                                                                                                    = max(c2 + 2, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
    c2                                                                                                    = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
    c3                                                                                                    = max(c3 + 2, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
    c3                                                                                                    = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
    c4                                                                                                    = max(c4 + 2, static_cast<uint16_t>((localObs >> 48)));
    c4                                                                                                    = min(c4, static_cast<uint16_t>((localCo >> 48)));
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + (turnIncrementR20[theta] & 3) - 1) & 127)]     = c1;
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 1 + (turnIncrementR20[theta] & 3) - 1) & 127)] = c2;
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 2 + (turnIncrementR20[theta] & 3) - 1) & 127)] = c3;
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 3 + (turnIncrementR20[theta] & 3) - 1) & 127)] = c4;

    i = threadIdx.x + blockIdx.x * blockDim.x + 511 * 4096;

    c1 = outRow[threadIdx.x * 4];
    c2 = outRow[threadIdx.x * 4 + 1];
    c3 = outRow[threadIdx.x * 4 + 2];
    c4 = outRow[threadIdx.x * 4 + 3];

    for (int32_t theta = 510; theta >= 0; theta--)
    {
        localCo  = Co[i];
        localObs = Obs[i];

        c1 = max(c1 + 2, static_cast<uint16_t>(localObs & 0xFFFFu));
        c1 = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
        c2 = max(c2 + 2, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
        c2 = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
        c3 = max(c3 + 2, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
        c3 = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
        c4 = max(c4 + 2, static_cast<uint16_t>((localObs >> 48)));
        c4 = min(c4, static_cast<uint16_t>((localCo >> 48)));

        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + (turnIncrementR20[theta] & 3) - 1) & 127)]     = c1;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 1 + (turnIncrementR20[theta] & 3) - 1) & 127)] = c2;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 2 + (turnIncrementR20[theta] & 3) - 1) & 127)] = c3;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 3 + (turnIncrementR20[theta] & 3) - 1) & 127)] = c4;

        Co[i] = (static_cast<uint64_t>(c4) << 48 | static_cast<uint64_t>(c3) << 32 | static_cast<uint64_t>(c2) << 16 | c1);
        i     = (((i - 4096) & 0x1FF000) | ((i + ((((turnIncrementR20[theta] >> 2) - 1) & 127) << 5)) & 4095));

        c1 = outRow[threadIdx.x * 4];
        c2 = outRow[threadIdx.x * 4 + 1];
        c3 = outRow[threadIdx.x * 4 + 2];
        c4 = outRow[threadIdx.x * 4 + 3];
    }
    localCo  = Co[i];
    localObs = Obs[i];

    c1 = max(c1 + 2, static_cast<uint16_t>(localObs & 0xFFFFu));
    c1 = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
    c2 = max(c2 + 2, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
    c2 = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
    c3 = max(c3 + 2, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
    c3 = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
    c4 = max(c4 + 2, static_cast<uint16_t>((localObs >> 48)));
    c4 = min(c4, static_cast<uint16_t>((localCo >> 48)));

    Co[i] = (static_cast<uint64_t>(c4) << 48 | static_cast<uint64_t>(c3) << 32 | static_cast<uint64_t>(c2) << 16 | c1);
}

__global__ void _costSweepR20LeftForward64(uint64_t* Co, uint64_t* Obs)
{
    __shared__ uint16_t outRow[256];

    uint64_t localCo{};
    uint64_t localObs{};

    uint16_t c1 = 64250u;
    uint16_t c2 = 64250u;
    uint16_t c3 = 64250u;
    uint16_t c4 = 64250u;
    int32_t i   = threadIdx.x + blockIdx.x * blockDim.x;

    for (int32_t theta = 0; theta < 511; theta++)
    {
        localCo  = Co[i];
        localObs = Obs[i];

        c1                                                                                                    = max(c1 + 2, static_cast<uint16_t>(localObs & 0xFFFFu));
        c1                                                                                                    = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
        c2                                                                                                    = max(c2 + 2, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
        c2                                                                                                    = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
        c3                                                                                                    = max(c3 + 2, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
        c3                                                                                                    = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
        c4                                                                                                    = max(c4 + 2, static_cast<uint16_t>((localObs >> 48)));
        c4                                                                                                    = min(c4, static_cast<uint16_t>((localCo >> 48)));
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + (turnIncrementR20[theta] & 3) - 1) & 127)]     = c1;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 1 + (turnIncrementR20[theta] & 3) - 1) & 127)] = c2;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 2 + (turnIncrementR20[theta] & 3) - 1) & 127)] = c3;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 3 + (turnIncrementR20[theta] & 3) - 1) & 127)] = c4;

        i = (((i + 4096) & 0x1FF000) | ((i + ((((turnIncrementR20[theta] >> 2) - 1) & 127) << 5)) & 4095));

        c1 = outRow[threadIdx.x * 4];
        c2 = outRow[threadIdx.x * 4 + 1];
        c3 = outRow[threadIdx.x * 4 + 2];
        c4 = outRow[threadIdx.x * 4 + 3];
    }
    localCo                                                                                               = Co[i];
    localObs                                                                                              = Obs[i];
    int32_t theta                                                                                         = 511;
    c1                                                                                                    = max(c1 + 2, static_cast<uint16_t>(localObs & 0xFFFFu));
    c1                                                                                                    = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
    c2                                                                                                    = max(c2 + 2, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
    c2                                                                                                    = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
    c3                                                                                                    = max(c3 + 2, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
    c3                                                                                                    = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
    c4                                                                                                    = max(c4 + 2, static_cast<uint16_t>((localObs >> 48)));
    c4                                                                                                    = min(c4, static_cast<uint16_t>((localCo >> 48)));
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + (turnIncrementR20[theta] & 3) - 1) & 127)]     = c1;
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 1 + (turnIncrementR20[theta] & 3) - 1) & 127)] = c2;
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 2 + (turnIncrementR20[theta] & 3) - 1) & 127)] = c3;
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 3 + (turnIncrementR20[theta] & 3) - 1) & 127)] = c4;

    i = threadIdx.x + blockIdx.x * blockDim.x;

    c1 = outRow[threadIdx.x * 4];
    c2 = outRow[threadIdx.x * 4 + 1];
    c3 = outRow[threadIdx.x * 4 + 2];
    c4 = outRow[threadIdx.x * 4 + 3];

    for (int32_t theta = 0; theta < 511; theta++)
    {
        localCo  = Co[i];
        localObs = Obs[i];

        c1                                                                                                    = max(c1 + 2, static_cast<uint16_t>(localObs & 0xFFFFu));
        c1                                                                                                    = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
        c2                                                                                                    = max(c2 + 2, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
        c2                                                                                                    = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
        c3                                                                                                    = max(c3 + 2, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
        c3                                                                                                    = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
        c4                                                                                                    = max(c4 + 2, static_cast<uint16_t>((localObs >> 48)));
        c4                                                                                                    = min(c4, static_cast<uint16_t>((localCo >> 48)));
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + (turnIncrementR20[theta] & 3) - 1) & 127)]     = c1;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 1 + (turnIncrementR20[theta] & 3) - 1) & 127)] = c2;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 2 + (turnIncrementR20[theta] & 3) - 1) & 127)] = c3;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 3 + (turnIncrementR20[theta] & 3) - 1) & 127)] = c4;

        Co[i] = (static_cast<uint64_t>(c4) << 48 | static_cast<uint64_t>(c3) << 32 | static_cast<uint64_t>(c2) << 16 | c1);
        i     = (((i + 4096) & 0x1FF000) | ((i + ((((turnIncrementR20[theta] >> 2) - 1) & 127) << 5)) & 4095));

        c1 = outRow[threadIdx.x * 4];
        c2 = outRow[threadIdx.x * 4 + 1];
        c3 = outRow[threadIdx.x * 4 + 2];
        c4 = outRow[threadIdx.x * 4 + 3];
    }
    localCo  = Co[i];
    localObs = Obs[i];

    c1 = max(c1 + 2, static_cast<uint16_t>(localObs & 0xFFFFu));
    c1 = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
    c2 = max(c2 + 2, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
    c2 = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
    c3 = max(c3 + 2, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
    c3 = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
    c4 = max(c4 + 2, static_cast<uint16_t>((localObs >> 48)));
    c4 = min(c4, static_cast<uint16_t>((localCo >> 48)));

    Co[i] = (static_cast<uint64_t>(c4) << 48 | static_cast<uint64_t>(c3) << 32 | static_cast<uint64_t>(c2) << 16 | c1);
}

__global__ void _costSweepR10LeftForward64(uint64_t* Co, uint64_t* Obs)
{
    __shared__ uint16_t outRow[256];

    uint64_t localCo{};
    uint64_t localObs{};

    uint16_t c1 = 64250u;
    uint16_t c2 = 64250u;
    uint16_t c3 = 64250u;
    uint16_t c4 = 64250u;
    int32_t i   = threadIdx.x + blockIdx.x * blockDim.x;

    for (int32_t theta = 0; theta < 511; theta++)
    {
        localCo  = Co[i];
        localObs = Obs[i];

        c1                                                                                                    = max(c1 + 1, static_cast<uint16_t>(localObs & 0xFFFFu));
        c1                                                                                                    = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
        c2                                                                                                    = max(c2 + 1, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
        c2                                                                                                    = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
        c3                                                                                                    = max(c3 + 1, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
        c3                                                                                                    = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
        c4                                                                                                    = max(c4 + 1, static_cast<uint16_t>((localObs >> 48)));
        c4                                                                                                    = min(c4, static_cast<uint16_t>((localCo >> 48)));
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + (turnIncrementR10[theta] & 3) - 1) & 127)]     = c1;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 1 + (turnIncrementR10[theta] & 3) - 1) & 127)] = c2;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 2 + (turnIncrementR10[theta] & 3) - 1) & 127)] = c3;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 3 + (turnIncrementR10[theta] & 3) - 1) & 127)] = c4;

        i = (((i + 4096) & 0x1FF000) | ((i + ((((turnIncrementR10[theta] >> 2) - 1) & 127) << 5)) & 4095));

        c1 = outRow[threadIdx.x * 4];
        c2 = outRow[threadIdx.x * 4 + 1];
        c3 = outRow[threadIdx.x * 4 + 2];
        c4 = outRow[threadIdx.x * 4 + 3];
    }
    localCo                                                                                               = Co[i];
    localObs                                                                                              = Obs[i];
    int32_t theta                                                                                         = 511;
    c1                                                                                                    = max(c1 + 1, static_cast<uint16_t>(localObs & 0xFFFFu));
    c1                                                                                                    = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
    c2                                                                                                    = max(c2 + 1, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
    c2                                                                                                    = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
    c3                                                                                                    = max(c3 + 1, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
    c3                                                                                                    = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
    c4                                                                                                    = max(c4 + 1, static_cast<uint16_t>((localObs >> 48)));
    c4                                                                                                    = min(c4, static_cast<uint16_t>((localCo >> 48)));
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + (turnIncrementR10[theta] & 3) - 1) & 127)]     = c1;
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 1 + (turnIncrementR10[theta] & 3) - 1) & 127)] = c2;
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 2 + (turnIncrementR10[theta] & 3) - 1) & 127)] = c3;
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 3 + (turnIncrementR10[theta] & 3) - 1) & 127)] = c4;

    i = threadIdx.x + blockIdx.x * blockDim.x;

    c1 = outRow[threadIdx.x * 4];
    c2 = outRow[threadIdx.x * 4 + 1];
    c3 = outRow[threadIdx.x * 4 + 2];
    c4 = outRow[threadIdx.x * 4 + 3];
    for (int32_t theta = 0; theta < 511; theta++)
    {
        localCo  = Co[i];
        localObs = Obs[i];

        c1                                                                                                    = max(c1 + 1, static_cast<uint16_t>(localObs & 0xFFFFu));
        c1                                                                                                    = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
        c2                                                                                                    = max(c2 + 1, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
        c2                                                                                                    = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
        c3                                                                                                    = max(c3 + 1, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
        c3                                                                                                    = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
        c4                                                                                                    = max(c4 + 1, static_cast<uint16_t>((localObs >> 48)));
        c4                                                                                                    = min(c4, static_cast<uint16_t>((localCo >> 48)));
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + (turnIncrementR10[theta] & 3) - 1) & 127)]     = c1;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 1 + (turnIncrementR10[theta] & 3) - 1) & 127)] = c2;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 2 + (turnIncrementR10[theta] & 3) - 1) & 127)] = c3;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 3 + (turnIncrementR10[theta] & 3) - 1) & 127)] = c4;

        Co[i] = (static_cast<uint64_t>(c4) << 48 | static_cast<uint64_t>(c3) << 32 | static_cast<uint64_t>(c2) << 16 | c1);
        i     = (((i + 4096) & 0x1FF000) | ((i + ((((turnIncrementR10[theta] >> 2) - 1) & 127) << 5)) & 4095));

        c1 = outRow[threadIdx.x * 4];
        c2 = outRow[threadIdx.x * 4 + 1];
        c3 = outRow[threadIdx.x * 4 + 2];
        c4 = outRow[threadIdx.x * 4 + 3];
    }
    localCo  = Co[i];
    localObs = Obs[i];

    c1 = max(c1 + 1, static_cast<uint16_t>(localObs & 0xFFFFu));
    c1 = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
    c2 = max(c2 + 1, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
    c2 = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
    c3 = max(c3 + 1, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
    c3 = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
    c4 = max(c4 + 1, static_cast<uint16_t>((localObs >> 48)));
    c4 = min(c4, static_cast<uint16_t>((localCo >> 48)));

    Co[i] = (static_cast<uint64_t>(c4) << 48 | static_cast<uint64_t>(c3) << 32 | static_cast<uint64_t>(c2) << 16 | c1);
}

__global__ void _costSweepR10RightReverse64(uint64_t* Co, uint64_t* Obs)
{
    __shared__ uint16_t outRow[256];

    uint64_t localCo{};
    uint64_t localObs{};

    uint16_t c1 = 64250u;
    uint16_t c2 = 64250u;
    uint16_t c3 = 64250u;
    uint16_t c4 = 64250u;
    int32_t i   = threadIdx.x + blockIdx.x * blockDim.x;

    for (int32_t theta = 0; theta < 511; theta++)
    {
        localCo  = Co[i];
        localObs = Obs[i];

        c1                                                                                                    = max(c1 + 1, static_cast<uint16_t>(localObs & 0xFFFFu));
        c1                                                                                                    = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
        c2                                                                                                    = max(c2 + 1, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
        c2                                                                                                    = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
        c3                                                                                                    = max(c3 + 1, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
        c3                                                                                                    = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
        c4                                                                                                    = max(c4 + 1, static_cast<uint16_t>((localObs >> 48)));
        c4                                                                                                    = min(c4, static_cast<uint16_t>((localCo >> 48)));
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 - (turnIncrementR10[theta] & 3) + 1) & 127)]     = c1;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 1 - (turnIncrementR10[theta] & 3) + 1) & 127)] = c2;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 2 - (turnIncrementR10[theta] & 3) + 1) & 127)] = c3;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 3 - (turnIncrementR10[theta] & 3) + 1) & 127)] = c4;

        i = (((i + 4096) & 0x1FF000) | ((i - (((turnIncrementR10[theta] >> 2) - 1) << 5)) & 4064) | (i & 31));

        c1 = outRow[threadIdx.x * 4];
        c2 = outRow[threadIdx.x * 4 + 1];
        c3 = outRow[threadIdx.x * 4 + 2];
        c4 = outRow[threadIdx.x * 4 + 3];
    }
    localCo                                                                                               = Co[i];
    localObs                                                                                              = Obs[i];
    int32_t theta                                                                                         = 511;
    c1                                                                                                    = max(c1 + 1, static_cast<uint16_t>(localObs & 0xFFFFu));
    c1                                                                                                    = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
    c2                                                                                                    = max(c2 + 1, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
    c2                                                                                                    = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
    c3                                                                                                    = max(c3 + 1, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
    c3                                                                                                    = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
    c4                                                                                                    = max(c4 + 1, static_cast<uint16_t>((localObs >> 48)));
    c4                                                                                                    = min(c4, static_cast<uint16_t>((localCo >> 48)));
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 - (turnIncrementR10[theta] & 3) + 1) & 127)]     = c1;
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 1 - (turnIncrementR10[theta] & 3) + 1) & 127)] = c2;
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 2 - (turnIncrementR10[theta] & 3) + 1) & 127)] = c3;
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 3 - (turnIncrementR10[theta] & 3) + 1) & 127)] = c4;

    i = threadIdx.x + blockIdx.x * blockDim.x;

    c1 = outRow[threadIdx.x * 4];
    c2 = outRow[threadIdx.x * 4 + 1];
    c3 = outRow[threadIdx.x * 4 + 2];
    c4 = outRow[threadIdx.x * 4 + 3];

    for (int32_t theta = 0; theta < 511; theta++)
    {
        localCo  = Co[i];
        localObs = Obs[i];

        c1                                                                                                    = max(c1 + 1, static_cast<uint16_t>(localObs & 0xFFFFu));
        c1                                                                                                    = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
        c2                                                                                                    = max(c2 + 1, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
        c2                                                                                                    = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
        c3                                                                                                    = max(c3 + 1, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
        c3                                                                                                    = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
        c4                                                                                                    = max(c4 + 1, static_cast<uint16_t>((localObs >> 48)));
        c4                                                                                                    = min(c4, static_cast<uint16_t>((localCo >> 48)));
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 - (turnIncrementR10[theta] & 3) + 1) & 127)]     = c1;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 1 - (turnIncrementR10[theta] & 3) + 1) & 127)] = c2;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 2 - (turnIncrementR10[theta] & 3) + 1) & 127)] = c3;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 3 - (turnIncrementR10[theta] & 3) + 1) & 127)] = c4;

        Co[i] = (static_cast<uint64_t>(c4) << 48 | static_cast<uint64_t>(c3) << 32 | static_cast<uint64_t>(c2) << 16 | c1);
        i     = (((i + 4096) & 0x1FF000) | ((i - ((((turnIncrementR10[theta] >> 2) - 1) & 127) << 5)) & 4064) | (i & 31));

        c1 = outRow[threadIdx.x * 4];
        c2 = outRow[threadIdx.x * 4 + 1];
        c3 = outRow[threadIdx.x * 4 + 2];
        c4 = outRow[threadIdx.x * 4 + 3];
    }
    localCo  = Co[i];
    localObs = Obs[i];

    c1 = max(c1 + 1, static_cast<uint16_t>(localObs & 0xFFFFu));
    c1 = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
    c2 = max(c2 + 1, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
    c2 = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
    c3 = max(c3 + 1, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
    c3 = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
    c4 = max(c4 + 1, static_cast<uint16_t>((localObs >> 48)));
    c4 = min(c4, static_cast<uint16_t>((localCo >> 48)));

    Co[i] = (static_cast<uint64_t>(c4) << 48 | static_cast<uint64_t>(c3) << 32 | static_cast<uint64_t>(c2) << 16 | c1);
}

__global__ void _costSweepR20RightReverse64(uint64_t* Co, uint64_t* Obs)
{
    __shared__ uint16_t outRow[256];

    uint64_t localCo{};
    uint64_t localObs{};

    uint16_t c1 = 64250u;
    uint16_t c2 = 64250u;
    uint16_t c3 = 64250u;
    uint16_t c4 = 64250u;
    int32_t i   = threadIdx.x + blockIdx.x * blockDim.x;

    for (int32_t theta = 0; theta < 511; theta++)
    {
        localCo  = Co[i];
        localObs = Obs[i];

        c1                                                                                                    = max(c1 + 2, static_cast<uint16_t>(localObs & 0xFFFFu));
        c1                                                                                                    = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
        c2                                                                                                    = max(c2 + 2, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
        c2                                                                                                    = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
        c3                                                                                                    = max(c3 + 2, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
        c3                                                                                                    = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
        c4                                                                                                    = max(c4 + 2, static_cast<uint16_t>((localObs >> 48)));
        c4                                                                                                    = min(c4, static_cast<uint16_t>((localCo >> 48)));
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 - (turnIncrementR20[theta] & 3) + 1) & 127)]     = c1;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 1 - (turnIncrementR20[theta] & 3) + 1) & 127)] = c2;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 2 - (turnIncrementR20[theta] & 3) + 1) & 127)] = c3;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 3 - (turnIncrementR20[theta] & 3) + 1) & 127)] = c4;

        i = (((i + 4096) & 0x1FF000) | ((i - (((turnIncrementR20[theta] >> 2) - 1) << 5)) & 4064) | (i & 31));

        c1 = outRow[threadIdx.x * 4];
        c2 = outRow[threadIdx.x * 4 + 1];
        c3 = outRow[threadIdx.x * 4 + 2];
        c4 = outRow[threadIdx.x * 4 + 3];
    }
    localCo                                                                                               = Co[i];
    localObs                                                                                              = Obs[i];
    int32_t theta                                                                                         = 511;
    c1                                                                                                    = max(c1 + 2, static_cast<uint16_t>(localObs & 0xFFFFu));
    c1                                                                                                    = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
    c2                                                                                                    = max(c2 + 2, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
    c2                                                                                                    = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
    c3                                                                                                    = max(c3 + 2, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
    c3                                                                                                    = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
    c4                                                                                                    = max(c4 + 2, static_cast<uint16_t>((localObs >> 48)));
    c4                                                                                                    = min(c4, static_cast<uint16_t>((localCo >> 48)));
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 - (turnIncrementR20[theta] & 3) + 1) & 127)]     = c1;
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 1 - (turnIncrementR20[theta] & 3) + 1) & 127)] = c2;
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 2 - (turnIncrementR20[theta] & 3) + 1) & 127)] = c3;
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 3 - (turnIncrementR20[theta] & 3) + 1) & 127)] = c4;

    i = threadIdx.x + blockIdx.x * blockDim.x;

    c1 = outRow[threadIdx.x * 4];
    c2 = outRow[threadIdx.x * 4 + 1];
    c3 = outRow[threadIdx.x * 4 + 2];
    c4 = outRow[threadIdx.x * 4 + 3];

    for (int32_t theta = 0; theta < 511; theta++)
    {
        localCo  = Co[i];
        localObs = Obs[i];

        c1                                                                                                    = max(c1 + 2, static_cast<uint16_t>(localObs & 0xFFFFu));
        c1                                                                                                    = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
        c2                                                                                                    = max(c2 + 2, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
        c2                                                                                                    = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
        c3                                                                                                    = max(c3 + 2, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
        c3                                                                                                    = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
        c4                                                                                                    = max(c4 + 2, static_cast<uint16_t>((localObs >> 48)));
        c4                                                                                                    = min(c4, static_cast<uint16_t>((localCo >> 48)));
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 - (turnIncrementR20[theta] & 3) + 1) & 127)]     = c1;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 1 - (turnIncrementR20[theta] & 3) + 1) & 127)] = c2;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 2 - (turnIncrementR20[theta] & 3) + 1) & 127)] = c3;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 3 - (turnIncrementR20[theta] & 3) + 1) & 127)] = c4;

        Co[i] = (static_cast<uint64_t>(c4) << 48 | static_cast<uint64_t>(c3) << 32 | static_cast<uint64_t>(c2) << 16 | c1);
        i     = (((i + 4096) & 0x1FF000) | ((i - ((((turnIncrementR20[theta] >> 2) - 1) & 127) << 5)) & 4064) | (i & 31));

        c1 = outRow[threadIdx.x * 4];
        c2 = outRow[threadIdx.x * 4 + 1];
        c3 = outRow[threadIdx.x * 4 + 2];
        c4 = outRow[threadIdx.x * 4 + 3];
    }
    localCo  = Co[i];
    localObs = Obs[i];

    c1 = max(c1 + 2, static_cast<uint16_t>(localObs & 0xFFFFu));
    c1 = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
    c2 = max(c2 + 2, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
    c2 = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
    c3 = max(c3 + 2, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
    c3 = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
    c4 = max(c4 + 2, static_cast<uint16_t>((localObs >> 48)));
    c4 = min(c4, static_cast<uint16_t>((localCo >> 48)));

    Co[i] = (static_cast<uint64_t>(c4) << 48 | static_cast<uint64_t>(c3) << 32 | static_cast<uint64_t>(c2) << 16 | c1);
}

__global__ void _costSweepR20LeftReverse64(uint64_t* Co, uint64_t* Obs)
{
    __shared__ uint16_t outRow[256];

    uint64_t localCo{};
    uint64_t localObs{};

    uint16_t c1 = 64250u;
    uint16_t c2 = 64250u;
    uint16_t c3 = 64250u;
    uint16_t c4 = 64250u;
    int32_t i   = threadIdx.x + blockIdx.x * blockDim.x + 511 * blockDim.x * gridDim.x;

    for (int32_t theta = 510; theta >= 0; theta--)
    {
        localCo  = Co[i];
        localObs = Obs[i];

        c1                                                                                                    = max(c1 + 2, static_cast<uint16_t>(localObs & 0xFFFFu));
        c1                                                                                                    = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
        c2                                                                                                    = max(c2 + 2, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
        c2                                                                                                    = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
        c3                                                                                                    = max(c3 + 2, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
        c3                                                                                                    = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
        c4                                                                                                    = max(c4 + 2, static_cast<uint16_t>((localObs >> 48)));
        c4                                                                                                    = min(c4, static_cast<uint16_t>((localCo >> 48)));
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 - (turnIncrementR20[theta] & 3) + 1) & 127)]     = c1;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 1 - (turnIncrementR20[theta] & 3) + 1) & 127)] = c2;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 2 - (turnIncrementR20[theta] & 3) + 1) & 127)] = c3;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 3 - (turnIncrementR20[theta] & 3) + 1) & 127)] = c4;

        i = (((i - 4096) & 0x1FF000) | ((i - (((turnIncrementR20[theta] >> 2) - 1) << 5)) & 4064) | (i & 31));

        c1 = outRow[threadIdx.x * 4];
        c2 = outRow[threadIdx.x * 4 + 1];
        c3 = outRow[threadIdx.x * 4 + 2];
        c4 = outRow[threadIdx.x * 4 + 3];
    }
    localCo       = Co[i];
    localObs      = Obs[i];
    int32_t theta = 511;
    c1            = max(c1 + 2, static_cast<uint16_t>(localObs & 0xFFFFu));
    c1            = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
    c2            = max(c2 + 2, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
    c2            = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
    c3            = max(c3 + 2, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
    c3            = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
    c4            = max(c4 + 2, static_cast<uint16_t>((localObs >> 48)));
    c4            = min(c4, static_cast<uint16_t>((localCo >> 48)));

    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 - (turnIncrementR20[theta] & 3) + 1) & 127)]     = c1;
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 1 - (turnIncrementR20[theta] & 3) + 1) & 127)] = c2;
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 2 - (turnIncrementR20[theta] & 3) + 1) & 127)] = c3;
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 3 - (turnIncrementR20[theta] & 3) + 1) & 127)] = c4;

    i = threadIdx.x + blockIdx.x * blockDim.x + 511 * blockDim.x * gridDim.x;

    c1 = outRow[threadIdx.x * 4];
    c2 = outRow[threadIdx.x * 4 + 1];
    c3 = outRow[threadIdx.x * 4 + 2];
    c4 = outRow[threadIdx.x * 4 + 3];

    for (int32_t theta = 510; theta >= 0; theta--)
    {
        localCo  = Co[i];
        localObs = Obs[i];

        c1 = max(c1 + 2, static_cast<uint16_t>(localObs & 0xFFFFu));
        c1 = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
        c2 = max(c2 + 2, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
        c2 = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
        c3 = max(c3 + 2, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
        c3 = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
        c4 = max(c4 + 2, static_cast<uint16_t>((localObs >> 48)));
        c4 = min(c4, static_cast<uint16_t>((localCo >> 48)));

        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 - (turnIncrementR20[theta] & 3) + 1) & 127)]     = c1;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 1 - (turnIncrementR20[theta] & 3) + 1) & 127)] = c2;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 2 - (turnIncrementR20[theta] & 3) + 1) & 127)] = c3;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 3 - (turnIncrementR20[theta] & 3) + 1) & 127)] = c4;

        Co[i] = (static_cast<uint64_t>(c4) << 48 | static_cast<uint64_t>(c3) << 32 | static_cast<uint64_t>(c2) << 16 | c1);
        i     = (((i - 4096) & 0x1FF000) | ((i - (((turnIncrementR20[theta] >> 2) - 1) << 5)) & 4064) | (i & 31));

        c1 = outRow[threadIdx.x * 4];
        c2 = outRow[threadIdx.x * 4 + 1];
        c3 = outRow[threadIdx.x * 4 + 2];
        c4 = outRow[threadIdx.x * 4 + 3];
    }
    localCo  = Co[i];
    localObs = Obs[i];

    c1 = max(c1 + 2, static_cast<uint16_t>(localObs & 0xFFFFu));
    c1 = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
    c2 = max(c2 + 2, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
    c2 = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
    c3 = max(c3 + 2, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
    c3 = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
    c4 = max(c4 + 2, static_cast<uint16_t>((localObs >> 48)));
    c4 = min(c4, static_cast<uint16_t>((localCo >> 48)));

    Co[i] = (static_cast<uint64_t>(c4) << 48 | static_cast<uint64_t>(c3) << 32 | static_cast<uint64_t>(c2) << 16 | c1);
}

__global__ void _costSweepR10LeftReverse64(uint64_t* Co, uint64_t* Obs)
{
    __shared__ uint16_t outRow[256];

    uint64_t localCo{};
    uint64_t localObs{};

    uint16_t c1 = 64250u;
    uint16_t c2 = 64250u;
    uint16_t c3 = 64250u;
    uint16_t c4 = 64250u;
    int32_t i   = threadIdx.x + blockIdx.x * blockDim.x + 511 * blockDim.x * gridDim.x;

    for (int32_t theta = 510; theta >= 0; theta--)
    {
        localCo  = Co[i];
        localObs = Obs[i];

        c1                                                                                                    = max(c1 + 1, static_cast<uint16_t>(localObs & 0xFFFFu));
        c1                                                                                                    = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
        c2                                                                                                    = max(c2 + 1, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
        c2                                                                                                    = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
        c3                                                                                                    = max(c3 + 1, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
        c3                                                                                                    = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
        c4                                                                                                    = max(c4 + 1, static_cast<uint16_t>((localObs >> 48)));
        c4                                                                                                    = min(c4, static_cast<uint16_t>((localCo >> 48)));
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 - (turnIncrementR10[theta] & 3) + 1) & 127)]     = c1;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 1 - (turnIncrementR10[theta] & 3) + 1) & 127)] = c2;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 2 - (turnIncrementR10[theta] & 3) + 1) & 127)] = c3;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 3 - (turnIncrementR10[theta] & 3) + 1) & 127)] = c4;

        i = (((i - 4096) & 0x1FF000) | ((i - (((turnIncrementR10[theta] >> 2) - 1) << 5)) & 4064) | (i & 31));

        c1 = outRow[threadIdx.x * 4];
        c2 = outRow[threadIdx.x * 4 + 1];
        c3 = outRow[threadIdx.x * 4 + 2];
        c4 = outRow[threadIdx.x * 4 + 3];
    }
    localCo       = Co[i];
    localObs      = Obs[i];
    int32_t theta = 511;

    c1                                                                                                    = max(c1 + 1, static_cast<uint16_t>(localObs & 0xFFFFu));
    c1                                                                                                    = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
    c2                                                                                                    = max(c2 + 1, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
    c2                                                                                                    = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
    c3                                                                                                    = max(c3 + 1, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
    c3                                                                                                    = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
    c4                                                                                                    = max(c4 + 1, static_cast<uint16_t>((localObs >> 48)));
    c4                                                                                                    = min(c4, static_cast<uint16_t>((localCo >> 48)));
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 - (turnIncrementR10[theta] & 3) + 1) & 127)]     = c1;
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 1 - (turnIncrementR10[theta] & 3) + 1) & 127)] = c2;
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 2 - (turnIncrementR10[theta] & 3) + 1) & 127)] = c3;
    outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 3 - (turnIncrementR10[theta] & 3) + 1) & 127)] = c4;
    i                                                                                                     = threadIdx.x + blockIdx.x * blockDim.x + 511 * blockDim.x * gridDim.x;
    c1                                                                                                    = outRow[threadIdx.x * 4];
    c2                                                                                                    = outRow[threadIdx.x * 4 + 1];
    c3                                                                                                    = outRow[threadIdx.x * 4 + 2];
    c4                                                                                                    = outRow[threadIdx.x * 4 + 3];

    for (int32_t theta = 510; theta >= 0; theta--)
    {
        localCo  = Co[i];
        localObs = Obs[i];

        c1 = max(c1 + 1, static_cast<uint16_t>(localObs & 0xFFFFu));
        c1 = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
        c2 = max(c2 + 1, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
        c2 = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
        c3 = max(c3 + 1, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
        c3 = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
        c4 = max(c4 + 1, static_cast<uint16_t>((localObs >> 48)));
        c4 = min(c4, static_cast<uint16_t>((localCo >> 48)));

        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 - (turnIncrementR10[theta] & 3) + 1) & 127)]     = c1;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 1 - (turnIncrementR10[theta] & 3) + 1) & 127)] = c2;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 2 - (turnIncrementR10[theta] & 3) + 1) & 127)] = c3;
        outRow[((threadIdx.x >> 5) << 7) + ((threadIdx.x * 4 + 3 - (turnIncrementR10[theta] & 3) + 1) & 127)] = c4;

        Co[i] = (static_cast<uint64_t>(c4) << 48 | static_cast<uint64_t>(c3) << 32 | static_cast<uint64_t>(c2) << 16 | c1);
        i     = (((i - 4096) & 0x1FF000) | ((i - (((turnIncrementR10[theta] >> 2) - 1) << 5)) & 4064) | (i & 31));

        c1 = outRow[threadIdx.x * 4];
        c2 = outRow[threadIdx.x * 4 + 1];
        c3 = outRow[threadIdx.x * 4 + 2];
        c4 = outRow[threadIdx.x * 4 + 3];
    }
    localCo  = Co[i];
    localObs = Obs[i];

    c1 = max(c1 + 1, static_cast<uint16_t>(localObs & 0xFFFFu));
    c1 = min(c1, static_cast<uint16_t>(localCo & 0xFFFFu));
    c2 = max(c2 + 1, static_cast<uint16_t>(((localObs >> 16) & 0xFFFFu)));
    c2 = min(c2, static_cast<uint16_t>(((localCo >> 16) & 0xFFFFu)));
    c3 = max(c3 + 1, static_cast<uint16_t>(((localObs >> 32) & 0xFFFFu)));
    c3 = min(c3, static_cast<uint16_t>(((localCo >> 32) & 0xFFFFu)));
    c4 = max(c4 + 1, static_cast<uint16_t>((localObs >> 48)));
    c4 = min(c4, static_cast<uint16_t>((localCo >> 48)));

    Co[i] = (static_cast<uint64_t>(c4) << 48 | static_cast<uint64_t>(c3) << 32 | static_cast<uint64_t>(c2) << 16 | c1);
}

void ParkingPlanner2::costSweepAll64(cudaStream_t cuStream)
{

    _costSweepR10RightForward64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data()),
                                                         reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweepR20RightForward64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data() + 1 * getGridSize3()),
                                                         reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweepR20LeftForward64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data() + 2 * getGridSize3()),
                                                        reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweepR10LeftForward64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data() + 3 * getGridSize3()),
                                                        reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweepR10RightReverse64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data() + 4 * getGridSize3()),
                                                         reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweepR20RightReverse64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data() + 5 * getGridSize3()),
                                                         reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweepR20LeftReverse64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data() + 6 * getGridSize3()),
                                                        reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    _costSweepR10LeftReverse64<<<64, 64, 0, cuStream>>>(reinterpret_cast<uint64_t*>(m_cuGridTurns.get().get().data() + 7 * getGridSize3()),
                                                        reinterpret_cast<uint64_t*>(m_cuGridObs.get().get().data()));
    cudaStreamSynchronize(cuStream);
}

__global__ void _copy32To16(uint16_t* Out16, uint32_t* In32)
{
    int32_t i        = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t inReg   = In32[i];
    Out16[i * 2]     = (inReg & 65535u);
    Out16[i * 2 + 1] = inReg >> 16;
}

void ParkingPlanner2::copy32To16(cudaStream_t cuStream)
{
    _copy32To16<<<4096, 1024, 0, cuStream>>>(m_cuGridMain.get().get().data(), m_cuOut32.get().get().data());
    cudaStreamSynchronize(cuStream);
}

__global__ void _copy64To16(uint16_t* Out16, uint64_t* In64)
{
    int32_t i        = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t inReg   = In64[i];
    Out16[i * 4]     = (inReg & 65535u);
    Out16[i * 4 + 1] = ((inReg >> 16) & 65535u);
    Out16[i * 4 + 2] = ((inReg >> 32) & 65535u);
    Out16[i * 4 + 3] = ((inReg >> 48) & 65535u);
}

void ParkingPlanner2::copy64To16(cudaStream_t cuStream)
{
    _copy64To16<<<2048, 1024, 0, cuStream>>>(m_cuGridMain.get().get().data(), m_cuOut64.get().get().data());
    cudaStreamSynchronize(cuStream);
}

__global__ void _copy16To64(uint64_t* Out64, uint16_t* In16)
{
    int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    Out64[i]  = ((static_cast<uint64_t>(In16[i * 4 + 3]) << 48) | (static_cast<uint64_t>(In16[i * 4 + 2]) << 32) |
                (static_cast<uint64_t>(In16[i * 4 + 1]) << 16) | (static_cast<uint64_t>(In16[i * 4])));
}

/*void ParkingPlanner2::copy16To64(cudaStream_t cuStream)
{
    _copy16To64<<<2048, 1024, 0, cuStream>>>(m_cuGridObs64.get().get().data(),m_cuGridObs.get().get().data());
    cudaStreamSynchronize(cuStream);
}*/

/*__global__ void _sectionFirst16(uint16_t* Cf, uint16_t* Cr, uint16_t* O, uint16_t* Obs, uint16_t* Co)
{
    int32_t i = threadIdx.x + 4*blockIdx.x*blockDim.x;
    //is initialize
    uint16_t Co1 = Co[i];
    uint16_t Co2 = Co[i+4096];
    uint16_t Co3 = Co[i+8192];
    uint16_t Co4 = Co[i+12288];
    uint16_t Obs1 = Obs[i];
    uint16_t Obs2 = Obs[i+4096];
    uint16_t Obs3 = Obs[i+8192];
    uint16_t Obs4 = Obs[i+12288];
    O[is] = max(max(Obs1, Obs2), max(Obs3, Obs4));
	Cf[is] = min(max(min(max(min(max(min(max(c+1, Obs1), Co1)+1, Obs2),Co2)+1,Obs3),Co3)+1, Obs4), Co4);
	Cr[is] = min(max(min(max(min(max(min(max(c+1, Obs4), Co4)+1, Obs3),Co3)+1,Obs2),Co2)+1, Obs1), Co1);
    
}
void ParkingPlanner2::sectionFirst64(cudaStream_t cuStream)
{
    _sectionFirst64<<<4096, 128, 0, cuStream>>>(m_cuGridMain.get().get().data(),m_cuOut64.get().get().data());
    cudaStreamSynchronize(cuStream);
}
__global__ void _sectionMid64(uint64_t* Cf, uint64_t* Cr, uint64_t* O)
{
    uint16_t c = 64250u;
    //is start
    for(int32_t section = 0; section <128; section++)
    {
       uint16_t localOs =  Os[is];
       uint16_t localCf = Cf[is];
       c = max(c+4, Os[is]);
       c = min(c, Cf[is]); 
       //is update;
    }
    //is start
    for(int32_t section = 0; section <128; section++)
    {
       uint16_t localOs =  Os[is];
       uint16_t localCf = Cf[is];
       c = max(c+4, Os[is]);
       c = min(c, Cf[is]);
       Cf[is] = c; 
       //is update;
    }
    for(int32_t section = 127; section >= 0; section--)
    {
        //similar to forawrd
    }
}
__global__ void _sectionLast64(uint64_t* Co, uint64_t* Obs, uint64_t* Cf, uint64_t* Cr)
{
    int32_t i = threadIdx.x + 4*blockIdx.x*blockDim.x;
    //is =  threadIdx.x + blockIdx.x*blockDim.x
    uint16_t Co1 = Co[i];
    uint16_t Co2 = Co[i+4096];
    uint16_t Co3 = Co[i+8192];
    uint16_t Co4 = Co[i+12288];
    uint16_t Obs1 = Obs[i];
    uint16_t Obs2 = Obs[i+4096];
    uint16_t Obs3 = Obs[i+8192];
    uint16_t Obs4 = Obs[i+12288];

    Cf = Cf[is];
    Cr = Cr[is];

   Co1 = min(max(Cf+1, Obs1), Co1); Co4 = min(max(Cr+1, Obs4), Co4);
   Co2 = min(max(Co1+1, Obs2), Co2); Co3 = min(max(Co4+1, Obs3), Co3);
   Co3 = min(max(Co2+1, Obs3), Co3); Co2 = min(max(Co3+1, Obs2), Co2);
   Co4 = min(max(Co3+1, Obs1), Co4); Co1 = min(max(Co2+1, Obs1), Co1);

   Co[i] = Co1;
   Co[i+4096] = Co2;
   Co[i+8192] = Co2;
   Co[i+12288] = Co2;
}*/

__global__ void _sections16(uint16_t* Co, uint16_t* Obs)
{
    int32_t i     = threadIdx.x + blockDim.x * blockIdx.x;
    int32_t theta = i / 16384;
    i             = (((i & 0x7FC000) * 4) | (i & 16383));
    int32_t i2{};
    int32_t i3{};
    int32_t i4{};

    uint16_t c1 = 64250u;
    uint16_t c2 = 64250u;
    uint16_t c3 = 64250u;
    uint16_t c4 = 64250u;

    int32_t inc = 16384;

    /*c1 = Obs[i];
    //i2 = i + inc;
    //i3 = i2 +inc;
    //i4 = i3 + inc;
    c2 = Obs[i + inc];
    c3 = Obs[i + 2*inc];
    c4 = Obs[i + 3*inc];

    Co[i] = c1;
    Co[i + inc] = c2;
    Co[i + 2*inc] = c3;
    Co[i + 3*inc] = c4;*/

    /*i = threadIdx.x + blockDim.x*blockIdx.x;
    i = (((i&0x7FC000)*4)|(i&16383));

    c1 = Obs[i];
    i2 = i + inc;
    i3 = i2 +inc;
    i4 = i3 + inc;
    c2 = Obs[i2];
    c3 = Obs[i3];
    c4 = Obs[i4];

    Co[i] = c1;
    Co[i2] = c2;
    Co[i3] = c3;
    Co[i4] = c4;*/

    /*c1 = Obs[i];
    c1 -= Co[i];
    i2 = i + inc;
    c2 = Obs[i2];
    c2 -= Co[i2];
    i3 = i2 +inc;
    c3 = Obs[i3];
    c3 -= Co[i3];
    i4 = i3 + inc;
    c4 = Obs[i4];
    c4 -= Co[i4];

    Co[i] = c1;
    Co[i2] = c2;
    Co[i3] = c3;
    Co[i4] = c4;

    c1 = c4;

    i = threadIdx.x + blockDim.x*blockIdx.x;
    i = (((i&0x7FC000)*4)|(i&16383));

    c1 = Obs[i];
    c1 -= Co[i];
    i2 = i + inc;
    c2 = Obs[i2];
    c2 -= Co[i2];
    i3 = i2 +inc;
    c3 = Obs[i3];
    c3 -= Co[i3];
    i4 = i3 + inc;
    c4 = Obs[i4];
    c4 -= Co[i4];

    Co[i] = c1;
    Co[i2] = c2;
    Co[i3] = c3;
    Co[i4] = c4;*/
    c1 = max(c1 + 1, Obs[i]);
    c1 = min(c1, Co[i]);

    inc = turnIncrementR10[theta];
    i2  = (((i + 16384) & 0x7FC000) | ((i + 128 * (((inc & 12) >> 2) - 1)) & 16256) | ((i + ((inc & 3) - 1)) & 127));
    inc = turnIncrementR10[theta + 1];
    i3  = (((i2 + 16384) & 0x7FC000) | ((i2 + 128 * (((inc & 12) >> 2) - 1)) & 16256) | ((i2 + ((inc & 3) - 1)) & 127));
    inc = turnIncrementR10[theta + 2];
    i4  = (((i3 + 16384) & 0x7FC000) | ((i3 + 128 * (((inc & 12) >> 2) - 1)) & 16256) | ((i3 + ((inc & 3) - 1)) & 127));
    c2  = max(c1 + 1, Obs[i2]);
    c2  = min(c2, Co[i2]);

    c3 = max(c2 + 1, Obs[i3]);
    c3 = min(c3, Co[i3]);

    c4 = max(c3 + 1, Obs[i4]);
    c4 = min(c4, Co[i4]);

    Co[i]  = c1;
    Co[i2] = c2;
    Co[i3] = c3;
    Co[i4] = c4;

    inc = turnIncrementR10[theta + 3];
    i   = (((i4 + 16384) & 0x7FC000) | ((i4 + 128 * (((inc & 12) >> 2) - 1)) & 16256) | ((i4 + ((inc & 3) - 1)) & 127));
    c1  = c4;

    /*i = threadIdx.x + blockDim.x*blockIdx.x;
    i = (((i&0x7FC000)*4)|(i&16383));

    c1 = max(c1 + 1, Obs[i]);
    c1 = min(c1, Co[i]);

    inc = turnIncrementR10[theta];
    i2 = (((i+16384)&0x7FC000)|((i+128*(((inc&12)>>2)-1))&16256) | ((i+((inc&3)-1))&127));
    inc = turnIncrementR10[theta+1];
    i3 = (((i2+16384)&0x7FC000)|((i2+128*(((inc&12)>>2)-1))&16256) | ((i2+((inc&3)-1))&127));
    inc = turnIncrementR10[theta+2];
    i4 = (((i3+16384)&0x7FC000)|((i3+128*(((inc&12)>>2)-1))&16256) | ((i3+((inc&3)-1))&127));
    c2 = max(c1 + 1, Obs[i2]);
    c2 = min(c2, Co[i2]);

    c3 = max(c2 + 1, Obs[i3]);
    c3 = min(c3, Co[i3]);

    c4 = max(c3 + 1, Obs[i4]);
    c4 = min(c4, Co[i4]);

    Co[i] = c1;
    Co[i2] = c2;
    Co[i3] = c3;
    Co[i4] = c4;


    inc = turnIncrementR10[theta+3];
    i = (((i4+16384)&0x7FC000)|((i4+128*(((inc&12)>>2)-1))&16256) | ((i4+((inc&3)-1))&127));
    c1 = c4;*/
    //c1 = max(c1 + 1, Obs[i]);
    //c1 = min(c1, Co[i]);

    //inc = turnIncrementR10[theta];
    //i2 = (((i+16384)&0x7FC000)|((i+128*(((inc&12)>>2)-1))&16256) | ((i+((inc&3)-1))&127));
}

void ParkingPlanner2::sections16(cudaStream_t cuStream)
{
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _sections16<<<2048, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());

    cudaStreamSynchronize(cuStream);
}

__global__ void _directCopy16(uint16_t* Co, uint16_t* Obs)
{
    int32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    Co[i]     = Obs[i];
    //i += (64*128*512);
    //Co[i] = Obs[i];
}

void ParkingPlanner2::directCopy16(cudaStream_t cuStream)
{
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());
    _directCopy16<<<8192, 1024, 0, cuStream>>>(m_cuGridTurns.get().get().data() + 3 * getGridSize3(), m_cuGridObs.get().get().data());

    cudaStreamSynchronize(cuStream);
}

__global__ void _directCopy32(uint32_t* Co, uint32_t* Obs)
{
    int32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    Co[i]     = Obs[i];
}

void ParkingPlanner2::directCopy32(cudaStream_t cuStream)
{
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopy32<<<4096, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    cudaDeviceSynchronize();
}

__global__ void _directCopyUnroll32(uint32_t* Co, uint32_t* Obs)
{
    int32_t i1    = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t reg1 = Obs[i1];
    int32_t i2    = i1 + (2048 * 1024);
    uint32_t reg2 = Obs[i2];
    Co[i1]        = reg1;
    Co[i2]        = reg2;
}

void ParkingPlanner2::directCopyUnroll32(cudaStream_t cuStream)
{
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    _directCopyUnroll32<<<2048, 1024, 0, cuStream>>>(m_cuOut32.get().get().data(), m_cuIn32.get().get().data());
    cudaDeviceSynchronize();
}

__global__ void _directCopy64(uint64_t* Co, uint64_t* Obs)
{
    int32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    Co[i]     = Obs[i];
}

void ParkingPlanner2::directCopy64(cudaStream_t cuStream)
{
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    _directCopy64<<<2048, 1024, 0, cuStream>>>(m_cuOut64.get().get().data(), m_cuIn64.get().get().data());
    cudaDeviceSynchronize();
}

__constant__ uint16_t transitionCost[(ParkingPlanner2::TURN_TYPES_NUM+1) * ParkingPlanner2::TURN_TYPES_NUM];

/*__global__ void _costTransition(uint16_t* CTurns, int32_t turnTypesNum, int32_t gridSize)
{
    
    int32_t index = blockDim.x*blockIdx.x + threadIdx.x;
    
    
    uint16_t R10RF = CTurns[index];
    uint16_t R20RF = CTurns[index + 1*gridSize];
    uint16_t R20LF = CTurns[index + 2*gridSize];
    uint16_t R10LF = CTurns[index + 3*gridSize];
    uint16_t R10RR = CTurns[index + 4*gridSize];
    uint16_t R20RR = CTurns[index + 5*gridSize];
    uint16_t R20LR = CTurns[index + 6*gridSize];
    uint16_t R10LR = CTurns[index + 7*gridSize];

    if(index == ParkingPlanner2::volIndex(0,0,256))
    {
        printf("dest costtransistion: %hu, %hu, %hu, %hu, %hu, %hu, %hu, %hu\n", R10RF, R20RF, R20LF, R10LF, R10RR, R20RR, R20LR, R10LR);
    }

    CTurns[index] = min(min(min(R10RF + transitionCost[0 + turnTypesNum*0], R20RF + transitionCost[0 + turnTypesNum*1]), 
                            min(R20LF + transitionCost[0 + turnTypesNum*2], R10LF + transitionCost[0 + turnTypesNum*3])),
                        min(min(R10RR + transitionCost[0 + turnTypesNum*4], R20RR + transitionCost[0 + turnTypesNum*5]),
                            min(R20LR + transitionCost[0 + turnTypesNum*6], R10LR + transitionCost[0 + turnTypesNum*7])));
    CTurns[index + 1*gridSize] = min(min(min(R10RF + transitionCost[1 + turnTypesNum*0], R20RF + transitionCost[1 + turnTypesNum*1]), 
                            min(R20LF + transitionCost[1 + turnTypesNum*2], R10LF + transitionCost[1 + turnTypesNum*3])),
                        min(min(R10RR + transitionCost[1 + turnTypesNum*4], R20RR + transitionCost[1 + turnTypesNum*5]),
                            min(R20LR + transitionCost[1 + turnTypesNum*6], R10LR + transitionCost[1 + turnTypesNum*7])));
    CTurns[index + 2*gridSize] = min(min(min(R10RF + transitionCost[2 + turnTypesNum*0], R20RF + transitionCost[2 + turnTypesNum*1]), 
                            min(R20LF + transitionCost[2 + turnTypesNum*2], R10LF + transitionCost[2 + turnTypesNum*3])),
                        min(min(R10RR + transitionCost[2 + turnTypesNum*4], R20RR + transitionCost[2 + turnTypesNum*5]),
                            min(R20LR + transitionCost[2 + turnTypesNum*6], R10LR + transitionCost[2 + turnTypesNum*7])));
    CTurns[index + 3*gridSize] = min(min(min(R10RF + transitionCost[3 + turnTypesNum*0], R20RF + transitionCost[3 + turnTypesNum*1]), 
                            min(R20LF + transitionCost[3 + turnTypesNum*2], R10LF + transitionCost[3 + turnTypesNum*3])),
                        min(min(R10RR + transitionCost[3 + turnTypesNum*4], R20RR + transitionCost[3 + turnTypesNum*5]),
                            min(R20LR + transitionCost[3 + turnTypesNum*6], R10LR + transitionCost[3 + turnTypesNum*7])));
    CTurns[index + 4*gridSize] = min(min(min(R10RF + transitionCost[4 + turnTypesNum*0], R20RF + transitionCost[4 + turnTypesNum*1]), 
                            min(R20LF + transitionCost[4 + turnTypesNum*2], R10LF + transitionCost[4 + turnTypesNum*3])),
                        min(min(R10RR + transitionCost[4 + turnTypesNum*4], R20RR + transitionCost[4 + turnTypesNum*5]),
                            min(R20LR + transitionCost[4 + turnTypesNum*6], R10LR + transitionCost[4 + turnTypesNum*7])));
    CTurns[index + 5*gridSize] = min(min(min(R10RF + transitionCost[5 + turnTypesNum*0], R20RF + transitionCost[5 + turnTypesNum*1]), 
                            min(R20LF + transitionCost[5 + turnTypesNum*2], R10LF + transitionCost[5 + turnTypesNum*3])),
                        min(min(R10RR + transitionCost[5 + turnTypesNum*4], R20RR + transitionCost[5 + turnTypesNum*5]),
                            min(R20LR + transitionCost[5 + turnTypesNum*6], R10LR + transitionCost[5 + turnTypesNum*7])));
    CTurns[index + 6*gridSize] = min(min(min(R10RF + transitionCost[6 + turnTypesNum*0], R20RF + transitionCost[6 + turnTypesNum*1]), 
                            min(R20LF + transitionCost[6 + turnTypesNum*2], R10LF + transitionCost[6 + turnTypesNum*3])),
                        min(min(R10RR + transitionCost[6 + turnTypesNum*4], R20RR + transitionCost[6 + turnTypesNum*5]),
                            min(R20LR + transitionCost[6 + turnTypesNum*6], R10LR + transitionCost[6 + turnTypesNum*7])));
    CTurns[index + 7*gridSize] = min(min(min(R10RF + transitionCost[7 + turnTypesNum*0], R20RF + transitionCost[7 + turnTypesNum*1]), 
                            min(R20LF + transitionCost[7 + turnTypesNum*2], R10LF + transitionCost[7 + turnTypesNum*3])),
                        min(min(R10RR + transitionCost[7 + turnTypesNum*4], R20RR + transitionCost[7 + turnTypesNum*5]),
                            min(R20LR + transitionCost[7 + turnTypesNum*6], R10LR + transitionCost[7 + turnTypesNum*7])));

}*/

__global__ void _costTransition(uint16_t* CTurns, int32_t turnTypesNum, int32_t gridSize)
{

    int32_t index = blockDim.x * blockIdx.x + threadIdx.x;

    uint16_t R10RF = CTurns[index];
    uint16_t R20RF = CTurns[index + 1 * gridSize];
    uint16_t R20LF = CTurns[index + 2 * gridSize];
    uint16_t R10LF = CTurns[index + 3 * gridSize];
    uint16_t R10RR = CTurns[index + 4 * gridSize];
    uint16_t R20RR = CTurns[index + 5 * gridSize];
    uint16_t R20LR = CTurns[index + 6 * gridSize];
    uint16_t R10LR = CTurns[index + 7 * gridSize];

    CTurns[index] = min(min(min(R10RF + transitionCost[0 + turnTypesNum * 0], R20RF + transitionCost[0 + turnTypesNum * 1]),
                            min(R20LF + transitionCost[0 + turnTypesNum * 2], R10LF + transitionCost[0 + turnTypesNum * 3])),
                        min(R20RR + transitionCost[0 + turnTypesNum * 5],
                            min(R20LR + transitionCost[0 + turnTypesNum * 6], R10LR + transitionCost[0 + turnTypesNum * 7])));
    CTurns[index + 1 * gridSize] = min(min(min(R10RF + transitionCost[1 + turnTypesNum * 0], R20RF + transitionCost[1 + turnTypesNum * 1]),
                                           min(R20LF + transitionCost[1 + turnTypesNum * 2], R10LF + transitionCost[1 + turnTypesNum * 3])),
                                       min(R10RR + transitionCost[1 + turnTypesNum * 4],
                                           min(R20LR + transitionCost[1 + turnTypesNum * 6], R10LR + transitionCost[1 + turnTypesNum * 7])));
    CTurns[index + 2 * gridSize] = min(min(min(R10RF + transitionCost[2 + turnTypesNum * 0], R20RF + transitionCost[2 + turnTypesNum * 1]),
                                           min(R20LF + transitionCost[2 + turnTypesNum * 2], R10LF + transitionCost[2 + turnTypesNum * 3])),
                                       min(min(R10RR + transitionCost[2 + turnTypesNum * 4], R20RR + transitionCost[2 + turnTypesNum * 5]),
                                           R10LR + transitionCost[2 + turnTypesNum * 7]));
    CTurns[index + 3 * gridSize] = min(min(min(R10RF + transitionCost[3 + turnTypesNum * 0], R20RF + transitionCost[3 + turnTypesNum * 1]),
                                           min(R20LF + transitionCost[3 + turnTypesNum * 2], R10LF + transitionCost[3 + turnTypesNum * 3])),
                                       min(min(R10RR + transitionCost[3 + turnTypesNum * 4], R20RR + transitionCost[3 + turnTypesNum * 5]),
                                           R20LR + transitionCost[3 + turnTypesNum * 6]));
    CTurns[index + 4 * gridSize] = min(min(R20RF + transitionCost[4 + turnTypesNum * 1],
                                           min(R20LF + transitionCost[4 + turnTypesNum * 2], R10LF + transitionCost[4 + turnTypesNum * 3])),
                                       min(min(R10RR + transitionCost[4 + turnTypesNum * 4], R20RR + transitionCost[4 + turnTypesNum * 5]),
                                           min(R20LR + transitionCost[4 + turnTypesNum * 6], R10LR + transitionCost[4 + turnTypesNum * 7])));
    CTurns[index + 5 * gridSize] = min(min(R10RF + transitionCost[5 + turnTypesNum * 0],
                                           min(R20LF + transitionCost[5 + turnTypesNum * 2], R10LF + transitionCost[5 + turnTypesNum * 3])),
                                       min(min(R10RR + transitionCost[5 + turnTypesNum * 4], R20RR + transitionCost[5 + turnTypesNum * 5]),
                                           min(R20LR + transitionCost[5 + turnTypesNum * 6], R10LR + transitionCost[5 + turnTypesNum * 7])));
    CTurns[index + 6 * gridSize] = min(min(min(R10RF + transitionCost[6 + turnTypesNum * 0], R20RF + transitionCost[6 + turnTypesNum * 1]),
                                           R10LF + transitionCost[6 + turnTypesNum * 3]),
                                       min(min(R10RR + transitionCost[6 + turnTypesNum * 4], R20RR + transitionCost[6 + turnTypesNum * 5]),
                                           min(R20LR + transitionCost[6 + turnTypesNum * 6], R10LR + transitionCost[6 + turnTypesNum * 7])));
    CTurns[index + 7 * gridSize] = min(min(min(R10RF + transitionCost[7 + turnTypesNum * 0], R20RF + transitionCost[7 + turnTypesNum * 1]),
                                           R20LF + transitionCost[7 + turnTypesNum * 2]),
                                       min(min(R10RR + transitionCost[7 + turnTypesNum * 4], R20RR + transitionCost[7 + turnTypesNum * 5]),
                                           min(R20LR + transitionCost[7 + turnTypesNum * 6], R10LR + transitionCost[7 + turnTypesNum * 7])));
}

void ParkingPlanner2::setTransitionCost()
{
    cudaMemcpyToSymbol(transitionCost, m_trCost, static_cast<uint32_t>(TURN_TYPES_NUM * TURN_TYPES_NUM * sizeof(uint16_t)));
}

void ParkingPlanner2::costTransition(cudaStream_t cuStream)
{
    dim3 block(256);
    dim3 grid((getGridSize3()) / block.x);
    _costTransition<<<grid, block, 0, cuStream>>>(m_cuGridTurns.get().get().data(), TURN_TYPES_NUM, getGridSize3());
    //cudaStreamSynchronize(cuStream);
    cudaDeviceSynchronize();
}

__global__ void _destinationCheck(bool* reached, int32_t destIndex,
                                  uint16_t* CTurns, int32_t gridSize)
{
    uint16_t minCost = min(min(min(CTurns[destIndex], CTurns[destIndex + 1 * gridSize]), min(CTurns[destIndex + 2 * gridSize], CTurns[destIndex + 3 * gridSize])),
                           min(min(CTurns[destIndex + 4 * gridSize], CTurns[destIndex + 5 * gridSize]),
                               min(CTurns[destIndex + 6 * gridSize], CTurns[destIndex + 7 * gridSize])));
    *reached = (minCost < 64250u);
}

bool ParkingPlanner2::destinationCheck(cudaStream_t cuStream)
{
    int32_t destIndex = volIndex(m_destination.x, m_destination.y, m_destination.hdg);
    bool* destReached;
    cudaMallocManaged(&destReached, sizeof(bool)); //Free this
    _destinationCheck<<<1, 1, 0, cuStream>>>(destReached, destIndex, m_cuGridTurns.get().get().data(), getGridSize3());
    cudaStreamSynchronize(cuStream);
    cudaDeviceSynchronize();

    DW_CHECK_CUDA_ERROR(cudaGetLastError());
    return *destReached;
}

__global__ void _minCost(uint16_t* out,
                         uint16_t* CTurns, int32_t gridSize)
{
    int32_t index = blockIdx.x * blockDim.x + threadIdx.x;

    out[index] = min(min(min(CTurns[index], CTurns[index + 1 * gridSize]), min(CTurns[index + 2 * gridSize], CTurns[index + 3 * gridSize])),
                     min(min(CTurns[index + 4 * gridSize], CTurns[index + 5 * gridSize]), min(CTurns[index + 6 * gridSize], CTurns[index + 7 * gridSize])));
}

void ParkingPlanner2::minCost(cudaStream_t cuStream)
{
    dim3 block(256);
    dim3 grid((getGridSize3()) / block.x);
    _minCost<<<grid, block, 0, cuStream>>>(m_cuGridMain.get().get().data(),
                                           m_cuGridTurns.get().get().data(), getGridSize3());
    //cudaStreamSynchronize(cuStream);
    cudaDeviceSynchronize();
}

__global__ void _initObstacle(uint16_t* obsGrid, core::span<const dwObstacle> obstacles, size_t obstacleCount, int32_t dim, float32_t posRes, float32_t hdgRes)
{
    int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if ((index & 127) == 0 || (index & 127) == 127 || ((index >> 7) & 127) == 0 || ((index >> 7) & 127) == 127)
        obsGrid[index] = 64250u;
    else
        obsGrid[index] = 0u;
    // TODO: (JK) Obstacle scenarios: Replace with obstacle initialization in testCode
    // Uncomment for immediate testing

    //if((index&127)>=30 && (index&127)<=50 && ((index>>7)&127)>=32 && ((index>>7)&127)<=96)
    //obsGrid[index] = 64250u;

    /*if((index&127)>66 && (index&127)<74)
        obsGrid[index] = 64250u;

    if(((index>>7)&127)>66 && ((index>>7)&127)<74)
        obsGrid[index] = 64250u;*/

    /*if((((index>>7)&127)>=69 && ((index>>7)&127)<=71) || (((index>>7)&127)>=91 && ((index>>7)&127)<=93) || (((index>>7)&127)>=113 && ((index>>7)&127)<=115))
        obsGrid[index] = 64250u;
    if((((index>>7)&127)>=69 && ((index>>7)&127)<=71 && (index&127)>=12 && (index&127)<=16) || 
       (((index>>7)&127)>=91 && ((index>>7)&127)<=93 && (index&127)>=110 && (index&127)<=114) || 
       (((index>>7)&127)>=113 && ((index>>7)&127)<=115 && (index&127)>=12 && (index&127)<=16))
       obsGrid[index] = 0u;*/

    Vector3f cellPose = ParkingPlanner2::index2Coord(index, dim).getPose(posRes, hdgRes);
    math::ConvexPolygon<4> egoPoly = math::ConvexPolygon<4>::createRectangle(
                                                                        {cellPose.x(), cellPose.y()},
                                                                        deg2Rad(cellPose.z()),
                                                                        2.0f,
                                                                        5.0f);

    for (int32_t i = 0; i < obstacleCount; i++)
    {
        constexpr float32_t SQR_OBSTACLE_RADIUS = 4.0f;

        const dwObstacle& obs = obstacles[i];

        Vector2f fences[4];
        for (uint32_t i = 0u; i < obs.boundaryPointCount; ++i)
        {
            fences[i] = Vector2f{obs.boundaryPoints[i].x,
                                 obs.boundaryPoints[i].y}; // TODO: positional scaling
        }

        // alternative 1: convex shape intersection
        math::ConvexPolygon<4> obsPoly
            {core::make_span<Vector2f>(&fences[0], obs.boundaryPointCount)};

        if (egoPoly.intersect(obsPoly))
        {
            obsGrid[index] = 64250u;
        }
        // TODO(JK): backtrace problem is capacity is DW_OBSTACLE_BOUNDARY_POINT_MAX_COUNT 
        // the same issue occurs even if no obstacle is added.

        // alternative 2: naive occupancy. ego is a point mass
        // if (math::Polygon::isPointInside(Vector2f{cellPose.x(), cellPose.y()},
        //                                  &fences[0],
        //                                  obs.boundaryPointCount))
        // {
        //     obsGrid[index] = 64250u;
        // }

        // alternative 3: disk-shaped
        // float32_t sqrDistance = (Vector2f{cellPose.x(), cellPose.y()} -
        //                          Vector2f{obs.position.x, obs.position.y})
        //                             .squaredNorm();

        // if (sqrDistance < SQR_OBSTACLE_RADIUS)
        // {
        //     obsGrid[index] = 64250u;
        // }
    }
}

void ParkingPlanner2::initObstacle(cudaStream_t cuStream)
{
    dim3 block(1024);
    dim3 grid(getGridSize3() / block.x);
    _initObstacle<<<grid, block, 0, cuStream>>>(m_cuGridObs.get().get().data(), m_cuObstacles.get().get(), m_obstacleCount, DIM3, POS_RES3, HDG_RES_DEG3);
    cudaStreamSynchronize(cuStream);
}

__global__ void _ourMemCpy32(uint32_t* out, uint32_t* in)
{

    for (int32_t theta = 0; theta < 360; theta++)
    {
        int32_t i1 = theta * blockDim.x * gridDim.x + blockIdx.x * blockDim.x + threadIdx.x;
        out[i1]    = in[i1];
    }
}

void ParkingPlanner2::ourMemCpy32(uint32_t* out, uint32_t* in, int32_t nElemsXY, cudaStream_t cuStream)
{
    dim3 block(512);
    dim3 grid((nElemsXY + block.x - 1) / block.x);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32<<<grid, block, 0, cuStream>>>(out, in);
    cudaStreamSynchronize(cuStream);
}

__global__ void _ourMemCpy32Direct(uint32_t* out, uint32_t* in)
{

    int32_t i = blockDim.x * blockIdx.x + threadIdx.x;

    out[i] = in[i];
}

void ParkingPlanner2::ourMemCpy32Direct(uint32_t* out, uint32_t* in, int32_t nElems, cudaStream_t cuStream)
{
    dim3 block(512);
    dim3 grid((nElems + block.x - 1) / block.x);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    _ourMemCpy32Direct<<<grid, block, 0, cuStream>>>(out, in);
    cudaStreamSynchronize(cuStream);
}

__global__ void _prepareOccupancyGrid(GridCell* out,
                                      core::span<const dwObstacle> obstacles,
                                      size_t obstacleCount,
                                      float32_t posRes,
                                      float32_t hdgRes)
{
    int32_t x     = blockIdx.x - (ParkingPlanner2::getSize() >> 1); //transforming indices from [0,size] range to [-size/2, size/2] range
    int32_t y     = blockIdx.y - (ParkingPlanner2::getSize() >> 1);
    int32_t theta = threadIdx.x;

    if (!ParkingPlanner2::withinBounds(x, y))
    {
        return;
    }

    for (size_t i = 0; i < obstacleCount; ++i)
    {
        // Just simply treat each obstacle as a ball of radius
        // TODO(JK/yizhou) use dwObstacle::boundaryPoints
        constexpr float32_t SQR_OBSTACLE_RADIUS = 2.0f;

        const dwObstacle& obs = obstacles[i];

        Vector3f cellPose = Coord3d(x, y, theta).getPose(posRes, hdgRes);

        float32_t sqrDistance = (Vector2f{cellPose.x(), cellPose.y()} -
                                 Vector2f{obs.position.x, obs.position.y})
                                    .squaredNorm();

        if (sqrDistance < SQR_OBSTACLE_RADIUS)
        {
            ParkingPlanner2::getCell(make_span(out, ParkingPlanner2::getGridSize()), x, y, theta).obstacle = true;
        }
    }
}

void ParkingPlanner2::prepareOccupancyGrid(GridCell* out,                          // device ptr
                                           core::span<const dwObstacle> obstacles, // device memory
                                           size_t obstacleCount,
                                           float32_t posRes,
                                           float32_t hdgRes,
                                           cudaStream_t cuStream)
{
    dim3 GridSize(X_SIZE, Y_SIZE);
    _prepareOccupancyGrid<<<GridSize, THETA_STEP, 0, cuStream>>>(out,
                                                                 obstacles,
                                                                 obstacleCount,
                                                                 posRes,
                                                                 hdgRes);
    cudaStreamSynchronize(cuStream);
}

void ParkingPlanner2::prepareOccupancyGrid()
{
    prepareOccupancyGrid(m_cuGrid.get().get().data(),
                         m_cuObstacles.get().get(),
                         m_obstacleCount,
                         POS_RES,
                         HDG_RES,
                         m_cuStream);
}

} // namespace planner
} // namespace dw