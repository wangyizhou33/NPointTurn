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

//C Interface
#include <dw/experimental/parkingplanner/ParkingPlanner.h>

//C++ Interface
#include <dw/core/Core.hpp>
#include <dw/core/Profiling.hpp>
#include <dw/experimental/parkingplanner/ParkingPlanner2.hpp>

using dw::planner::ParkingPlanner2;

dwStatus dwParkingPlanner_initDefaultParams(dwParkingPlannerParams* params)
{
    return dw::core::Exception::guard([&] {
        THROW_IF_PARAM_NULL(params);
        dw::checkClassSize<dwParkingPlannerParams, 8>();

        params->max_turns    = 15;
        params->turnRadius_m = 5.0f;
    });
}

dwStatus dwParkingPlanner_initialize(const dwParkingPlannerParams* params,
                                     dwContextHandle_t contextHandle,
                                     dwParkingPlannerHandle_t* parkingPlannerHandle)
{
    return dw::core::Exception::guard([&] {
        THROW_IF_PARAM_NULL(parkingPlannerHandle);
        THROW_IF_PARAM_NULL(params);
        dw::core::Context* context = dw::CHandle::cast(contextHandle);

        std::unique_ptr<ParkingPlanner2> parkingPlanner(new ParkingPlanner2(params, context));

        dw::makeUniqueCHandle(parkingPlannerHandle, std::move(parkingPlanner));
    });
}

dwStatus dwParkingPlanner_release(dwParkingPlannerHandle_t parkingPlannerHandle)
{
    FUNCTION_RANGE;
    return dw::core::Exception::guard([&] {
        dw::deleteUniqueCHandle(parkingPlannerHandle);
    });
}

dwStatus dwParkingPlanner_reset(dwParkingPlannerHandle_t parkingPlannerHandle)
{
    FUNCTION_RANGE;
    return dw::Exception::guard([&] {
        THROW_IF_PARAM_NULL(parkingPlannerHandle);

        ParkingPlanner2* parkingPlanner = dw::CHandle::cast(parkingPlannerHandle);
        parkingPlanner->reset();
    });
}

dwStatus dwParkingPlanner_setTarget(const dwUnstructuredTarget t,
                                    dwParkingPlannerHandle_t parkingPlannerHandle)
{
    FUNCTION_RANGE;
    return dw::core::Exception::guard([&] {
        THROW_IF_PARAM_NULL(parkingPlannerHandle);

        ParkingPlanner2* parkingPlanner = dw::CHandle::cast(parkingPlannerHandle);
        parkingPlanner->setTargetPose(t);
    });
}

dwStatus dwParkingPlanner_setObstacles(const dwObstacle* obs,
                                       size_t size,
                                       dwParkingPlannerHandle_t parkingPlannerHandle)
{
    FUNCTION_RANGE;
    return dw::core::Exception::guard([&] {
        THROW_IF_PARAM_NULL(parkingPlannerHandle);
        THROW_IF_PARAM_NULL(obs);

        ParkingPlanner2* parkingPlanner = dw::CHandle::cast(parkingPlannerHandle);
        parkingPlanner->setObstacles(dw::core::make_span(obs, size));
    });
}

dwStatus dwParkingPlanner_computePath(dwParkingPlannerHandle_t parkingPlannerHandle)
{
    FUNCTION_RANGE;
    return dw::core::Exception::guard([&] {
        THROW_IF_PARAM_NULL(parkingPlannerHandle);

        ParkingPlanner2* parkingPlanner = dw::CHandle::cast(parkingPlannerHandle);

        parkingPlanner->process();
    });
}

dwStatus dwParkingPlanner_getPathDetails(dwParkingState* path,
                                         dwPathPlannerDrivingState* dir,
                                         size_t* size,
                                         size_t capacity,
                                         dwConstParkingPlannerHandle_t parkingPlannerHandle)
{
    FUNCTION_RANGE;
    return dw::core::Exception::guard([&] {
        THROW_IF_PARAM_NULL(parkingPlannerHandle);
        THROW_IF_PARAM_NULL(path);
        THROW_IF_PARAM_NULL(dir);
        THROW_IF_PARAM_NULL(size);

        const ParkingPlanner2* parkingPlanner                   = dw::CHandle::cast(parkingPlannerHandle);
        dw::core::span<const dw::core::Vector3f> pathSpan       = parkingPlanner->getPath();
        dw::core::span<const dwPathPlannerDrivingState> dirSpan = parkingPlanner->getDrivingDirs();
        *size                                                   = pathSpan.size();

        if (parkingPlanner->hasPath())
        {
            if (*size < capacity)
            {
                for (size_t i = 0; i < *size; ++i)
                {
                    path[i].x       = pathSpan[i].x();
                    path[i].y       = pathSpan[i].y();
                    path[i].heading = pathSpan[i].z();
                    dir[i]          = dirSpan[i];
                }
            }
            else
            {
                throw dw::core::Exception(DW_OUT_OF_BOUNDS, "Input array size insufficient to take in output\n");
            }
        }
    });
}

dwStatus dwParkingPlanner_getPathSegment(dwParkingState* pathSegment,
                                         size_t* size,
                                         size_t capacity,
                                         dwConstParkingPlannerHandle_t parkingPlannerHandle)
{
    FUNCTION_RANGE;
    return dw::core::Exception::guard([&] {
        THROW_IF_PARAM_NULL(parkingPlannerHandle);
        THROW_IF_PARAM_NULL(pathSegment);
        THROW_IF_PARAM_NULL(size);

        const ParkingPlanner2* parkingPlanner             = dw::CHandle::cast(parkingPlannerHandle);
        dw::core::span<const dw::core::Vector3f> pathSpan = parkingPlanner->getPathSegment();
        *size                                             = pathSpan.size();
        if (parkingPlanner->hasPath())
        {
            if (*size < capacity)
            {
                for (size_t i = 0; i < *size; ++i)
                {
                    pathSegment[i].x       = pathSpan[i].x();
                    pathSegment[i].y       = pathSpan[i].y();
                    pathSegment[i].heading = pathSpan[i].z();
                }
            }
            else
            {
                throw dw::core::Exception(DW_OUT_OF_BOUNDS, "Input array size insufficient to take in output\n");
            }
        }
    });
}

dwStatus dwParkingPlanner_getCUDAStream(cudaStream_t stream,
                                        dwConstParkingPlannerHandle_t parkingPlannerHandle)
{
    FUNCTION_RANGE;
    return dw::core::Exception::guard([&] {
        THROW_IF_PARAM_NULL(parkingPlannerHandle);

        const ParkingPlanner2* parkingPlanner = dw::CHandle::cast(parkingPlannerHandle);
        stream                                = parkingPlanner->getCUDAStream();
    });
}

dwStatus dwParkingPlanner_setCUDAStream(cudaStream_t stream,
                                        dwParkingPlannerHandle_t parkingPlannerHandle)
{
    FUNCTION_RANGE;
    return dw::core::Exception::guard([&] {
        THROW_IF_PARAM_NULL(parkingPlannerHandle);

        ParkingPlanner2* parkingPlanner = dw::CHandle::cast(parkingPlannerHandle);
        parkingPlanner->setCUDAStream(stream);
    });
}
