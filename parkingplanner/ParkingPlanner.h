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

#ifndef DW_EXPERIMENTAL_PARKINGPLANNER_H_
#define DW_EXPERIMENTAL_PARKINGPLANNER_H_

#include <dw/core/Context.h>
#include <dw/experimental/behaviorplanner/BehaviorPlannerExtra.h> //dwUnstructuredTarget
#include <dw/experimental/pathunstructured/Path.h>                //dwPathPlannerDrivingState

#ifdef __cplusplus
extern "C" {
#endif

// Handle representing a Parking Planner
typedef struct dwParkingPlannerObject* dwParkingPlannerHandle_t;

// Constant handle representing a Parking Planner
typedef struct dwParkingPlannerObject const* dwConstParkingPlannerHandle_t;

// Configurable parameters of the Parking Planner
typedef struct dwParkingPlannerParams
{
    float32_t turnRadius_m; //!< Turning radius of the left/right turn primitives in meters
    uint32_t max_turns;     //!< The maximum number of composite iterations (staright+left+right) before termination
} dwParkingPlannerParams;

// Struct describes states of the parking planner planning space {x,y,heading}
typedef struct dwParkingState
{
    float32_t x;       //!< x position in the planning space
    float32_t y;       //!< y position in the planning space
    float32_t heading; //!< heading angle
} dwParkingState;

/**
 * @brief Initializes the parking planner configurable parameters to default values.
 * 
 * @param[out] params Parking Planner configurable parameters.
 * 
 * @return DW_SUCCESS params initialized successfully. <br>
 *         DW_INVALID_ARGUMENTS params is null pointer. <br>
 */
DW_API_PUBLIC
dwStatus dwParkingPlanner_initDefaultParams(dwParkingPlannerParams* params);

/**
 * @brief Initializes the parking planner handle
 * 
 * @param[in] params Configurable parameters of the Parking Planner.
 * @param[in] contextHandle Context of the Parking Planner.
 * @param[out] parkingPlannerHandle Resulting Parking Planner. 
 * @return DW_SUCCESS Parking Planner initialized successfully. <br>
 *         DW_INVALID_ARGUMENTS parkingPlannerHandle or params is null pointer. <br>
 */
DW_API_PUBLIC
dwStatus dwParkingPlanner_initialize(const dwParkingPlannerParams* params,
                                     dwContextHandle_t contextHandle,
                                     dwParkingPlannerHandle_t* parkingPlannerHandle);

/**
 * @brief Releases the Parking Planner
 * 
 * @param[in,out] parkingPlannerHandle Parking Planner to be released.
 * 
 * @return DW_SUCCESS Parking Planner hanlde released successfully. <br>
 *         DW_INVALID_ARGUMENTS parkingPlannerHandle is null pointer. <br>
 */
DW_API_PUBLIC
dwStatus dwParkingPlanner_release(dwParkingPlannerHandle_t parkingPlannerHandle);

/**
 * @brief Reset the Parking Planner
 * 
 * @param[in,out] parkingPlannerHandle Parking Planner to be reset.
 * 
 * @return DW_SUCCESS Parking Planner reset successfully. <br>
 *         DW_INVALID_ARGUMENTS parkingPlannerHandle is null pointer. <br>
 */
DW_API_PUBLIC
dwStatus dwParkingPlanner_reset(dwParkingPlannerHandle_t parkingPlannerHandle);

/**
 * @brief Sets the target of the Parking Planner
 * 
 * @param[in] t 3d position, heading and timestamp of the goal pose.
 * @param[out] parkingPlannerHandle Parking Planner object handle 
 * 
 * @note 2d position (t.position.x, t.position.y) and heading (t.heading) are used by the Parking planner currently.
 * 
 * @return DW_SUCCESS Parking Planner target set successfully. <br>
 *         DW_INVALID_ARGUMENTS parkingPlannerHandle is null pointer. <br>
 */
DW_API_PUBLIC
dwStatus dwParkingPlanner_setTarget(const dwUnstructuredTarget t,
                                    dwParkingPlannerHandle_t parkingPlannerHandle);

/**
 * @brief Marks the obstacles in the Parking Planner
 * 
 * @param[in] obs Pointer to the first dwObstacle object in the obstacle list.
 * @param[in] size Size of the obstacle list
 * @param[out] parkingPlannerHandle Parking Planner object handle.
 * 
 * @return DW_SUCCESS Parking planner Obstacles set successfully. <br>
 *         DW_INVALID_ARGUMENTS parkingPlannerHandle or obs is null pointer. <br>
 */
DW_API_PUBLIC
dwStatus dwParkingPlanner_setObstacles(const dwObstacle* obs,
                                       size_t size,
                                       dwParkingPlannerHandle_t parkingPlannerHandle);

/**
 * @brief Plan the path for the given scenario.
 * 
 * @note Set the target and obstacles before calling this function.
 * 
 * @param[in,out] parkingPlannerHandle Parking Planner object handle.
 * 
 * @return DW_SUCCESS Path computed successfully. <br>
 *         DW_INVALID_ARGUMENTS parkingPlannerHandle is null pointer. <br>
 */
DW_API_PUBLIC
dwStatus dwParkingPlanner_computePath(dwParkingPlannerHandle_t parkingPlannerHandle);

/**
 * @brief Get the planned path and driving direction at each path point
 * !> Path is defined by a list of points with 2D position and heading at each point
 * !> Driving direction is defined at each path point; Either forward or reverse.
 * 
 * @param[out] path Pointer to the first path point in the list
 * @param[out] dir Pointer to the driving direction of the first path point in the list
 * @param[out] size Size of the lists given by the Parking Planner
 * @param[in] capacity Size of the lists passed in as argument
 * @param[in] parkingPlannerHandle const Parking planner object handle
 * 
 * @note If path doesn't exist, the size is set to 0.
 * 
 * @return DW_SUCCESS Path details obtained successfully. <br>
 *         DW_INVALID_ARGUMENTS parkingPlannerHandle / path / dir / size is null pointer. <br>
 *         DW_OUT_OF_BOUNDS Input array size insufficient to take in output. <br>
 */
DW_API_PUBLIC
dwStatus dwParkingPlanner_getPathDetails(dwParkingState* path,
                                         dwPathPlannerDrivingState* dir,
                                         size_t* size,
                                         size_t capacity,
                                         dwConstParkingPlannerHandle_t parkingPlannerHandle);

/**
 * @brief Get the first segment of the path
 * !> Path is defined by a list of points with 2D position and heading at each point
 * 
 * @note !> Each segment is a portion of the path that can be driven without changing the reverse gear.
 *       !> This API is provided because the motionplanner will query a rail at every step.
 *       !> And driving direction will not change in a rail. 
 * 
 * @param[out] pathSegment Pointer to the first path point
 * @param[out] size Size of the path segment list given by the parking planner
 * @param[in] capacity Size of the list passed in as argument
 * @param[in] parkingPlannerHandle const Parking planner object handle
 * 
 * @return DW_SUCCESS Path Segment obtained successfully. <br>
 *         DW_INVALID_ARGUMENTS parkingPlannerHandle / pathSegment / size is null pointer. <br>
 *         DW_OUT_OF_BOUNDS Input array size insufficient to take in output. <br>
 */
DW_API_PUBLIC
dwStatus dwParkingPlanner_getPathSegment(dwParkingState* pathSegment,
                                         size_t* size,
                                         size_t capacity,
                                         dwConstParkingPlannerHandle_t parkingPlannerHandle);

/**
 * @brief Get the CUDA Stream 
 * 
 * @param[out] stream CUDA stream
 * @param[in] parkingPlannerHandle const Parking planner object handle
 * 
 * @return DW_SUCCESS CUDA Stream obtained successfully.<br>
 *         DW_INVALID_ARGUMENT parkingPlannerHandle is null pointer.<br>
 */
DW_API_PUBLIC
dwStatus dwParkingPlanner_getCUDAStream(cudaStream_t stream,
                                        dwConstParkingPlannerHandle_t parkingPlannerHandle);

/**
 * @brief Set the CUDA Stream 
 * 
 * @param[out] stream CUDA stream; If it is not set explicitly, the default is null pointer
 * @param[in,out] parkingPlannerHandle Parking planner object handle
 * 
 * @return DW_SUCCESS CUDA Stream set successfully. <br>
 *         DW_INVALID_ARGUMENTS parkingPlannerHandle is null pointer. <br>
 */
DW_API_PUBLIC
dwStatus dwParkingPlanner_setCUDAStream(cudaStream_t stream,
                                        dwParkingPlannerHandle_t parkingPlannerHandle);

#ifdef __cplusplus
}
#endif

#endif