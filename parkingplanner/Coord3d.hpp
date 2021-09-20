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

#ifndef DW_PARKINGPLANNER2_COORD_3D_HPP_
#define DW_PARKINGPLANNER2_COORD_3D_HPP_

#include <dw/core/Matrix.hpp>

namespace dw
{
namespace planner
{
/**
 * This struct is used to index the gridCells in the parking space. 
 * The parking space is represented as a 3D grid composed of xy coordinates and heading angle.
 * x denotes index of a position in the x direction
 * y denotes index of a position in the y direction
 * hdg denotes index based on heading angle
 */
struct Coord3d
{
    int32_t x   = 0;
    int32_t y   = 0;
    int32_t hdg = 0;

    Coord3d() = default;

    constexpr CUDA_BOTH_INLINE Coord3d(int32_t x_, int32_t y_, int32_t hdg_)
        : x(x_)
        , y(y_)
        , hdg(hdg_)
    {
    }

    Coord3d(const Coord3d&) = default;

    /**
     * @param[in] pose {x,y,hdg}
     * @param[in] posRes position resolution in [m]
     * @param[in] hdgRes heading resolution in [deg]
     */
    CUDA_BOTH_INLINE Coord3d(const Vector3f& pose,
                             float32_t posRes,
                             float32_t hdgRes)
    {
        x   = static_cast<int32_t>(floor((pose.x() + 0.5f * posRes) / posRes));
        y   = static_cast<int32_t>(floor((pose.y() + 0.5f * posRes) / posRes));
        hdg = static_cast<int32_t>(floor((pose.z() + 0.5f * hdgRes) / hdgRes));
    }

    CUDA_BOTH_INLINE Vector3f getPose(float32_t posRes, float32_t hdgRes) const
    {
        Vector3f pose{};

        pose.x() = static_cast<float32_t>(x) * posRes;
        pose.y() = static_cast<float32_t>(y) * posRes;
        pose.z() = static_cast<float32_t>(hdg) * hdgRes;

        return pose;
    }

}; // Coord3d

CUDA_BOTH_INLINE bool operator==(const Coord3d& lhs, const Coord3d& rhs)
{
    return (lhs.x == rhs.x) &&
           (lhs.y == rhs.y) &&
           (lhs.hdg == rhs.hdg);
}

CUDA_BOTH_INLINE bool operator!=(const Coord3d& lhs, const Coord3d& rhs)
{
    return !(lhs == rhs);
} //Coord3d operators

} // namespace dw
} // namespace planner

#endif // DW_PARKINGPLANNER2_COORD_2D_HPP_