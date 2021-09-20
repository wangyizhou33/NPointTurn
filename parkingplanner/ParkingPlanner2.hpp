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

#ifndef DW_PARKINGPLANNER_PLANNERPLANNER2_HPP_
#define DW_PARKINGPLANNER_PLANNERPLANNER2_HPP_

#include <dw/core/Types.hpp>
#include <dw/core/Object.hpp>
#include <dw/worldmodel/Obstacle.h> // dwObstacle
#include <dw/experimental/parkingplanner/Coord3d.hpp>
#include <algorithm> // std::reverse
#include <dw/experimental/behaviorplanner/BehaviorPlannerExtra.h>
#include <dw/experimental/pathunstructured/Path.h> //dwPathPlannerDrivingState
#include <dw/math/MathUtils.hpp>
#include <dw/experimental/parkingplanner/ParkingPlanner.h>
#include <dw/core/container/UniqueSpan.hpp> // UniqueSpan, UniqueDeviceSpan
#include <functional>

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

#define FRIEND_TEST(test_case_name, test_name) \
    friend class test_case_name##_##test_name##_Test

namespace dw
{
namespace planner
{

constexpr float32_t ROOT2 = 1.414213f; // Root of 2

enum class ManeuverType
{
    STRAIGHT = 0,
    LEFT,
    RIGHT
};

enum ObstacleName
{
    NONE = 0,
    WALL,
    THREE_WALL_MAZE,
    BAY,
    HURDLE,
    PARALLEL
};

struct ParkingPlannerParams
{
    uint32_t max_turns{15};       //Maximum number of composite turns (straight+left+right) //
    float32_t turnRadius_m{5.0f}; //turn radius of the left and right turn primitives //init_default param struct
};

struct GridCell
{
    Coord3d unwarpedPose{};
    Coord3d prevPose{};
    ManeuverType maneuver{};
    bool obstacle{};
    bool reachable{};
    bool reverse{};
    bool gap{true};
    uint8_t iterCount{};
    int16_t src{};
};

CUDA_BOTH_INLINE float32_t deg2Rad(int32_t theta)
{
    return math::deg2Rad(static_cast<float32_t>(theta));
}

/// We exploit parallelism here compared to ParkingPlanner which is
/// conventionally D* & A* graph search
class ParkingPlanner2 : public dw::core::Object
{
    FRIEND_TEST(ParkingPlanner2Test, CAPIThreePointTurn_L0);
    FRIEND_TEST(ParkingPlanner2Test, CAPIObstacles_L0);
    FRIEND_TEST(ParkingPlanner2Test, CAPIBay_L0);
    FRIEND_TEST(ParkingPlanner2Test, CAPIHurdle_L0);
    FRIEND_TEST(ParkingPlanner2Test, CAPIParallel_L0);
    FRIEND_TEST(ParkingPlanner2Test, WarpUnwarp_L0);
    FRIEND_TEST(ParkingPlanner2Test, MiniWarpUnwarp_L0);
    FRIEND_TEST(ParkingPlanner2Test, NextIndex_L0);
    FRIEND_TEST(ParkingPlanner2Test, CheckIncrement_L0);
    friend class ParkingPlanner2TestFixtures;

public:
    /** Constructor
     * Initializing m_start and m_destination
     * Managed memory allocated for Grid table (m_grid) and warped Grid table (m_cuGridWarped) 
     * unwarpedPose initialized and boundaries marked as obstacles using setupTable()
     */
    explicit ParkingPlanner2(const dwParkingPlannerParams* params = nullptr, core::Context* context = nullptr); //add optional arg with passed turn params

    /**
     * m_grid & m_cuGridWarped reset to zero. m_start & m_destination reset to zero.
     * unwarpedPose and boundary obstacls reset using setupTable()
     */
    void reset() override;
    void reset2();

    /**** Functions setting obstacles ****/

    /**
     * bool obstacle in Grid cell corresponding to span of dwObstacle objects marked true
     * @param[in] obstacles is a List of obstacles
     */
    void setObstacles(span<const dwObstacle> obstacles);

    /**** getter functions for parameters ****/

    /**
     * @return: X_SIZE the dimension of the grid (including padding)
     */
    CUDA_BOTH_INLINE static int32_t getSize() { return X_SIZE; }
    CUDA_BOTH_INLINE static int32_t getDim3() { return DIM3; }
    CUDA_BOTH_INLINE static int32_t getHalfDim3() { return HALF_DIM3; }

    /**
     * @return: THETA_STEP the theta dimension of the grid
     */
    CUDA_BOTH_INLINE static uint32_t getThetaStep() { return THETA_STEP; }
    CUDA_BOTH_INLINE static uint32_t getThetaStep3() { return THETA_DIM3; }

    /**
     * @return: RANGE the dimension of the planning space 
     */
    CUDA_BOTH_INLINE static float32_t getRange() { return RANGE; }
    CUDA_BOTH_INLINE static float32_t getRange3() { return RANGE3; }

    /**
     * @return: POS_RES position resolution of the grid
     */
    CUDA_BOTH_INLINE static float32_t getPosRes() { return POS_RES; }
    CUDA_BOTH_INLINE static float32_t getPosRes3() { return POS_RES3; }

    /**
     * @return: HDG_RES heading resolution of the grid
     */
    CUDA_BOTH_INLINE static float32_t getHdgRes() { return HDG_RES; }
    CUDA_BOTH_INLINE static float32_t getHdgRes3() { return HDG_RES_DEG3; }

    /**
     * @return Total No. of cells in the Grid table (including padding)
     */
    CUDA_BOTH_INLINE static uint32_t getGridSize() { return X_SIZE * Y_SIZE * THETA_STEP; }

    /**
     * @return Total No. of cells in the Grid table (No Padding)
     */
    CUDA_BOTH_INLINE static uint32_t getGridSize3() { return DIM3 * DIM3 * THETA_DIM3; }

    /**
     * @param[in]: t is a dwUnstructuredTarget describing the desired destination position/positions
     */
    void setTargetPose(const dwUnstructuredTarget& t);

    /**
     * @return: True if m_path is not empty; false if empty. 
     */
    bool hasPath() const { return !m_path.empty(); }

    /**
     *@return: The complete generated path as a span of Vector3f path points
     */
    core::span<const Vector3f> getPath() const { return make_span(m_path.data(), m_path.size()); }

    /**
     * @return: The first segment of the path, i.e. the portion that can be driven without changing reverse gear
     */
    core::span<const Vector3f> getPathSegment() const;

    /**
     * @return: The driving direction forward/reverse at each path point 
     */
    core::span<const dwPathPlannerDrivingState> getDrivingDirs() const { return make_span(m_pathDrivingDirs.data(), m_pathDrivingDirs.size()); }
    core::span<const int32_t> getTurnTypes() const { return make_span(m_turnEachStep.data(), m_turnEachStep.size()); }

    /**
     * @param[in] stream is set as the cuda stream
     */
    void setCUDAStream(cudaStream_t stream) { m_cuStream = stream; }

    /**
     * @return: m_cuStream the member denoting the cuda stream
     */
    cudaStream_t getCUDAStream() const { return m_cuStream; }

    /**** process function ****/
    /**
     * Attempts to make the Grid cell corresponding to m_destination reachable by starting from Grid cell corresponding to m_start.
     * Progresses one composite turn (staright + left + right) at a time until destination is reachable
     * Upperbound for number of composite turn attempts = max_turns
     */
    void process();

    void process2();

    void processNew();

    void timeGPU();
    void timeSOL();

    /**** static inline functions ****/
    /**
     * @param[in]: x index of the gridcell
     * @param[in]: y index of the gridcell
     * @param[in]: theta index of the gridcell
     * @return: memory offset between pointed to the first Grid cell and Grid cell corresponding to {x,y,theta} in the table
     */
    CUDA_BOTH static int32_t getCell(int32_t x, int32_t y, int32_t theta);

    /**
     * @param[in] table is a span denoting the Grid
     * @param[in]: x index of the gridcell
     * @param[in]: y index of the gridcell
     * @param[in]: theta index of the gridcell
     * @return: pointer to Grid cell corresponding to {x,y,theta} in the grid corresponding to table 
     */
    CUDA_BOTH static GridCell& getCell(core::span<GridCell> table, int32_t x, int32_t y, int32_t theta);

    /**
     * @param[in] table is a span denoting the Grid
     * @param[in] pos is a Coord3d corresponding to a cell position
     * @return: pointer to Grid cell corresponding to   
     */
    CUDA_BOTH static GridCell& getCell(core::span<GridCell> table, const Coord3d& pos);

    /**** Bound check functions ****/
    /**
     * @param[in] pos is a Coord3d pose corresponding to a cell position
     * @return: bool is true if pos is within plan space (without padding)
     */
    CUDA_BOTH_INLINE static bool withinPlanSpace(Coord3d& pos);

    /**
     * @param[in]: x index of the gridcell
     * @param[in]: y index of the gridcell
     * @param[in]: theta index of the gridcell
     * @return: bool is true if {x,y,theta} is within plan space (without padding)
     */
    CUDA_BOTH_INLINE static bool withinPlanSpace(int32_t x, int32_t y, int32_t theta = 0);

    /**
     * @param[in] pos is a Coord3d pose corresponding to a cell position
     * @return: bool is true if pos is within plan space appended with padding
     */
    CUDA_BOTH_INLINE static bool withinBounds(Coord3d& pos);

    /**
     * @param[in]: x index of the gridcell
     * @param[in]: y index of the gridcell
     * @param[in]: theta index of the gridcell
     * @return: bool is true if {x,y,theta} is within plan space appended with padding
     */
    CUDA_BOTH_INLINE static bool withinBounds(int32_t x, int32_t y, int32_t theta = 0);

    /**** Cell Warping functions ****/

    /**
     * Warping: The act of rotating/translating each theta slice (theta = constant plane) in the {x,y,theta} grid based on its theta value. 
     * Straight warp: We rotate the theta slice in such a way that the heading direction aligns with the x-direction.
     *                Effectively, each theta slice is rotated by -theta. As a result, all poses reachable from a given pose lie along x-axis. 
     * 
     * Left/Right warp: The points on a turn of fixed curvature form a helix in the {x,y,theta} grid. 
     *                  We translate each theta slice long the xy plane in such a way that the helix becomes a straight line along theta axis. 
     *                  Let's say the poses {x1,y1,theta1}, {x2,y2,theta2} lie on the same circle as {x0,y0,0}. After translating the theta slices,
     *                  the points will be moved to {x0,y0,theta1} and {x0,y0,theta2} respectively. 
     * 
     * Note: The direction of translation of theta slices would be different for left and right turns. 
     */

    /**
     * Finds the warped cell position corresponding to the given cell position (straight warp)
     * @param[in] dst is a Coord3d unwarped (a position denoting a cell in the m_grid)
     * @return Coord3d warped (a position denoting a cell in the m_cuGridWarped during straight warp)
     */
    CUDA_BOTH static Coord3d warpStraight(const Coord3d& dst, float32_t posRes, float32_t hdgRes);

    /** Straight Warps the given x,y,theta. 
     * Straight warp: Finds the x,y in the coordinate system rotated by theta. Same theta is maintained.
     * @param[in] dstPose is a configuration x,y,theta in the planning space
     * @return x, y in the coordinate system rotated by theta. Theta remains the same.
     */
    CUDA_BOTH static Vector3f warpStraight(const Vector3f& dstPose);

    /** Finds the unwarped cell position corresponding to the given warped cell position (straight warp)
     * @param[in] dst is a Coord3d warped (a position denoting a cell in the m_cuGridWarped during straight warp)
     * @return Coord3d unwarped ( a position denoting a cell in the m_grid)
     */
    CUDA_BOTH static Coord3d unwarpStraight(const Coord3d& dst, float32_t posRes, float32_t hdgRes);

    /** Unwarps the given x,y,theta. (from straight warp)
     * Straight warp: Finds the x,y in the coordinate system rotated by theta. Same theta is maintained.
     * This function performs the inverse.
     * @param[in] dstPose is a warped x,y,theta pose 
     * @return configuration x,y,theta in the planning space.
     */
    CUDA_BOTH static Vector3f unwarpStraight(const Vector3f& dstPose);

    /** Finds the warped cell position corresponding to a given cell position (left warp)
     * @param[in] dst is a Coord3d unwarped (a position denoting a cell in the m_grid)
     * @return Coord3d warped pose (a position denoting a cell in the m_cuGridWarped during left warp)
     */
    CUDA_BOTH static Coord3d warpLeft(const Coord3d& dst, float32_t posRes, float32_t hdgRes, float32_t turnRadius_m);

    /** Left Warps the given x,y,theta. 
     * Left warp: x,y position of theta = 0 point on the left turn circle found. Same theta value maintained.
     * @param[in] dstPose is a configuration x,y,theta in planning space
     * @return left warped x,y,theta 
     */
    CUDA_BOTH static Vector3f warpLeft(const Vector3f& dstPose, float32_t turnRadius_m);

    /** Finds the unwarped cell position corresponding to the given warped cell position (left warp)
     * @param[in] dst is a Coord3d warped (a position denoting a cell in the m_cuGridWarped during left warp)
     * @return Coord3d unwarped ( a position denoting a cell in the m_grid)
     */
    CUDA_BOTH static Coord3d unwarpLeft(const Coord3d& dst, float32_t posRes, float32_t hdgRes, float32_t turnRadius_m);

    /** Unwarps the given x,y,theta. (from left warp)
     * Left warp: x,y position of theta = 0 point on the left turn circle found. Same theta value maintained.
     * This function performs the inverse.
     * @param[in] dstPose is a warped x,y,theta pose. 
     * @return configuration x,y,theta in the planning space.
     */
    CUDA_BOTH static Vector3f unwarpLeft(const Vector3f& dstPose, float32_t turnRadius_m);

    /** Finds the warped cell position corresponding to a given cell position (right warp)
     * @param[in] dst is a Coord3d unwarped (a position denoting a cell in the m_grid)
     * @return Coord3d warped pose (a position denoting a cell in the m_cuGridWarped during right warp)
     */
    CUDA_BOTH static Coord3d warpRight(const Coord3d& dst, float32_t posRes, float32_t hdgRes, float32_t turnRadius_m);

    /** Right Warps the given x,y,theta. 
     * Right warp: x,y position of theta = 0 point on the right turn circle found. Same theta value maintained.
     * @param[in] configuration x,y,theta in planning space
     * @return right warped x,y,theta 
     */
    CUDA_BOTH static Vector3f warpRight(const Vector3f& dstPose, float32_t turnRadius_m);

    /** Finds the unwarped cell position corresponding to the given warped cell position (right warp)
     * @param[in] dst is a Coord3d warped (a position denoting a cell in the m_cuGridWarped during right warp)
     * @return Coord3d unwarped ( a position denoting a cell in the m_grid)
     */
    CUDA_BOTH static Coord3d unwarpRight(const Coord3d& dst, float32_t posRes, float32_t hdgRes, float32_t turnRadius_m);

    /** Unwarps the given x,y,theta. (from right warp)
     * Right warp: x,y position of theta = 0 point on the right turn circle found. Same theta value maintained.
     * This function performs the inverse.
     * @param[in] dstPose is warped x,y,theta pose. 
     * @return configuration x,y,theta in the planning space.
     */
    CUDA_BOTH static Vector3f unwarpRight(const Vector3f& dstPose, float32_t turnRadius_m);

    //**** Generic Cost functions;
    CUDA_BOTH_INLINE static int32_t volIndex(int32_t x, int32_t y, int32_t theta);

    CUDA_BOTH_INLINE static bool boundCheck(int32_t x, int32_t y, int32_t theta);

    CUDA_BOTH_INLINE static int32_t turnIndex(int32_t x, int32_t y, int32_t theta, bool left, float32_t turnRadius);
    CUDA_BOTH_INLINE static Coord3d turnIndexPlain(int32_t x, int32_t y, int32_t theta, bool left, float32_t turnRadius);

    inline static int32_t nextIndex(int32_t x, int32_t y, int32_t theta, float32_t turnRadius, bool left, bool reverse);

    CUDA_BOTH_INLINE static Coord3d index2Coord(int32_t index, int32_t dim);

    CUDA_BOTH_INLINE static Coord3d originCoord(int32_t x, int32_t y, int32_t theta, bool left, float32_t turnRadius);

    void costSweepA(uint16_t* Co, uint16_t* Obs, bool left, float32_t turnRadius, cudaStream_t cuStream = nullptr);
    void costSweepB(uint16_t* Co, uint16_t* Obs, bool left, float32_t turnRadius, cudaStream_t cuStream = nullptr);
    void costSweepR10RightForward(cudaStream_t cuStream = nullptr);
    void costSweepR20RightForward(cudaStream_t cuStream = nullptr);
    void costSweepR20LeftForward(cudaStream_t cuStream = nullptr);
    void costSweepR10LeftForward(cudaStream_t cuStream = nullptr);
    void costSweepR10RightReverse(cudaStream_t cuStream = nullptr);
    void costSweepR20RightReverse(cudaStream_t cuStream = nullptr);
    void costSweepR20LeftReverse(cudaStream_t cuStream = nullptr);
    void costSweepR10LeftReverse(cudaStream_t cuStream = nullptr);
    void costSweepAll(cudaStream_t cuStream = nullptr);
    void costSweepAll64(cudaStream_t cuStream = nullptr);

    void costUp16(cudaStream_t cuStream = nullptr);
    void costSweep32(cudaStream_t cuStream = nullptr);
    void costSweep64(cudaStream_t cuStream = nullptr);
    void copy32To16(cudaStream_t cuStream = nullptr);
    void copy64To16(cudaStream_t cuStream = nullptr);
    void sections16(cudaStream_t cuStream = nullptr);
    void directCopy16(cudaStream_t cuStream = nullptr);
    void directCopy32(cudaStream_t cuStream = nullptr);
    void directCopyUnroll32(cudaStream_t cuStream = nullptr);
    void directCopy64(cudaStream_t cuStream = nullptr);
    //void loopUnroll16(cudaStream_t cuStream = nullptr);
    void costTransition(cudaStream_t cuStream = nullptr);

    void setTransitionCost();
    void setTurnIncrement();
    void processCost();
    void processCostStep();
    void computeIndexIncrement();
    void computeIndexIncrement4();
    void printTrCost(); //Remove function after tuning
    void intermediateTestKernel();
    void testSweep32();
    void testSweep64();

    static constexpr int32_t TURN_TYPES_NUM = 8;
    static constexpr int32_t THETA_DIM3     = 512;

private:
    //**** Generic cost functions;
    void processSweepStep();
    bool destinationCheck(cudaStream_t cuStream = nullptr);
    void minCost(cudaStream_t cuStream = nullptr);
    void copyDeviceToHostMain();
    void copyDeviceToHostGrid();
    void copyDeviceToHostObstacle();
    void initializeGrid();
    void initializeGrid32();
    void initializeGrid64();
    void initializeTrCost();
    void initObstacle(cudaStream_t cuStream = nullptr);
    bool backTraceCost();
    void printBackTrace();
    void printFullPath();
    uint16_t pathToCost(uint32_t startIndex, uint32_t endIndex);
    int32_t turnNextIndex(Coord3d coord, int32_t turnType);
    uint16_t turnStepCost(int32_t turnType);

    /**** parameters ****/

    static constexpr float32_t POS_RES = 0.5f;  // position resolution 50 cm // need to change to 10 finally
    static constexpr float32_t HDG_RES = 1.0f;  // heading resolution 1 deg
    static constexpr float32_t RANGE   = 30.0f; // x and y range from -30 meters to 30 meters

    static constexpr int32_t X_LENGTH = static_cast<int32_t>(2 * RANGE / POS_RES); //X direction, planning space No. of Grid cells
    static constexpr int32_t Y_LENGTH = static_cast<int32_t>(2 * RANGE / POS_RES); //Y direction, planning space No. of Grid cells

    static constexpr int32_t X_SIZE     = static_cast<int32_t>(X_LENGTH * ROOT2); //X Length in No. of Grid cells including padding
    static constexpr int32_t Y_SIZE     = static_cast<int32_t>(Y_LENGTH * ROOT2); //Y Length in No. of Grid cells including padding
    static constexpr int32_t THETA_STEP = static_cast<int32_t>(360.0f / HDG_RES); //theta direction, Length in No. of Grid cells
    static const Coord3d INVALID_POSE;

    /**** parameters Sol test ****/
    static constexpr uint32_t PADDING = static_cast<uint32_t>(2.0f * 5.0f / POS_RES);
    static constexpr uint32_t X_DIM   = static_cast<uint32_t>(X_LENGTH) + 2 * (PADDING);
    static constexpr uint32_t Y_DIM   = static_cast<uint32_t>(Y_LENGTH) + 2 * (PADDING);
    static constexpr uint32_t X_CELLS = static_cast<uint32_t>(static_cast<float32_t>(X_DIM) / 32.0f); //TODO (JK): If not divisible by 32?
    static constexpr uint32_t Y_CELLS = Y_DIM;

    /**** V3 parameters generic cost ****/

    static constexpr float32_t RANGE3          = 32.0f; // actual range will be less than this
    static constexpr float32_t POS_RES3        = 0.25f;
    static constexpr int32_t DIM3              = static_cast<int32_t>((RANGE3 + 0.5f * POS_RES3) / POS_RES3);
    static constexpr int32_t HALF_DIM3         = (DIM3 + 1) / 2;
    static constexpr float32_t MAX_TURNRADIUS3 = 20.0f;
    static constexpr int32_t PADDING3          = static_cast<int32_t>((MAX_TURNRADIUS3 + 0.5f * POS_RES3) / POS_RES3);
    static constexpr int32_t KERNEL_DIM3       = (DIM3 + 2 * PADDING3);
    static constexpr int32_t HALF_KERNEL3      = (KERNEL_DIM3 + 1) / 2;
    static constexpr float32_t HDG_RES_RAD3    = 2 * M_PI / static_cast<float32_t>(THETA_DIM3);
    static constexpr float32_t HDG_RES_DEG3    = 360.0f / static_cast<float32_t>(THETA_DIM3);
    static constexpr float32_t MIN_TURNRADIUS3 = 10.0f;
    static constexpr float32_t WHEELBASE3      = 3.8f;
    static constexpr float32_t STEER_RATIO3    = 2 * M_PI * MIN_TURNRADIUS3 / WHEELBASE3; //Bicycle model of vehicle
    static constexpr int32_t MAX_TURNS         = 8;
    static constexpr uint16_t MAX_COST         = static_cast<uint16_t>(64250u);

    /**
     * @return number of primitives needed to reach destination
     * private function because used only by the gui
     */
    uint32_t getNumberOfPrimitives() const { return static_cast<uint32_t>(m_maneuverList.size()); }
    uint32_t getNumberOfPrimitives3() const { return static_cast<uint32_t>(m_turns.size()); }

    /**** Functions initializing a parking scenario ****/
    /**
     * @param[in] start is assigned to m_start
     */
    void setStart(const Coord3d& start);

    /**
     * m_start is set as the input indices and reachability of the corresponding cell is set to true
     * @param[in]: x index of the gridcell
     * @param[in]: y index of the gridcell
     * @param[in]: theta index of the gridcell
     */
    void setStart(int32_t x, int32_t y, int32_t theta);

    void setResetStart(const Coord3d& start);
    void setResetStart(int32_t x, int32_t y, int32_t theta);

    /**
     * set origin of cuGrid reachable
     */
    void setStartGPU();

    /**
     * set m_start to (0,0,0) 
     */
    inline void setEmptyStart() { setStart({}); }

    /**
     * @param[in] dest is assigned to m_destination
     */
    void setDestination(const Coord3d& dest);

    /**
     * Set m_destination as the input indices
     * @param[in]: x index of the destination cell
     * @param[in]: y index of the destination cell
     * @param[in]: theta index of the destination cell
     */
    void setDestination(int32_t x, int32_t y, int32_t theta);

    /**
     * bool obstacle in Grid cell corresponding to input Coord3d marked true
     * @param[in]: obs is a Coord3d position correspoding to a cell
     */
    void setObstacle(const Coord3d& obs);

    /**
     * bool obstacle in Grid cell corresponding to input x,y,theta marked true
     * @param[in]: x index of the gridcell
     * @param[in]: y index of the gridcell
     * @param[in]: theta index of the gridcell
     */
    void setObstacle(int32_t x, int32_t y, int32_t theta);

    /**
     * Sets up a special obstacle
     * Obstacle description: Three straight parellel walls with small openings at different points
     * @param[in]: cuStream sets the cuda stream of the operation
     */
    void setThreeWallMaze(cudaStream_t cuStream = nullptr);

    /**
     * Sets up a special obstacle
     * Obstacle description: wall parellel to y axis. 
     * @param[in]: cuStream sets the cuda stream of the operation
     */
    void setWall(cudaStream_t cuStream = nullptr);

    /**
     * Sets up a special obstacle
     * Obstacle description: Closed bay with a small opening
     * @param[in]: cuStream sets the cuda stream of the operation
     */
    void setBay(cudaStream_t cuStream = nullptr);

    /**
     * Sets up a special obstacle
     * Obstacle description: Rectangular hurdle close to the start pose
     * @param[in]: cuStream sets the cuda stream of the operation
     */
    void setHurdle(cudaStream_t cuStream = nullptr);

    /**
     * Sets up a special obstacle
     * Obstacle description: Two Rectangular hurdles emulating the parellel parking situation approximately
     * @param[in]: cuStream sets the cuda stream of the operation
     */
    void setParallel(cudaStream_t cuStream = nullptr);

    /**
     * set the unwarpedPose of all Grid cells in the m_grid, based on the position of the cell in the Grid.
     * @param[in]: cuStream sets the cuda stream of the operation
     */
    void setUnwarpedPose(cudaStream_t cuStream = nullptr);

    /**
     * Pick one among the special obstacles
     * @param[in]: obs ObstacleName that denotes a special obstacle
     */
    void setSpecialObstacle(ObstacleName obs);

    /**** sweep methods ****/

    /**
     * Calls a kernel function that sweeps the warped array
     * Cells corresponding configurations reachable by the straight maneuver (straight/right/left) will be made reachable
     * @param[in] table pointer to the first cell of the warped table
     * @param[in] cuStream sets the cuda stream of the operation
     */
    static void sweepStraight(GridCell* table, cudaStream_t cuStream = nullptr);

    static void sweepStraight2(GridCell* table, uint8_t iter, cudaStream_t cuStream = nullptr);

    static void sweepStraightNew(GridCell* table, uint8_t iter, cudaStream_t cuStream = nullptr);

    /**
     * Calls a kernel function that sweeps the warped array
     * Cells corresponding configurations reachable by the respective maneuver (right/left) will be made reachable
     * @param[in] table pointer to the first cell of the warped table
     * @param[in] maneuver specifies the type of maneuver left or right
     * @param[in] cuStream sets the cuda stream of the operation
     */
    static void sweepArc(GridCell* table, ManeuverType maneuver, cudaStream_t cuStream = nullptr);

    static void sweepArc2(GridCell* table, ManeuverType maneuver, uint8_t iter, cudaStream_t cuStream = nullptr);

    /**** Table Warping methods ****/

    /**
     * Warping: The act of rotating/translating each theta slice (theta = constant plane) in the {x,y,theta} grid based on its theta value. 
     * Straight warp: We rotate the theta slice in such a way that the heading direction aligns with the x-direction.
     *                Effectively, each theta slice is rotated by -theta. As a result, all poses reachable from a given pose lie along x-axis. 
     * 
     * Left/Right warp: The points on a turn of fixed curvature form a helix in the {x,y,theta} grid. 
     *                  We translate each theta slice long the xy plane in such a way that the helix becomes a straight line along theta axis. 
     *                  Let's say the poses {x1,y1,theta1}, {x2,y2,theta2} lie on the same circle as {x0,y0,0}. After translating the theta slices,
     *                  the points will be moved to {x0,y0,theta1} and {x0,y0,theta2} respectively. 
     * 
     * Note: The direction of translation of theta slices would be different for left and right turns. 
     */

    /**
     * Straight warp: Finds the x,y in the coordinate system rotated by theta. Same theta is maintained.
     * Each theta slice is rotated by the theta angle
     * @param[out] out is the pointer to the first cell in output warped/unwarped table (Straight warp)
     * @param[in] in is the pointer to the first cell in input unwarped/warped table
     * @param[in] posRes position resolution of the grid 
     * @param[in] hdgRes heading resolution of the grid 
     * @param[in]: cuStream sets the cuda stream of the operation
     * @param [in] unwarp: bool denoting if the function warp or unwarp
     */
    static void warpStraight(GridCell* out, GridCell* in, float32_t posRes, float32_t hdgRes, cudaStream_t cuStream = nullptr, bool unwarp = false);

    static void warpStraightNew(GridCell* out, GridCell* in, float32_t posRes, float32_t hdgRes, cudaStream_t cuStream = nullptr, bool unwarp = false);

    /**
     * Left warp: x,y position of theta = 0 point on the left turn circle found. Same theta value maintained.
     * Left turn corresponds to a helix in the x,y,theta configuration space
     * Each such helix is warped into a straight line parellel to theta axis
     * @param[out] out is the pointer to the first cell in output warped/unwarped table (Left warp)
     * @param[in] in is the pointer to the first cell in input unwarped/warped table
     * @param[in] posRes position resolution of the grid 
     * @param[in] hdgRes heading resolution of the grid 
     * @param[in]: cuStream sets the cuda stream of the operation
     * @param [in] unwarp: bool denoting if the function warp or unwarp
     */
    static void warpLeft(GridCell* out, GridCell* in, float32_t posRes, float32_t hdgRes, float32_t turnRadius_m, cudaStream_t cuStream = nullptr, bool unwarp = false);

    /**
     * Right warp: x,y position of theta = 0 point on the right turn circle found. Same theta value maintained.
     * Right turn corresponds to a helix in the x,y,theta configuration space
     * Each such helix is warped into a straight line parellel to theta axis
     * @param[out] out is the pointer to the first cell in output warped/unwarped table (Right warp)
     * @param[in] in is the pointer to the first cell in input unwarped/warped table
     * @param[in] posRes : position resolution of the grid 
     * @param[in] hdgRes : heading resolution of the grid
     * @param[in]: cuStream sets the cuda stream of the operation
     * @param [in] unwarp : bool denoting if the function warp or unwarp
     */
    static void warpRight(GridCell* out, GridCell* in, float32_t posRes, float32_t hdgRes, float32_t turnRadius_m, cudaStream_t cuStream = nullptr, bool unwarp = false);

    /**
     * Analysis functions:
     * sweepAdd() only add and straight sweep 
     * sweepAddLeft() only add and sweep along the access pattern
     */
    static void sweepAdd1(uint8_t* cell, cudaStream_t cuStream = nullptr);
    static void sweepAdd2(uint8_t* cell, cudaStream_t cuStream = nullptr);

    static void sweepAdd16(uint16_t* cell, cudaStream_t cuStream = nullptr);

    static void sweepAddLeft(uint8_t* cell, float32_t posRes, float32_t hdgRes, float32_t turnRadius, cudaStream_t cuStream = nullptr);
    static void sweepAddLeft16(uint16_t* cell, float32_t posRes, float32_t hdgRes, float32_t turnRadius, cudaStream_t cuStream = nullptr);
    static void kernelLeft(uint8_t* cell, float32_t posRes, float32_t hdgRes, float32_t turnRadius, uint8_t iter, cudaStream_t cuStream = nullptr);
    static void kernelRight(uint8_t* cell, float32_t posRes, float32_t hdgRes, float32_t turnRadius, uint8_t iter, cudaStream_t cuStream = nullptr);

    static void bitSweepLeft(uint32_t* RbO, const uint32_t* Fb, const uint32_t* RbI, float32_t turnRadius, cudaStream_t cuStream = nullptr);
    static void bitSweepLeftCPU(uint32_t* RbO, const uint32_t* Fb, const uint32_t* RbI, float32_t turnRadius, cudaStream_t cuStream = nullptr);
    static void bitSweepRightCPU(uint32_t* RbO, const uint32_t* Fb, const uint32_t* RbI, float32_t turnRadius, cudaStream_t cuStream = nullptr);

    static void ourMemCpy32(uint32_t* out, uint32_t* in, int32_t nElemsXY, cudaStream_t cuStream = nullptr);
    static void ourMemCpy32Direct(uint32_t* out, uint32_t* in, int32_t nElems, cudaStream_t cuStream = nullptr);

    /**
     * 
     * Sets the bool obstacle in the cells corresponding to the position of list of obstacle object as true
     * @param[out] out is a pointer to first cell in the unwarped table
     * @param[in] obstacles is a list of obstacle objects
     * @param[in] obstacleCount is a the size of the obstacle list
     * @param[in] posRes : position resolution of the grid 
     * @param[in] hdgRes : heading resolution of the grid
     * @param[in]: cuStream sets the cuda stream of the operation
     */
    static void prepareOccupancyGrid(GridCell* out,
                                     core::span<const dwObstacle> obstacles, // device memory
                                     size_t obstacleCount,
                                     float32_t posRes,
                                     float32_t hdgRes,
                                     cudaStream_t cuStream = nullptr);
    /**
     * Sets the bool obstacle true in m_pTable for cells corresponding to obstacle objects in m_cuObstacles
     */
    void prepareOccupancyGrid();

    /**** process methods ****/

    /**
     * @return: true if cell corresponding to m_destination is reaachable
     */
    inline bool isDestinationReached() const { return getCell(m_grid.get(), m_destination).reachable; }

    bool isDestinationReachedGPU();

    /**
     * Asserts if the No.of turns is less than max_turns
     * @param[in] turnCount Number of turns made
     * @return: bool true if turnCount is greater than max_turns
     */
    inline bool isMaxTurnsReached(uint32_t turnCount) const { return turnCount > m_parkingPlannerParams.max_turns; }

    /**
     * Makes one composite turn (straight + left + right) reachable from every reachable pose
     * warps straight -> sweeps -> unwarps
     * warps left -> sweeps -> unwarps
     * warps right -> sweeps -> unwarps
     */
    void processOneTurn();

    void processOneTurn2(uint8_t iter);

    void processOneTurnNew(uint8_t iter);

    /** Makes all straight poses reachable from every reachable pose
     * warps straight -> sweeps -> unwarps
     */
    void processStraight();

    void processStraight2(uint8_t iter);

    void processStraightNew(uint8_t iter);

    /** Makes all poses accessible with a left turn from every reachable pose reachable
     * warps left -> sweeps -> unwarps
     */
    void processLeft();

    void processLeft2(uint8_t iter);

    /** Makes all poses accessible with a right turn from every reachable pose reachable
     * warps right -> sweeps -> unwarps
     */
    void processRight();

    void processRight2(uint8_t iter);

    /**
     *  fills m_vertices, end points of segments in path start -> destination. Only valid if return is true. Empty otherwise.
     *  fills m_maneuverList, primitive maneuvers of each segment from start -> destination. 
     *  fills m_segmentDirs, List of booleans telling if a segment is driver with reverse or straight. 
     * @return true if a trace is found from m_destination to m_start
     */
    bool backtrace();

    bool backtrace2();

    /**** Output generation functions ****/
    /**
     * Intermediate way points are generated between segment end points  & tail for straight maneuver
     * @param[in] head : segment start pose
     * @param[in] tail : segment end pose
     * @param[in] maneuver : maneuver taken in the segment
     * @param[in] reverse : true if revgear is engaged
     */
    void generateStraight(const Vector3f& head, const Vector3f& tail, const bool reverse);

    /**
     * Intermediate way points are generated between segment end points  & tail for left maneuver
     * @param[in] head : segment start pose
     * @param[in] tail : segment end pose
     * @param[in] reverse : true if revgear is engaged
     * @param[in] left : true if maneuver is left, false for right
     * @param[in] warpFunc : warp function warpLeft or warpRight
     * @param[in] unwarpFunc :unwarp function unwarpLeft or unwarpRight
     */
    void generateArc(const Vector3f& head,
                     const Vector3f& tail,
                     const bool reverse,
                     const bool left,
                     std::function<Vector3f(const Vector3f&, float32_t)> warpFunc,
                     std::function<Vector3f(const Vector3f&, float32_t)> unwarpFunc);

    /**
     * Build complete path with incremental way points based on the segment end points and primitive maneuvers
     * Uses m_vertices, m_maneuverList and m_segmentDirs to fill m_path and m_pathDrivingDirs
     */
    void buildPath();
    void buildPath2();

    Coord3d m_start;
    Coord3d m_destination; //starting and destination poses for a simulation

    void copyGridHostToDevice();
    void copyGridDeviceToHost();

    void copyGridNewHostToDevice();
    void copyGrid16HostToDevice();
    void copyGrid32HostToDevice();
    void copyGridNewDeviceToHost();
    void copyGrid16DeviceToHost();
    void copyGrid32DeviceToHost();
    //void analyseKernels();

    /** 
     * host memory of grid (unwarped)
     */
    UniqueSpan<GridCell> m_grid;
    UniqueSpan<uint8_t> m_gridNew;
    UniqueSpan<uint16_t> m_grid16;
    UniqueSpan<uint32_t> m_grid32;

    UniqueSpan<uint32_t> m_reach0;
    UniqueSpan<uint32_t> m_reach1;
    UniqueSpan<uint32_t> m_reach2;
    UniqueSpan<uint32_t> m_reach3;
    UniqueSpan<uint32_t> m_free;

    /**
     * device memory of the grid, unwarped and warped
     */
    UniqueDeviceSpan<GridCell> m_cuGrid{};
    UniqueDeviceSpan<GridCell> m_cuGridWarped{}; // no need to have a host warped table

    UniqueDeviceSpan<uint8_t> m_cuGridNew{};
    UniqueDeviceSpan<uint16_t> m_cuGrid16{};
    UniqueDeviceSpan<uint32_t> m_cuGrid32{};

    UniqueDeviceSpan<uint32_t> m_cuReach0{};
    UniqueDeviceSpan<uint32_t> m_cuReach1{};
    UniqueDeviceSpan<uint32_t> m_cuReach2{};
    UniqueDeviceSpan<uint32_t> m_cuReach3{};
    UniqueDeviceSpan<uint32_t> m_cuFree{};

    /**
     * New host memory 
     */
    UniqueSpan<uint16_t> m_gridMain{};
    UniqueSpan<uint16_t> m_gridObs{};
    UniqueSpan<uint16_t> m_gridTurns{};

    /**
     * New device memory
     */
    UniqueDeviceSpan<uint16_t> m_cuGridObs{};
    UniqueDeviceSpan<uint16_t> m_cuGridMain{};
    UniqueDeviceSpan<uint16_t> m_cuGridTurns{};

    /**
     * 64 Implementation
     */
    UniqueDeviceSpan<uint64_t> m_cuGridTurns64{};
    UniqueDeviceSpan<uint64_t> m_cuGridObs64{};
    /**
     * Copy test 32 memory
     */
    UniqueDeviceSpan<uint32_t> m_cuOut32{};
    UniqueDeviceSpan<uint32_t> m_cuIn32{};

    /**
     * Copy test 64 memory
     */
    UniqueDeviceSpan<uint64_t> m_cuOut64{};
    UniqueDeviceSpan<uint64_t> m_cuIn64{};

    uint16_t m_trCost[(TURN_TYPES_NUM+1) * TURN_TYPES_NUM]{};
    uint8_t m_turnIncrementR10[THETA_DIM3]{};
    uint8_t m_turnIncrementR20[THETA_DIM3]{};
    uint8_t m_turnIncrementR10Four[THETA_DIM3]{};

    /**
     * obstacle info
     */
    UniqueDeviceSpan<dwObstacle> m_cuObstacles{};
    static constexpr size_t MAX_OBSTABLE_NUM = 200; // m_cuObstacles capacity
    size_t m_obstacleCount                   = 0;   // m_cuObstacles size

    /**** path related members ****/
    // container capacity
    static constexpr size_t MAX_MANEUVER_NUM   = 11;
    static constexpr size_t MAX_PATH_POINT_NUM = 5000;

    ParkingPlannerParams m_parkingPlannerParams{};

    VectorFixed<ManeuverType, MAX_MANEUVER_NUM> m_maneuverList{};
    VectorFixed<bool, MAX_MANEUVER_NUM> m_segmentDirs{};
    VectorFixed<Vector3f, MAX_PATH_POINT_NUM> m_path{};
    VectorFixed<int32_t, MAX_PATH_POINT_NUM> m_turnEachStep{};
    VectorFixed<dwPathPlannerDrivingState, MAX_PATH_POINT_NUM> m_pathDrivingDirs{};
    VectorFixed<int32_t, MAX_TURNS + 20> m_turns{};
    VectorFixed<Vector3f, MAX_TURNS + 21> m_vertices{};
    int32_t m_startTurnType{};
    VectorFixed<int32_t, MAX_PATH_POINT_NUM> m_pathTheta{};

    cudaStream_t m_cuStream = nullptr;
}; // ParkingPlanner2

// @note: the bound includes padding
CUDA_BOTH_INLINE bool ParkingPlanner2::withinBounds(Coord3d& pos)
{
    return (pos.x >= (-1 * (X_SIZE >> 1))) && (pos.x <= ((X_SIZE >> 1) - 1)) &&
           (pos.y >= (-1 * (Y_SIZE >> 1))) && (pos.y <= ((Y_SIZE >> 1) - 1)) &&
           (pos.hdg < THETA_STEP);
}

CUDA_BOTH_INLINE bool ParkingPlanner2::withinBounds(int32_t x, int32_t y, int32_t theta)
{
    Coord3d pos = {x, y, theta};
    return ParkingPlanner2::withinBounds(pos);
}

// @note: the bound does NOT include padding
CUDA_BOTH_INLINE bool ParkingPlanner2::withinPlanSpace(Coord3d& pos)
{
    return (pos.x >= (-1 * (X_LENGTH >> 1))) && (pos.x <= ((X_LENGTH >> 1))) &&
           (pos.y >= (-1 * (Y_LENGTH >> 1))) && (pos.y <= ((Y_LENGTH >> 1))) &&
           (pos.hdg < THETA_STEP);
}

CUDA_BOTH_INLINE bool ParkingPlanner2::withinPlanSpace(int32_t x, int32_t y, int32_t theta)
{
    Coord3d pos = {x, y, theta};
    return ParkingPlanner2::withinPlanSpace(pos);
}

CUDA_BOTH_INLINE int32_t ParkingPlanner2::volIndex(int32_t x, int32_t y, int32_t theta)
{
    return (x + HALF_DIM3) + DIM3 * ((y + HALF_DIM3) + DIM3 * (theta));
}

CUDA_BOTH_INLINE bool ParkingPlanner2::boundCheck(int32_t x, int32_t y, int32_t theta)
{
    return ((x >= -HALF_DIM3) && (x <= HALF_DIM3 - 1) && (y >= -HALF_DIM3) && (y <= HALF_DIM3 - 1) && (theta >= 0) && (theta < THETA_DIM3));
}

CUDA_BOTH_INLINE int32_t ParkingPlanner2::turnIndex(int32_t x, int32_t y, int32_t theta, bool left, float32_t turnRadius)
{
    float32_t s       = left ? 1.0f : -1.0f;
    float32_t actualX = static_cast<float32_t>(x) * POS_RES3 + s * turnRadius * sin(static_cast<float32_t>(theta * HDG_RES_RAD3));
    float32_t actualY = static_cast<float32_t>(y) * POS_RES3 + s * turnRadius * (1.0f - cos(static_cast<float32_t>(theta * HDG_RES_RAD3)));
    int32_t newIndexX = (DIM3 + (static_cast<int32_t>(round((actualX / POS_RES3))) + HALF_DIM3) % DIM3) % DIM3 - HALF_DIM3;
    int32_t newIndexY = (DIM3 + (static_cast<int32_t>(round((actualY / POS_RES3))) + HALF_DIM3) % DIM3) % DIM3 - HALF_DIM3;

    return ParkingPlanner2::volIndex(newIndexX, newIndexY, theta);
}

CUDA_BOTH_INLINE Coord3d ParkingPlanner2::turnIndexPlain(int32_t x, int32_t y, int32_t theta, bool left, float32_t turnRadius)
{
    float32_t s       = left ? 1.0f : -1.0f;
    float32_t actualX = static_cast<float32_t>(x) * POS_RES3 + s * turnRadius * sin(static_cast<float32_t>(theta * HDG_RES_RAD3));
    float32_t actualY = static_cast<float32_t>(y) * POS_RES3 + s * turnRadius * (1.0f - cos(static_cast<float32_t>(theta * HDG_RES_RAD3)));
    int32_t newIndexX = static_cast<int32_t>(round((actualX / POS_RES3)));
    int32_t newIndexY = static_cast<int32_t>(round((actualY / POS_RES3)));

    return Coord3d(newIndexX, newIndexY, theta);
}

CUDA_BOTH_INLINE Coord3d ParkingPlanner2::originCoord(int32_t x, int32_t y, int32_t theta, bool left, float32_t turnRadius)
{
    float32_t s       = left ? 1.0f : -1.0f;
    float32_t actualX = static_cast<float32_t>(x) * POS_RES3 - s * turnRadius * sin(static_cast<float32_t>(theta * HDG_RES_RAD3));
    float32_t actualY = static_cast<float32_t>(y) * POS_RES3 - s * turnRadius * (1.0f - cos(static_cast<float32_t>(theta * HDG_RES_RAD3)));
    int32_t newIndexX = static_cast<int32_t>(round((actualX / POS_RES3)));
    int32_t newIndexY = static_cast<int32_t>(round((actualY / POS_RES3)));
    Coord3d coord     = {newIndexX, newIndexY, 0};
    return coord;
}

CUDA_BOTH_INLINE Coord3d ParkingPlanner2::index2Coord(int32_t index, int32_t dim)
{
    Coord3d coord{};
    coord.x   = (index % dim) - ((dim + 1) / 2);
    coord.y   = ((index / dim) % dim) - ((dim + 1) / 2);
    coord.hdg = (index / (dim * dim));
    return coord;
}

inline int32_t ParkingPlanner2::nextIndex(int32_t x, int32_t y, int32_t theta, float32_t turnRadius, bool left, bool reverse)
{
    float32_t s = left ? 1.0f : -1.0f;
    int32_t nextTheta{};
    if (left == reverse)
        nextTheta = (theta == THETA_DIM3 - 1) ? 0 : theta + 1;
    else
        nextTheta = (theta == 0) ? THETA_DIM3 - 1 : theta - 1;

    float32_t actualX = static_cast<float32_t>(x) * POS_RES3 -
                        s * turnRadius * sin(static_cast<float32_t>(theta * HDG_RES_RAD3));
    float32_t actualY = static_cast<float32_t>(y) * POS_RES3 -
                        s * turnRadius * (1.0f - cos(static_cast<float32_t>(theta * HDG_RES_RAD3)));
    int32_t newIndexX = static_cast<int32_t>(round((actualX / POS_RES3)));
    int32_t newIndexY = static_cast<int32_t>(round((actualY / POS_RES3)));

    actualX = static_cast<float32_t>(newIndexX) * POS_RES3 +
              s * turnRadius * sin(static_cast<float32_t>(nextTheta * HDG_RES_RAD3));
    actualY = static_cast<float32_t>(newIndexY) * POS_RES3 +
              s * turnRadius * (1.0f - cos(static_cast<float32_t>(nextTheta * HDG_RES_RAD3)));

    newIndexX = static_cast<int32_t>(round((actualX / POS_RES3)));
    newIndexY = static_cast<int32_t>(round((actualY / POS_RES3)));

    if (ParkingPlanner2::boundCheck(newIndexX, newIndexY, nextTheta))
        return ParkingPlanner2::volIndex(newIndexX, newIndexY, nextTheta);
    else
        return -1;
}

} // namespace planner
} // namespace dw

//Macro to declare an opaque type for the C++ Parking planner type
DECLARE_HANDLE_TYPE(dw::planner::ParkingPlanner2, dwParkingPlannerObject);

#endif