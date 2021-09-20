/* Copyright (c) 2020 NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <gtest/gtest.h>
#include <samples/framework/SampleFramework.hpp> // getConsoleLoggerCallback
#include <samples/framework/ProgramArguments.hpp>
#include <tests/common/DriveworksTest.hpp>
#include <dw/experimental/parkingplanner/ParkingPlanner2.hpp>
#include <dw/experimental/parkingplanner/ParkingPlanner.h>
#include <dwvisualization/core/renderengine/RenderEngine.hpp>
#include <dwvisualization/experimental/renderview/ImGuiRendererCallback.hpp>
#include <dwvisualization/experimental/renderview/ImGuiView.hpp>
#include <dwvisualization/experimental/renderview/Imgui.hpp>
#include <dwvisualization/experimental/view3d/MouseView3D.hpp>
#include <dw/math/MathUtils.hpp>
#include <dw/mapc/MatrixMap.hpp>

// clang-format off
// result in milli-seconds
#define TIME_IT(result, a) \
        auto start = std::chrono::high_resolution_clock::now(); \
        a; \
        auto elapsed = std::chrono::high_resolution_clock::now() - start; \
        result = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count() / 1000000.0f; \
// clang-format on

// window size
constexpr uint32_t g_windowWidthInteractive  = 1280;
constexpr uint32_t g_windowHeightInteractive = 720;

constexpr float32_t POS_SCALE = 3.0f;
constexpr float32_t HDG_SCALE = 0.3f;

extern ProgramArguments g_arguments;

namespace dw
{
namespace planner
{

inline Vector4f transparentColor2Vector(const Vector4f& solidColor, float32_t percentage)
{
    Vector4f ret{solidColor};
    ret[3] *= percentage;

    return ret;
}

enum class WarpType
{
    NONE,
    STRAIGHT,
    LEFT,
    RIGHT
};

class ParkingPlanner2TestFixtures : public ::testing::Test
{
public:
    ParkingPlanner2TestFixtures()
        : m_renderOn(stoi(g_arguments.get("render"))){};

    void SetUp() override;
    void TearDown() override;

    void initRenderViewsInteractive();
    void initRenderViewsInteractive2();
    void initRendering(uint32_t windowWidth, uint32_t windowHeight);
    void initRendering2(uint32_t windowWidth, uint32_t windowHeight);
    void render();
    void render2();

    static std::unique_ptr<glimgui::renderer> m_imguiRenderer;
    static Vector2f m_mousePos;
    static std::unique_ptr<visualization::MouseView3D> m_mouseView;

    bool m_renderOn{false};

private:
    void renderResult();
    void renderResult2();
    void setRenderEngine();
    void renderGrid();       // the reference grid
    void renderGrid2();       // the reference grid
    void renderGridBuffer(); // the grid for planning purpose
    void renderGridBuffer2(); // the grid for planning purpose
    void updateGridBufferPos(WarpType type = WarpType::NONE);
    void updateGridBufferColor();
    void updateGridBufferColor2();
    void updatePathBuffer(); // render the solution
    void updatePathBuffer2(); // render the solution
    void showMenu();
    void showMenu2();
    void initializeView(); // initialize/reset m_mouseView

    // mock obstacle button callbacks
    void setMockParallel();
    void setMockSlanted();
    void setMockPerpendicular();

    // button callbacks
    void setUpParallel();
    void setUpSlanted();
    void setUpPerpendicular();
    static dwObstacle createObstacle(const Vector3f& obj,  // {x, y ,hdg}
                                     const Vector3f& ego);
    void simulate();
    void move();
    void updateWorld();

    void simulateChangeStart();
    void moveStart();

    Vector3f m_ego{}; // represented in a global frame
    Vector3f m_des{}; // represented in a global frame
    static constexpr size_t MAX_OBSTABLE_NUM = 10; // this is intentionally much smaller
                                                   // than ParkingPlanner2::MAX_OBSTABLE_NUM
    StaticVectorFixed<dwObstacle, MAX_OBSTABLE_NUM> m_obstacles{}; // represented in ego body
    StaticVectorFixed<Vector3f, MAX_OBSTABLE_NUM> m_objs{}; // represented in a global frame

    std::unique_ptr<visualization::RenderEngine> m_renderEnginePtr{};
    std::unique_ptr<ParkingPlanner2> m_parkingPlanner{};
    dwContextHandle_t m_context{DW_NULL_HANDLE};

    // render buffer
    using DataLayout     = visualization::renderengine::DataLayout;
    using Buffer         = visualization::renderengine::Buffer;
    using ColoredPoint3D = visualization::renderengine::ColoredPoint3D;

    uint32_t m_gridBufferId{0}; // the grid cells of the parking planner
    void initGridBuffer();
    void initGridBuffer2();
    uint32_t m_pathBufferId{0};   // solution
    uint32_t m_reachableCells{0}; // show statistics in the menu
    uint32_t m_obstacleCells{0};  // show statistics in the menu
    Vector3f m_userDest{0.f, 0.f, 180.f}; 
    ObstacleName m_splObstacle{NONE};

    bool m_showOneTheta{false};  // represented in ego body

    uint16_t m_costBefore{}; //cost of completed path in simulation

    float32_t m_processTime{0.0f};
}; // ParkingPlanner2TestFixtures

std::unique_ptr<glimgui::renderer> ParkingPlanner2TestFixtures::m_imguiRenderer{};
Vector2f ParkingPlanner2TestFixtures::m_mousePos{};
std::unique_ptr<visualization::MouseView3D> ParkingPlanner2TestFixtures::m_mouseView{};
bool m_mouseViewLock{false}; // lock the view when mouse is hovered on the menu

void ParkingPlanner2TestFixtures::SetUp()
{
    // init driveworks
    dwVersion version;
    dwGetVersion(&version);
    dwContextParameters sdkParams = {};

    dwStatus status = DW_SUCCESS;
    status          = dwInitialize(&m_context, version, &sdkParams);
    ASSERT_EQ(DW_SUCCESS, status);

    dwParkingPlannerParams ppParams{};
    dwParkingPlanner_initDefaultParams(&ppParams);

    m_parkingPlanner = std::make_unique<ParkingPlanner2>(&ppParams, CHandle::cast(m_context));
    m_parkingPlanner->setStart(0, 0, 0);
    m_parkingPlanner->setDestination(0, 0, 256);
    m_parkingPlanner->m_startTurnType = 8;
    m_costBefore = static_cast<uint16_t>(0u);
}

void ParkingPlanner2TestFixtures::TearDown()
{
    if (m_renderEnginePtr)
    {
        m_renderEnginePtr->destroyBuffer(m_gridBufferId);
        m_renderEnginePtr->destroyBuffer(m_pathBufferId);
    }

    releaseSampleApp();

    if (m_context != DW_NULL_HANDLE)
    {
        dwStatus status = dwRelease(m_context);
        ASSERT_EQ(DW_SUCCESS, status);
    }
}

void ParkingPlanner2TestFixtures::initRenderViewsInteractive()
{
    // first to initialize render buffers
    initRendering(g_windowWidthInteractive, g_windowHeightInteractive);

    // mouse input handling
    if (gWindow)
    {

        gWindow->setOnMouseDownCallback([](int button, float x, float y, int mods) {
            int32_t action = (button == 0) ? GLFW_PRESS : GLFW_RELEASE;
            m_imguiRenderer->mouse_position_callback(x, y);
            m_imguiRenderer->mouse_button_callback(button, action, mods);

            m_mouseView->mouseDown(button, m_mousePos(0), m_mousePos(1));

        });

        gWindow->setOnMouseUpCallback([](int button, float x, float y, int mods) {
            int32_t action = (button == 0) ? GLFW_RELEASE : GLFW_PRESS;
            m_imguiRenderer->mouse_position_callback(x, y);
            m_imguiRenderer->mouse_button_callback(button, action, mods);

            m_mouseView->mouseUp(button, m_mousePos(0), m_mousePos(1));

        });

        gWindow->setOnMouseMoveCallback([](float x, float y) {
            m_imguiRenderer->mouse_position_callback(x, y);

            m_mousePos = {x, y}; // intentionally outside if
            if (!m_mouseViewLock)
            {
                m_mouseView->mouseMove(x, y);
            }
        });

        gWindow->setOnMouseWheelCallback([](float dx, float dy) {
            m_mouseView->mouseWheel(dx, dy);
        });
    }

    m_imguiRenderer.reset(new glimgui::renderer(g_windowWidthInteractive, g_windowHeightInteractive));
}

void ParkingPlanner2TestFixtures::initRenderViewsInteractive2()
{
    // first to initialize render buffers
    initRendering2(g_windowWidthInteractive, g_windowHeightInteractive);

    // mouse input handling
    if (gWindow)
    {

        gWindow->setOnMouseDownCallback([](int button, float x, float y, int mods) {
            int32_t action = (button == 0) ? GLFW_PRESS : GLFW_RELEASE;
            m_imguiRenderer->mouse_position_callback(x, y);
            m_imguiRenderer->mouse_button_callback(button, action, mods);

            m_mouseView->mouseDown(button, m_mousePos(0), m_mousePos(1));

        });

        gWindow->setOnMouseUpCallback([](int button, float x, float y, int mods) {
            int32_t action = (button == 0) ? GLFW_RELEASE : GLFW_PRESS;
            m_imguiRenderer->mouse_position_callback(x, y);
            m_imguiRenderer->mouse_button_callback(button, action, mods);

            m_mouseView->mouseUp(button, m_mousePos(0), m_mousePos(1));

        });

        gWindow->setOnMouseMoveCallback([](float x, float y) {
            m_imguiRenderer->mouse_position_callback(x, y);

            m_mousePos = {x, y}; // intentionally outside if
            if (!m_mouseViewLock)
            {
                m_mouseView->mouseMove(x, y);
            }
        });

        gWindow->setOnMouseWheelCallback([](float dx, float dy) {
            m_mouseView->mouseWheel(dx, dy);
        });
    }

    m_imguiRenderer.reset(new glimgui::renderer(g_windowWidthInteractive, g_windowHeightInteractive));
}

void ParkingPlanner2TestFixtures::initializeView()
{
    m_mouseView->setCenter(0.0f, 0.0f, 50.0f);
    m_mouseView->setViewAngle(static_cast<float32_t>(M_PI_2) - 0.1f,
                              static_cast<float32_t>(M_PI));
    m_mouseView->setRadius(200.0f);
}

void ParkingPlanner2TestFixtures::initRendering(uint32_t windowWidth, uint32_t windowHeight)
{
    // init window
    if (!initSampleApp(0, nullptr, nullptr, nullptr, windowWidth, windowHeight))
        return;

    // init renderengine
    dwRenderEngineParams params{};
    dwRenderEngine_initDefaultParams(&params, gWindow->width(), gWindow->height());
    m_renderEnginePtr = makeUnique<visualization::RenderEngine>(params, CHandle::cast(m_context));
    m_mouseView       = makeUnique<visualization::MouseView3D>();

    initializeView();

    m_gridBufferId = m_renderEnginePtr->createBuffer(DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                                     m_parkingPlanner->getGridSize(),
                                                     DataLayout::ForColoredPoint3D);

    m_pathBufferId = m_renderEnginePtr->createBuffer(DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_3D,
                                                     ParkingPlanner2::MAX_PATH_POINT_NUM,
                                                     DataLayout::ForColoredPoint3D);

    initGridBuffer();
}

void ParkingPlanner2TestFixtures::initRendering2(uint32_t windowWidth, uint32_t windowHeight)
{
    // init window
    if (!initSampleApp(0, nullptr, nullptr, nullptr, windowWidth, windowHeight))
        return;

    // init renderengine
    dwRenderEngineParams params{};
    dwRenderEngine_initDefaultParams(&params, gWindow->width(), gWindow->height());
    m_renderEnginePtr = makeUnique<visualization::RenderEngine>(params, CHandle::cast(m_context));
    m_mouseView       = makeUnique<visualization::MouseView3D>();

    initializeView();

    m_gridBufferId = m_renderEnginePtr->createBuffer(DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                                     m_parkingPlanner->getGridSize3(),
                                                     DataLayout::ForColoredPoint3D);

    m_pathBufferId = m_renderEnginePtr->createBuffer(DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_3D,
                                                     ParkingPlanner2::MAX_PATH_POINT_NUM,
                                                     DataLayout::ForColoredPoint3D);

    initGridBuffer2();
}

void ParkingPlanner2TestFixtures::render()
{
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_ALWAYS);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glfwSwapInterval(1); // Enable vsync

    // global imgui style setting
    ImGui::GetIO().FontGlobalScale   = 0.75f;
    ImGui::GetStyle().WindowRounding = 1.0f;

    while (!gWindow->shouldClose() && gRun)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        m_imguiRenderer->start_new_frame();

        ImGui::SetNextWindowPos({0, 0}, ImGuiCond_Always);
        ImGui::SetNextWindowSize({static_cast<float32_t>(gWindow->width()), static_cast<float32_t>(gWindow->height())},
                                 ImGuiCond_Always);

        ImGui::Begin("Parking planner", nullptr, ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoResize);
        {
            showMenu();

            static visualization::ImGuiRendererCallback callback;
            callback.addCallback([&]() {
                this->renderResult();
            },
                                 {ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y});
        }
        ImGui::End();

        // Rendering
        m_imguiRenderer->render();

        // Display
        gWindow->swapBuffers();
    }
}

void ParkingPlanner2TestFixtures::render2()
{
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_ALWAYS);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glfwSwapInterval(1); // Enable vsync

    // global imgui style setting
    ImGui::GetIO().FontGlobalScale   = 0.75f;
    ImGui::GetStyle().WindowRounding = 1.0f;

    while (!gWindow->shouldClose() && gRun)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        m_imguiRenderer->start_new_frame();

        ImGui::SetNextWindowPos({0, 0}, ImGuiCond_Always);
        ImGui::SetNextWindowSize({static_cast<float32_t>(gWindow->width()), static_cast<float32_t>(gWindow->height())},
                                 ImGuiCond_Always);

        ImGui::Begin("Parking planner", nullptr, ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoResize);
        {
            showMenu2();

            static visualization::ImGuiRendererCallback callback;
            callback.addCallback([&]() {
                this->renderResult2();
            },
                                 {ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y});
        }
        ImGui::End();

        // Rendering
        m_imguiRenderer->render();

        // Display
        gWindow->swapBuffers();
    }
}

void ParkingPlanner2TestFixtures::renderResult()
{
    setRenderEngine();
    renderGrid();
    renderGridBuffer();
}

void ParkingPlanner2TestFixtures::renderResult2()
{
    setRenderEngine();
    renderGrid2();
    renderGridBuffer2();
}

void ParkingPlanner2TestFixtures::setRenderEngine()
{
    m_renderEnginePtr->setTile(0);
    m_renderEnginePtr->setBackgroundColor(visualization::COLOR_WHITE);

    m_renderEnginePtr->setModelView(m_mouseView->getModelView());
    m_renderEnginePtr->setProjection(m_mouseView->getProjection());
}

void ParkingPlanner2TestFixtures::renderGrid()
{
    m_renderEnginePtr->setColorByValue(DW_RENDER_ENGINE_COLOR_BY_VALUE_COLOR, 1.0f);
    m_renderEnginePtr->setColor(visualization::COLOR_LIGHTGREY * 0.2);
    m_renderEnginePtr->setLineWidth(1.0f);
    m_renderEnginePtr->renderPlanarGrid3D({0.0f, 0.0f, 2.0f * ParkingPlanner2::getRange(), 2.0f * ParkingPlanner2::getRange()},
                                          1.0f,
                                          1.0f,
                                          Matrix4f::Identity());
}

void ParkingPlanner2TestFixtures::renderGrid2()
{
    m_renderEnginePtr->setColorByValue(DW_RENDER_ENGINE_COLOR_BY_VALUE_COLOR, 1.0f);
    m_renderEnginePtr->setColor(visualization::COLOR_LIGHTGREY * 0.2);
    m_renderEnginePtr->setLineWidth(1.0f);
    m_renderEnginePtr->renderPlanarGrid3D({0.0f, 0.0f, 50.0f, 50.0f},
                                          1.0f,
                                          1.0f,
                                          Matrix4f::Identity());
}

void ParkingPlanner2TestFixtures::renderGridBuffer()
{
    m_renderEnginePtr->setColorByValue(DW_RENDER_ENGINE_COLOR_BY_VALUE_MODE_ATTRIBUTE_RGBA, 1.0f);
    m_renderEnginePtr->setPointSize(2.0f);
    auto& buffer = m_renderEnginePtr->getBuffer(m_gridBufferId);
    buffer.setPrimitiveCount(m_parkingPlanner->getGridSize());
    m_renderEnginePtr->renderBuffer(buffer);

    if (!m_parkingPlanner->getPath().empty())
    {
        m_renderEnginePtr->setLineWidth(3.0f);
        auto& buffer1 = m_renderEnginePtr->getBuffer(m_pathBufferId);
        buffer1.setPrimitiveCount((m_parkingPlanner->getPath().size() - 1));
        m_renderEnginePtr->renderBuffer(buffer1);
    }
}
void ParkingPlanner2TestFixtures::renderGridBuffer2()
{
    m_renderEnginePtr->setColorByValue(DW_RENDER_ENGINE_COLOR_BY_VALUE_MODE_ATTRIBUTE_RGBA, 1.0f);
    m_renderEnginePtr->setPointSize(2.0f);
    auto& buffer = m_renderEnginePtr->getBuffer(m_gridBufferId);

    // render the volume or one theta slice
    buffer.setPrimitiveCount((m_showOneTheta) 
                                 ? ParkingPlanner2::getDim3() * ParkingPlanner2::getDim3()
                                 : m_parkingPlanner->getGridSize3());
    m_renderEnginePtr->renderBuffer(buffer);

    if (!m_parkingPlanner->getPath().empty())
    {
        m_renderEnginePtr->setLineWidth(3.0f);
        auto& buffer1 = m_renderEnginePtr->getBuffer(m_pathBufferId);
        buffer1.setPrimitiveCount((m_parkingPlanner->getPath().size() - 1));
        m_renderEnginePtr->renderBuffer(buffer1);
    }
}

void ParkingPlanner2TestFixtures::updateGridBufferColor()
{
    auto& renderBuffer          = m_renderEnginePtr->getBuffer(m_gridBufferId);
    span<ColoredPoint3D> buffer = renderBuffer.mapWriteVertices<ColoredPoint3D>();

    m_reachableCells = 0;
    m_obstacleCells  = 0;
    for (int32_t x = 0; x < ParkingPlanner2::getSize(); ++x)
    {
        for (int32_t y = 0; y < ParkingPlanner2::getSize(); ++y)
        {
            for (uint32_t theta = 0; theta < ParkingPlanner2::getThetaStep(); ++theta)
            {
                Coord3d coord = Coord3d(x - (ParkingPlanner2::getSize() >> 1),
                                        y - (ParkingPlanner2::getSize() >> 1),
                                        theta);

                int32_t index = ParkingPlanner2::getCell(coord.x, coord.y, coord.hdg);

                GridCell& cell = ParkingPlanner2::getCell(m_parkingPlanner->m_grid.get(),
                                                          coord);
                Vector4f newColor{};
                if (cell.obstacle && ParkingPlanner2::withinPlanSpace(coord))
                {
                    newColor = transparentColor2Vector(visualization::COLOR_RED, 1.0f);
                    ++m_obstacleCells;
                }
                else if (cell.reachable)
                {
                    newColor = transparentColor2Vector(visualization::COLOR_GREEN, 0.3f);
                    ++m_reachableCells;
                }
                else
                {
                    newColor = transparentColor2Vector(visualization::COLOR_LIGHTGREY, 0.02f);
                }

                buffer[index].color = newColor; // preserve the position, change the color
            }
        }
    }

    renderBuffer.unmap(m_parkingPlanner->getGridSize());
}

void ParkingPlanner2TestFixtures::updateGridBufferColor2()
{
    auto& renderBuffer          = m_renderEnginePtr->getBuffer(m_gridBufferId);
    span<ColoredPoint3D> buffer = renderBuffer.mapWriteVertices<ColoredPoint3D>();

    m_reachableCells = 0;
    m_obstacleCells  = 0;
    for (int32_t x = 0; x < ParkingPlanner2::getDim3(); ++x)
    {
        for (int32_t y = 0; y < ParkingPlanner2::getDim3(); ++y)
        {
            for (uint32_t theta = 0; theta < ParkingPlanner2::getThetaStep3(); ++theta)
            {
                Coord3d coord = Coord3d(x - (ParkingPlanner2::getHalfDim3()),
                                        y - (ParkingPlanner2::getHalfDim3()),
                                        theta);

                int32_t index = ParkingPlanner2::volIndex(coord.x, coord.y, coord.hdg);

                uint16_t cost = m_parkingPlanner->m_gridMain[index];
                uint16_t obs  = m_parkingPlanner->m_gridObs[index];
                Vector4f newColor{};

                if (cost < 64250u)
                {
                    newColor = transparentColor2Vector(visualization::COLOR_GREEN, 0.3f);
                    ++m_reachableCells;
                }
                else
                {
                    newColor = transparentColor2Vector(visualization::COLOR_LIGHTGREY, 0.02f);
                }
                //Not displaying the outer boundary obstacle
                if (obs == 64250u && x*y != 0 && x != ParkingPlanner2::getDim3()-1 && y != ParkingPlanner2::getDim3()-1) 
                {
                    newColor = transparentColor2Vector(visualization::COLOR_RED, 0.3f);
                    ++m_obstacleCells;
                }

                buffer[index].color = newColor; // preserve the position, change the color
            }
        }
    }

    renderBuffer.unmap(m_parkingPlanner->getGridSize3());
}

void ParkingPlanner2TestFixtures::updateGridBufferPos(WarpType type)
{
    auto& renderBuffer          = m_renderEnginePtr->getBuffer(m_gridBufferId);
    span<ColoredPoint3D> buffer = renderBuffer.mapWriteVertices<ColoredPoint3D>();

    const float32_t POS_RES = ParkingPlanner2::getPosRes();
    const float32_t HDG_RES = ParkingPlanner2::getHdgRes();

    for (int32_t x = 0; x < ParkingPlanner2::getSize(); ++x)
    {
        for (int32_t y = 0; y < ParkingPlanner2::getSize(); ++y)
        {
            for (uint32_t theta = 0; theta < ParkingPlanner2::getThetaStep(); ++theta)
            {
                Coord3d coord = Coord3d(x - (ParkingPlanner2::getSize() >> 1),
                                        y - (ParkingPlanner2::getSize() >> 1),
                                        theta);
                Vector3f pose = coord.getPose(POS_RES, HDG_RES);

                switch (type)
                {
                case WarpType::NONE:
                    break;
                case WarpType::STRAIGHT:
                    pose = ParkingPlanner2::unwarpStraight(pose);
                    break;
                case WarpType::LEFT:
                    pose = ParkingPlanner2::unwarpLeft(pose, m_parkingPlanner->m_parkingPlannerParams.turnRadius_m);
                    break;
                case WarpType::RIGHT:
                    pose = ParkingPlanner2::unwarpRight(pose, m_parkingPlanner->m_parkingPlannerParams.turnRadius_m);
                    break;
                default:
                    break;
                }

                pose.x() = pose.x() * POS_SCALE;
                pose.y() = pose.y() * POS_SCALE;
                pose.z() = pose.z() * HDG_SCALE;

                int32_t index = ParkingPlanner2::getCell(coord.x, coord.y, coord.hdg);

                buffer[index].position = pose;
            }
        }
    }

    renderBuffer.unmap(m_parkingPlanner->getGridSize());
}

void ParkingPlanner2TestFixtures::updatePathBuffer()
{
    auto& renderBuffer          = m_renderEnginePtr->getBuffer(m_pathBufferId);
    span<ColoredPoint3D> buffer = renderBuffer.mapWriteVertices<ColoredPoint3D>();

    uint32_t cnt = 0;

    if (m_parkingPlanner->hasPath())
    {
        auto itDir = m_parkingPlanner->getDrivingDirs().begin() + 1;
        for (auto it = m_parkingPlanner->getPath().begin() + 1; it < m_parkingPlanner->getPath().end(); ++it)
        {
            const Vector3f& pt                        = *it;
            const Vector3f& prevPt                    = *(it - 1);
            const dwPathPlannerDrivingState ptDir     = (*itDir);
            const dwPathPlannerDrivingState prevPtDir = (*(itDir - 1));

            ColoredPoint3D& coloredPt = buffer[cnt++];
            coloredPt.position        = Vector3f{pt.x() * POS_SCALE,
                                          pt.y() * POS_SCALE,
                                          pt.z() * HDG_SCALE};
            if (ptDir == DW_PATH_FORWARD)
            {
                coloredPt.color = visualization::COLOR_BLUE;
            }
            else
            {
                coloredPt.color = visualization::COLOR_PURPLE;
            }

            ColoredPoint3D& prevColoredPt = buffer[cnt++];
            prevColoredPt.position        = Vector3f{prevPt.x() * POS_SCALE,
                                              prevPt.y() * POS_SCALE,
                                              prevPt.z() * HDG_SCALE};
            if (prevPtDir == DW_PATH_FORWARD)
            {
                prevColoredPt.color = visualization::COLOR_BLUE;
            }
            else
            {
                prevColoredPt.color = visualization::COLOR_PURPLE;
            }
            ++itDir;
        }
    }

    renderBuffer.unmap(cnt);
}

void ParkingPlanner2TestFixtures::updatePathBuffer2()
{
    auto& renderBuffer          = m_renderEnginePtr->getBuffer(m_pathBufferId);
    span<ColoredPoint3D> buffer = renderBuffer.mapWriteVertices<ColoredPoint3D>();

    uint32_t cnt = 0;

    if (m_parkingPlanner->hasPath())
    {
        auto itTurn = m_parkingPlanner->getTurnTypes().begin() + 1;
        for (auto it = m_parkingPlanner->getPath().begin() + 1; it < m_parkingPlanner->getPath().end(); ++it)
        {
            const Vector3f& pt                        = *it;
            const Vector3f& prevPt                    = *(it - 1);
            const int32_t ptTurn     = (*itTurn);
            const int32_t prevPtTurn = (*(itTurn - 1));

            ColoredPoint3D& coloredPt = buffer[cnt++];
            coloredPt.position        = Vector3f{pt.x() * POS_SCALE,
                                          pt.y() * POS_SCALE,
                                          ((m_showOneTheta) ? 0.0f : pt.z())* HDG_SCALE}; // squash 3D to 2D
            if (ptTurn < 4)
            {
                if(ptTurn == 1 || ptTurn == 2)
                    coloredPt.color = visualization::COLOR_BLUE;
                else
                    coloredPt.color = visualization::COLOR_PURPLE;
            }
            else
            {
                if(ptTurn == 5 || ptTurn == 6)
                    coloredPt.color = visualization::COLOR_BLACK;
                else
                    coloredPt.color = visualization::COLOR_WHITE;
            }

            ColoredPoint3D& prevColoredPt = buffer[cnt++];
            prevColoredPt.position        = Vector3f{prevPt.x() * POS_SCALE,
                                              prevPt.y() * POS_SCALE,
                                              ((m_showOneTheta) ? 0.0f : prevPt.z()) * HDG_SCALE};
            if (prevPtTurn < 4)
            {
                if(ptTurn == 1 || ptTurn == 2)
                    prevColoredPt.color = visualization::COLOR_BLUE;
                else
                    prevColoredPt.color = visualization::COLOR_PURPLE;
            }
            else
            {
                if(ptTurn == 5 || ptTurn == 6)
                    prevColoredPt.color = visualization::COLOR_BLACK;
                else
                    prevColoredPt.color = visualization::COLOR_WHITE;
            }
            ++itTurn;
        }
    }

    renderBuffer.unmap(cnt);
}

void ParkingPlanner2TestFixtures::showMenu()
{
    ImGuiWindowFlags windowFlags       = 0;
    ImGui::GetStyle().WindowBorderSize = 1.0f;
    windowFlags |= ImGuiWindowFlags_NoSavedSettings;
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(600, 250), ImGuiCond_FirstUseEver);

    // controls to change ego state
    ImGui::Begin("Menu", nullptr, windowFlags);
    {
        ImGui::PushItemWidth(100);

        // mock environment
        if (ImGui::Button("Do one turn2"))
        {
            Coord3d tempDest(m_userDest,
                             ParkingPlanner2::getPosRes(),
                             ParkingPlanner2::getHdgRes());
            m_parkingPlanner->setDestination(tempDest);
            m_parkingPlanner->setSpecialObstacle(m_splObstacle);
            m_parkingPlanner->setObstacles(m_obstacles);
            TIME_IT(m_processTime,
            {
                m_parkingPlanner->copyGridHostToDevice();
                m_parkingPlanner->processOneTurn();
                m_parkingPlanner->copyGridDeviceToHost();
            });
            m_parkingPlanner->buildPath();
            updateGridBufferColor();
            updatePathBuffer();
        }
        ImGui::SameLine();
        if (ImGui::Button("Do straight turn"))
        {
            Coord3d tempDest(m_userDest,
                             ParkingPlanner2::getPosRes(),
                             ParkingPlanner2::getHdgRes());
            m_parkingPlanner->setDestination(tempDest);
            TIME_IT(m_processTime,
            {
                m_parkingPlanner->copyGridHostToDevice();
                m_parkingPlanner->processStraight();
                m_parkingPlanner->copyGridDeviceToHost();
            });
            m_parkingPlanner->buildPath();
            updateGridBufferColor();
            updatePathBuffer();
        }
        ImGui::SameLine();
        if (ImGui::Button("Do left turn"))
        {
            Coord3d tempDest(m_userDest,
                             ParkingPlanner2::getPosRes(),
                             ParkingPlanner2::getHdgRes());
            m_parkingPlanner->setDestination(tempDest);
            TIME_IT(m_processTime,
            {
                m_parkingPlanner->copyGridHostToDevice();
                m_parkingPlanner->processLeft();
                m_parkingPlanner->copyGridDeviceToHost();
            });
            m_parkingPlanner->buildPath();
            updateGridBufferColor();
            updatePathBuffer();
        }
        ImGui::SameLine();
        if (ImGui::Button("Do right turn"))
        {
            Coord3d tempDest(m_userDest,
                             ParkingPlanner2::getPosRes(),
                             ParkingPlanner2::getHdgRes());
            m_parkingPlanner->setDestination(tempDest);
            TIME_IT(m_processTime,
            {
                m_parkingPlanner->copyGridHostToDevice();
                m_parkingPlanner->processRight();
                m_parkingPlanner->copyGridDeviceToHost();
            });
            m_parkingPlanner->buildPath();
            updateGridBufferColor();
            updatePathBuffer();
        }
        ImGui::SameLine();
        if (ImGui::Button("Process"))
        {
            m_parkingPlanner->reset();

            Coord3d tempDest(m_userDest,
                             ParkingPlanner2::getPosRes(),
                             ParkingPlanner2::getHdgRes());
            m_parkingPlanner->setDestination(tempDest);
            m_parkingPlanner->setObstacles(m_obstacles);
            m_parkingPlanner->setSpecialObstacle(m_splObstacle);

            TIME_IT(m_processTime,
                    m_parkingPlanner->process());
            updateGridBufferColor();
            updatePathBuffer();
        }
        if (ImGui::Button("Process2"))
        {
            m_parkingPlanner->reset();

            Coord3d tempDest(m_userDest,
                             ParkingPlanner2::getPosRes(),
                             ParkingPlanner2::getHdgRes());
            m_parkingPlanner->setDestination(tempDest);
            m_parkingPlanner->setObstacles(m_obstacles);
            m_parkingPlanner->setSpecialObstacle(m_splObstacle);

            TIME_IT(m_processTime,
                    m_parkingPlanner->process2());
            updateGridBufferColor();
            updatePathBuffer();
        }
        ImGui::SameLine();
        if (ImGui::Button("ProcessNew"))
        {
            m_parkingPlanner->reset();

            Coord3d tempDest(m_userDest,
                             ParkingPlanner2::getPosRes(),
                             ParkingPlanner2::getHdgRes());
            m_parkingPlanner->setDestination(tempDest);
            m_parkingPlanner->setObstacles(m_obstacles);
            m_parkingPlanner->setSpecialObstacle(m_splObstacle);

            TIME_IT(m_processTime,
                    m_parkingPlanner->processNew());
            updateGridBufferColor();
            updatePathBuffer();
        }
        ImGui::SameLine();
        if (ImGui::Button("Time Profiling"))
        {
            m_parkingPlanner->timeGPU();
        }

        ImGui::SameLine();
        if (ImGui::Button("Process Cost"))
        {
            m_parkingPlanner->setObstacles(m_obstacles);
            m_parkingPlanner->processCost();
        }

        ImGui::SameLine();
        if (ImGui::Button("sleep")) // test the clock
        {
            TIME_IT(m_processTime,
                    usleep(1e6));
        }

        if (m_parkingPlanner->isDestinationReached())
        {
            if (m_parkingPlanner->hasPath())
            {
                ImGui::Text("Destination reached and path found with %d primitives.",
                            m_parkingPlanner->getNumberOfPrimitives());
            }
            else
            {
                ImGui::Text("Destination reached but path not traceable.");
            }
        }
        else
        {
            ImGui::Text("Destination not reached.");
        }

        // display cell info
        {
            core::FixedString<50> textString{""};
            textString += "Reachable cells: ";
            textString += m_reachableCells;

            textString += ", obstacle cells ";
            textString += m_obstacleCells;

            ImGui::Text("%s", textString.c_str());
        }
        // display time to execute the last process call
        {
            ImGui::Text("Last process takes %.3f milli-seconds", m_processTime);
        }

        // visualize warps
        if (ImGui::Button("unwarp"))
        {
            updateGridBufferPos();
        }
        ImGui::SameLine();
        if (ImGui::Button("Straight"))
        {
            updateGridBufferPos(WarpType::STRAIGHT);
        }
        ImGui::SameLine();
        if (ImGui::Button("Left"))
        {
            updateGridBufferPos(WarpType::LEFT);
        }
        ImGui::SameLine();
        if (ImGui::Button("Right"))
        {
            updateGridBufferPos(WarpType::RIGHT);
        }

        if (ImGui::Button("Reset"))
        {
            m_parkingPlanner->reset();
            updateGridBufferColor();
            updatePathBuffer();
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset view"))
        {
            initializeView();
        }

        ImGui::SliderFloat(":Dest x", &m_userDest.x(), -ParkingPlanner2::getRange(), ParkingPlanner2::getRange());
        ImGui::SameLine();
        ImGui::SliderFloat(":Dest y", &m_userDest.y(), -ParkingPlanner2::getRange(), ParkingPlanner2::getRange());
        ImGui::SameLine();
        ImGui::SliderFloat(":Dest theta", &m_userDest.z(), 0.0f, 360.0f);

        // mock obstacle
        if (ImGui::Button("Add obstacle"))
        {
            m_obstacles.emplace_back_maybe();
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset obstacle"))
        {
            m_obstacles.clear();
        }
        if(ImGui::Button("NONE"))
        {
            m_splObstacle = NONE;
        }
        ImGui::SameLine();
        if(ImGui::Button("WALL"))
        {
            m_splObstacle = WALL;
            m_userDest = {15.0f,0.0f,0.0f};
        }
        ImGui::SameLine();
        if(ImGui::Button("3WallMaze"))
        {
            m_splObstacle = THREE_WALL_MAZE;
            m_userDest = {20.0f,20.0f,0.0f};
        }
        ImGui::SameLine();
        if(ImGui::Button("BAY"))
        {
            m_splObstacle = BAY;
            m_userDest = {5.0f,5.0f,0.0f};
        }
        ImGui::SameLine();
        if(ImGui::Button("HURDLE"))
        {
            m_splObstacle = HURDLE;
            m_userDest = {10.0f,0.0f,180.0f};
        }
        ImGui::SameLine();
        if(ImGui::Button("PARALLEL"))
        {
            m_splObstacle = PARALLEL;
            m_userDest = {17.0f,0.0f,0.0f};
        }

        // Imgui uses label to distinguish widget
        // Here we are creating sliders within a loop
        // We have to push a unique identifier
        int32_t widgetId{0};
        for (dwObstacle& obs : m_obstacles)
        {
            ImGui::PushID(widgetId++);
            ImGui::SliderFloat(":x", &obs.position.x,
                               -ParkingPlanner2::getRange(), ParkingPlanner2::getRange());
            ImGui::SameLine();
            ImGui::SliderFloat(":y", &obs.position.y,
                               -ParkingPlanner2::getRange(), ParkingPlanner2::getRange());
            ImGui::PopID();
        }

        // prevent view change when user is interacting with the menu and widgets
        // e.g. drag the slider
        if (ImGui::IsWindowHovered() ||
            ImGui::IsWindowFocused())
        {
            m_mouseViewLock = true;
        }
        else
        {
            m_mouseViewLock = false;
        }
    }
    ImGui::End();
}

void ParkingPlanner2TestFixtures::showMenu2()
{
    ImGuiWindowFlags windowFlags       = 0;
    ImGui::GetStyle().WindowBorderSize = 1.0f;
    windowFlags |= ImGuiWindowFlags_NoSavedSettings;
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(600, 250), ImGuiCond_FirstUseEver);

    // controls to change ego state
    ImGui::Begin("Menu", nullptr, windowFlags);
    {
        ImGui::PushItemWidth(100);

        if (ImGui::Button("Time Profiling"))
        {
            m_parkingPlanner->timeGPU();
        }

        ImGui::SameLine();
        if (ImGui::Button("Sweep Step"))
        {            
            Coord3d tempDest(m_userDest,
                             ParkingPlanner2::getPosRes3(),
                             ParkingPlanner2::getHdgRes3());
            m_parkingPlanner->setDestination(tempDest);
            m_parkingPlanner->setObstacles(m_obstacles);
            m_parkingPlanner->processCostStep();
            updateGridBufferColor2();
        }

        ImGui::SameLine();
        if (ImGui::Button("Transition step"))
        {            
            TIME_PRINT("costTransition",m_parkingPlanner->costTransition();)
        }

        ImGui::SameLine();
        if (ImGui::Button("Process Cost"))
        {            
            Coord3d tempDest(m_userDest,
                             ParkingPlanner2::getPosRes3(),
                             ParkingPlanner2::getHdgRes3());

            m_parkingPlanner->setDestination(tempDest);
            m_parkingPlanner->setObstacles(m_obstacles);
            TIME_PRINT("processCost",m_parkingPlanner->processCost();)
            updateGridBufferColor2();
            updatePathBuffer2();
        }

        ImGui::SameLine();
        if (ImGui::Button("Time SOL"))
        {
            m_parkingPlanner->timeSOL();
        }
        
        if (ImGui::Button("Test Kernel"))
        {
            m_parkingPlanner->intermediateTestKernel();
        }
        ImGui::SameLine();
        if (ImGui::Button("Test Sweep 32"))
        {
            m_parkingPlanner->testSweep32();
            updateGridBufferColor2();
        }
        ImGui::SameLine();
        if (ImGui::Button("Test Sweep 64"))
        {
            m_parkingPlanner->testSweep64();
            updateGridBufferColor2();
        }
        // display cell info
        {
            core::FixedString<50> textString{""};
            textString += "Reachable cells: ";
            textString += m_reachableCells;

            textString += ", obstacle cells ";
            textString += m_obstacleCells;

            ImGui::Text("%s", textString.c_str());
        }


        if (ImGui::Button("Reset"))
        {
            m_parkingPlanner->setObstacles(m_obstacles);
            m_parkingPlanner->reset2();
            m_parkingPlanner->minCost();
            m_parkingPlanner->copyDeviceToHostMain();
            updateGridBufferColor2();
            updatePathBuffer2();
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset view"))
        {
            initializeView();
        }

        ImGui::SliderFloat(":Dest x", &m_userDest.x(), -ParkingPlanner2::getRange3()/2, ParkingPlanner2::getRange3()/2);
        ImGui::SameLine();
        ImGui::SliderFloat(":Dest y", &m_userDest.y(), -ParkingPlanner2::getRange3()/2, ParkingPlanner2::getRange3()/2);
        ImGui::SameLine();
        ImGui::SliderFloat(":Dest theta", &m_userDest.z(), 0.0f, 360.0f);

        if (ImGui::Button("Add obstacle"))
        {
            m_obstacles.emplace_back_maybe();
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset obstacle"))
        {
            m_obstacles.clear();
        }

        // Imgui uses label to distinguish widget
        // Here we are creating sliders within a loop
        // We have to push a unique identifier
        int32_t widgetId{0};
        for (dwObstacle& obs : m_obstacles)
        {
            ImGui::PushID(widgetId++);
            ImGui::SliderFloat(":x", &obs.position.x,
                               -ParkingPlanner2::getRange3()/2, ParkingPlanner2::getRange3()/2);
            ImGui::SameLine();
            ImGui::SliderFloat(":y", &obs.position.y,
                               -ParkingPlanner2::getRange3()/2, ParkingPlanner2::getRange3()/2);
            ImGui::SameLine();
            float32_t heading = std::atan2(obs.direction.y, obs.direction.x);
            ImGui::SliderFloat(":theta", &heading,
                               -3.14f, 3.14f);

            obs = createObstacle({obs.position.x, obs.position.y, heading},
                                 {0.0f, 0.0f, 0.0f});

            ImGui::PopID();
        }

        //Mock parking scenarios
        if (ImGui::Button("mock parallel"))
        {
            std::cout << "mock parallel parking scenario" << std::endl;
            setMockParallel();
        }
        ImGui::SameLine();
        if (ImGui::Button("mock slanted"))
        {
            std::cout << "mock slanted parking scenario" << std::endl;
            setMockSlanted();
        }
        ImGui::SameLine();
        if (ImGui::Button("mock perpendicular"))
        {
            std::cout << "mock perpendicular parking scenario" << std::endl;
            setMockPerpendicular();
        }

        // right now, I have difficulty in getting GT fence from drivesim
        if (ImGui::Button("parallel"))
        {
            std::cout << "parallel parking scenario" << std::endl;
            setUpParallel();
        }
        ImGui::SameLine();
        if (ImGui::Button("slanted"))
        {
            std::cout << "slanted parking scenario" << std::endl;
            setUpSlanted();
        }
        ImGui::SameLine();
        if (ImGui::Button("perpendicular"))
        {
            std::cout << "perpendicular parking scenario" << std::endl;
            setUpPerpendicular();
        }

        // rendering
        if (ImGui::Checkbox("showOneTheta", &m_showOneTheta))
        {
            updatePathBuffer2();
            // volume does not need an explicit update.
        }
        ImGui::SameLine();
        if (ImGui::Button("simulation update world"))
        {
            simulate();
        }
        if (ImGui::Button("simulation update start"))
        {
            simulateChangeStart();
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset simulation"))
        {
            m_parkingPlanner->setResetStart(0,0,0);
            m_costBefore = 0;
            m_parkingPlanner->m_startTurnType = 8;
        }
        // prevent view change when user is interacting with the menu and widgets
        // e.g. drag the slider
        if (ImGui::IsWindowHovered() ||
            ImGui::IsWindowFocused())
        {
            m_mouseViewLock = true;
        }
        else
        {
            m_mouseViewLock = false;
        }
    }
    ImGui::End();
}

// mock -> plan -> move -> update world -> plan -> move 
// terminate when ego gets to the destination
void ParkingPlanner2TestFixtures::simulate()
{
     //while ((m_ego - m_userDest).norm() > 3.0f)
     //{
        Coord3d tempDest(m_userDest,
                            ParkingPlanner2::getPosRes3(),
                            ParkingPlanner2::getHdgRes3());

        m_parkingPlanner->setDestination(tempDest);
        m_parkingPlanner->setObstacles(m_obstacles);
        TIME_PRINT("processCost",m_parkingPlanner->processCost();)
        updateGridBufferColor2();
        updatePathBuffer2();
        printf("Cost Before: %hu, Cost current: %hu, Total cost: %hu\n", m_costBefore, m_parkingPlanner->pathToCost(0u, m_parkingPlanner->getPath().size()),
                                                                        m_costBefore + m_parkingPlanner->pathToCost(0u, m_parkingPlanner->getPath().size()));
        move();
        updateWorld();
     //}
}

void ParkingPlanner2TestFixtures::simulateChangeStart()
{
     //while ((m_ego - m_userDest).norm() > 3.0f)
     //{
        Coord3d tempDest(m_userDest,
                            ParkingPlanner2::getPosRes3(),
                            ParkingPlanner2::getHdgRes3());

        m_parkingPlanner->setDestination(tempDest);
        m_parkingPlanner->setObstacles(m_obstacles);
        TIME_PRINT("processCost",m_parkingPlanner->processCost();)
        updateGridBufferColor2();
        updatePathBuffer2();
        printf("Cost Before: %hu, Cost current: %hu, Total cost: %hu\n", m_costBefore, m_parkingPlanner->pathToCost(0u, m_parkingPlanner->getPath().size()),
                                                                        m_costBefore + m_parkingPlanner->pathToCost(0u, m_parkingPlanner->getPath().size()));
        moveStart();
     //}
}

void ParkingPlanner2TestFixtures::moveStart()
{
    Vector3f newPose{};

    if (m_parkingPlanner->hasPath())
    {
        if (m_parkingPlanner->getPath().size() > 10) // 5 is step size
        {
            newPose = m_parkingPlanner->getPath()[10];
            m_parkingPlanner->m_startTurnType = m_parkingPlanner->getTurnTypes()[10];
            m_costBefore += m_parkingPlanner->pathToCost(0u, 11u);
        }
        else
        {
            newPose = m_parkingPlanner->getPath().back();
            m_parkingPlanner->m_startTurnType = m_parkingPlanner->getTurnTypes().back();
            m_costBefore += m_parkingPlanner->pathToCost(0u, m_parkingPlanner->m_path.size());
        }
    }
    
    m_parkingPlanner->setResetStart(Coord3d(newPose,m_parkingPlanner->getPosRes3(), m_parkingPlanner->getHdgRes3()));
}

void ParkingPlanner2TestFixtures::move()
{
    Vector3f newPose{};

   if (m_parkingPlanner->hasPath())
    {
        if (m_parkingPlanner->getPath().size() > 10) // 5 is step size
        {
            newPose = m_parkingPlanner->getPath()[10];
            printf("turn turn turn:%d",m_parkingPlanner->getTurnTypes()[10]);
            m_parkingPlanner->m_startTurnType = m_parkingPlanner->getTurnTypes()[10];
            m_costBefore += m_parkingPlanner->pathToCost(0u, 11u);
            
        }
        else
        {
            newPose = m_parkingPlanner->getPath().back();
            printf("turn turn turn:%d",m_parkingPlanner->getTurnTypes().back());
            m_parkingPlanner->m_startTurnType = m_parkingPlanner->getTurnTypes().back();
            m_costBefore += m_parkingPlanner->pathToCost(0u, m_parkingPlanner->m_path.size());
        }
    }

    // convert angle degree to radian
    float32_t heading = deg2Rad(newPose.z());

    // update m_ego
    Isometry2f e2w{};
    e2w.setTranslation({m_ego.x(), m_ego.y()});
    e2w.setRotation({std::cos(m_ego.z()), -std::sin(m_ego.z()), std::sin(m_ego.z()), std::cos(m_ego.z())});

    // transform the ego
    auto newPose1 = e2w.apply(Vector2f{newPose.x(), newPose.y()});
    m_ego.x() = newPose1.x();
    m_ego.y() = newPose1.y();
    m_ego.z() += heading;
}

void ParkingPlanner2TestFixtures::updateWorld()
{
    Vector3f ego = m_ego;
    Vector3f des = m_des;

    Isometry2f o{};
    o.setTranslation({des.x(), des.y()});
    o.setRotation({std::cos(des.z()), -std::sin(des.z()), std::sin(des.z()), std::cos(des.z())});

    Isometry2f e{};
    e.setTranslation({ego.x(), ego.y()});
    e.setRotation({std::cos(ego.z()), -std::sin(ego.z()), std::sin(ego.z()), std::cos(ego.z())});

    auto o2e = e.inverse() * o;

    m_userDest.x() = o2e.getTranslation().x();
    m_userDest.y() = o2e.getTranslation().y();
    m_userDest.z() = math::rad2Deg(des.z()-ego.z());
    m_userDest.z() = std::fmod(m_userDest.z(), 360.0f);
    if(m_userDest.z()<0)
        m_userDest.z() += 360.0f;

    m_obstacles.clear();
    for (const auto& obj : m_objs)
    {
        m_obstacles.push_back(createObstacle(obj, ego));
    }
}

dwObstacle ParkingPlanner2TestFixtures::createObstacle(const Vector3f& obj, // {x, y, hdg}
                                                       const Vector3f& ego)
{
    Isometry2f o{};
    o.setTranslation({obj.x(), obj.y()});
    o.setRotation({std::cos(obj.z()), -std::sin(obj.z()), std::sin(obj.z()), std::cos(obj.z())});

    Isometry2f e{};
    e.setTranslation({ego.x(), ego.y()});
    e.setRotation({std::cos(ego.z()), -std::sin(ego.z()), std::sin(ego.z()), std::cos(ego.z())});

    constexpr float32_t WIDTH = 2.0f;
    constexpr float32_t LENGTH = 5.0f;

    VectorFixed<Vector2f, 4u> vertices{}; // right now represented in the obj body frame
    vertices.push_back({LENGTH / 2.0f, WIDTH / 2.0f});
    vertices.push_back({LENGTH / 2.0f, -WIDTH / 2.0f});
    vertices.push_back({-LENGTH / 2.0f, -WIDTH / 2.0f});
    vertices.push_back({-LENGTH / 2.0f, WIDTH / 2.0f});

    auto o2e = e.inverse() * o;

    dwObstacle ret{};
    ret.position.x = o2e.getTranslation().x();
    ret.position.y = o2e.getTranslation().y();
    ret.position.z = 0.0f;

    auto dir = o2e.getRotation() * Vector2f{1.0f, 0.0f};
    ret.direction = dwVector3f{dir.x(), dir.y(), 0.0f};

    ret.boundaryPointCount = 0u;
    for (auto& v: vertices)
    {
        v = o2e.apply(v);
        ret.boundaryPoints[ret.boundaryPointCount++] = dwVector3f{v.x(), v.y(), 0.0f};
    }

    // std::cerr << o2e.getMatrix() << std::endl;

    return ret;
}

void ParkingPlanner2TestFixtures::setMockParallel()
{
    Vector3f obj = {6.0f, 3.0f, 0.0f};
    m_obstacles.push_back(createObstacle(obj,{0.0f, 0.0f, 0.0f}));
    obj = {-6.0f, 3.0f,0.0f};
    m_obstacles.push_back(createObstacle(obj,{0.0f, 0.0f, 0.0f}));
    m_userDest = {0.0f, 3.0f, 0.0f};
}

void ParkingPlanner2TestFixtures::setMockSlanted()
{
    Vector3f obj = {2.5f, 4.5f, math::deg2Rad(45.0f)};
    m_obstacles.push_back(createObstacle(obj,{0.0f, 0.0f, 0.0f}));
    obj = {9.5f, 4.5f, math::deg2Rad(45.0f)};
    m_obstacles.push_back(createObstacle(obj,{0.0f, 0.0f, 0.0f}));
    m_userDest = {6.0f, 4.5f, 45.0f};
}

void ParkingPlanner2TestFixtures::setMockPerpendicular()
{
    Vector3f obj = {1.5f, 4.5f, math::deg2Rad(90.0f)};
    m_obstacles.push_back(createObstacle(obj,{0.0f, 0.0f, 0.0f}));
    obj = {6.5f, 4.5f, math::deg2Rad(90.0f)};
    m_obstacles.push_back(createObstacle(obj,{0.0f, 0.0f, 0.0f}));
    m_userDest = {4.0f, 4.5f, 90.0f};
}

void ParkingPlanner2TestFixtures::setUpParallel()
{
    Vector3f ego{-732.34f, -341.41f, math::deg2Rad(8.438f)};

    VectorFixed<Vector3f, 10> objs{};
    objs.push_back({-715.68, -339.77, math::deg2Rad(-120.f)});
    objs.push_back({-714.23, -343.77, math::deg2Rad(-120.f)});
    objs.push_back({-713.18, -349.09, math::deg2Rad(-120.f)});
    objs.push_back({-711.96, -353.86, math::deg2Rad(-120.f)});
    objs.push_back({-732.39, -337.09, math::deg2Rad(10.f)});

    // parking spot
    // -725.54, -335.87, -10
    Vector3f des{-725.54f, -335.87f, math::deg2Rad(10.0f)};
    Isometry2f o{};
    o.setTranslation({des.x(), des.y()});
    o.setRotation({std::cos(des.z()), -std::sin(des.z()), std::sin(des.z()), std::cos(des.z())});

    Isometry2f e{};
    e.setTranslation({ego.x(), ego.y()});
    e.setRotation({std::cos(ego.z()), -std::sin(ego.z()), std::sin(ego.z()), std::cos(ego.z())});

    auto o2e = e.inverse() * o;

    m_userDest.x() = o2e.getTranslation().x();
    m_userDest.y() = o2e.getTranslation().y();
    m_userDest.z() = math::rad2Deg(des.z()-ego.z());

    m_obstacles.clear();
    for (const auto& obj : objs)
    {
        m_obstacles.push_back(createObstacle(obj, ego));
    }
    m_ego = ego;
    m_des = des;
    m_objs.copyFrom(objs);
}

void ParkingPlanner2TestFixtures::setUpSlanted()
{
    Vector3f ego{-755.12, -344.83, math::deg2Rad(-71.719f)};

    VectorFixed<Vector3f, 10> objs{};
    objs.push_back({-759.22, -347.37, math::deg2Rad(-35.f)});
    objs.push_back({-756.73, -356.80, math::deg2Rad(-35.f)});
    objs.push_back({-755.17, -362.03, math::deg2Rad(-35.f)});

    // parking spot
    // -757.85, -352.57, 35
    Vector3f des{-757.85f, -352.57f, math::deg2Rad(-35.0f)};
    Isometry2f o{};
    o.setTranslation({des.x(), des.y()});
    o.setRotation({std::cos(des.z()), -std::sin(des.z()), std::sin(des.z()), std::cos(des.z())});

    Isometry2f e{};
    e.setTranslation({ego.x(), ego.y()});
    e.setRotation({std::cos(ego.z()), -std::sin(ego.z()), std::sin(ego.z()), std::cos(ego.z())});

    auto o2e = e.inverse() * o;

    m_userDest.x() = o2e.getTranslation().x();
    m_userDest.y() = o2e.getTranslation().y();
    m_userDest.z() = math::rad2Deg(des.z()-ego.z());

    m_obstacles.clear();
    for (const auto& obj : objs)
    {
        m_obstacles.push_back(createObstacle(obj, ego));
    }
    m_ego = ego;
    m_des = des;
    m_objs.copyFrom(objs);
}

void ParkingPlanner2TestFixtures::setUpPerpendicular()
{
    Vector3f ego{-710.1f, -367.28f, math::deg2Rad(-80.f)};

    VectorFixed<Vector3f, 10> objs{};
    objs.push_back({-715.43, -372.22, math::deg2Rad(10.f)});
    objs.push_back({-714.71, -375.26, math::deg2Rad(10.f)});
    objs.push_back({-714.59, -378.24, math::deg2Rad(10.f)});
    //objs.push_back({-713.77, -381.06, math::deg2Rad(10.f)}); //Parking spot
    objs.push_back({-713.34, -384.05, math::deg2Rad(10.f)});
    objs.push_back({-712.93, -387.23, math::deg2Rad(10.f)});
    objs.push_back({-712.38, -389.73, math::deg2Rad(10.f)});

    // parking spot
    // choose any of the obstacle spot above
    Vector3f des{-713.77, -381.06, math::deg2Rad(10.f)};
    Isometry2f o{};
    o.setTranslation({des.x(), des.y()});
    o.setRotation({std::cos(des.z()), -std::sin(des.z()), std::sin(des.z()), std::cos(des.z())});

    Isometry2f e{};
    e.setTranslation({ego.x(), ego.y()});
    e.setRotation({std::cos(ego.z()), -std::sin(ego.z()), std::sin(ego.z()), std::cos(ego.z())});

    auto o2e = e.inverse() * o;

    m_userDest.x() = o2e.getTranslation().x();
    m_userDest.y() = o2e.getTranslation().y();
    m_userDest.z() = math::rad2Deg(des.z()-ego.z());

    m_obstacles.clear();
    for (const auto& obj : objs)
    {
        m_obstacles.push_back(createObstacle(obj, ego));
    }
    m_ego = ego;
    m_des = des;
    m_objs.copyFrom(objs);
}

void ParkingPlanner2TestFixtures::initGridBuffer()
{
    auto& renderBuffer          = m_renderEnginePtr->getBuffer(m_gridBufferId);
    span<ColoredPoint3D> buffer = renderBuffer.mapWriteVertices<ColoredPoint3D>();

    const float32_t POS_RES = ParkingPlanner2::getPosRes();
    const float32_t HDG_RES = ParkingPlanner2::getHdgRes();

    for (int32_t x = 0; x < ParkingPlanner2::getSize(); ++x)
    {
        for (int32_t y = 0; y < ParkingPlanner2::getSize(); ++y)
        {
            for (uint32_t theta = 0; theta < ParkingPlanner2::getThetaStep(); ++theta)
            {
                Coord3d coord = Coord3d(x - (ParkingPlanner2::getSize() >> 1),
                                        y - (ParkingPlanner2::getSize() >> 1),
                                        theta);
                Vector3f pose = coord.getPose(POS_RES, HDG_RES);

                pose.x() = pose.x() * POS_SCALE;
                pose.y() = pose.y() * POS_SCALE;
                pose.z() = pose.z() * HDG_SCALE;

                int32_t index = ParkingPlanner2::getCell(coord.x, coord.y, coord.hdg);

                buffer[index].position = pose;
                buffer[index].color    = transparentColor2Vector(visualization::COLOR_LIGHTGREY, 0.02f);
            }
        }
    }

    renderBuffer.unmap(m_parkingPlanner->getGridSize());
}

void ParkingPlanner2TestFixtures::initGridBuffer2()
{
    auto& renderBuffer          = m_renderEnginePtr->getBuffer(m_gridBufferId);
    span<ColoredPoint3D> buffer = renderBuffer.mapWriteVertices<ColoredPoint3D>();

    const float32_t POS_RES = ParkingPlanner2::getPosRes3();
    const float32_t HDG_RES = ParkingPlanner2::getHdgRes3();

    for (int32_t x = 0; x < ParkingPlanner2::getDim3(); ++x)
    {
        for (int32_t y = 0; y < ParkingPlanner2::getDim3(); ++y)
        {
            for (uint32_t theta = 0; theta < ParkingPlanner2::getThetaStep3(); ++theta)
            {
                Coord3d coord = Coord3d(x - (ParkingPlanner2::getHalfDim3()),
                                        y - (ParkingPlanner2::getHalfDim3()),
                                        theta);
                Vector3f pose = coord.getPose(POS_RES, HDG_RES);

                pose.x() = pose.x() * POS_SCALE;
                pose.y() = pose.y() * POS_SCALE;
                pose.z() = pose.z() * HDG_SCALE;

                int32_t index = ParkingPlanner2::volIndex(coord.x, coord.y, coord.hdg);

                buffer[index].position = pose;
                buffer[index].color    = transparentColor2Vector(visualization::COLOR_LIGHTGREY, 0.02f);
            }
        }
    }

    renderBuffer.unmap(m_parkingPlanner->getGridSize3());
}

TEST_F(ParkingPlanner2TestFixtures, Gui_L0)
{
    if (m_renderOn)
    {
        //initRenderViewsInteractive();
        //render();
        initRenderViewsInteractive2();
        render2();
    }
}
} // namespace planner
} // namespace dw
