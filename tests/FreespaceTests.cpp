#include "gtest/gtest.h"
#include "../src/Freespace.hpp"
#include "../src/Visualization.hpp"
#include "../src/Paper.hpp"
#include <numeric> // accumulate
#include "json.hpp"

/**
 * Testing strategy:
 * - assert the CPU / GPU implementations create the same freespace volume
 * - assert that either volume (because they are the same) make the same free or not determine
 *   as the polygonal intersection check
 * 
 */

// mock a rectangular obstacle
Obstacle createObstacle()
{
    Obstacle obs{};
    obs.pos = {5.0f, 0.0f};
    obs.hdg = 0.0f;
    obs.boundaryPoints.push_back({3.0, 1.0});
    obs.boundaryPoints.push_back({7.0, 1.0});
    obs.boundaryPoints.push_back({7.0, -1.0});
    obs.boundaryPoints.push_back({3.0, -1.0});
    obs.boundaryPoints.push_back({3.0, 1.0});

    return obs;
}

const float32_t width  = 2.0f;
const float32_t length = 5.0f;
Obstacle createObstacle(const Vector2f& pos, float32_t hdg)
{
    Obstacle obs{};
    obs.pos = pos;
    obs.hdg = hdg;

    obs.boundaryPoints.push_back(pos + Vector2f{length / 2.0f, width / 2.0f}.rotate(hdg));
    obs.boundaryPoints.push_back(pos + Vector2f{length / 2.0f, -width / 2.0f}.rotate(hdg));
    obs.boundaryPoints.push_back(pos + Vector2f{-length / 2.0f, -width / 2.0f}.rotate(hdg));
    obs.boundaryPoints.push_back(pos + Vector2f{-length / 2.0f, width / 2.0f}.rotate(hdg));
    obs.boundaryPoints.push_back(pos + Vector2f{length / 2.0f, width / 2.0f}.rotate(hdg));

    return obs;
}

const nlohmann::json slanted =
    {
        {"ego",
         {{"x", -755.12},
          {"y", -344.83},
          {"yaw", -71.719}}},
        {"park",
         {{"x", -757.85},
          {"y", -352.57},
          {"yaw", -35.0}}},
        {"obstacles",
         {
             {{"x", -759.22},
              {"y", -347.37},
              {"yaw", -35.0}},
             {{"x", -756.73},
              {"y", -356.80},
              {"yaw", -35.0}},
             {{"x", -755.17},
              {"y", -362.03},
              {"yaw", -35.0}},
         }}};

const nlohmann::json test =
    {
        {"ego",
         {{"x", 0},
          {"y", 0},
          {"yaw", 0}}},
        {"park",
         {{"x", -757.85},
          {"y", -352.57},
          {"yaw", -35.0}}},
        {"obstacles",
         {
             {{"x", 10.0},
              {"y", 0.0},
              {"yaw", 45}},
         }}};

void createScene(std::vector<Obstacle>& vec)
{
    auto j = slanted;

    Vector2f ego  = {j["ego"]["x"], j["ego"]["y"]};
    float32_t psi = deg2Rad(j["ego"]["yaw"]);

    for (size_t i = 0; i < j["obstacles"].size(); ++i)
    {
        auto obstacle   = j["obstacles"].at(i);
        Vector2f obs    = {obstacle["x"], obstacle["y"]};
        float32_t theta = deg2Rad(obstacle["yaw"]);

        obs = obs.transform(psi, ego);
        vec.push_back(createObstacle(obs, theta - psi));
    }
}

TEST(FreespaceTests, temp)
{
    Freespace freespace{};
    Visualization vis{};
    vis.setFreespace(freespace.get());

    std::vector<Obstacle> vec{};
    // vec.push_back(createObstacle());
    createScene(vec);
    // 1430.49 ms
    // TIME_PRINT("CPU", freespace.computeFreespaceCPU(vec));

    // 4.61719 ms
    TIME_PRINT("GPU", freespace.computeFreespaceGPU(vec));

    vis.draw();
}

TEST(FreespaceTests, CPUvsGPU)
{
    Freespace freespace{};

    std::vector<Obstacle> vec{};
    vec.push_back(createObstacle());

    Dimension dim{};
    size_t size = dim.row * dim.col * dim.height;

    using value_type = Freespace::value_type;
    std::unique_ptr<value_type[]> cpuResult{};
    cpuResult = std::make_unique<value_type[]>(size);

    freespace.computeFreespaceCPU(vec);
    std::copy_n(freespace.get(), size, cpuResult.get());

    freespace.computeFreespaceGPU(vec);
    const value_type* gpuResult = freespace.get();

    // check element-wise equality
    uint32_t cnt{0u};
    for (uint32_t i = 0u; i < size; ++i)
    {
        // failure case
        if (cpuResult[i] != gpuResult[i])
        {
            cnt++;
            std::cerr << "Failure index: " << i
                      << " cpu " << cpuResult[i]
                      << " vs gpu " << gpuResult[i]
                      << std::endl;
            EXPECT_TRUE(false);
        }
    }
    std::cerr << "Total mismatch cnt: " << cnt << std::endl;
}

void xshear(uint32_t* out,
            const uint32_t* in,
            float32_t shear,
            int32_t width,
            int32_t height,
            float32_t res,
            bool antiAlias = false)
{
    for (int32_t y = 0; y < height; ++y)
    {
        float32_t skew  = shear * static_cast<float32_t>(y - height / 2); // how much to shift this line
        int32_t skewi   = std::floor(skew + 0.5f);                        // integer part
        float32_t skewf = skew - skewi;                                   // fractional part

        float32_t oleft = 0.f;

        for (int32_t x = width - 1; x >= 0; --x)
        {
            uint32_t pixel = in[x + y * width];
            float32_t left = (antiAlias) ? pixel * skewf : 0.f;
            pixel          = (pixel - left) + oleft;

            if (x + skewi >= 0 &&
                x + skewi < width)
            {
                out[(x + skewi) + y * width] = pixel;
            }
            oleft = left;
        }
        if (skewi >= 0 && skewi < width)
        {
            out[skewi + y * width] = oleft;
        }
    }
}

void yshear(uint32_t* out,
            const uint32_t* in,
            float32_t shear,
            int32_t width,
            int32_t height,
            float32_t res,
            bool antiAlias = false)
{
    for (int32_t x = 0; x < width; ++x)
    {
        float32_t skew  = shear * static_cast<float32_t>(x - width / 2); // how much to shift this line
        int32_t skewi   = std::floor(skew + 0.5f);                       // integer part
        float32_t skewf = skew - skewi;                                  // fractional part

        float32_t oleft = 0.f;

        for (int32_t y = height - 1; y >= 0; --y)
        {
            uint32_t pixel = in[x + y * width];
            float32_t left = (antiAlias) ? pixel * skewf : 0.f;
            pixel          = (pixel - left) + oleft;

            if (y + skewi >= 0 &&
                y + skewi < width)
            {
                out[x + (y + skewi) * width] = pixel;
            }
            oleft = left;
        }
        if (skewi >= 0 && skewi < height)
        {
            out[x + skewi * width] = oleft;
        }
    }
}

// url: https://www.ocf.berkeley.edu/~fricke/projects/israel/paeth/rotation_by_shearing.html
void shearRotate(uint32_t* out,
                 const uint32_t* in,
                 float32_t angle,
                 int32_t width,
                 int32_t height)
{
    float32_t alpha = -std::tan(angle / 2.0f);
    float32_t gamma = alpha;
    float32_t beta  = std::sin(angle);

    std::unique_ptr<uint32_t[]> tmp = std::make_unique<uint32_t[]>(width * height);
    std::fill_n(tmp.get(), width * height, 0u);

    std::cout << " alpha " << alpha
              << " beta " << beta
              << " gamma " << gamma
              << std::endl;

    xshear(out, in, alpha, width, height, 0.5f);
    yshear(tmp.get(), out, beta, width, height, 0.5f);
    xshear(out, tmp.get(), gamma, width, height, 0.5f);

    uint32_t in_ones  = std::accumulate(in, in + width * height, 0);
    uint32_t out_ones = std::accumulate(out, out + width * height, 0);
    EXPECT_EQ(in_ones, out_ones);
}

void shearRotateSimple(uint32_t* out,
                       const uint32_t* in,
                       float32_t angle,
                       int32_t width,
                       int32_t height)
{
    for (uint32_t i = 0u; i < width; ++i)
    {
        for (uint32_t j = 0u; j < height; ++j)
        {
            int32_t x = (int32_t)i - width / 2;
            int32_t y = (int32_t)j - height / 2;

            Vector2i pos{x, y};
            Vector2i newPos = pos.shearRotate(angle);

            uint32_t ii = newPos.x + width / 2;
            uint32_t jj = newPos.y + height / 2;

            if (ii >= 0 &&
                ii < width &&
                jj >= 0 &&
                jj <= height)
            {
                out[ii + jj * width] = in[i + j * width];
            }
        }
    }
}

void naiveRotate(
    uint32_t* out,
    const uint32_t* in,
    float32_t theta,
    int32_t width,
    int32_t height,
    float32_t res)
{
    for (uint32_t i = 0u; i < width; ++i)
    {
        for (uint32_t j = 0u; j < height; ++j)
        {
            Vector2f pos = toCartesian(i, j, width, height, res);
            uint32_t ind = i + width * j;

            Vector2f newPos = pos.rotate(-theta);

            if (isInBoundary(newPos.x, newPos.y, 30.f)) //TODO: fix the hack
            {
                Vector2ui newInd = toIndex(newPos.x, newPos.y, width, height, res);
                out[ind]         = in[index(newInd.x, newInd.y, 0u, width, height)];
            }
        }
    }
}

// investigate rotation
TEST(FreespaceTests, rotationByShearing)
{
    Dimension dim{};
    std::unique_ptr<uint32_t[]> mem = std::make_unique<uint32_t[]>(dim.row * dim.col);
    std::fill_n(mem.get(), dim.row * dim.col, 0u);

    std::unique_ptr<uint32_t[]> mem1 = std::make_unique<uint32_t[]>(dim.row * dim.col);
    std::fill_n(mem1.get(), dim.row * dim.col, 0u);

    // create a rectangle
    auto createRectangle = [dim](uint32_t* mem,
                                 uint32_t val,
                                 int32_t halfEdgeLength) {
        for (int32_t i = -halfEdgeLength; i < halfEdgeLength; ++i)
        {
            for (int32_t j = -halfEdgeLength; j < halfEdgeLength; ++j)
            {
                mem[index(i + dim.row / 2, j + dim.col / 2, 0, dim.row, dim.col)] = val;
            }
        }
    };

    auto createLine = [dim](uint32_t* mem,
                            uint32_t val,
                            int32_t halfEdgeLength) {
        int j = 0;

        for (int32_t i = -halfEdgeLength; i < halfEdgeLength; ++i)
        {
            mem[index(i + dim.row / 2, j + dim.col / 2, 0, dim.row, dim.col)] = val;
        }
    };

    auto checkSumTest = [](uint32_t* a, uint32_t* b, uint32_t size) {
        uint32_t sumA = std::accumulate(a, a + size, 0);
        uint32_t sumB = std::accumulate(b, b + size, 0);

        EXPECT_EQ(sumA, sumB);
    };

    // createRectangle(mem.get(), 1u, 20);
    // createRectangle(mem.get(), 0u, 19);

    createLine(mem.get(), 1u, 20);

    float32_t angle = M_PI * 3.0f / 10.0f;

    // rotation will be off if angle is close to pi
    // apply two opposite shearRotation should get you exactly the original image
    // shearRotate(mem1.get(), mem.get(), angle, dim.row, dim.col);
    // shearRotate(mem.get(), mem1.get(), -angle, dim.row, dim.col);

    // naiveRotate(mem1.get(), mem.get(), angle, dim.row, dim.col, dim.posRes);
    // naiveRotate(mem.get(), mem1.get(), -angle, dim.row, dim.col, dim.posRes);

    shearRotateSimple(mem1.get(), mem.get(), angle, dim.row, dim.col);
    shearRotateSimple(mem.get(), mem1.get(), -angle, dim.row, dim.col);

    checkSumTest(mem.get(), mem1.get(), dim.row * dim.col);

    Visualization vis{};
    vis.setFreespace(mem1.get());

    // vis.draw();
}