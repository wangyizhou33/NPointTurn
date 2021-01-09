#ifndef VECTOR_HPP_
#define VECTOR_HPP_

#include "Helper.hpp"
#include <limits>
#include <cmath>

template <typename T>
bool isFloatZero(T a)
{
    return a < std::numeric_limits<T>::epsilon() &&
           a > -std::numeric_limits<T>::epsilon();
}

struct Vector2f
{
    float32_t x{};
    float32_t y{};

    Vector2f(float32_t _x, float32_t _y)
        : x(_x)
        , y(_y){};

    static Vector2f Zero() { return {0.0f, 0.0f}; };

    static bool isZero(const Vector2f& a)
    {
        return isFloatZero(a.x) && isFloatZero(a.y);
    };

    bool isZero() const { return isZero(*this); };

    float32_t at(size_t i) const
    {
        if (i == 0)
            return x;
        else if (i == 1)
            return y;
        else
            throw "index access out of bound";
    };

    float32_t& at(size_t i)
    {
        if (i == 0)
            return x;
        else if (i == 1)
            return y;
        else
            throw "index access out of bound";
    };

    float32_t operator()(size_t i) const { return at(i); };
    float32_t& operator()(size_t i) { return at(i); };

    Vector2f operator+(const Vector2f& v) const
    {
        return {x + v.x, y + v.y};
    };

    Vector2f& operator+=(const Vector2f& v)
    {
        *this = *this + v;
        return *this;
    };

    Vector2f operator-(const Vector2f& v) const
    {
        return {x - v.x, y - v.y};
    };

    Vector2f& operator-=(const Vector2f& v)
    {
        *this = *this - v;
        return *this;
    };

    Vector2f operator/(float32_t d) const
    {
        return {x / d, y / d};
    };

    Vector2f& operator/=(float32_t d)
    {
        *this = *this / d;
        return *this;
    };

    float32_t norm() const { return std::sqrt(x * x + y * y); };

    void normalize()
    {
        if (isZero())
            throw "norm is zero";
        float32_t n = norm();
        x /= n;
        y /= n;
    };

    float32_t dot(const Vector2f& in) const
    {
        return x * in.x + y * in.y;
    };
};

#endif // VECTOR_HPP_