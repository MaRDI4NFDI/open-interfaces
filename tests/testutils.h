#pragma once
#include <cmath>
#include <exception>

/**
 * Version of `sqrt` function with `constexpr` qualifier.
 *
 * The code uses Newton-Raphson procedure to find a root of a nonlinear
 * function f(y) = 0 by truncated Taylor series.
 * In this case, f(y) = y^2 - x, where `x` is a given constant.
 *
 * @param x Value, square root of which is to be computed
 * @return Approximate value of the square root of `x`
 */
constexpr double constexpr_sqrt(double x) {
    double tol = 1e-8;
    double y = x / 2.0;
    int k = 0;
    const int max_iter = 50;

    while ((std::abs(y * y - x) > tol) && (k < max_iter)) {
        y = 0.5 * (y + x / y);
        k += 1;
    }

    if (k > max_iter) {
        throw std::exception();
    }

    return y;
}

/**
 * Solve a nonlinear equation f(x) = 0 using the Newton-Raphson method.
 * The implementation is simplistic so it expects also a derivative function
 * `fprime` to be provided by user. Also, it works only for scalar functions
 * `f: R -> R`.
 */
template <class F, class Fprime>
double fsolve(F const &f, Fprime const &fprime) {
    double x = 1.0;
    double tol = 1e-15;
    size_t k = 0;
    const size_t max_iter = 50;

    while (((std::abs(f(x))) > tol) && (k < max_iter)) {
        x = x - f(x) / fprime(x);
    }

    if (k > max_iter) {
        throw std::exception();
    }

    return x;
}
