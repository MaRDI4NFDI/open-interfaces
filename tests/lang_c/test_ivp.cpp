#include <cmath>
#include <cstdio>

#include <gtest/gtest.h>

extern "C" {
#include "oif/c_bindings.h"
#include "oif/interfaces/ivp.h"
}

class ScalarExpDecayProblem {
public:
    static constexpr int N = 1;
    static constexpr double y0[] = {1.0};
    static void
    rhs(double t, OIFArrayF64 *y, OIFArrayF64 *rhs_out) {
        int size = y->dimensions[0];
        for (int i = 0; i < size; ++i) {
            rhs_out->data[i] = -y->data[i];
        }
    }
    static double exact(double t, int i) {
        // Exact solution at time t, component i.
        return exp(-t);
    }
};

class LinearOscillatorProblem {
public:
    static constexpr int N = 2;
    static constexpr double y0[2] = {1.0, 0.5};
    static constexpr double omega = M_PI;
    static void
    rhs(double t, OIFArrayF64 *y, OIFArrayF64 *rhs_out) {
        rhs_out->data[0] = y->data[1];
        rhs_out->data[1] = -omega * omega * y->data[0];
    }
    static double exact(double t, int i) {
        // Exact solution at time t, component i.
        assert(i == 0);  // Check only the first component.
        return y0[0] * cos(omega*t) + y0[1] * sin(omega*t) / omega;
    }
};

TEST(IvpScipyOdeDopri5TestSuite, ScalarExpDecayTestCase) {
    double t0 = 0.0;
    intptr_t dims[] = {ScalarExpDecayProblem::N,};
    OIFArrayF64 *y0 = oif_init_array_f64_from_data(
        1, dims, ScalarExpDecayProblem::y0);
    OIFArrayF64 *y = oif_create_array_f64(1, dims);
    ImplHandle implh = oif_init_impl("ivp", "scipy_ode_dopri5", 1, 0);

    int status;
    status = oif_ivp_set_rhs_fn(implh, ScalarExpDecayProblem::rhs);
    status = oif_ivp_set_initial_value(implh, y0, t0);

    double t = 0.1;
    auto t_span = {0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0};
    for (auto t : t_span) {
        status = oif_ivp_integrate(implh, t, y);
        EXPECT_NEAR(y->data[0], ScalarExpDecayProblem::exact(t, 0), 1e-15);
    }
}


TEST(IvpScipyOdeDopri5TestSuite, LinearOscillatorTestCase) {
    double t0 = 0.0;
    intptr_t dims[] = {LinearOscillatorProblem::N,};
    OIFArrayF64 *y0 = oif_init_array_f64_from_data(
        1, dims, LinearOscillatorProblem::y0
    );
     OIFArrayF64 *y = oif_create_array_f64(1, dims);
    ImplHandle implh = oif_init_impl("ivp", "scipy_ode_dopri5", 1, 0);

    int status;
    status = oif_ivp_set_rhs_fn(implh, LinearOscillatorProblem::rhs);
    status = oif_ivp_set_initial_value(implh, y0, t0);

    double t = 0.1;
    auto t_span = {0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0};
    for (auto t : t_span) {
        status = oif_ivp_integrate(implh, t, y);
        EXPECT_NEAR(y->data[0], LinearOscillatorProblem::exact(t, 0), 1e-12);
    }
}
