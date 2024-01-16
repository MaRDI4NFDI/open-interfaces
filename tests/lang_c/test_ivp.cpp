#include "gtest/gtest.h"
#include <cmath>
#include <cstdio>

#include "testutils.h"
#include <gtest/gtest.h>

extern "C" {
#include "oif/c_bindings.h"
#include "oif/interfaces/ivp.h"
}

const double EPS = 0.9;

class ScalarExpDecayProblem {
  public:
    static constexpr int N = 1;
    static constexpr double y0[] = {1.0};
    static int rhs(double t, OIFArrayF64 *y, OIFArrayF64 *rhs_out) {
        int size = y->dimensions[0];
        for (int i = 0; i < size; ++i) {
            rhs_out->data[i] = -y->data[i];
        }
        return 0;
    }
    static void verify(double t, OIFArrayF64 *y) {
        EXPECT_NEAR(y->data[0], exp(-t), 1e-15);
    }
};

class LinearOscillatorProblem {
  public:
    static constexpr int N = 2;
    static constexpr double y0[2] = {1.0, 0.5};
    static constexpr double omega = M_PI;
    static int rhs(double t, OIFArrayF64 *y, OIFArrayF64 *rhs_out) {
        rhs_out->data[0] = y->data[1];
        rhs_out->data[1] = -omega * omega * y->data[0];
        return 0;
    }
    static void verify(double t, OIFArrayF64 *y) {
        double y_exact_0 =
            y0[0] * cos(omega * t) + y0[1] * sin(omega * t) / omega;
        EXPECT_NEAR(y->data[0], y_exact_0, 1e-12);
        double y_exact_1 =
            -y0[0] * omega * sin(omega * t) + y0[1] * cos(omega * t);
        EXPECT_NEAR(y->data[1], y_exact_1, 1e-12);
    }
};

class OrbitEquationsProblem {
  public:
    static constexpr int N = 4;
    static constexpr double eps = 0.9L;
    static constexpr double y0[4] = {
        1 - eps, 0.0, 0.0, constexpr_sqrt((1 + eps) / (1 - eps))};
    static int rhs(double t, OIFArrayF64 *y, OIFArrayF64 *rhs_out) {
        double r = sqrt(y->data[0] * y->data[0] + y->data[1] * y->data[1]);
        rhs_out->data[0] = y->data[2];
        rhs_out->data[1] = y->data[3];
        rhs_out->data[2] = -y->data[0] / (r * r * r);
        rhs_out->data[3] = -y->data[1] / (r * r * r);

        return 0;
    }
    static void verify(double t, OIFArrayF64 *y) {
        double u = fsolve([t](double u) { return u - eps * sin(u) - t; },
                          [](double u) { return 1 - eps * cos(u); });
        EXPECT_NEAR(y->data[0], cos(u) - eps, 1e-10);
        EXPECT_NEAR(y->data[1], sqrt(1 - pow(eps, 2)) * sin(u), 1e-10);
        EXPECT_NEAR(y->data[2], -sin(u) / (1 - eps * cos(u)), 1e-10);
        EXPECT_NEAR(y->data[3],
                    sqrt(1 - pow(eps, 2)) * cos(u) / (1 - eps * cos(u)),
                    1e-10);
    }
};

struct IvpImplementationsFixture : public testing::TestWithParam<const char *> {
};

TEST_P(IvpImplementationsFixture, ScalarExpDecayTestCase) {
    double t0 = 0.0;
    intptr_t dims[] = {
        ScalarExpDecayProblem::N,
    };
    OIFArrayF64 *y0 =
        oif_init_array_f64_from_data(1, dims, ScalarExpDecayProblem::y0);
    OIFArrayF64 *y = oif_create_array_f64(1, dims);
    ImplHandle implh = oif_init_impl("ivp", "scipy_ode_dopri5", 1, 0);

    int status;
    status = oif_ivp_set_initial_value(implh, y0, t0);
    status = oif_ivp_set_rhs_fn(implh, ScalarExpDecayProblem::rhs);

    auto t_span = {0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0};
    for (auto t : t_span) {
        status = oif_ivp_integrate(implh, t, y);
        ScalarExpDecayProblem::verify(t, y);
    }

    oif_free_array_f64(y0);
    oif_free_array_f64(y);
}

TEST_P(IvpImplementationsFixture, LinearOscillatorTestCase) {
    double t0 = 0.0;
    intptr_t dims[] = {
        LinearOscillatorProblem::N,
    };
    OIFArrayF64 *y0 =
        oif_init_array_f64_from_data(1, dims, LinearOscillatorProblem::y0);
    OIFArrayF64 *y = oif_create_array_f64(1, dims);
    ImplHandle implh = oif_init_impl("ivp", "scipy_ode_dopri5", 1, 0);

    int status;
    status = oif_ivp_set_initial_value(implh, y0, t0);
    status = oif_ivp_set_rhs_fn(implh, LinearOscillatorProblem::rhs);

    auto t_span = {0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0};
    for (auto t : t_span) {
        status = oif_ivp_integrate(implh, t, y);
        LinearOscillatorProblem::verify(t, y);
    }

    oif_free_array_f64(y0);
    oif_free_array_f64(y);
}

TEST_P(IvpImplementationsFixture, OrbitEquationsProblemTestCase) {
    double t0 = 0.0;
    intptr_t dims[] = {
        OrbitEquationsProblem::N,
    };
    OIFArrayF64 *y0 =
        oif_init_array_f64_from_data(1, dims, OrbitEquationsProblem::y0);
    OIFArrayF64 *y = oif_create_array_f64(1, dims);
    ImplHandle implh = oif_init_impl("ivp", "scipy_ode_dopri5", 1, 0);

    int status;
    status = oif_ivp_set_initial_value(implh, y0, t0);
    status = oif_ivp_set_rhs_fn(implh, OrbitEquationsProblem::rhs);

    auto t_span = {0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0};
    for (auto t : t_span) {
        status = oif_ivp_integrate(implh, t, y);
        OrbitEquationsProblem::verify(t, y);
    }

    oif_free_array_f64(y0);
    oif_free_array_f64(y);
}

INSTANTIATE_TEST_SUITE_P(IvpImplementationsTests,
                         IvpImplementationsFixture,
                         testing::Values("scipy_ode_dopri5", "sundials_cvode"));
