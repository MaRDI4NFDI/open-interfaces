#include <cmath>
#include <cstdio>

#include "testutils.h"
#include <gtest/gtest.h>

extern "C" {
#include "oif/c_bindings.h"
#include "oif/interfaces/ivp.h"
}

const double EPS = 0.9;

class ODEProblem {
public:
    int N;
    double *y0;

    ODEProblem(int N) : N(N) {
        y0 = new double[N];
    }

    ~ODEProblem() {
        delete[] y0;
    }

    static int rhs_wrapper(double t, OIFArrayF64 *y, OIFArrayF64 *ydot, void *user_data) {
        ODEProblem *problem = reinterpret_cast<ODEProblem*>(user_data);
        problem->rhs(t, y, ydot, NULL);
        return 0;
    }

    virtual int rhs(double t, OIFArrayF64 *y, OIFArrayF64 *ydot, void *user_data) = 0;

    virtual void verify(double t, OIFArrayF64 *y) = 0;
};

class ScalarExpDecayProblem : public ODEProblem {
   public:
    ScalarExpDecayProblem() : ODEProblem(1) {
        y0[0] = 1.0;
    }

    int
    rhs(double /* t */, OIFArrayF64 *y, OIFArrayF64 *rhs_out, void */* user_data */) override
    {
        int size = y->dimensions[0];
        for (int i = 0; i < size; ++i) {
            rhs_out->data[i] = -y->data[i];
        }
        return 0;
    }

    void
    verify(double t, OIFArrayF64 *y) override
    {
        EXPECT_NEAR(y->data[0], exp(-t), 1e-14);
    }
};

class LinearOscillatorProblem : public ODEProblem {
    public:
    LinearOscillatorProblem() : ODEProblem(2) {
        y0[0] = 1.0;
        y0[1] = 0.5;
    }

    int
    rhs(double /* t */, OIFArrayF64 *y, OIFArrayF64 *rhs_out, void */* user_data */) override
    {
        rhs_out->data[0] = y->data[1];
        rhs_out->data[1] = -omega * omega * y->data[0];
        return 0;
    }
    void
    verify(double t, OIFArrayF64 *y) override
    {
        double y_exact_0 = y0[0] * cos(omega * t) + y0[1] * sin(omega * t) / omega;
        EXPECT_NEAR(y->data[0], y_exact_0, 1e-12);
        double y_exact_1 = -y0[0] * omega * sin(omega * t) + y0[1] * cos(omega * t);
        EXPECT_NEAR(y->data[1], y_exact_1, 1e-12);
    }
    private:
    double omega = M_PI;
};

class OrbitEquationsProblem : public ODEProblem {
    public:
    OrbitEquationsProblem() : ODEProblem(4) {
        y0[0] = 1 - eps;
        y0[1] = 0.0;
        y0[2] = 0.0;
        y0[3] = sqrt((1 + eps) / (1 - eps));
    }

    int
    rhs(double /* t */, OIFArrayF64 *y, OIFArrayF64 *rhs_out, void */* user_data */) override
    {
        double r = sqrt(y->data[0] * y->data[0] + y->data[1] * y->data[1]);
        rhs_out->data[0] = y->data[2];
        rhs_out->data[1] = y->data[3];
        rhs_out->data[2] = -y->data[0] / (r * r * r);
        rhs_out->data[3] = -y->data[1] / (r * r * r);

        return 0;
    }

    void
    verify(double t, OIFArrayF64 *y) override
    {
        double u = fsolve([&, t](double u) { return u - eps * sin(u) - t; },
                          [&](double u) { return 1 - eps * cos(u); });
        EXPECT_NEAR(y->data[0], cos(u) - eps, 1e-10);
        EXPECT_NEAR(y->data[1], sqrt(1 - pow(eps, 2)) * sin(u), 1e-10);
        EXPECT_NEAR(y->data[2], -sin(u) / (1 - eps * cos(u)), 1e-10);
        EXPECT_NEAR(y->data[3], sqrt(1 - pow(eps, 2)) * cos(u) / (1 - eps * cos(u)), 1e-10);
    }
   private:
   double eps = 0.9L;
};

struct IvpImplementationsFixture : public testing::TestWithParam<std::tuple<const char *, ODEProblem *>> {};

TEST_P(IvpImplementationsFixture, ScalarExpDecayTestCase)
{
    const char *impl = std::get<0>(GetParam());
    ODEProblem *problem = std::get<1>(GetParam());
    double t0 = 0.0;
    intptr_t dims[] = {
        problem->N,
    };
    OIFArrayF64 *y0 = oif_init_array_f64_from_data(1, dims, problem->y0);
    OIFArrayF64 *y = oif_create_array_f64(1, dims);
    ImplHandle implh = oif_init_impl("ivp", impl, 1, 0);

    int status;
    status = oif_ivp_set_initial_value(implh, y0, t0);
    EXPECT_EQ(status, 0);
    status = oif_ivp_set_user_data(implh, &problem);
    EXPECT_EQ(status, 0);
    status = oif_ivp_set_rhs_fn(implh, ODEProblem::rhs_wrapper);
    EXPECT_EQ(status, 0);

    auto t_span = {0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0};
    for (auto t : t_span) {
        status = oif_ivp_integrate(implh, t, y);
        EXPECT_EQ(status, 0);
        problem->verify(t, y);
    }

    oif_free_array_f64(y0);
    oif_free_array_f64(y);
}

INSTANTIATE_TEST_SUITE_P(IvpImplementationsTests, IvpImplementationsFixture,
                         testing::Combine(
                             testing::Values("sundials_cvode"),
                             testing::Values(new ScalarExpDecayProblem(), new LinearOscillatorProblem(), new OrbitEquationsProblem())));
