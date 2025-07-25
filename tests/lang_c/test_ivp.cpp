#include <cmath>
#include <cstring>
#include <iostream>

#include "testutils.h"
#include <gtest/gtest.h>

#include "oif/c_bindings.h"
#include "oif/interfaces/ivp.h"

#include "oif/_platform.h"

using namespace std;

// ----------------------------------------------------------------------------
// BEGIN ODEProblem and its derived classes.

class ODEProblem {
   public:
    // Thank you, clang-tidy, but I can handle public data alright.
    // Supress warning about non-private data members in classes.
    // NOLINTBEGIN
    int N;
    double *y0;
    // NOLINTEND

    ODEProblem(int N) : N(N) { y0 = new double[N]; }

    ODEProblem(const ODEProblem &other)
    {
        N = other.N;
        y0 = new double[N];
        memcpy(y0, other.y0, sizeof(*y0) * N);
    }

    ODEProblem &
    operator=(const ODEProblem &other)
    {
        if (this == &other) {
            return *this;
        }

        this->N = other.N;
        delete[] y0;
        this->y0 = new double[N];
        memcpy(y0, other.y0, sizeof(*y0) * this->N);

        return *this;
    }

    virtual ~ODEProblem() { delete[] y0; }

    static int
    rhs_wrapper(double t, OIFArrayF64 *y, OIFArrayF64 *ydot, void *user_data)
    {
        ODEProblem *problem = reinterpret_cast<ODEProblem *>(user_data);
        problem->rhs(t, y, ydot, NULL);
        return 0;
    }

    virtual int
    rhs(double t, OIFArrayF64 *y, OIFArrayF64 *ydot, void *user_data) = 0;

    virtual void
    verify(double t, OIFArrayF64 *y) = 0;
};

class ScalarExpDecayProblem : public ODEProblem {
   public:
    ScalarExpDecayProblem() : ODEProblem(1) { y0[0] = 1.0; }

    int
    rhs(double /* t */, OIFArrayF64 *y, OIFArrayF64 *rhs_out, void * /* user_data */) override
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
        EXPECT_NEAR(y->data[0], exp(-t), 1e-4);
    }
};

class LinearOscillatorProblem : public ODEProblem {
   public:
    LinearOscillatorProblem() : ODEProblem(2)
    {
        y0[0] = 1.0;
        y0[1] = 0.5;
    }

    int
    rhs(double /* t */, OIFArrayF64 *y, OIFArrayF64 *rhs_out, void * /* user_data */) override
    {
        rhs_out->data[0] = y->data[1];
        rhs_out->data[1] = -omega * omega * y->data[0];
        return 0;
    }
    void
    verify(double t, OIFArrayF64 *y) override
    {
        double y_exact_0 = y0[0] * cos(omega * t) + y0[1] * sin(omega * t) / omega;
        EXPECT_NEAR(y->data[0], y_exact_0, 1e-4);
        double y_exact_1 = -y0[0] * omega * sin(omega * t) + y0[1] * cos(omega * t);
        EXPECT_NEAR(y->data[1], y_exact_1, 1e-4);
    }

   private:
    const double omega = M_PI;
};

class OrbitEquationsProblem : public ODEProblem {
   public:
    OrbitEquationsProblem() : ODEProblem(4)
    {
        y0[0] = 1 - eps;
        y0[1] = 0.0;
        y0[2] = 0.0;
        y0[3] = sqrt((1 + eps) / (1 - eps));
    }

    int
    rhs(double /* t */, OIFArrayF64 *y, OIFArrayF64 *rhs_out, void * /* user_data */) override
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
        EXPECT_NEAR(y->data[0], cos(u) - eps, 2e-4);
        EXPECT_NEAR(y->data[1], sqrt(1 - pow(eps, 2)) * sin(u), 2e-4);
        EXPECT_NEAR(y->data[2], -sin(u) / (1 - eps * cos(u)), 2e-4);
        EXPECT_NEAR(y->data[3], sqrt(1 - pow(eps, 2)) * cos(u) / (1 - eps * cos(u)), 2e-4);
    }

   private:
    double eps = 0.9L;
};

// END ODEProblem and its derived classes.

// ----------------------------------------------------------------------------
// BEGIN Tests that use combinations of implementations and ODE problems.

struct IvpImplementationsTimesODEProblemsFixture
    : public testing::TestWithParam<std::tuple<const char *, std::shared_ptr<ODEProblem>>> {};

INSTANTIATE_TEST_SUITE_P(
    IvpImplementationsTests, IvpImplementationsTimesODEProblemsFixture,
    testing::Combine(testing::Values("sundials_cvode"
#if !defined(OIF_SANITIZE_ADDRESS_ENABLED)
                                     ,
                                     "scipy_ode", "jl_diffeq"
#endif
                                     ),
                     testing::Values(std::make_shared<ScalarExpDecayProblem>(),
                                     std::make_shared<LinearOscillatorProblem>(),
                                     std::make_shared<OrbitEquationsProblem>())));

TEST_P(IvpImplementationsTimesODEProblemsFixture, BasicTestCase)
{
    const char *impl = std::get<0>(GetParam());
    ODEProblem *problem = std::get<1>(GetParam()).get();
    double t0 = 0.0;
    intptr_t dims[] = {
        problem->N,
    };
    OIFArrayF64 *y0 = oif_init_array_f64_from_data(1, dims, problem->y0);
    OIFArrayF64 *y = oif_create_array_f64(1, dims);
    ImplHandle implh = oif_load_impl("ivp", impl, 1, 0);
    ASSERT_GT(implh, 0);

    int status;
    status = oif_ivp_set_initial_value(implh, y0, t0);
    ASSERT_EQ(status, 0);
    status = oif_ivp_set_user_data(implh, problem);
    ASSERT_EQ(status, 0);
    status = oif_ivp_set_rhs_fn(implh, ODEProblem::rhs_wrapper);
    ASSERT_EQ(status, 0);

    auto t_span = {0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0};
    for (auto t : t_span) {
        status = oif_ivp_integrate(implh, t, y);
        ASSERT_EQ(status, 0);
        problem->verify(t, y);
    }

    oif_free_array_f64(y0);
    oif_free_array_f64(y);
    oif_unload_impl(implh);
}
// END Tests that use combinations of implementations and ODE problems.

// ----------------------------------------------------------------------------
// BEGIN Tests that implementations-integrators.
struct SolverIntegratorsCombination {
    const char *impl;
    std::vector<const char *> integrators;
};

struct ImplTimesIntegratorsFixture
    : public testing::TestWithParam<SolverIntegratorsCombination> {};

INSTANTIATE_TEST_SUITE_P(
    IvpChangeIntegratorsTests, ImplTimesIntegratorsFixture,
    testing::Values(SolverIntegratorsCombination{"sundials_cvode", {"bdf", "adams"}}
#if !defined(OIF_SANITIZE_ADDRESS_ENABLED)
                    ,
                    SolverIntegratorsCombination{"scipy_ode",
                                                 {"vode", "lsoda", "dopri5", "dop853"}},
                    SolverIntegratorsCombination{"jl_diffeq", {"Tsit5"}}
#endif
                    ));

TEST_P(ImplTimesIntegratorsFixture, SetIntegratorMethodWorks)
{
    auto param = GetParam();
    const char *impl = param.impl;
    ODEProblem *problem = new ScalarExpDecayProblem();
    double t0 = 0.0;
    double t1 = 0.1;
    intptr_t dims[] = {
        problem->N,
    };
    OIFArrayF64 *y0 = oif_init_array_f64_from_data(1, dims, problem->y0);
    OIFArrayF64 *y = oif_create_array_f64(1, dims);
    ImplHandle implh = oif_load_impl("ivp", impl, 1, 0);
    ASSERT_GT(implh, 0);

    for (auto integrator_name : param.integrators) {
        int status;
        status = oif_ivp_set_initial_value(implh, y0, t0);
        ASSERT_EQ(status, 0);
        status = oif_ivp_set_user_data(implh, problem);
        ASSERT_EQ(status, 0);
        status = oif_ivp_set_rhs_fn(implh, ODEProblem::rhs_wrapper);
        ASSERT_EQ(status, 0);

        status = oif_ivp_set_integrator(implh, (char *)integrator_name, NULL);
        ASSERT_EQ(status, 0);

        oif_ivp_integrate(implh, t1, y);
    }

    oif_unload_impl(implh);
    oif_free_array_f64(y);
    oif_free_array_f64(y0);
    delete problem;
}
// END Tests that use combinations of implementations and ODE problems.

// ----------------------------------------------------------------------------
// BEGIN Tests for integrator parameters via config dicts for `sundials_cvode`.
class SundialsCVODEConfigDictTest : public testing::Test {
   protected:
    void
    SetUp() override
    {
        const char *impl = "sundials_cvode";
        problem = new ScalarExpDecayProblem();
        double t0 = 0.0;
        intptr_t dims[] = {
            problem->N,
        };
        y0 = oif_init_array_f64_from_data(1, dims, problem->y0);
        y = oif_create_array_f64(1, dims);
        implh = oif_load_impl("ivp", impl, 1, 0);
        EXPECT_GT(implh, 0);

        int status;
        status = oif_ivp_set_initial_value(implh, y0, t0);
        EXPECT_EQ(status, 0);
        status = oif_ivp_set_user_data(implh, problem);
        EXPECT_EQ(status, 0);
        status = oif_ivp_set_rhs_fn(implh, ODEProblem::rhs_wrapper);
        EXPECT_EQ(status, 0);
    }

    void
    TearDown() override
    {
        oif_free_array_f64(y0);
        oif_free_array_f64(y);
        oif_unload_impl(implh);
        delete problem;
    }

    // NOLINTBEGIN
    ImplHandle implh;
    ODEProblem *problem;
    double t1 = 0.1;
    OIFArrayF64 *y0;
    OIFArrayF64 *y;
    // NOLINTEND
};

TEST_F(SundialsCVODEConfigDictTest, Test01)
{
    OIFConfigDict *dict = oif_config_dict_init();
    oif_config_dict_add_int(dict, "max_num_steps", 1000);
    int status = oif_ivp_set_integrator(implh, (char *)"bdf", dict);
    ASSERT_EQ(status, 0);

    oif_ivp_integrate(implh, t1, y);
    oif_config_dict_free(dict);
}

TEST_F(SundialsCVODEConfigDictTest, Test02)
{
    OIFConfigDict *dict = oif_config_dict_init();
    oif_config_dict_add_int(dict, "max_num_steps", 1000);
    int status = oif_ivp_set_integrator(implh, (char *)"adams", dict);
    ASSERT_EQ(status, 0);

    oif_ivp_integrate(implh, t1, y);
    oif_config_dict_free(dict);
}

TEST_F(SundialsCVODEConfigDictTest, Test03)
{
    OIFConfigDict *dict = oif_config_dict_init();
    oif_config_dict_add_int(dict, "max_num_steps", 1000);
    oif_config_dict_add_str(dict, "method", "wrong_key");
    int status = oif_ivp_set_integrator(implh, (char *)"adams", dict);
    ASSERT_NE(status, 0);

    oif_ivp_integrate(implh, t1, y);
    oif_config_dict_free(dict);
}
// END Tests for integrator parameters via config dicts for `sundials_cvode`.

// ----------------------------------------------------------------------------
// BEGIN Tests for integrator parameters via config dicts for `scipy_ode`.

#if !defined(OIF_SANITIZE_ADDRESS_ENABLED)
class ScipyODEConfigDictTest : public testing::Test {
   protected:
    ScipyODEConfigDictTest()
    {
        const char *impl = "scipy_ode";
        problem = new ScalarExpDecayProblem();
        double t0 = 0.0;
        intptr_t dims[] = {
            problem->N,
        };
        y0 = oif_init_array_f64_from_data(1, dims, problem->y0);
        y = oif_create_array_f64(1, dims);
        implh = oif_load_impl("ivp", impl, 1, 0);
        EXPECT_GT(implh, 0);

        int status;
        status = oif_ivp_set_initial_value(implh, y0, t0);
        EXPECT_EQ(status, 0);
        status = oif_ivp_set_user_data(implh, problem);
        EXPECT_EQ(status, 0);
        status = oif_ivp_set_rhs_fn(implh, ODEProblem::rhs_wrapper);
        EXPECT_EQ(status, 0);
    }

    ~ScipyODEConfigDictTest()
    {
        oif_free_array_f64(y0);
        oif_free_array_f64(y);
        delete problem;
        oif_unload_impl(implh);
    }

    // NOLINTBEGIN
    ImplHandle implh;
    ODEProblem *problem;
    double t1 = 0.1;
    OIFArrayF64 *y0;
    OIFArrayF64 *y;
    // NOLINTEND
};

TEST_F(ScipyODEConfigDictTest, ShouldAcceptIntegratorParamsForDopri5)
{
    OIFConfigDict *dict = oif_config_dict_init();
    oif_config_dict_add_int(dict, "nsteps", 1000);
    int status = oif_ivp_set_integrator(implh, (char *)"dopri5", dict);
    ASSERT_EQ(status, 0);

    oif_ivp_integrate(implh, t1, y);
    oif_config_dict_free(dict);
}

TEST_F(ScipyODEConfigDictTest, ShouldAcceptIntegratorParamsForVODE)
{
    OIFConfigDict *dict = oif_config_dict_init();
    oif_config_dict_add_int(dict, "nsteps", 1000);
    oif_config_dict_add_double(dict, "rtol", 1e-12);
    oif_config_dict_add_double(dict, "atol", 1e-12);
    oif_config_dict_add_str(dict, "method", "bdf");
    int status = oif_ivp_set_integrator(implh, (char *)"vode", dict);
    ASSERT_EQ(status, 0);

    oif_ivp_integrate(implh, t1, y);
    oif_config_dict_free(dict);
}

TEST_F(ScipyODEConfigDictTest, ShouldFailForWrongIntegratorParams)
{
    OIFConfigDict *dict = oif_config_dict_init();
    oif_config_dict_add_int(dict, "max_num_steps_wrong_key", 1000);
    int status = oif_ivp_set_integrator(implh, (char *)"vode", dict);
    ASSERT_NE(status, 0);

    oif_config_dict_free(dict);
}
#endif
// END Tests for integrator parameters via config dicts for `scipy_ode`.

// ----------------------------------------------------------------------------
// BEGIN Tests for `dopri5c` implementation.
TEST(Dopri5CConfigDictTest, DoesNotAllowSetIntegratorMethod)
{
    ImplHandle implh = oif_load_impl("ivp", "dopri5c", 1, 0);
    ASSERT_GT(implh, 0);

    int status = oif_ivp_set_integrator(implh, (char *)"does not matter", NULL);
    ASSERT_NE(status, 0);

    oif_unload_impl(implh);
}
// END Tests for `dopri5c` implementation.
