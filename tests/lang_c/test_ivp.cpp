#include <cmath>
#include <cstring>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "testutils.hpp"
#include <gtest/gtest.h>

#include <oif/api.h>
#include <oif/config_dict.h>
#include "oif/c_bindings.h"
#include "oif/interfaces/ivp.h"

#include "oif/_platform.h"  // IWYU pragma: keep

using namespace std;

constexpr auto PI = 3.14159265358979323846;

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
        delete[] y0;  // Free old memory.
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

        if (this->y0) {
            delete[] this->y0;  // Free old memory.
        }
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
        const int size = y->dimensions[0];
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
        const double y_exact_0 = y0[0] * cos(omega * t) + y0[1] * sin(omega * t) / omega;
        EXPECT_NEAR(y->data[0], y_exact_0, 1e-4);
        const double y_exact_1 = -y0[0] * omega * sin(omega * t) + y0[1] * cos(omega * t);
        EXPECT_NEAR(y->data[1], y_exact_1, 1e-4);
    }

   private:
    const double omega = PI;
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
        const double r = sqrt(y->data[0] * y->data[0] + y->data[1] * y->data[1]);
        rhs_out->data[0] = y->data[2];
        rhs_out->data[1] = y->data[3];
        rhs_out->data[2] = -y->data[0] / (r * r * r);
        rhs_out->data[3] = -y->data[1] / (r * r * r);

        return 0;
    }

    void
    verify(double t, OIFArrayF64 *y) override
    {
        const double u = fsolve([&, t](double u) { return u - eps * sin(u) - t; },
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
    testing::Combine(
        testing::Values(
            // "sundials_cvode",
            "dopri5c"
#if !defined(OIF_SANITIZE_ADDRESS_ENABLED)
            ,
            "scipy_ode", "jl_diffeq"
#endif
            ),
        testing::Values(std::make_shared<ScalarExpDecayProblem>()
                        // ,
                        //             std::make_shared<LinearOscillatorProblem>(),
                        //             std::make_shared<OrbitEquationsProblem>()
                        )));

TEST_P(IvpImplementationsTimesODEProblemsFixture, BasicTestCase)
{
    const char *impl = std::get<0>(GetParam());
    ODEProblem *problem = std::get<1>(GetParam()).get();
    const double t0 = 0.0;
    intptr_t dims[] = {
        problem->N,
    };
    OIFArrayF64 *y0 = oif_init_array_f64_from_data(1, dims, problem->y0);
    OIFArrayF64 *y = oif_create_array_f64(1, dims);
    const ImplHandle implh = oif_load_impl("ivp", impl, 1, 0);
    if (implh == OIF_BRIDGE_NOT_AVAILABLE_ERROR) {
        GTEST_SKIP()
            << "[TEST] Bridge component for the implementation '"
            << impl << "' is not available. Skipping the test.";
    }
    if (implh == OIF_IMPL_NOT_AVAILABLE_ERROR) {
        GTEST_SKIP()
            << "[TEST] Implementation '"
            << impl << "' is not available. Skipping the test.";
    }

    int status;
    status = oif_ivp_set_initial_value(implh, y0, t0);
    ASSERT_EQ(status, 0);
    status = oif_ivp_set_user_data(implh, problem);
    ASSERT_EQ(status, 0);
    status = oif_ivp_set_rhs_fn(implh, ODEProblem::rhs_wrapper);
    ASSERT_EQ(status, 0);
    status = oif_ivp_set_tolerances(implh, 1e-6, 1e-12);
    ASSERT_EQ(status, 0);

    auto t_span = {0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0};
    for (auto t : t_span) {
        status = oif_ivp_integrate(implh, t, y);
        ASSERT_EQ(status, 0);
        problem->verify(t, y);
    }

    oif_free_array_f64(y0);
    oif_free_array_f64(y);

    if (implh != OIF_BRIDGE_NOT_AVAILABLE_ERROR && implh != OIF_IMPL_NOT_AVAILABLE_ERROR) {
        oif_unload_impl(implh);
    }
}
// END Tests that use combinations of implementations and ODE problems.

// ---------------------------------------------------------------------------
// BEGIN Tests that do not depend on ODE problems.
struct IvpImplementationsFixture : public testing::TestWithParam<const char *> {};

INSTANTIATE_TEST_SUITE_P(IvpImplementationsTests, IvpImplementationsFixture,
                         testing::Values("sundials_cvode"
#if !defined(OIF_SANITIZE_ADDRESS_ENABLED)
                                         ,
                                         "scipy_ode", "jl_diffeq"
#endif
                                         ));

TEST_P(IvpImplementationsFixture, DoesNotAcceptUnknownIntegratorName)
{
    const char *impl = GetParam();
    const ImplHandle implh = oif_load_impl("ivp", impl, 1, 0);
    if (implh == OIF_BRIDGE_NOT_AVAILABLE_ERROR) {
        GTEST_SKIP()
            << "[TEST] Bridge component for the implementation '"
            << impl << "' is not available. Skipping the test.";
    }
    if (implh == OIF_IMPL_NOT_AVAILABLE_ERROR) {
        GTEST_SKIP()
            << "[TEST] Implementation '"
            << impl << "' is not available. Skipping the test.";
    }

    const int status = oif_ivp_set_integrator(implh, (char *)"unknown_integrator_name", NULL);
    EXPECT_NE(status, 0);

    if (implh != OIF_BRIDGE_NOT_AVAILABLE_ERROR && implh != OIF_IMPL_NOT_AVAILABLE_ERROR) {
        oif_unload_impl(implh);
    }
}

TEST_P(IvpImplementationsFixture, DoesNotAcceptUnknownIntegratorNameAndIgnoresConfig)
{
    const char *impl = GetParam();
    const ImplHandle implh = oif_load_impl("ivp", impl, 1, 0);
    if (implh == OIF_BRIDGE_NOT_AVAILABLE_ERROR) {
        GTEST_SKIP()
            << "[TEST] Bridge component for the implementation '"
            << impl << "' is not available. Skipping the test.";
    }
    if (implh == OIF_IMPL_NOT_AVAILABLE_ERROR) {
        GTEST_SKIP()
            << "[TEST] Implementation '"
            << impl << "' is not available. Skipping the test.";
    }

    OIFConfigDict *config = oif_config_dict_init();
    oif_config_dict_add_int(config, "max_num_steps", 1000);
    const int status =
        oif_ivp_set_integrator(implh, (char *)"unknown_integrator_name", config);
    EXPECT_NE(status, 0);

    oif_config_dict_free(config);

    if (implh != OIF_BRIDGE_NOT_AVAILABLE_ERROR && implh != OIF_IMPL_NOT_AVAILABLE_ERROR) {
        oif_unload_impl(implh);
    }
}
// END Tests that do not depend on ODE problems.

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
    const ODEProblem *problem = new ScalarExpDecayProblem();
    const double t0 = 0.0;
    const double t1 = 0.1;
    intptr_t dims[] = {
        problem->N,
    };
    OIFArrayF64 *y0 = oif_init_array_f64_from_data(1, dims, problem->y0);
    OIFArrayF64 *y = oif_create_array_f64(1, dims);
    const ImplHandle implh = oif_load_impl("ivp", impl, 1, 0);
    if (implh == OIF_BRIDGE_NOT_AVAILABLE_ERROR) {
        GTEST_SKIP()
            << "[TEST] Bridge component for the implementation '"
            << impl << "' is not available. Skipping the test.";
    }
    if (implh == OIF_IMPL_NOT_AVAILABLE_ERROR) {
        GTEST_SKIP()
            << "[TEST] Implementation '"
            << impl << "' is not available. Skipping the test.";
    }

    for (auto integrator_name : param.integrators) {
        int status;
        status = oif_ivp_set_initial_value(implh, y0, t0);
        ASSERT_EQ(status, 0);

        status = oif_ivp_set_user_data(implh, (void *)problem);
        ASSERT_EQ(status, 0);
        status = oif_ivp_set_rhs_fn(implh, ODEProblem::rhs_wrapper);
        ASSERT_EQ(status, 0);

        status = oif_ivp_set_integrator(implh, (char *)integrator_name, NULL);
        ASSERT_EQ(status, 0);

        oif_ivp_integrate(implh, t1, y);
    }

    oif_free_array_f64(y);
    oif_free_array_f64(y0);
    delete problem;

    if (implh != OIF_BRIDGE_NOT_AVAILABLE_ERROR && implh != OIF_IMPL_NOT_AVAILABLE_ERROR) {
        oif_unload_impl(implh);
    }
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
        const double t0 = 0.0;
        intptr_t dims[] = {
            problem->N,
        };
        y0 = oif_init_array_f64_from_data(1, dims, problem->y0);
        y = oif_create_array_f64(1, dims);
        implh = oif_load_impl("ivp", impl, 1, 0);
        if (implh == OIF_BRIDGE_NOT_AVAILABLE_ERROR) {
            GTEST_SKIP()
                << "[TEST] Bridge component for the implementation '"
                << impl << "' is not available. Skipping the test.";
        }
        if (implh == OIF_IMPL_NOT_AVAILABLE_ERROR) {
            GTEST_SKIP()
                << "[TEST] Implementation '"
                << impl << "' is not available. Skipping the test.";
        }

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
        delete problem;

        if (implh != OIF_BRIDGE_NOT_AVAILABLE_ERROR && implh != OIF_IMPL_NOT_AVAILABLE_ERROR) {
            oif_unload_impl(implh);
        }
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

    status = oif_ivp_integrate(implh, t1, y);
    ASSERT_EQ(status, 0);

    oif_config_dict_free(dict);
}

TEST_F(SundialsCVODEConfigDictTest, Test02)
{
    OIFConfigDict *dict = oif_config_dict_init();
    oif_config_dict_add_int(dict, "max_num_steps", 1000);
    int status = oif_ivp_set_integrator(implh, (char *)"adams", dict);
    ASSERT_EQ(status, 0);

    status = oif_ivp_integrate(implh, t1, y);
    ASSERT_EQ(status, 0);

    oif_config_dict_free(dict);
}

TEST_F(SundialsCVODEConfigDictTest, Test03)
{
    OIFConfigDict *dict = oif_config_dict_init();
    oif_config_dict_add_int(dict, "max_num_steps", 1000);
    oif_config_dict_add_str(dict, "method", "wrong_key");
    const int status = oif_ivp_set_integrator(implh, (char *)"adams", dict);
    ASSERT_NE(status, 0);

    oif_config_dict_free(dict);
}
// END Tests for integrator parameters via config dicts for `sundials_cvode`.

// ----------------------------------------------------------------------------
// BEGIN Tests for integrator parameters via config dicts for `scipy_ode`.

#if !defined(OIF_SANITIZE_ADDRESS_ENABLED)
class ScipyODEConfigDictTest : public testing::Test {
   protected:
       void
    SetUp() override
    {
        const char *impl = "scipy_ode";
        problem = new ScalarExpDecayProblem();
        const double t0 = 0.0;
        const intptr_t dims[] = {
            problem->N,
        };
        y0 = oif_init_array_f64_from_data(1, dims, problem->y0);
        y = oif_create_array_f64(1, dims);
        implh = oif_load_impl("ivp", impl, 1, 0);
        if (implh == OIF_BRIDGE_NOT_AVAILABLE_ERROR) {
            GTEST_SKIP()
                << "[TEST] Bridge component for the implementation '"
                << impl << "' is not available. Skipping the test.";
        }
        if (implh == OIF_IMPL_NOT_AVAILABLE_ERROR) {
            GTEST_SKIP()
                << "[TEST] Implementation '"
                << impl << "' is not available. Skipping the test.";
        }

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
        delete problem;

        if (implh != OIF_BRIDGE_NOT_AVAILABLE_ERROR && implh != OIF_IMPL_NOT_AVAILABLE_ERROR) {
            oif_unload_impl(implh);
        }
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
    int status;
    OIFConfigDict *dict = oif_config_dict_init();
    oif_config_dict_add_int(dict, "nsteps", 1000);
    status = oif_ivp_set_integrator(implh, (char *)"dopri5", dict);
    ASSERT_EQ(status, 0);

    status = oif_ivp_integrate(implh, t1, y);
    ASSERT_EQ(status, 0);
    oif_config_dict_free(dict);
}

TEST_F(ScipyODEConfigDictTest, ShouldAcceptIntegratorParamsForVODE)
{
    int status;
    OIFConfigDict *dict = oif_config_dict_init();
    oif_config_dict_add_int(dict, "nsteps", 1000);
    oif_config_dict_add_double(dict, "rtol", 1e-12);
    oif_config_dict_add_double(dict, "atol", 1e-12);
    oif_config_dict_add_str(dict, "method", "bdf");
    status = oif_ivp_set_integrator(implh, (char *)"vode", dict);
    ASSERT_EQ(status, 0);

    status = oif_ivp_integrate(implh, t1, y);
    ASSERT_EQ(status, 0);
    oif_config_dict_free(dict);
}

TEST_F(ScipyODEConfigDictTest, ShouldFailForWrongIntegratorParams)
{
    OIFConfigDict *dict = oif_config_dict_init();
    oif_config_dict_add_int(dict, "max_num_steps_wrong_key", 1000);
    const int status = oif_ivp_set_integrator(implh, (char *)"vode", dict);
    ASSERT_NE(status, 0);

    oif_config_dict_free(dict);
}
#endif
// END Tests for integrator parameters via config dicts for `scipy_ode`.

// ----------------------------------------------------------------------------
// BEGIN Tests for `dopri5c` implementation.

class Dopri5OOPFixture : public ::testing::Test {
   protected:
    void
    SetUp() override
    {
        implh1 = oif_load_impl("ivp", "dopri5c", 1, 0);
        if (implh1 == OIF_BRIDGE_NOT_AVAILABLE_ERROR) {
            GTEST_SKIP()
                << "[TEST] Bridge component for the implementation '"
                << "dopri5c" << "' is not available. Skipping the test.";
        }
        if (implh1 == OIF_IMPL_NOT_AVAILABLE_ERROR) {
            GTEST_SKIP()
                << "[TEST] Implementation '"
                << "dopri5c" << "' is not available. Skipping the test.";
        }
        implh2 = oif_load_impl("ivp", "dopri5c", 1, 0);

        problem_exp_decay = new ScalarExpDecayProblem();
        problem_oscillator = new LinearOscillatorProblem();

        dims_exp_decay[0] = problem_exp_decay->N;
        dims_oscillator[0] = problem_oscillator->N;

        y0_exp_decay_1 =
            oif_init_array_f64_from_data(1, dims_exp_decay, problem_exp_decay->y0);
        y0_exp_decay_2 =
            oif_init_array_f64_from_data(1, dims_exp_decay, problem_exp_decay->y0);
        y0_oscillator_1 =
            oif_init_array_f64_from_data(1, dims_oscillator, problem_oscillator->y0);
        y0_oscillator_2 =
            oif_init_array_f64_from_data(1, dims_oscillator, problem_oscillator->y0);

        y_exp_decay_1 = oif_create_array_f64(1, dims_exp_decay);
        y_exp_decay_2 = oif_create_array_f64(1, dims_exp_decay);
        y_oscillator_1 = oif_create_array_f64(1, dims_oscillator);
        y_oscillator_2 = oif_create_array_f64(1, dims_oscillator);
    }

    void
    TearDown() override
    {
        oif_free_array_f64(y0_exp_decay_1);
        oif_free_array_f64(y0_exp_decay_2);
        oif_free_array_f64(y0_oscillator_1);
        oif_free_array_f64(y0_oscillator_2);

        oif_free_array_f64(y_exp_decay_1);
        oif_free_array_f64(y_exp_decay_2);
        oif_free_array_f64(y_oscillator_1);
        oif_free_array_f64(y_oscillator_2);

        oif_unload_impl(implh1);
        oif_unload_impl(implh2);
        delete problem_exp_decay;
        delete problem_oscillator;
    }

    // NOLINTBEGIN
    ImplHandle implh1;
    ImplHandle implh2;
    ODEProblem *problem_exp_decay = nullptr;
    ODEProblem *problem_oscillator = nullptr;
    intptr_t dims_exp_decay[1];
    intptr_t dims_oscillator[1];
    const double t0 = 0.0;

    OIFArrayF64 *y0_exp_decay_1;
    OIFArrayF64 *y0_exp_decay_2;
    OIFArrayF64 *y0_oscillator_1;
    OIFArrayF64 *y0_oscillator_2;
    OIFArrayF64 *y_exp_decay_1;
    OIFArrayF64 *y_exp_decay_2;
    OIFArrayF64 *y_oscillator_1;
    OIFArrayF64 *y_oscillator_2;
    // NOLINTEND
};

class MockCallbackProvider {
   public:
    static int
    rhs_1(double /* t */, OIFArrayF64 * /* y */, OIFArrayF64 * /* ydot */,
          void * /* user_data */)
    {
        count_1++;
        return 0;
    }

    static int
    rhs_2(double /* t */, OIFArrayF64 * /* y */, OIFArrayF64 * /* ydot */,
          void * /* user_data */)
    {
        count_2++;
        return 0;
    }

    static unsigned count_1;
    static unsigned count_2;
};

// This is ugly way in which storage for static data members
// is allocated in C++ before C++17.
unsigned MockCallbackProvider::count_1 = 0;
unsigned MockCallbackProvider::count_2 = 0;

TEST_F(Dopri5OOPFixture, NoSharedDataInImplementation1)
{
    // This test checks that two instances of the same implementation
    // do not share data.
    // Precisely, we load the `dopri5c` implementation twice,
    // and then we integrate the same problem.
    // We check that the results are the same
    // after integration to the same time point
    // by the results are different if one solver is integrated
    // to, e.g. time t3, while the other is integrated
    // only to time t2 < t3.

    oif_ivp_set_initial_value(implh1, y0_exp_decay_1, t0);
    oif_ivp_set_initial_value(implh2, y0_exp_decay_2, t0);

    oif_ivp_set_user_data(implh1, problem_exp_decay);
    oif_ivp_set_user_data(implh2, problem_exp_decay);

    oif_ivp_set_rhs_fn(implh1, ODEProblem::rhs_wrapper);
    oif_ivp_set_rhs_fn(implh2, ODEProblem::rhs_wrapper);

    const double t_span[] = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0};
    oif_ivp_integrate(implh1, t_span[1], y_exp_decay_1);
    oif_ivp_integrate(implh1, t_span[2], y_exp_decay_1);
    oif_ivp_integrate(implh1, t_span[3], y_exp_decay_1);

    oif_ivp_integrate(implh2, t_span[1], y_exp_decay_2);
    ASSERT_NE(y_exp_decay_1->data[0], y_exp_decay_2->data[0]);

    oif_ivp_integrate(implh2, t_span[2], y_exp_decay_2);
    ASSERT_NE(y_exp_decay_1->data[0], y_exp_decay_2->data[0]);

    oif_ivp_integrate(implh2, t_span[3], y_exp_decay_2);
    ASSERT_EQ(y_exp_decay_1->data[0], y_exp_decay_2->data[0]);
}

TEST_F(Dopri5OOPFixture, TwoDifferentProblems__SolutionsMustNeverMatch)
{
    // This test checks that two instances of the same implementation
    // do not share data.
    // Precisely, we load the `dopri5c` implementation twice,
    // and then we integrate two **different** problem.
    // We check after each integration step
    // that the results are always different for the first component
    // of the solution vector (because dimension is different
    // for the two problems).

    oif_ivp_set_initial_value(implh1, y0_exp_decay_1, t0);
    oif_ivp_set_initial_value(implh2, y0_oscillator_2, t0);

    oif_ivp_set_user_data(implh1, problem_exp_decay);
    oif_ivp_set_user_data(implh2, problem_oscillator);

    oif_ivp_set_rhs_fn(implh1, ODEProblem::rhs_wrapper);
    oif_ivp_set_rhs_fn(implh2, ODEProblem::rhs_wrapper);

    const double t_span[] = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0};
    oif_ivp_integrate(implh1, t_span[1], y_exp_decay_1);
    oif_ivp_integrate(implh2, t_span[1], y_oscillator_2);
    ASSERT_NE(y_exp_decay_1->data[0], y_oscillator_2->data[0]);

    oif_ivp_integrate(implh1, t_span[1], y_exp_decay_1);
    oif_ivp_integrate(implh2, t_span[1], y_oscillator_2);
    ASSERT_NE(y_exp_decay_1->data[0], y_oscillator_2->data[0]);

    oif_ivp_integrate(implh1, t_span[1], y_exp_decay_1);
    oif_ivp_integrate(implh2, t_span[1], y_oscillator_2);
    ASSERT_NE(y_exp_decay_1->data[0], y_oscillator_2->data[0]);
}

TEST_F(Dopri5OOPFixture, TwoDifferentProblems__MustHaveDifferentCallbacks)
{
    MockCallbackProvider::count_1 = 0;
    MockCallbackProvider::count_2 = 0;

    // Right-hand side function is called twice in `set_rh_fn`
    // for `dopri5c` because it computes the initial step size.
    const unsigned ninvokations_per_set = 2;

    oif_ivp_set_initial_value(implh1, y0_exp_decay_1, t0);
    oif_ivp_set_initial_value(implh2, y0_oscillator_2, t0);

    oif_ivp_set_rhs_fn(implh1, MockCallbackProvider::rhs_1);
    ASSERT_EQ(MockCallbackProvider::count_1, 1 * ninvokations_per_set);
    ASSERT_EQ(MockCallbackProvider::count_2, 0);
    oif_ivp_set_rhs_fn(implh1, MockCallbackProvider::rhs_1);
    ASSERT_EQ(MockCallbackProvider::count_1, 2 * ninvokations_per_set);
    ASSERT_EQ(MockCallbackProvider::count_2, 0);
    oif_ivp_set_rhs_fn(implh1, MockCallbackProvider::rhs_1);
    ASSERT_EQ(MockCallbackProvider::count_1, 3 * ninvokations_per_set);
    ASSERT_EQ(MockCallbackProvider::count_2, 0);

    oif_ivp_set_rhs_fn(implh2, MockCallbackProvider::rhs_2);
    ASSERT_EQ(MockCallbackProvider::count_1, 3 * ninvokations_per_set);
    ASSERT_EQ(MockCallbackProvider::count_2, 1 * ninvokations_per_set);
}

TEST_F(Dopri5OOPFixture, DoesNotAllowSetIntegratorMethod)
{
    ASSERT_GT(implh1, 0);
    int const status = oif_ivp_set_integrator(implh1, (char *)"does not matter", NULL);
    ASSERT_NE(status, 0);
}

// END Tests for `dopri5c` implementation.
