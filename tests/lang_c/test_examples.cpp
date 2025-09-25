#include <cstdlib>
#include <string>
#ifdef __unix__
#include <sys/wait.h>
#endif

#include <gtest/gtest.h>

// ----------------------------------------------------------------------------
// BEGIN Tests for call_qeq_from_c
class CallQEQFromCParameterizedTestFixture : public ::testing::TestWithParam<std::string> {
   protected:
    std::string impl;
};

INSTANTIATE_TEST_SUITE_P(CallQEQFromCTests, CallQEQFromCParameterizedTestFixture,
                         ::testing::Values("c_qeq_solver", "jl_qeq_solver", "py_qeq_solver"));

TEST_P(CallQEQFromCParameterizedTestFixture, RunsSuccessfully)
{
    // We add these tests to `ctest` with WORKING DIRECTORY set to the
    // root of the build directory ($CMAKE_BINARY_DIR).
    const std::string program = "examples/call_qeq_from_c";
    const std::string arg = GetParam();
    const std::string command = program + " " + arg;
    int status = std::system(command.c_str());
#ifdef __unix__
    status = WEXITSTATUS(status);
#endif
    ASSERT_EQ(status, 0);
}

// ----------------------------------------------------------------------------
// BEGIN Tests for call_qeq_from_c
class CallLinsolveFromCParameterizedTestFixture
    : public ::testing::TestWithParam<std::string> {
   protected:
    std::string impl;
};

INSTANTIATE_TEST_SUITE_P(CallLinsolveFromCTests, CallLinsolveFromCParameterizedTestFixture,
                         ::testing::Values("c_lapack", "jl_backslash", "numpy"));

TEST_P(CallLinsolveFromCParameterizedTestFixture, RunsSuccessfully)
{
    // We add these tests to `ctest` with WORKING DIRECTORY set to the
    // root of the build directory ($CMAKE_BINARY_DIR).
    const std::string program = "examples/call_linsolve_from_c";
    const std::string arg = GetParam();
    const std::string command = program + " " + arg;
    int status = std::system(command.c_str());
#ifdef __unix__
    status = WEXITSTATUS(status);
#endif
    ASSERT_EQ(status, 0);
}

// ----------------------------------------------------------------------------
// BEGIN Tests for call_ivp_from_c
class CallIVPFromCParameterizedTestFixture : public ::testing::TestWithParam<std::string> {
   protected:
    std::string impl;
};

INSTANTIATE_TEST_SUITE_P(CallIVPFromCTests, CallIVPFromCParameterizedTestFixture,
                         ::testing::Values("scipy_ode", "sundials_cvode", "jl_diffeq"));

TEST_P(CallIVPFromCParameterizedTestFixture, RunsSuccessfully)
{
    // We add these tests to `ctest` with WORKING DIRECTORY set to the
    // root of the build directory ($CMAKE_BINARY_DIR).
    const std::string program = "examples/call_ivp_from_c";
    const std::string arg = GetParam();
    const std::string command = program + " " + arg;
    int status = std::system(command.c_str());
#ifdef __unix__
    status = WEXITSTATUS(status);
#endif
    ASSERT_EQ(status, 0);
}

// ----------------------------------------------------------------------------
// BEGIN Tests for call_ivp_from_c_burgers_eq
class CallIVPFromCBurgersEqParameterizedTestFixture
    : public ::testing::TestWithParam<std::string> {
   protected:
    std::string impl;
};

INSTANTIATE_TEST_SUITE_P(CallIVPFromCBurgersEqTests,
                         CallIVPFromCBurgersEqParameterizedTestFixture,
                         ::testing::Values("sundials_cvode", "scipy_ode", "jl_diffeq"));

TEST_P(CallIVPFromCBurgersEqParameterizedTestFixture, RunsSuccessfully)
{
    // We add these tests to `ctest` with WORKING DIRECTORY set to the
    // root of the build directory ($CMAKE_BINARY_DIR).
    const std::string program = "examples/call_ivp_from_c_burgers_eq";
    const std::string arg = GetParam();
    const std::string command = program + " " + arg;
    int status = std::system(command.c_str());
#ifdef __unix__
    status = WEXITSTATUS(status);
#endif
    ASSERT_EQ(status, 0);
}

// ----------------------------------------------------------------------------
// BEGIN Tests for call_ivp_from_c_vdp_eq
class CallIVPFromCVdPEqParameterizedTestFixture
    : public ::testing::TestWithParam<std::tuple<std::string, std::string, bool>> {
   protected:
    std::string impl;
    std::string integrator;
    bool should_succeed;
};

INSTANTIATE_TEST_SUITE_P(
    CallIVPFromCVdPEqTests, CallIVPFromCVdPEqParameterizedTestFixture,
    ::testing::Values(std::make_tuple("scipy_ode", "dopri5", false),
                      std::make_tuple("scipy_ode", "dopri5-100k", false),
                      std::make_tuple("scipy_ode", "vode-40k", true)
                      // std::make_tuple("jl_diffeq", "Rosenbrock23", true)  // FIXME
                      ));

TEST_P(CallIVPFromCVdPEqParameterizedTestFixture, RunsAsExpected)
{
    // We add these tests to `ctest` with WORKING DIRECTORY set to the
    // root of the build directory ($CMAKE_BINARY_DIR).
    const std::string program = "examples/call_ivp_from_c_vdp_eq";
    const std::string arg1 = std::get<0>(GetParam());
    const std::string arg2 = std::get<1>(GetParam());
    const bool should_succeed = std::get<2>(GetParam());
    const std::string command = program + " " + arg1 + " " + arg2;
    int status = std::system(command.c_str());
#ifdef __unix__
    status = WEXITSTATUS(status);
#endif
    std::cout << "Status is: " << status << std::endl;
    if (should_succeed)
        ASSERT_EQ(status, 0);
    else {
        ASSERT_EQ(status,
                  42);  // The program is programmed to return 42 on (expected) failure.
    }
}
