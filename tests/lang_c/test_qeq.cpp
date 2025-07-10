#include <gtest/gtest.h>

#include "oif/c_bindings.h"
#include "oif/interfaces/qeq.h"

#include <oif/_platform.h>


class QeqGatewayFixture : public ::testing::TestWithParam<const char *> {
protected:
    OIFArrayF64 *roots;
    ImplHandle implh;

    void SetUp() override {
        intptr_t dimensions[] = {
            2,
        };
        roots = oif_create_array_f64(1, dimensions);
        implh = oif_load_impl("qeq", GetParam(), 1, 0);

        ASSERT_NE(roots, nullptr);
        ASSERT_GT(implh, 0);
    }

    void TearDown() override {
        oif_free_array_f64(roots);
        oif_unload_impl(implh);
    }
};

INSTANTIATE_TEST_SUITE_P(
    QeqGatewayParameterizedTestSuite,
    QeqGatewayFixture,
    ::testing::Values(
        "c_qeq_solver"
#if !defined(OIF_SANITIZE_ADDRESS_ENABLED)
        ,
        "jl_qeq_solver",
        "py_qeq_solver"
#endif
    )
);

TEST_P(QeqGatewayFixture, LinearCase) {
    int status = oif_solve_qeq(implh, 0.0, 2.0, -1.0, roots);
    ASSERT_EQ(status, 0);

    EXPECT_EQ(roots->data[0], 0.5);
    EXPECT_EQ(roots->data[1], 0.5);
}

TEST_P(QeqGatewayFixture, TwoEqualRoots)
{
    int status = oif_solve_qeq(implh, 1.0, 2.0, 1.0, roots);
    ASSERT_EQ(status, 0);

    EXPECT_EQ(roots->data[0], -1.0);
    EXPECT_EQ(roots->data[1], -1.0);
}

TEST_P(QeqGatewayFixture, TwoDistinctRoots)
{
    int status = oif_solve_qeq(implh, 1, 5, -14, roots);
    ASSERT_EQ(status, 0);

    EXPECT_EQ(roots->data[0], -7);
    EXPECT_EQ(roots->data[1], +2);
}

TEST_P(QeqGatewayFixture, ExtremeRoots)
{
    int status = oif_solve_qeq(implh, 1, -20'000, 1.0, roots);
    ASSERT_EQ(status, 0);

    EXPECT_EQ(roots->data[0], 19999.999949999998);
    EXPECT_EQ(roots->data[1], 5.000000012500001e-05);
}
