#include <gtest/gtest.h>

extern "C" {
#include "oif/c_bindings.h"
#include "oif/interfaces/linsolve.h"
}

class LinearSolverFixture : public ::testing::TestWithParam<const char *> {};

TEST_P(LinearSolverFixture, TestCase1)
{
    intptr_t A_dims[] = {2, 2};
    double A_data[] = {1.0, 1.0, -3.0, 1.0};
    OIFArrayF64 *A = oif_init_array_f64_from_data(2, A_dims, A_data);
    intptr_t b_dims[] = {2};
    double b_data[] = {6.0, 2.0};
    OIFArrayF64 *b = oif_init_array_f64_from_data(1, b_dims, b_data);
    intptr_t roots_dims[] = {2};
    OIFArrayF64 *roots = oif_create_array_f64(1, roots_dims);
    ImplHandle implh = oif_init_impl("linsolve", GetParam(), 1, 0);

    int status = oif_solve_linear_system(implh, A, b, roots);
    EXPECT_EQ(status, 0);

    int M = A->dimensions[0];
    int N = A->dimensions[1];
    for (size_t i = 0; i < M; ++i) {
        float scalar_product = 0.0;
        for (size_t j = 0; j < N; ++j) {
            scalar_product += A->data[i * N + j] * roots->data[j];
        }
        EXPECT_FLOAT_EQ(scalar_product, b->data[i]);
    }
    oif_free_array_f64(A);
    oif_free_array_f64(b);
    oif_free_array_f64(roots);
    oif_unload_impl(implh);
}

INSTANTIATE_TEST_SUITE_P(LinearSolverTestSuite, LinearSolverFixture,
                         ::testing::Values("c_lapack", "numpy"));
