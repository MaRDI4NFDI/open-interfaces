#include <gtest/gtest.h>

extern "C" {
#include "oif/backend_c/qeq.h"
}


TEST(QeqTestSuite, LinearCase) {
    double roots[2];
    solve_qeq_v1(0.0, 2.0, -1.0, roots);
    EXPECT_EQ(roots[0], 0.5);
    EXPECT_EQ(roots[1], 0.5);
}


TEST(QeqTestSuite, TwoRoots) {
    double roots[2];
    solve_qeq_v1(1.0, 2.0, 1.0, roots);
    EXPECT_EQ(roots[0], -1.0);
    EXPECT_EQ(roots[1], -1.0);
}
