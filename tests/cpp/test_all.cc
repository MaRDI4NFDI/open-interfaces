extern "C" {
#include <oif_connector/oif_connector.h>
}

#include "oif_constants.h"
#include <array>
#include <iostream>
#include <string>

#include "catch2/catch_amalgamated.hpp"

template <std::size_t N> void print_vector(std::array<double, N> &x);

TEST_CASE("Convert cpp main", "[convert]") {
  using namespace std;
  string lang{"julia"};
  string expr{"print(42);"};

  oif_connector_init(lang.c_str());

  constexpr int N{2};
  array<double, N * N> A{2, 0, 0, 1};
  array<double, N> b{1, 1};
  array<double, N> x;

  REQUIRE(oif_connector_solve(N, A.data(), b.data(), x.data()) == OIF_OK);
  REQUIRE(oif_connector_eval_expression(expr.c_str()) == OIF_OK);

  oif_connector_deinit();
}

template <std::size_t N> void print_vector(std::array<double, N> &x) {
  std::cout << "Solution [";
  for (auto &&xi : x) {
    std::cout << xi << ",";
  }
  std::cout << "]" << std::endl;
}
