extern "C" {
#include <oif_connector/oif_connector.h>
}

#include "oif_constants.h"
#include <array>
#include <iostream>
#include <string>

template <std::size_t N> void print_vector(std::array<double, N> &x);

int main(int argc, char *argv[]) {
  using namespace std;
  string lang{"julia"};
  string expr{"print(42);"};
  if (argc > 2) {
    lang = string{argv[1]};
    expr = string{argv[2]};
  }

  oif_connector_init(lang.c_str());

  constexpr int N{2};
  array<double, N * N> A{2, 0, 0, 1};
  array<double, N> b{1, 1};
  array<double, N> x;

  if (oif_connector_solve(N, A.data(), b.data(), x.data()) != OIF_OK) {
    print_vector(x);
    return -1;
  }
  print_vector(x);

  oif_connector_eval_expression(expr.c_str());
  oif_connector_deinit();
}

template <std::size_t N> void print_vector(std::array<double, N> &x) {
  std::cout << "Solution [";
  for (auto &&xi : x) {
    std::cout << xi << ",";
  }
  std::cout << "]" << std::endl;
}
