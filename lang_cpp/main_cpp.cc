extern "C" {
#include <oif_connector/oif_connector.h>
}

#include <string>

int main(int argc, char *argv[]) {
  std::string lang{"julia"};
  std::string expr{"print(42);"};
  if (argc > 2) {
    lang = std::string{argv[1]};
    expr = std::string{argv[2]};
  }

  oif_connector_init(lang.c_str());
  oif_connector_eval_expression(expr.c_str());

  oif_connector_deinit();
}
