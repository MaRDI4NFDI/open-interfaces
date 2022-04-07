#include <oif_connector/oif_connector.h>

#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
  char *lang = "julia";
  char *expr = "print(42);";
  if (argc > 2) {
    lang = strdup(argv[1]);
    expr = strdup(argv[2]);
  }

  oif_connector_init(lang);
  oif_connector_eval_expression(expr);

  oif_connector_deinit();
}
