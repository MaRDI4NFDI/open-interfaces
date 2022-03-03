#include <oif_config.h>

#include <oif_connector.h>

#include <stdlib.h>

int main(OIF_UNUSED int argc, OIF_UNUSED char *argv[]) {
  if (oif_init_lang() != EXIT_SUCCESS)
    return EXIT_FAILURE;
  oif_eval_expression("print(6*7)");
  oif_deinit_lang();
  return EXIT_SUCCESS;
}
