#include <oif_config.h>
#include <oif_connector.h>

#include <stdlib.h>

int main(OIF_UNUSED int argc, OIF_UNUSED char *argv[]) {
  if (oif_connector_init("python") != EXIT_SUCCESS)
    return EXIT_FAILURE;
  oif_connector_eval_expression("print(6*7)");
  oif_connector_deinit();
  return EXIT_SUCCESS;
}
