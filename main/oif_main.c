#include <oif_config.h>

#include <oif_connector.h>


int main(int argc, char *argv[]) {
  oif_init_lang();
  oif_eval_expression("print(6*7)");
  oif_deinit_lang();
  return 0;
}
