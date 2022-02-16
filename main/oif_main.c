#include <oif_config.h>

#ifdef OIF_USE_JULIA
#include <oif_julia.h>
#endif

int main(int argc, char *argv[]) {
#ifdef OIF_USE_JULIA
  oif_init_lang();
  oif_eval_expression("print(6*7)");
  oif_deinit_lang();
#else
  printf("no language bindings");
#endif
  return 0;
}
