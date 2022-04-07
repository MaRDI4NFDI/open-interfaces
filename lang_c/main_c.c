#include <oif_config.h>

#include <stdlib.h>

int main(OIF_UNUSED int argc, OIF_UNUSED char *argv[]) {
  char *libname;
  if (0 > asprintf(&libname, "liboif_%s.so", __oif_current_lang))
    return OIF_LOAD_ERROR;
  __oif_lib_handle = dlopen("liboif_r.so", RTLD_LAZY);
  free(libname);

  if (oif_connector_init("python") != EXIT_SUCCESS)
    return EXIT_FAILURE;
  oif_connector_eval_expression("print(6*7)");
  oif_connector_deinit();
  return EXIT_SUCCESS;
}
