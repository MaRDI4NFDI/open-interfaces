#ifndef JULIA_EMBED_OIF_CONNECTOR_H
#define JULIA_EMBED_OIF_CONNECTOR_H

#include <oif_config.h>

#ifdef OIF_USE_R
#include <Rinternals.h>
#define oif_connector_init(X)                                                  \
  _Generic((X), \
                         SEXP: oif_connector_init_r,     \
                         const char*: oif_connector_init_c, \
                         default: oif_connector_init_c      \
                         )(X)

int oif_connector_init_r(SEXP lang);
#else
#define oif_connector_init(X) oif_connector_init_c((X))
#endif

int oif_connector_init_c(const char *lang);

int oif_connector_eval_expression(const char *str);
void oif_connector_deinit();

#endif // JULIA_EMBED_OIF_CONNECTOR_H
