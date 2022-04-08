#ifndef JULIA_EMBED_OIF_CONNECTOR_H
#define JULIA_EMBED_OIF_CONNECTOR_H

/** Special functions that act on R data structs
 *
 */
typedef struct SEXPREC *SEXP;
int oif_connector_init_r(SEXP lang);
int oif_connector_eval_expression_r(SEXP str);
void oif_connector_deinit_r();

// Plain C functions for other languages
int oif_connector_init(const char *lang);
int oif_connector_eval_expression(const char *str);
void oif_connector_deinit();

#endif // JULIA_EMBED_OIF_CONNECTOR_H
