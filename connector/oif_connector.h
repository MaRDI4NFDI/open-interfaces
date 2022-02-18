#ifndef JULIA_EMBED_OIF_CONNECTOR_H
#define JULIA_EMBED_OIF_CONNECTOR_H

int oif_init_lang();
void oif_eval_expression(const char *str);
void oif_deinit_lang();

#endif // JULIA_EMBED_OIF_CONNECTOR_H
