#ifndef JULIA_EMBED_OIF_CONNECTOR_H
#define JULIA_EMBED_OIF_CONNECTOR_H

int oif_init_connector(const char *lang);
int oif_init_lang();
int oif_eval_expression(const char *str);
void oif_deinit_lang();

#endif // JULIA_EMBED_OIF_CONNECTOR_H
