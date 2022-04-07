#ifndef JULIA_EMBED_OIF_CONNECTOR_H
#define JULIA_EMBED_OIF_CONNECTOR_H

int oif_connector_init(const char *lang);

int oif_connector_eval_expression(const char *str);
void oif_connector_deinit();

#endif // JULIA_EMBED_OIF_CONNECTOR_H
