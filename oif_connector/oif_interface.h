#ifndef OIF_INTERFACE_H
#define OIF_INTERFACE_H

/* This is the Interface that Language-specific Implementors must fullfil
 *
 *
 */

int oif_lang_init();
int oif_lang_eval_expression(const char *str);
void oif_lang_deinit();

#endif // OIF_INTERFACE_H
