#ifndef OIF_INTERFACE_H
#define OIF_INTERFACE_H

/* This is the Interface that Language-specific Implementors must fulfill
 *
 */

/// language-specific implementation of \link oif_connector_init
/// language-specific implementation of \link oif_connector_eval_expression
int oif_lang_init(void);
int oif_lang_eval_expression(const char *str);
/// language-specific implementation of \link oif_connector_deinit
/// language-specific implementation of \link oif_connector_solve
int oif_lang_deinit(void);
int oif_lang_solve(int N, double *A, double *b, double *x);

#endif // OIF_INTERFACE_H
