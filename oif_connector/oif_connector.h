#ifndef OIF_CONNECTOR_H
#define OIF_CONNECTOR_H

/* This is the Interface which language-specific Drivers can call into
 *
 * \warning Currently only one Implementor can be loaded at once.
 */

/** Try to load the Implementor DSO
 *
 * The connector expects all Implementor DSOs residing in subdirectories:
 *      lang_c/liboif_c.so
 *      lang_LANG/liboif_LANG.so
 * @param lang specifies which implementor to load
 * @return `OIF_OK` or a fitting error code
 */
int oif_connector_init(const char *lang);

/** Evaluate a given string
 *
 * @param str
 * @return
 */
int oif_connector_eval_expression(const char *str);

/** Solve Ax=b
 *
 * @param N system size, integer > 0
 * @param A a square `N`x`N` Matrix
 * @param b righthand-side of `N` values
 * @param x output array
 * @return `OIF_OK` or a fitting error code
 *
 * \note All arrays must be pre-allocated
 */
int oif_connector_solve(int N, double *A, double *b, double *x);

/** Unload Implementor, free acquired resources
 *
 * @return `OIF_OK` or a fitting error code
 */
int oif_connector_deinit(void);

/** Special functions that act on R data structs
 * \todo move
 * \addtogroup r_module
 *  @{
 */
typedef struct SEXPREC *SEXP;
int oif_connector_init_r(SEXP lang);
int oif_connector_eval_expression_r(SEXP str);
void oif_connector_deinit_r(void);
/** @}*/

#endif // OIF_CONNECTOR_H
