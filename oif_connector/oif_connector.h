#ifndef OIF_CONNECTOR_H
#define OIF_CONNECTOR_H

/* This is the Interface which language-specific Drivers can call into
 *
 * \warning Currently only one Implementor can be loaded at once.
 */

/* Error codes */
#define OIF_OK 0
#define OIF_LOAD_ERROR 1
#define OIF_SYMBOL_ERROR 2
#define OIF_TYPE_ERROR 3
#define OIF_NOT_IMPLEMENTED 4
#define OIF_RUNTIME_ERROR 5
/* *********** */

#define OIF_UNUSED __attribute__((unused))

/** Try to load the Implementor DSO
 *
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

/** Unload Implementor, free acquired resources
 *
 */
void oif_connector_deinit();

/** Special functions that act on R data structs
 * \todo move
 * \addtogroup r_module
 *  @{
 */
typedef struct SEXPREC *SEXP;
int oif_connector_init_r(SEXP lang);
int oif_connector_eval_expression_r(SEXP str);
void oif_connector_deinit_r();
/** @}*/

#endif // OIF_CONNECTOR_H
