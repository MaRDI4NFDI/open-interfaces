// clang-format Language: C
#pragma once

#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>

#include <oif/api.h>

/**
 * Initialize interface implementation.
 * @param interface Name of the interface
 * @param implementation Name of the implementation
 * @param version_major  Major version number of the implementation
 * @param version_minor  Minor version number of the implementation
 * @return positive number that identifies the requested implementation
 *         or OIF_IMPL_INIT_ERROR in case of the error
 */
ImplHandle
oif_load_impl(const char *interface, const char *impl, int version_major, int version_minor);

int
oif_unload_impl(ImplHandle implh);

OIFArrayF64 *
oif_create_array_f64(int nd, const intptr_t *const dimensions);

OIFArrayF64 *
oif_init_array_f64_from_data(int nd, const intptr_t *const dimensions,
                             const double *const data);

void
oif_free_array_f64(OIFArrayF64 *x);

void
oif_print_matrix(const OIFArrayF64 *const mat);

void
oif_print_vector(const OIFArrayF64 *const vec);

#ifdef __cplusplus
}
#endif
