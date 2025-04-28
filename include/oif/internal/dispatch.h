// clang-format Language: C
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>

#include <oif/api.h>

/**
 * Load implementation of an interface.
 *
 * @param interface     Name of the interface
 * @param impl          Name of the implementation for the interface
 * @param version_major Major version number of the implementation
 * @param version_minor Minor version number of the implementation
 * @return positive number that identifies the requested implementation
 *         or OIF_IMPL_INIT_ERROR in case of the error
 */
ImplHandle
load_interface_impl(const char *interface, const char *impl, size_t version_major,
                    size_t version_minor);

/**
 * Unload implementation of an interface.
 *
 * @param implh Implementation handle that identifies the implementation
 * @return nonnegative number that identifies whether the operation
 *         was successful
 */
int
unload_interface_impl(ImplHandle implh);

/**
 * Call implementation of an interface.
 * @param implh Implementat handle that identifies the implementation
 * @param method Name of the method (function) to invoke
 * @param in_args Array of input arguments
 * @param out_args Array of output arguments
 * @return status code that signals about an error if non-zero
 */
int
call_interface_impl(ImplHandle implh, const char *method, OIFArgs *in_args, OIFArgs *out_args);

#ifdef __cplusplus
}
#endif
