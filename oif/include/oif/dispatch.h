#pragma once
#ifndef OIF_DISPATCH_H
#define OIF_DISPATCH_H
#include <stdint.h>
#include <stdlib.h>

#include <oif/api.h>


/**
 * Load implementation of the interface.
 *
 * @param interface     Name of the interface
 * @param impl          Name of the implementation for the interface
 * @param version_major Major version number of the implementation
 * @param version_minor Minor version number of the implementation
 * @return positive number that identifies the requested implementation
 *         or OIF_IMPL_INIT_ERROR in case of the error
 */
ImplHandle load_interface_impl(
        const char *interface,
        const char *impl,
        size_t version_major,
        size_t version_minor
);


int call_interface_method(
    ImplHandle implh,
    const char *method,
    OIFArgs *args,
    OIFArgs *retvals
);
#endif
