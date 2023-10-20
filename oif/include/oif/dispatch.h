#pragma once
#ifndef OIF_DISPATCH_H
#define OIF_DISPATCH_H
#include <stdint.h>
#include <stdlib.h>

#include <oif/api.h>


/**
 * Load implementation by its name and version information.
 */
ImplHandle load_backend_by_name(
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
