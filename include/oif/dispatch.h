#pragma once
#ifndef OIF_DISPATCH_H
#define OIF_DISPATCH_H
#include <stdint.h>
#include <stdlib.h>

#include <oif/api.h>


/**
 * Load backend by its name and version information.
 */
BackendHandle load_backend_by_name(
        const char *backend_name,
        const char *operation,
        size_t version_major,
        size_t version_minor
);


int call_interface_method(
    BackendHandle bh,
    const char *method,
    OIFArgs *args,
    OIFArgs *retvals
);
#endif
