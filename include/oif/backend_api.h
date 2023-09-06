#pragma once
/**
 * Interface that backends should implement.
 */
#include <oif/api.h>


BackendHandle load_backend(
        const char *operation,
        size_t version_major,
        size_t version_minor
);


int
run_interface_method(
    const char *method, OIFArgs *in_args, OIFArgs *out_args
);

