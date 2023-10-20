#pragma once
/**
 * Interface that language-specific dispatches must implement.
 */
#include <oif/api.h>


ImplHandle load_backend(
        const char *impl_details,
        size_t version_major,
        size_t version_minor
);


int
run_interface_method(
    const char *method, OIFArgs *in_args, OIFArgs *out_args
);

