#pragma once
/**
 * Interface that language-specific dispatches must implement.
 */
#include <oif/api.h>

enum
{
    OIF_LANG_C = 1,
    OIF_LANG_CXX = 2,
    OIF_LANG_PYTHON = 3,
    OIF_LANG_JULIA = 4,
    OIF_LANG_R = 5,
    OIF_LANG_COUNT = 6,
};


ImplHandle load_backend(
        const char *impl_details,
        size_t version_major,
        size_t version_minor
);


int
run_interface_method(
    const char *method, OIFArgs *in_args, OIFArgs *out_args
);

