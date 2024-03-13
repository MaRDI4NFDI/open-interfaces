#pragma once
/**
 * Interface that language-specific dispatches must implement.
 */
#include <oif/api.h>

enum {
    OIF_LANG_C = 1,
    OIF_LANG_CXX = 2,
    OIF_LANG_PYTHON = 3,
    OIF_LANG_JULIA = 4,
    OIF_LANG_R = 5,
    OIF_LANG_COUNT = 6,
};

// Identifier for the language-specific dispatch library (C, Python, etc.).
typedef unsigned int DispatchHandle;

/**
 * Base structure for implementation details.
 * Language-specific implementations can add extra members to the subtypes
 * of `Impl`, however, they must not set `implh` and `dh` themselves
 * as this is a responsibility of the `dispatch` library.
 */
typedef struct {
    ImplHandle implh;
    DispatchHandle dh;
} ImplInfo;

ImplInfo *
load_backend(const char *impl_details, size_t version_major, size_t version_minor);

int
unload_impl(ImplInfo *impl_info);

int
run_interface_method(ImplInfo *impl_info, const char *method, OIFArgs *in_args,
                     OIFArgs *out_args);
