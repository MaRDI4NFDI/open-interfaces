#pragma once

#include <oif/api.h>

BackendHandle
oif_init_backend(
    const char *backend, const char *interface, int major, int minor
);
