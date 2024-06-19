#pragma once
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>


/**
 * Convert `size_t` input to `uint32_t`.
 *
 * If the input value is larger than the maximum value for `uint32_t`,
 * then program exits with an error message.
 */
inline static uint32_t
u32_from_size_t(size_t val)
{
    uint32_t result;

    if (val <= UINT32_MAX) {
        result = (uint32_t) val;
    }
    else {
        fprintf(stderr, "Could not convert safely to `uint32_t` from `size_t`\n");
        exit(2);
    }

    return result;
}
