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
uint32_t
oif_util_u32_from_size_t(size_t val);

/**
 * Duplicate a given null-terminated string along with memory allocation.
 *
 * It is the user responsibility to free the allocated memory.
 * This function is basically a copy of `strdup` because it is not in ISO C.
 *
 * @return The pointer to the duplicated null-terminated string or NULL in case
 * of an error.
 */
char *
oif_util_str_duplicate(const char *src);
