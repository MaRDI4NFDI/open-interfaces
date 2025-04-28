// clang-format Language: C
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <limits.h>
#include <stdint.h>
#include <stdio.h>

/**
 * Convert `size_t` input to `uint32_t`.
 *
 * If the input value is larger than the maximum value for `uint32_t`,
 * then program exits with an error message.
 */
uint32_t
oif_util_u32_from_size_t(size_t val);

/**
 * Duplicate a null-terminated string `src`, allocating memory for the copy.
 *
 * It is the user responsibility to free the allocated memory.
 * This function is basically the same as `strdup` in POSIX C.
 *
 * @return The pointer to the duplicated null-terminated string or NULL in case
 * of an error.
 */
char *
oif_util_str_duplicate(const char *src);

/**
 * Compare two C strings case-insensitively.
 *
 * @return The same values as `string.h::strcmp`
 */
int
oif_strcmp_nocase(const char s1[static 1], const char s2[static 1]);

#ifdef __cplusplus
}
#endif
