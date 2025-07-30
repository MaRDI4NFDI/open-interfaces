// clang-format Language: C
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>

void *
oif_util_malloc_(size_t nbytes);
void
oif_util_free_(void *ptr);

void *
oif_util_malloc_verbose(size_t nbytes, const char *file, const char *func, int line);
void
oif_util_free_verbose(void *ptr, const char *file, const char *func, int line);

/**
 * Wrap `malloc` to collect debugging information about allocated memory.
 *
 * The function also does a check and **exits if the `malloc` fails**.
 * The reason for exiting is that if we cannot allocate memory,
 * then something is really wrong with the system,
 * and there is no point to continue.
 *
 * @param nbytes Number of bytes to allocate.
 * @return Pointer to the allocated memory.
 */
#define oif_util_malloc(nbytes) oif_util_malloc_verbose((nbytes), __FILE__, __func__, __LINE__)

#define oif_util_free(ptr) oif_util_free_verbose((ptr), __FILE__, __func__, __LINE__)

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
oif_util_str_duplicate_(const char *src);

char *
oif_util_str_duplicate_verbose(const char *src, const char *file, const char *func, int line);

#define oif_util_str_duplicate(src)                                                           \
    oif_util_str_duplicate_verbose((src), __FILE__, __func__, __LINE__)

/**
 * Compare two C strings case-insensitively.
 *
 * @return The same values as `string.h::strcmp`
 */
int
oif_strcmp_nocase(const char *s1, const char *s2);

/**
 * Check if a string is contained in a null-trerminated array of strings.
 *
 * @param arr The null-terminated array of strings to search in.
 * @param s The string to search for.
 * @return true if the key is found, false otherwise.
 */
bool
oif_util_str_contains(const char **arr, const char *s);

/**
 * Log a warning message to stderr.
 *
 * The error message starts with "[prefix] WARNING: "
 * followed by the formatted message.
 *
 * @param prefix The prefix to be printed before the message.
 * @param fmt The format string for the message.
 * @param ... The arguments for the format string.
 */
int
logwarn(const char *prefix, const char *fmt, ...);

/**
 * Log an error message to stderr.
 *
 * The error message starts with "[prefix] ERROR: "
 * followed by the formatted message.
 *
 * @param prefix The prefix to be printed before the message.
 * @param fmt The format string for the message.
 * @param ... The arguments for the format string.
 */
int
logerr(const char *prefix, const char *fmt, ...);

#ifdef __cplusplus
}
#endif
