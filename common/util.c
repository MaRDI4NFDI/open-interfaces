#define _GNU_SOURCE
#include <ctype.h>
#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <oif/util.h>
#include <oif/_platform.h>

static size_t NALLOCS_ = 0;
static size_t NBYTES_ = 0;

void *
oif_util_malloc_(size_t nbytes)
{
    // void *p_raw = malloc(nbytes + sizeof(size_t));
    // if (p_raw == NULL) {
    //     fprintf(stderr, "Could not allocate memory\n");
    //     exit(1);
    // }
    // NALLOCS_++;
    // NBYTES_ += nbytes;

    // size_t *p2 = (size_t *)p_raw;
    // *p2 = nbytes;
    // void *p_oif = (void *)(p2 + 1);
    // fprintf(stderr, "Pointer malloc: %p\n", p_raw);
    // fprintf(stderr, "    Size in it: %zu bytes\n", nbytes);
    // fprintf(stderr, "   Pointer oif: %p\n", p_oif);
    //
    void *p_oif = malloc(nbytes);
    NALLOCS_++;

    return p_oif;
}

void *
oif_util_malloc_verbose(size_t nbytes, const char *file, const char *func, int line)
{
#if defined(OIF_OPTION_VERBOSE_DEBUG_INFO)
    fprintf(stderr,
            "\033[31m"
            "[oif_util_malloc]"
            "\033[0m"
            " Allocating %zu bytes of memory\n"
            "                  File: %s\n"
            "              Function: %s\n"
            "                  Line: %d\n",
            nbytes, file, func, line);
#else
    (void)file;  // Suppress unused parameter warning
    (void)func;  // Suppress unused parameter warning
    (void)line;  // Suppress unused parameter warning
#endif
    return oif_util_malloc_(nbytes);
}

void
oif_util_free_(void *ptr)
{
    if (ptr == NULL) {
        fprintf(stderr, "Cannot free a NULL pointer\n");
        return;
    }

    size_t *p_oif = (size_t *)ptr;
    // size_t *p_raw = p_oif - 1;
    // size_t nbytes = *p_raw;
    // fprintf(stderr, "Pointer to free: %p\n", ptr);
    // fprintf(stderr, "    Pointer raw: %p\n", p_raw);
    // fprintf(stderr, "     Size in it: %zu bytes\n", nbytes);
    free(p_oif);
    p_oif = NULL;
    if (NALLOCS_ == 0) {
        fprintf(stderr,
                "\033[31m[oif_util_free] ERROR:\033[0m For some reason, number of non-freed "
                "allocations is already zero. Exiting...\n");
        exit(1);
    }
    NALLOCS_--;
    // NBYTES_ -= nbytes;
}

void
oif_util_free_verbose(void *ptr, const char *file, const char *func, int line)
{
#if defined(OIF_OPTION_VERBOSE_DEBUG_INFO)
    fprintf(stderr,
            "\033[31m[oif_util_free]\033[0m Freeing memory\n"
            "                  File: %s\n"
            "              Function: %s\n"
            "                  Line: %d\n",
            file, func, line);
#else
    (void)file;  // Suppress unused parameter warning
    (void)func;  // Suppress unused parameter warning
    (void)line;  // Suppress unused parameter warning
#endif
    oif_util_free_(ptr);
}

uint32_t
oif_util_u32_from_size_t(size_t val)
{
    uint32_t result;

    if (val <= UINT32_MAX) {
        result = (uint32_t)val;
    }
    else {
        fprintf(stderr, "Could not convert safely to `uint32_t` from `size_t`\n");
        exit(2);
    }

    return result;
}

char *
oif_util_str_duplicate_(const char *src)
{
    size_t len = strlen(src);
    char *dest = oif_util_malloc_((len + 1) * sizeof(*dest));
    if (dest == NULL) {
        fprintf(stderr, "[str_duplicate] Could not allocate memory\n");
        return NULL;
    }
    strcpy(dest, src);
    return dest;
}

char *
oif_util_str_duplicate_verbose(const char *src, const char *file, const char *func, int line)
{
#if defined(OIF_OPTION_VERBOSE_DEBUG_INFO)
    fprintf(stderr,
            "\033[31m"
            "[oif_util_str_duplicate]"
            "\033[0m"
            " Duplicate string '%s'\n"
            "                  File: %s\n"
            "              Function: %s\n"
            "                  Line: %d\n",
            src, file, func, line);
#else
    (void)file;  // Suppress unused parameter warning
    (void)func;  // Suppress unused parameter warning
    (void)line;  // Suppress unused parameter warning
#endif
    return oif_util_str_duplicate_(src);
}

int
oif_strcmp_nocase(const char s1[static 1], const char s2[static 1])
{
    size_t n1 = strlen(s1);
    size_t n2 = strlen(s2);
    size_t n3 = n1 < n2 ? n1 : n2;

    for (size_t i = 0; i < n3; ++i) {
        if (tolower(s1[i]) < tolower(s2[i])) {
            return -1;
        }
        else if (tolower(s1[i]) > tolower(s2[i])) {
            return 1;
        }
    }

    return 0;
}

bool
oif_util_str_contains(const char **arr, const char *s)
{
    for (size_t i = 0; arr[i] != NULL; ++i) {
        if (strcmp(arr[i], s) == 0) {
            return true;
        }
    }
    return false;
}

int
logwarn(const char *prefix, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);

    fprintf(stderr, "[%s] \033[1m\033[33mWARNING:\033[0m ", prefix);
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");

    va_end(ap);

    return 0;
}

int
logerr(const char *prefix, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);

    fprintf(stderr, "[%s] \033[1m\033[31mERROR:\033[0m ", prefix);
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");

    va_end(ap);

    return 0;
}

#if defined(__GNUC__)
#if !defined(__OPTIMIZE__)
void __attribute((destructor))
oif_util_dtor_(void)
{
    fprintf(stderr,
            "\033[31m[oif_util]\033[0m Final statistics on memory allocs via malloc/free:\n");
    fprintf(stderr, "\033[31m[oif_util]\033[0m Number of not-freed allocations: %zu\n",
            NALLOCS_);
    // fprintf(stderr, "[oif_util] Number of not-freed bytes: %zu\n", NBYTES_);
    if (NALLOCS_ > 0) {
        fprintf(stderr, "\033[31m[oif_util]\033[0m Memory leaks detected!\n");
        exit(1);
    }
}
#endif
#endif
