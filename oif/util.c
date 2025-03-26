#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <oif/util.h>

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
oif_util_str_duplicate(const char *src)
{
    size_t len = strlen(src);
    char *dest = malloc((len + 1) * sizeof(*dest));
    if (dest == NULL) {
        fprintf(stderr, "[str_duplicate] Could not allocate memory\n");
        return NULL;
    }
    strcpy(dest, src);
    return dest;
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
