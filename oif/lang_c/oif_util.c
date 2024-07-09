#include <oif/util.h>
#include <string.h>

uint32_t
oif_util_u32_from_size_t(size_t val)
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


char *
oif_util_str_duplicate(const char *src)
{
    size_t len = strlen(src);
    char *dest = malloc(len * sizeof(*dest));
    if (dest == NULL) {
        fprintf(stderr, "[str_duplicate] Could not allocate memory\n");
        return NULL;
    }
    strcpy(dest, src);
    return dest;
}
