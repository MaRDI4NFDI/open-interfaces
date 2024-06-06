#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include "oif_impl/_util.h"


int oif_strcmp_nocase(const char s1[static 1], const char s2[static 1])
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

