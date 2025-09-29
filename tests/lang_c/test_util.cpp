#include <string.h>
#include <stdio.h>

#include <gtest/gtest.h>

#include <oif/util.h>

TEST(OIFUtilTest, oif)
char *
oif_util_str_duplicate_(const char *src)
{
    const char src[] = "Hello, World!";
    char *dst = NULL;

    dst = oif_util_str_duplicate_(src);

    oif_config_dict_add_int(dict, "int_option", 42);
    oif_config_dict_add_double(dict, "double_option", 2.718);

    oif_config_dict_free(dict);
}
