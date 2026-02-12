#include <gtest/gtest.h>

#include <oif/util.h>

TEST(OIFUtilTestSuite, oif_util_make_string_1)
{
    char *actual = oif_util_make_string("%s: %d, %s: %.3f %s!", "value1", 42, "value2", 3.14159, "Hello World");
    char const *const expected = "value1: 42, value2: 3.142 Hello World!";
    ASSERT_STREQ(actual, expected);
    oif_util_free(actual);
}
