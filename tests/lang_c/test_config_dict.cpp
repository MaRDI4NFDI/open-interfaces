#include <string.h>

#include <gtest/gtest.h>

#include <oif/api.h>
#include <oif/config_dict.h>
#include <oif/util.h>

TEST(OIFConfigDictTest, SimpleCase)
{
    OIFConfigDict *dict = oif_config_dict_init();

    oif_config_dict_add_int(dict, "int_option", 42);
    oif_config_dict_add_double(dict, "double_option", 2.718);

    oif_config_dict_free(dict);
}

TEST(OIFConfigDictTest, TraverseCase)
{
    OIFConfigDict *dict = oif_config_dict_init();

    oif_config_dict_add_int(dict, "int_option", 42);
    oif_config_dict_add_double(dict, "double_option", 2.718);

    ASSERT_EQ(oif_config_dict_key_exists(dict, "int_option"), true);
    ASSERT_EQ(oif_config_dict_key_exists(dict, "double_option"), true);

    oif_config_dict_free(dict);
}

TEST(OIFConfigDictTest, GetValues)
{
    OIFConfigDict *dict = oif_config_dict_init();

    oif_config_dict_add_int(dict, "int_option", 42);
    oif_config_dict_add_double(dict, "double_option", 2.718);

    ASSERT_EQ(oif_config_dict_get_int(dict, "int_option"), 42);
    ASSERT_EQ(oif_config_dict_get_double(dict, "double_option"), 2.718);

    oif_config_dict_free(dict);
}

TEST(OIFConfigDictTest, SerializeDeserializeBasicCase)
{
    OIFConfigDict *dict = oif_config_dict_init();

    oif_config_dict_add_int(dict, "int_option", 42);
    oif_config_dict_add_int(dict, "int_neg_option", -12345);
    oif_config_dict_add_double(dict, "double_option", 2.718);

    oif_config_dict_serialize(dict);
    OIFConfigDict *new_dict = oif_config_dict_init();
    oif_config_dict_copy_serialization(new_dict, dict);
    int status = oif_config_dict_deserialize(new_dict);
    ASSERT_EQ(status, 0);

    ASSERT_EQ(oif_config_dict_get_int(new_dict, "int_option"), 42);
    ASSERT_EQ(oif_config_dict_get_int(new_dict, "int_neg_option"), -12345);
    ASSERT_EQ(oif_config_dict_get_double(new_dict, "double_option"), 2.718);

    oif_config_dict_free(dict);
    oif_config_dict_free(new_dict);
}

TEST(OIFConfigDictTest, SerializeDeserializeVeryLongStringCase)
{
    OIFConfigDict *dict = oif_config_dict_init();

    const int N = 20;

    char buffer[1024];
    for (size_t i = 0; i < N; ++i) {
        sprintf(buffer, "int_option_%zu", i);
        oif_config_dict_add_int(dict, buffer, i);
    }
    for (size_t i = 0; i < N; ++i) {
        sprintf(buffer, "int_neg_option_%zu", i);
        oif_config_dict_add_int(dict, buffer, -i);
    }
    for (size_t i = 0; i < N; ++i) {
        sprintf(buffer, "double_option_%.6f", 3.14 * i);
        oif_config_dict_add_double(dict, buffer, 3.14 * i);
    }

    oif_config_dict_serialize(dict);
    OIFConfigDict *new_dict = oif_config_dict_init();
    oif_config_dict_copy_serialization(new_dict, dict);
    int status = oif_config_dict_deserialize(new_dict);
    ASSERT_EQ(status, 0);

    for (size_t i = 0; i < N; ++i) {
        sprintf(buffer, "int_option_%zu", i);
        ASSERT_EQ(oif_config_dict_get_int(new_dict, buffer), i);
    }

    for (size_t i = 0; i < N; ++i) {
        sprintf(buffer, "int_neg_option_%zu", i);
        ASSERT_EQ(oif_config_dict_get_int(new_dict, buffer), -i);
    }

    for (size_t i = 0; i < N; ++i) {
        sprintf(buffer, "double_option_%.6f", 3.14 * i);
        ASSERT_EQ(oif_config_dict_get_double(new_dict, buffer), 3.14 * i);
    }

    oif_config_dict_free(dict);
    oif_config_dict_free(new_dict);
}

TEST(OIFConfigDictTest, GetCopyOfKeys)
{
    OIFConfigDict *dict = oif_config_dict_init();

    oif_config_dict_add_int(dict, "int_option", 42);
    oif_config_dict_add_double(dict, "double_option", 2.718);

    const char **keys = oif_config_dict_get_keys(dict);
    ASSERT_EQ(oif_util_str_contains(keys, "int_option"), true);
    ASSERT_EQ(oif_util_str_contains(keys, "double_option"), true);
    ASSERT_EQ(keys[2], nullptr);

    oif_util_free(keys);
    oif_config_dict_free(dict);
}
