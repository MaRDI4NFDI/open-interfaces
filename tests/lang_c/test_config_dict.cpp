#include <gtest/gtest.h>

#include <oif/api.h>
#include <oif/config_dict.h>


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
