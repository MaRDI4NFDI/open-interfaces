#include <gtest/gtest.h>
#include <array>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <memory>

namespace {
std::string
capture_stdout(const std::string &command)
{
    std::array<char, 1024> buffer;
    std::string result;

    // Use popen to run the command and capture stdout
    const std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);

    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }

    // Read the output line by line
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }

    return result;
}

const std::string RUN_AUX_TESTS_CONFIG_DICT("tests/lang_c/run_aux_tests_config_dict");
}  // namespace

TEST(OIFConfigDictPrint, EmptyDict)
{
    const std::string output = capture_stdout(RUN_AUX_TESTS_CONFIG_DICT);
    EXPECT_EQ(output, "Config dict has no entries\n");
}

TEST(OIFConfigDictPrint, Case1)
{
    const std::string output = capture_stdout(RUN_AUX_TESTS_CONFIG_DICT + " case1");
    EXPECT_EQ(output,
              "Key = 'key2', value = '42'\nKey = 'key1', value = "
              "'1'\nSerialization\n\\xa4key2*\\xa4key1\\x01\n");
}

TEST(OIFConfigDictPrint, Case2)
{
    const std::string output = capture_stdout(RUN_AUX_TESTS_CONFIG_DICT + " case2");
    EXPECT_EQ(output,
              "Key = 'key2', value = '3.140000'\nKey = 'key3', value = 'hello'\nKey = 'key1', "
              "value = "
              "'1'\nSerialization\n\\xa4key2\\xcb@"
              "\\x09\\x1e\\xb8Q\\xeb\\x85\\x1f\\xa4key3\\xa5hello\\xa4key1\\x01\n");
}
