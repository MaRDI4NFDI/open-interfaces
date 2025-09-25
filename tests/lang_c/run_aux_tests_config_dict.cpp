#include <cstring>
#include <oif/c_bindings.h>
#include <oif/config_dict.h>
#include <cstdio>
#include <cstdlib>
using namespace std;


int main(int argc, char *argv[])
{
    OIFConfigDict *dict = oif_config_dict_init();

    if (argc == 1) {
        // Do nothing
    }
    else if (strcmp(argv[1], "case1") == 0) {
        oif_config_dict_add_int(dict, "key1", 1);
        oif_config_dict_add_int(dict, "key2", 42);
    }
    else if (strcmp(argv[1], "case2") == 0) {
        oif_config_dict_add_int(dict, "key1", 1);
        oif_config_dict_add_double(dict, "key2", 3.14);
        oif_config_dict_add_str(dict, "key3", "hello");
    }
    else {
        fprintf(stderr, "Unknown argument: %s\n", argv[1]);
        return EXIT_FAILURE;
    }

    oif_config_dict_serialize(dict);

    oif_config_dict_print(dict);

    oif_config_dict_free(dict);
}
