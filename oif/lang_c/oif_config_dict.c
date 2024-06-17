#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>


#include <hashmap.h>

#include <oif/api.h>
#include <oif/config_dict.h>

struct oif_config_dict_t {
    HASHMAP(char, OIFConfigEntry) map;
};

static size_t SIZE_ = 65;


OIFConfigDict *oif_config_dict_init(void)
{
    OIFConfigDict *dict = malloc(sizeof(OIFConfigDict));
    assert(dict != NULL);

    hashmap_init(&dict->map, hashmap_hash_string, strcmp);

    return dict;
}

void oif_config_dict_free(OIFConfigDict *dict)
{
    const char *key;
    OIFConfigEntry *entry;
    hashmap_foreach(key, entry, &dict->map) {
        free(entry->value);
    }

    hashmap_cleanup(&dict->map);
    free(dict);
}

void oif_config_dict_add_int(OIFConfigDict *dict, const char *key, int value)
{
    OIFConfigEntry *entry = malloc(sizeof(OIFConfigEntry));
    if (entry == NULL) {
        fprintf(stderr, "Could not add an entry to the config dictionary\n");
        exit(1);
    }
    entry->type = OIF_INT;
    entry->value = malloc(sizeof(int));
    if (entry->value == NULL) {
        fprintf(stderr, "Could not allocate memory for adding an int entry\n");
        exit(1);
    }
    memcpy(entry->value, &value, sizeof(int));

    int result = hashmap_put(&dict->map, key, entry);
}

void oif_config_dict_add_double(OIFConfigDict *dict, const char *key, double value)
{
    OIFConfigEntry *entry = malloc(sizeof(OIFConfigEntry));
    if (entry == NULL) {
        fprintf(stderr, "Could not add an entry to the config dictionary\n");
        exit(1);
    }
    entry->type = OIF_FLOAT64;
    entry->value = malloc(sizeof(double));
    if (entry->value == NULL) {
        fprintf(stderr, "Could not allocate memory for adding a double entry\n");
        exit(1);
    }
    memcpy(entry->value, &value, sizeof(double));

    int result = hashmap_put(&dict->map, key, entry);
}

const char **oif_config_dict_get_keys(OIFConfigDict *dict)
{
    const char **keys = malloc(SIZE_ * sizeof(char *));
    if (keys == NULL) {
        fprintf(stderr, "Could not allocate memory for keys\n");
        exit(1);
    }
    const char *key;
 
    HASHMAP_ITER(dict->map) it;

    size_t i = 0;
    for (it = hashmap_iter(&dict->map); hashmap_iter_valid(&it); hashmap_iter_next(&it)) {
        if (i == SIZE_ - 1) {
            fprintf(stderr, "Dictionary has too many options. Cannot proceed\n");
            exit(1);
        }
        key = hashmap_iter_get_key(&it);
        keys[i] = key;
        i++;
    }
    keys[i] = NULL;

    return keys;
}

bool oif_config_dict_key_exists(OIFConfigDict *dict, const char *key)
{
    OIFConfigEntry *entry = hashmap_get(&dict->map, key);
    if (entry == NULL) {
        return false;
    }
    return true;
}

int oif_config_dict_get_int(OIFConfigDict *dict, const char *key)
{
    OIFConfigEntry *entry = hashmap_get(&dict->map, key);
    if (entry == NULL) {
        fprintf(stderr, "Could not find an entry with key '%s'\n", key);
        exit(1);
    }

    assert(entry->type == OIF_INT);
    int int_value = *(int *) entry->value;
    return int_value;
}

double oif_config_dict_get_double(OIFConfigDict *dict, const char *key)
{
    OIFConfigEntry *entry = hashmap_get(&dict->map, key);
    if (entry == NULL) {
        fprintf(stderr, "Could not find an entry with key '%s'\n", key);
        exit(1);
    }

    assert(entry->type == OIF_FLOAT64);
    double double_value = *(double *) entry->value;
    return double_value;
}

