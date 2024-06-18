#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <cwpack.h>
#include <hashmap.h>

#include <oif/api.h>
#include <oif/config_dict.h>

struct oif_config_dict_t {
    HASHMAP(char, OIFConfigEntry) map;
    cw_pack_context *pc;
    size_t size;
};

static size_t SIZE_ = 65;


static char *copy_key_(const char *key)
{
    char *key_copy = malloc(strlen(key) * sizeof(char));
    if (key_copy == NULL) {
        fprintf(stderr, "Could not allocate memory for a key copy\n");
        exit(1);
    }
    strcpy(key_copy, key);
    return key_copy;
}

static void free_key_(char *key) {
    free(key);
}


OIFConfigDict *oif_config_dict_init(void)
{
    OIFConfigDict *dict = malloc(sizeof(OIFConfigDict));
    assert(dict != NULL);

    hashmap_init(&dict->map, hashmap_hash_string, strcmp);
    hashmap_set_key_alloc_funcs(&dict->map, copy_key_, free_key_);
    dict->size = 0;

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
    dict->size++;
    assert(dict->size < SIZE_);
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
    dict->size++;
    assert(dict->size < SIZE_);
}

const char **oif_config_dict_get_keys(OIFConfigDict *dict)
{
    const char **keys = malloc(dict->size * sizeof(char *));
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

void oif_config_dict_serialize(OIFConfigDict *dict)
{
    cw_pack_context *pc = malloc(sizeof(*pc));
    if (pc == NULL) {
        fprintf(
            stderr,
            "Could not allocate memory required for serializing a config dictionary\n"
        );
    }
    char buffer[512];
    cw_pack_context_init(pc, buffer, 512, 0);

    cw_pack_map_size(pc, dict->size);

    const char *key;
    OIFConfigEntry *entry;
    hashmap_foreach(key, entry, &dict->map) {
        printf("Packing '%s'\n", key);
        cw_pack_str(pc, key, strlen(key));
        if (entry->type == OIF_INT) {
            int64_t i64_value = *(int *) entry->value;
            cw_pack_signed(pc, i64_value);
        }
        else if (entry->type == OIF_FLOAT64) {
            cw_pack_double(pc, *(double *) entry->value);
        }
        else {
            fprintf(
                stderr,
                "Unsupported type for serialization\n"
            );
            exit(1);
        }
    }

    if (pc->return_code != CWP_RC_OK) {
        fprintf(
            stderr,
            "Serialization of config dictionary was not successful. pc->return_code = %d\n", pc->return_code
        );
        exit(1);
    }

    dict->pc = pc;
}

OIFConfigDict *oif_config_dict_deserialize(OIFConfigDict *dict)
{
    OIFConfigDict *new_dict = oif_config_dict_init();
    cw_unpack_context uctx;

    cw_unpack_context_init(&uctx, dict->pc->start, dict->pc->current - dict->pc->start, 0);

    const char *key;
    while (uctx.return_code != CWP_RC_END_OF_INPUT) {
        cw_unpack_next(&uctx);
        key = uctx.item.as.str.start;
        printf("uctx.item = %p\n", &(uctx.item));
        cw_unpack_next(&uctx);
        printf("uctx.item = %p\n", &(uctx.item));
        if (uctx.item.type == CWP_ITEM_NEGATIVE_INTEGER) {
            oif_config_dict_add_int(new_dict, key, uctx.item.as.i64);
        }
        else if (uctx.item.type == CWP_ITEM_DOUBLE) {
            oif_config_dict_add_double(new_dict, key, uctx.item.as.long_real);
        }
        else {
            fprintf(
                stderr,
                "[oif_config_dict_deserialize] Unknown type: %d\n", uctx.item.type
            );
            exit(1);
        }
    }

    return new_dict;
}
