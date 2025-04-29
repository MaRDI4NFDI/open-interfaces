// clang-format Language: C
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

#include <oif/api.h>

typedef struct oif_config_dict_t OIFConfigDict;

typedef struct oif_config_entry_t {
    OIFArgType type;
    void *value;
} OIFConfigEntry;

OIFConfigDict *
oif_config_dict_init(void);
void
oif_config_dict_free(void *dict);
void
oif_config_dict_add_int(OIFConfigDict *dict, const char *key, int value);
void
oif_config_dict_add_double(OIFConfigDict *dict, const char *key, double value);
void
oif_config_dict_add_str(OIFConfigDict *dict, const char *key, const char *value);
const char **
oif_config_dict_get_keys(OIFConfigDict *dict);
bool
oif_config_dict_key_exists(OIFConfigDict *dict, const char *key);
int
oif_config_dict_get_int(OIFConfigDict *dict, const char *key);
double
oif_config_dict_get_double(OIFConfigDict *dict, const char *key);
void
oif_config_dict_serialize(OIFConfigDict *dict);
int
oif_config_dict_deserialize(OIFConfigDict *dict);
int
oif_config_dict_copy_serialization(OIFConfigDict *to, const OIFConfigDict *from);
void
oif_config_dict_print(OIFConfigDict *dict);
const uint8_t *
oif_config_dict_get_serialized(OIFConfigDict *dict);
size_t
oif_config_dict_get_serialized_object_length(OIFConfigDict *dict);

#ifdef __cplusplus
}
#endif
