// Dispatch library that is called from other languages, and dispatches it
// to the appropriate language-specific dispatch.
#include <assert.h>
#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// This is required to avoid clang-tidy issues with `hashmap.h`.
#if __STDC_VERSION__ < 202300L
#define typeof __typeof__
#endif

#include <hashmap.h>

#include "oif/api.h"
#include <oif/util.h>

#include "oif/internal/bridge_api.h"
#include "oif/internal/dispatch.h"

static char OIF_DISPATCH_C_SO[] = "liboif_dispatch_c.so";
static char OIF_DISPATCH_PYTHON_SO[] = "liboif_dispatch_python.so";
static char OIF_DISPATCH_JULIA_SO[] = "liboif_dispatch_julia.so";

static const char *OIF_IMPL_PATH;

static char *OIF_LANG_FROM_LANG_ID[] = {
    [OIF_LANG_C] = "C",         [OIF_LANG_CXX] = "C++", [OIF_LANG_PYTHON] = "Python",
    [OIF_LANG_JULIA] = "Julia", [OIF_LANG_R] = "R",
};

/**
 * Array of handles to the dynamically loaded libraries
 * for the language-specific dispatches.
 */
void *OIF_DISPATCH_HANDLES[OIF_LANG_COUNT];

// cppcheck-suppress unusedStructMember
static HASHMAP(ImplHandle, ImplInfo) IMPL_MAP;

static bool INITIALIZED_ = false;

static int IMPL_COUNTER_ = 1000;

size_t
hash_fn(const ImplHandle *key)
{
    if (*key < 0) {
        fprintf(stderr, "[dispatch] Was expecting a non-negative number, got %d\n", *key);
        exit(1);
    }
    return *key;
}

int
compare_fn(const ImplHandle *key1, const ImplHandle *key2)
{
    return *key1 - *key2;
}

static int
init_module_(void)
{
    OIF_IMPL_PATH = getenv("OIF_IMPL_PATH");
    if (OIF_IMPL_PATH == NULL) {
        fprintf(stderr,
                "[dispatch] Environment variable 'OIF_IMPL_PATH' must be "
                "set so that implementations can be found. Cannot proceed\n");
        return -1;
    }
    hashmap_init(&IMPL_MAP, hash_fn, compare_fn);
    INITIALIZED_ = true;

    return 0;
}

ImplHandle
load_interface_impl(const char *interface, const char *impl, size_t version_major,
                    size_t version_minor)
{
    if (!INITIALIZED_) {
        int status = init_module_();
        if (status) {
            return -1;
        }
    }
    DispatchHandle dh;
    const char *dispatch_lang_so;
    void *lib_handle = NULL;
    FILE *conf_file = NULL;
    char *buffer;
    /* One must be a pessimist, while programming in C. */
    ImplHandle retval = OIF_IMPL_INIT_ERROR;

    char conf_filename_fixed_part[512] = {'\0'};
    strcat(conf_filename_fixed_part, interface);
    strcat(conf_filename_fixed_part, "/");
    strcat(conf_filename_fixed_part, impl);
    strcat(conf_filename_fixed_part, "/");
    strcat(conf_filename_fixed_part, impl);
    strcat(conf_filename_fixed_part, ".conf");

    // We cannot tokenize OIF_IMPL_PATH because `strtok`
    // modifies the original string during tokenization
    // by replacing tokens with nul-terminators.
    // Then, when loading new implementations,
    // OIF_IMPL_PATH will not have the original value.
    char *oif_impl_path_dup = oif_util_str_duplicate(OIF_IMPL_PATH);
    char *path = strtok((char *)oif_impl_path_dup, ":");
    char conf_filename[1024] = "";
    char *conf_filename_p = conf_filename;
    while (path) {
        strcat(conf_filename_p, path);
        strcat(conf_filename_p, "/");
        strcat(conf_filename_p, conf_filename_fixed_part);

        conf_file = fopen(conf_filename, "re");
        if (conf_file != NULL) {
            break;
        }
        path = strtok(NULL, ":");
        conf_filename_p = conf_filename;
        conf_filename_p[0] = '\0';
    }

    if (conf_file == NULL) {
        fprintf(stderr, "[dispatch] Cannot open conf file '%s'\n", conf_filename_fixed_part);
        fprintf(stderr, "[dispatch] Search was done in the following paths: %s\n",
                oif_impl_path_dup);
        free(oif_impl_path_dup);
        perror("Error message is: ");
        return -1;
    }
    else {
        fprintf(stderr, "[dispatch] Configuration file: %s\n", conf_filename);
    }
    free(oif_impl_path_dup);

    // Temporary buffer to read lines from file.
    const int buffer_size = 512;
    size_t len;
    char *fgets_status;
    buffer = malloc(sizeof(char) * buffer_size);
    if (buffer == NULL) {
        fprintf(stderr,
                "[dispatch] Could not allocate buffer for parsing "
                "implementation configuration files\n");
        goto cleanup;
    }
    char backend_name[16];
    fgets_status = fgets(buffer, buffer_size, conf_file);
    if (fgets_status == NULL) {
        fprintf(stderr,
                "[dispatch] Could not read backend line from configuration "
                "file '%s'\n",
                conf_filename);
        goto cleanup;
    }
    len = strlen(buffer);
    if (buffer[len - 1] != '\n') {
        fprintf(stderr, "Backend name is longer than allocated buffer\n");
        goto cleanup;
    }
    else {
        // Trim the new line character.
        buffer[len - 1] = '\0';
    }
    strcpy(backend_name, buffer);
    fprintf(stderr, "[dispatch] Backend name: %s\n", backend_name);

    fgets_status = fgets(buffer, buffer_size, conf_file);
    if (fgets_status == NULL) {
        fprintf(stderr,
                "[dispatch] Could not read implementation details line "
                "from the configuration file\n");
        goto cleanup;
    }
    len = strlen(buffer);
    if (buffer[len - 1] != '\n') {
        fprintf(stderr, "[dispatch] Backend name is longer than allocated array\n");
        goto cleanup;
    }
    else {
        // Trim new line character.
        buffer[len - 1] = '\0';
    }
    char impl_details[512];
    strcpy(impl_details, buffer);
    fprintf(stderr, "[dispatch] Implementation details: '%s'\n", impl_details);

    if (strcmp(backend_name, "c") == 0) {
        dh = OIF_LANG_C;
        dispatch_lang_so = OIF_DISPATCH_C_SO;
    }
    else if (strcmp(backend_name, "python") == 0) {
        dh = OIF_LANG_PYTHON;
        dispatch_lang_so = OIF_DISPATCH_PYTHON_SO;
    }
    else if (strcmp(backend_name, "julia") == 0) {
        dh = OIF_LANG_JULIA;
        dispatch_lang_so = OIF_DISPATCH_JULIA_SO;
    }
    else {
        fprintf(stderr, "[dispatch] Implementation has unknown backend: '%s'\n", backend_name);
        goto cleanup;
    }

    if (OIF_DISPATCH_HANDLES[dh] == NULL) {
        lib_handle = dlopen(dispatch_lang_so, RTLD_LOCAL | RTLD_LAZY);
        if (lib_handle == NULL) {
            fprintf(stderr, "[dispatch] Cannot load shared library '%s'\n", dispatch_lang_so);
            fprintf(stderr, "Error message: %s\n", dlerror());
            goto cleanup;
        }
        OIF_DISPATCH_HANDLES[dh] = lib_handle;
    }
    else {
        lib_handle = OIF_DISPATCH_HANDLES[dh];
    }

    ImplInfo *(*load_impl_fn)(const char *, size_t, size_t);
    load_impl_fn = dlsym(lib_handle, "load_impl");

    if (load_impl_fn == NULL) {
        fprintf(stderr, "[dispatch] Could not load function %s: %s\n", "load_impl", dlerror());
        goto cleanup;
    }

    ImplInfo *impl_info = load_impl_fn(impl_details, version_major, version_minor);
    if (impl_info == NULL) {
        fprintf(stderr, "[dispatch] Could not load implementation\n");
        goto cleanup;
    }
    impl_info->implh = IMPL_COUNTER_;
    impl_info->dh = dh;
    impl_info->interface = oif_util_str_duplicate(interface);
    int result = hashmap_put(&IMPL_MAP, &impl_info->implh, impl_info);
    if (result != 0) {
        fprintf(stderr, "[dispatch] hashmap_put had error, result %d\n", result);
        goto cleanup;
    }
    IMPL_COUNTER_++;
    retval = impl_info->implh;

cleanup:
    if (buffer != NULL) {
        free(buffer);
    }
    if (conf_file != NULL) {
        fclose(conf_file);
    }

    return retval;
}

int
unload_interface_impl(ImplHandle implh)
{
    ImplInfo *impl_info = hashmap_get(&IMPL_MAP, &implh);
    DispatchHandle dh = impl_info->dh;
    if (OIF_DISPATCH_HANDLES[dh] == NULL) {
        fprintf(stderr,
                "[dispatch] Cannot unload interface implementation "
                "for language '%s'\n",
                OIF_LANG_FROM_LANG_ID[dh]);
        exit(EXIT_FAILURE);
    }
    void *lib_handle = OIF_DISPATCH_HANDLES[dh];

    int (*unload_impl_fn)(ImplInfo *);
    unload_impl_fn = dlsym(lib_handle, "unload_impl");
    if (unload_impl_fn == NULL) {
        fprintf(stderr,
                "[dispatch] Cannot find function 'unload_impl' "
                "for language '%s'\n",
                OIF_LANG_FROM_LANG_ID[dh]);
        return -1;
    }
    // Free resources added by subtypes of ImplInfo.
    unload_impl_fn(impl_info);
    ImplInfo *result = hashmap_remove(&IMPL_MAP, &implh);
    if (result == NULL || result->implh != implh) {
        fprintf(stderr,
                "[dispatch] Error occured when unloading implementation "
                "from the implementations table.");
    }
    free(impl_info->interface);
    free(impl_info);
    impl_info = NULL;
    printf("[dispatch] Unload implementation with id '%d'\n", implh);

    return 0;
}

int
call_interface_impl(ImplHandle implh, const char *method, OIFArgs *in_args, OIFArgs *out_args)
{
    int status;

    ImplInfo *impl_info = hashmap_get(&IMPL_MAP, &implh);
    DispatchHandle dh = impl_info->dh;
    if (OIF_DISPATCH_HANDLES[dh] == NULL) {
        fprintf(stderr,
                "[dispatch] Cannot call interface implementation "
                "for language '%s'\n",
                OIF_LANG_FROM_LANG_ID[dh]);
        exit(EXIT_FAILURE);
    }
    void *lib_handle = OIF_DISPATCH_HANDLES[dh];

    int (*call_impl_fn)(ImplInfo *, const char *, OIFArgs *, OIFArgs *);
    call_impl_fn = dlsym(lib_handle, "call_impl");
    if (call_impl_fn == NULL) {
        fprintf(stderr,
                "[dispatch] Could not load function 'call_impl' "
                "for language '%s'\n",
                OIF_LANG_FROM_LANG_ID[dh]);
        return -1;
    }
    status = call_impl_fn(impl_info, method, in_args, out_args);

    if (status) {
        fprintf(stderr,
                "[dispatch] ERROR: during execution of the function "
                "'%s::%s' an error occurred\n",
                impl_info->interface, method);
    }
    return status;
}

#if defined(__GNUC__)
#if !defined(__OPTIMIZE__)
void __attribute__((destructor))
dtor()
{
    fprintf(stderr, "[dispatch] WARNING: running non-optimized build\n");
}
#endif
#endif
