// clang-format Language: C
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(static_assert)
#define static_assert _Static_assert
#endif

#ifdef __cplusplus
}
#if defined(__SANITIZE_ADDRESS__)
    #define OIF_SANITIZE_ADDRESS_ENABLED 1
#endif

#if defined(__clang__) || defined (__GNUC__)
# define OIF_ATTRIBUTE_NO_SANITIZE_ADDRESS __attribute__((no_sanitize_address))
#else
# define OIF_ATTRIBUTE_NO_SANITIZE_ADDRESS
#endif
// Usage example:
// OIF_ATTRIBUTE_NO_SANITIZE_ADDRESS
// void my_function()  {
//     ...Function implementation...
// }

