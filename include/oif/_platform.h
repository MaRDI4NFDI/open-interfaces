// clang-format Language: C
#pragma once

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
#if !defined(static_assert)
#define static_assert _Static_assert
#endif
#endif

#if defined(__SANITIZE_ADDRESS__)
#define OIF_SANITIZE_ADDRESS_ENABLED 1
#endif

#if defined(__clang__) || defined(__GNUC__)
#define OIF_ATTRIBUTE_NO_SANITIZE_ADDRESS __attribute__((no_sanitize_address))
#else
#define OIF_ATTRIBUTE_NO_SANITIZE_ADDRESS
#endif
// Usage example:
// OIF_ATTRIBUTE_NO_SANITIZE_ADDRESS
// void my_function()  {
//     ...Function implementation...
// }

#if defined(OIF_OPTION_VERBOSE_DEBUG_INFO)
// #define OIF_OPTION_VERBOSE_DEBUG_INFO
// #error "OIF_OPTION_VERBOSE_DEBUG_INFO is defined"
#else
// #error "OIF_OPTION_VERBOSE_DEBUG_INFO is NOT defined"
#endif
