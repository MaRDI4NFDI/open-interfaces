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

#if defined(__GNUC__) && !defined(__OPTIMIZE__)
#define OIF_FLAG_PRINT_DEBUG_VERBOSE_INFO
#else
#endif
