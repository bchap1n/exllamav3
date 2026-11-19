#include "avx512_target.h"

bool is_avx512_supported()
{
    static bool avx512_check = false;
    static bool avx512_supported = false;
    if (avx512_check) return avx512_supported;

#if !(defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
    // no AVX on this architecture
#elif defined(__linux__)
    // Check for AVX-512F and AVX-512BW support
    avx512_supported = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");
#else
    // Non-x86 (ARM): AVX-512 not applicable
    avx512_supported = false;
#endif

    avx512_check = true;
    // if (avx512_supported) printf("AVX-512 supported\n");
    // else printf("AVX-512 not supported\n");
    return avx512_supported;
}
