# jm_math
My custom math library that I use for developing games and graphics programs

Note: You only need simd_sincos.h if you want the simd sin() and cos() functions. Otherwise you can just use jm_math.h.
When including simd_sincos.h you need to #define JM_MATH_USE_SSE_TRANSCENDENTALS to include the code.

You can use the following defines _before_ include this file to change what is included

* JM_MATH_IMPLEMENTATION : will include the implementation. Otherwise this file acts like a regular header ile)
* JM_MATH_USE_SSE : will include sse functions
* JM_MATH_USE_AVX : will include avx functions
* JM_MATH_USE_SSE_TRANSCENDENTALS : Needed for some sse functions. Currently requires an external file for sin_ps() and cos_ps() implementations
* JM_USE_SIMD_ALL : include all simd related functions(may exclude transcendentals)
