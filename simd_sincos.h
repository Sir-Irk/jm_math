/* SIMD (SSE1+MMX or SSE2) implementation of sin, cos, exp and log

   Inspired by Intel Approximate Math library, and based on the
   corresponding algorithms of the cephes math library

   The default is to use the SSE1 version. If you define USE_SSE2 the
   the SSE2 intrinsics will be used in place of the MMX intrinsics. Do
   not expect any significant performance improvement with SSE2.
  
   Modified by Jeremy Montgomery for AVX. Also removed mmx code and 
   cleaned up constants
*/

/* Copyright (C) 2007  Julien Pommier
 *

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

#include <xmmintrin.h>
#include <emmintrin.h>

#define _PS_CONST(name, val) \
	static const __m128 _ps_##name = _mm_set1_ps(val); \
	static const __m256 _256_ps_##name = _mm256_set1_ps(val);
#define _PS_HEX_CONST(name, val) \
	static const __m128 _ps_##name = _mm_castsi128_ps(_mm_set1_epi32(val)); \
	static const __m256 _256_ps_##name = _mm256_castsi256_ps(_mm256_set1_epi32(val)); 
#define _PI32_CONST(name, val) \
	static const __m128i _pi32_##name = _mm_set1_epi32(val); \
	static const __m256i _256_pi32_##name = _mm256_set1_epi32(val);

_PS_CONST(1, 1.0f);
_PS_CONST(0p5, 0.5f);
/* the smallest non denormalized float number */
_PS_HEX_CONST(min_norm_pos, 0x00800000);
_PS_HEX_CONST(mant_mask, 0x7f800000);
_PS_HEX_CONST(inv_mant_mask, ~0x7f800000);

_PS_HEX_CONST(sign_mask, (int)0x80000000);
_PS_HEX_CONST(inv_sign_mask, ~0x80000000);

_PI32_CONST(1, 1);
_PI32_CONST(inv1, ~1);
_PI32_CONST(2, 2);
_PI32_CONST(4, 4);
_PI32_CONST(0x7f, 0x7f);

_PS_CONST(cephes_SQRTHF, 0.707106781186547524f);
_PS_CONST(cephes_log_p0, 7.0376836292E-2f);
_PS_CONST(cephes_log_p1, -1.1514610310E-1f);
_PS_CONST(cephes_log_p2, 1.1676998740E-1f);
_PS_CONST(cephes_log_p3, -1.2420140846E-1f);
_PS_CONST(cephes_log_p4, +1.4249322787E-1f);
_PS_CONST(cephes_log_p5, -1.6668057665E-1f);
_PS_CONST(cephes_log_p6, +2.0000714765E-1f);
_PS_CONST(cephes_log_p7, -2.4999993993E-1f);
_PS_CONST(cephes_log_p8, +3.3333331174E-1f);
_PS_CONST(cephes_log_q1, -2.12194440e-4f);
_PS_CONST(cephes_log_q2, 0.693359375f);

/* natural logarithm computed for 4 simultaneous float
   return NaN for x <= 0
*/
__m128
log_ps(__m128 x) {
    __m128i emm0;
    __m128 one = _ps_1;

    __m128 invalid_mask = _mm_cmple_ps(x, _mm_setzero_ps());

    x = _mm_max_ps(x, _ps_min_norm_pos); /* cut off denormalized stuff */

    emm0     = _mm_srli_epi32(_mm_castps_si128(x), 23);
    /* keep only the fractional part */
    x = _mm_and_ps(x, _ps_inv_mant_mask);
    x = _mm_or_ps(x, _ps_0p5);

    emm0     = _mm_sub_epi32(emm0, _pi32_0x7f);
    __m128 e = _mm_cvtepi32_ps(emm0);
    e = _mm_add_ps(e, one);

    /* part2:
       if( x < SQRTHF ) {
         e -= 1;
         x = x + x - 1.0;
       } else { x = x - 1.0; }
    */
    __m128 mask = _mm_cmplt_ps(x, _ps_cephes_SQRTHF);
    __m128 tmp  = _mm_and_ps(x, mask);
    x           = _mm_sub_ps(x, one);
    e           = _mm_sub_ps(e, _mm_and_ps(one, mask));
    x           = _mm_add_ps(x, tmp);

    __m128 z = _mm_mul_ps(x, x);

    __m128 y = _ps_cephes_log_p0;
    y        = _mm_mul_ps(y, x);
    y        = _mm_add_ps(y, _ps_cephes_log_p1);
    y        = _mm_mul_ps(y, x);
    y        = _mm_add_ps(y, _ps_cephes_log_p2);
    y        = _mm_mul_ps(y, x);
    y        = _mm_add_ps(y, _ps_cephes_log_p3);
    y        = _mm_mul_ps(y, x);
    y        = _mm_add_ps(y, _ps_cephes_log_p4);
    y        = _mm_mul_ps(y, x);
    y        = _mm_add_ps(y, _ps_cephes_log_p5);
    y        = _mm_mul_ps(y, x);
    y        = _mm_add_ps(y, _ps_cephes_log_p6);
    y        = _mm_mul_ps(y, x);
    y        = _mm_add_ps(y, _ps_cephes_log_p7);
    y        = _mm_mul_ps(y, x);
    y        = _mm_add_ps(y, _ps_cephes_log_p8);
    y        = _mm_mul_ps(y, x);

    y = _mm_mul_ps(y, z);

    tmp = _mm_mul_ps(e, _ps_cephes_log_q1);
    y   = _mm_add_ps(y, tmp);

    tmp = _mm_mul_ps(z, _ps_0p5);
    y   = _mm_sub_ps(y, tmp);

    tmp = _mm_mul_ps(e, _ps_cephes_log_q2);
    x   = _mm_add_ps(x, y);
    x   = _mm_add_ps(x, tmp);
    x   = _mm_or_ps(x, invalid_mask); // negative arg will be NAN
    return x;
}

_PS_CONST(exp_hi, 88.3762626647949f);
_PS_CONST(exp_lo, -88.3762626647949f);

_PS_CONST(cephes_LOG2EF, 1.44269504088896341f);
_PS_CONST(cephes_exp_C1, 0.693359375f);
_PS_CONST(cephes_exp_C2, -2.12194440e-4f);

_PS_CONST(cephes_exp_p0, 1.9875691500E-4f);
_PS_CONST(cephes_exp_p1, 1.3981999507E-3f);
_PS_CONST(cephes_exp_p2, 8.3334519073E-3f);
_PS_CONST(cephes_exp_p3, 4.1665795894E-2f);
_PS_CONST(cephes_exp_p4, 1.6666665459E-1f);
_PS_CONST(cephes_exp_p5, 5.0000001201E-1f);

__m128
exp_ps(__m128 x) {
    __m128 tmp = _mm_setzero_ps(), fx;
    __m128i emm0;
    __m128 one = _ps_1;

    x = _mm_min_ps(x, _ps_exp_hi);
    x = _mm_max_ps(x, _ps_exp_lo);

    /* express exp(x) as exp(g + n*log(2)) */
    fx = _mm_mul_ps(x, _ps_cephes_LOG2EF);
    fx = _mm_add_ps(fx, _ps_0p5);

/* how to perform a floorf with SSE: just below */
    emm0         = _mm_cvttps_epi32(fx);
    tmp          = _mm_cvtepi32_ps(emm0);
    /* if greater, substract 1 */
    __m128 mask = _mm_cmpgt_ps(tmp, fx);
    mask        = _mm_and_ps(mask, one);
    fx          = _mm_sub_ps(tmp, mask);

    tmp      = _mm_mul_ps(fx, _ps_cephes_exp_C1);
    __m128 z = _mm_mul_ps(fx, _ps_cephes_exp_C2);
    x        = _mm_sub_ps(x, tmp);
    x        = _mm_sub_ps(x, z);

    z = _mm_mul_ps(x, x);

    __m128 y = _ps_cephes_exp_p0;
    y        = _mm_mul_ps(y, x);
    y        = _mm_add_ps(y, _ps_cephes_exp_p1);
    y        = _mm_mul_ps(y, x);
    y        = _mm_add_ps(y, _ps_cephes_exp_p2);
    y        = _mm_mul_ps(y, x);
    y        = _mm_add_ps(y, _ps_cephes_exp_p3);
    y        = _mm_mul_ps(y, x);
    y        = _mm_add_ps(y, _ps_cephes_exp_p4);
    y        = _mm_mul_ps(y, x);
    y        = _mm_add_ps(y, _ps_cephes_exp_p5);
    y        = _mm_mul_ps(y, z);
    y        = _mm_add_ps(y, x);
    y        = _mm_add_ps(y, one);

/* build 2^n */
    emm0         = _mm_cvttps_epi32(fx);
    emm0         = _mm_add_epi32(emm0, _pi32_0x7f);
    emm0         = _mm_slli_epi32(emm0, 23);
    __m128 pow2n = _mm_castsi128_ps(emm0);
    y = _mm_mul_ps(y, pow2n);
    return y;
}

_PS_CONST(minus_cephes_DP1, -0.78515625f);
_PS_CONST(minus_cephes_DP2, -2.4187564849853515625e-4f);
_PS_CONST(minus_cephes_DP3, -3.77489497744594108e-8f);
_PS_CONST(sincof_p0, -1.9515295891E-4f);
_PS_CONST(sincof_p1, 8.3321608736E-3f);
_PS_CONST(sincof_p2, -1.6666654611E-1f);
_PS_CONST(coscof_p0, 2.443315711809948E-005f);
_PS_CONST(coscof_p1, -1.388731625493765E-003f);
_PS_CONST(coscof_p2, 4.166664568298827E-002f);
_PS_CONST(cephes_FOPI, 1.27323954473516f); // 4 / M_PI

/* evaluation of 4 sines at onces, using only SSE1+MMX intrinsics so
   it runs also on old athlons XPs and the pentium III of your grand
   mother.

   The code is the exact rewriting of the cephes sinf function.
   Precision is excellent as long as x < 8192 (I did not bother to
   take into account the special handling they have for greater values
   -- it does not return garbage for arguments over 8192, though, but
   the extra precision is missing).

   Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
   surprising but correct result.

   Performance is also surprisingly good, 1.33 times faster than the
   macos vsinf SSE2 function, and 1.5 times faster than the
   __vrs4_sinf of amd's ACML (which is only available in 64 bits). Not
   too bad for an SSE1 function (with no special tuning) !
   However the latter libraries probably have a much better handling of NaN,
   Inf, denormalized and other special arguments..

   On my core 1 duo, the execution of this function takes approximately 95 cycles.

   From what I have observed on the experiments with Intel AMath lib, switching to an
   SSE2 version would improve the perf by only 10%.

   Since it is based on SSE intrinsics, it has to be compiled at -O2 to
   deliver full speed.
*/
__m128 sin_ps(__m128 x) { // any x
    __m128 xmm1, xmm2 = _mm_setzero_ps(), xmm3, sign_bit, y;

    __m128i emm0, emm2;
    sign_bit = x;
    /* take the absolute value */
    x = _mm_and_ps(x, _ps_inv_sign_mask);
    /* extract the sign bit (upper one) */
    sign_bit = _mm_and_ps(sign_bit, _ps_sign_mask);

    /* scale by 4/Pi */
    y = _mm_mul_ps(x, _ps_cephes_FOPI);

    /* store the integer part of y in mm0 */
    emm2 = _mm_cvttps_epi32(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    emm2 = _mm_add_epi32(emm2, _pi32_1);
    emm2 = _mm_and_si128(emm2, _pi32_inv1);
    y    = _mm_cvtepi32_ps(emm2);

    /* get the swap sign flag */
    emm0 = _mm_and_si128(emm2, _pi32_4);
    emm0 = _mm_slli_epi32(emm0, 29);
    /* get the polynom selection mask
       there is one polynom for 0 <= x <= Pi/4
       and another one for Pi/4<x<=Pi/2

       Both branches will be computed.
    */
    emm2 = _mm_and_si128(emm2, _pi32_2);
    emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());

    __m128 swap_sign_bit = _mm_castsi128_ps(emm0);
    __m128 poly_mask     = _mm_castsi128_ps(emm2);
    sign_bit             = _mm_xor_ps(sign_bit, swap_sign_bit);

    /* The magic pass: "Extended precision modular arithmetic"
       x = ((x - y * DP1) - y * DP2) - y * DP3; */
    xmm1 = _ps_minus_cephes_DP1;
    xmm2 = _ps_minus_cephes_DP2;
    xmm3 = _ps_minus_cephes_DP3;
    xmm1 = _mm_mul_ps(y, xmm1);
    xmm2 = _mm_mul_ps(y, xmm2);
    xmm3 = _mm_mul_ps(y, xmm3);
    x    = _mm_add_ps(x, xmm1);
    x    = _mm_add_ps(x, xmm2);
    x    = _mm_add_ps(x, xmm3);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    y        = _ps_coscof_p0;
    __m128 z = _mm_mul_ps(x, x);

    y          = _mm_mul_ps(y, z);
    y          = _mm_add_ps(y, _ps_coscof_p1);
    y          = _mm_mul_ps(y, z);
    y          = _mm_add_ps(y, _ps_coscof_p2);
    y          = _mm_mul_ps(y, z);
    y          = _mm_mul_ps(y, z);
    __m128 tmp = _mm_mul_ps(z, _ps_0p5);
    y          = _mm_sub_ps(y, tmp);
    y          = _mm_add_ps(y, _ps_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    __m128 y2 = _ps_sincof_p0;
    y2        = _mm_mul_ps(y2, z);
    y2        = _mm_add_ps(y2, _ps_sincof_p1);
    y2        = _mm_mul_ps(y2, z);
    y2        = _mm_add_ps(y2, _ps_sincof_p2);
    y2        = _mm_mul_ps(y2, z);
    y2        = _mm_mul_ps(y2, x);
    y2        = _mm_add_ps(y2, x);

    /* select the correct result from the two polynoms */
    xmm3 = poly_mask;
    y2   = _mm_and_ps(xmm3, y2); //, xmm3);
    y    = _mm_andnot_ps(xmm3, y);
    y    = _mm_add_ps(y, y2);
    /* update the sign */
    y = _mm_xor_ps(y, sign_bit);
    return y;
}

/* almost the same as sin_ps */
__m128 cos_ps(__m128 x) { // any x
    __m128 xmm1, xmm2 = _mm_setzero_ps(), xmm3, y;
    __m128i emm0, emm2;
    /* take the absolute value */
    x = _mm_and_ps(x, _ps_inv_sign_mask);

    /* scale by 4/Pi */
    y = _mm_mul_ps(x, _ps_cephes_FOPI);

    /* store the integer part of y in mm0 */
    emm2 = _mm_cvttps_epi32(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    emm2 = _mm_add_epi32(emm2, _pi32_1);
    emm2 = _mm_and_si128(emm2, _pi32_inv1);
    y    = _mm_cvtepi32_ps(emm2);

    emm2 = _mm_sub_epi32(emm2, _pi32_2);

    /* get the swap sign flag */
    emm0 = _mm_andnot_si128(emm2, _pi32_4);
    emm0 = _mm_slli_epi32(emm0, 29);
    /* get the polynom selection mask */
    emm2 = _mm_and_si128(emm2, _pi32_2);
    emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());

    __m128 sign_bit  = _mm_castsi128_ps(emm0);
    __m128 poly_mask = _mm_castsi128_ps(emm2);
    /* The magic pass: "Extended precision modular arithmetic"
       x = ((x - y * DP1) - y * DP2) - y * DP3; */
    xmm1 = _ps_minus_cephes_DP1;
    xmm2 = _ps_minus_cephes_DP2;
    xmm3 = _ps_minus_cephes_DP3;
    xmm1 = _mm_mul_ps(y, xmm1);
    xmm2 = _mm_mul_ps(y, xmm2);
    xmm3 = _mm_mul_ps(y, xmm3);
    x    = _mm_add_ps(x, xmm1);
    x    = _mm_add_ps(x, xmm2);
    x    = _mm_add_ps(x, xmm3);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    y        = _ps_coscof_p0;
    __m128 z = _mm_mul_ps(x, x);

    y          = _mm_mul_ps(y, z);
    y          = _mm_add_ps(y, _ps_coscof_p1);
    y          = _mm_mul_ps(y, z);
    y          = _mm_add_ps(y, _ps_coscof_p2);
    y          = _mm_mul_ps(y, z);
    y          = _mm_mul_ps(y, z);
    __m128 tmp = _mm_mul_ps(z, _ps_0p5);
    y          = _mm_sub_ps(y, tmp);
    y          = _mm_add_ps(y, _ps_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    __m128 y2 = _ps_sincof_p0;
    y2        = _mm_mul_ps(y2, z);
    y2        = _mm_add_ps(y2, _ps_sincof_p1);
    y2        = _mm_mul_ps(y2, z);
    y2        = _mm_add_ps(y2, _ps_sincof_p2);
    y2        = _mm_mul_ps(y2, z);
    y2        = _mm_mul_ps(y2, x);
    y2        = _mm_add_ps(y2, x);

    /* select the correct result from the two polynoms */
    xmm3 = poly_mask;
    y2   = _mm_and_ps(xmm3, y2); //, xmm3);
    y    = _mm_andnot_ps(xmm3, y);
    y    = _mm_add_ps(y, y2);
    /* update the sign */
    y = _mm_xor_ps(y, sign_bit);

    return y;
}

/* since sin_ps and cos_ps are almost identical, sincos_ps could replace both of them..
   it is almost as fast, and gives you a free cosine with your sine */
void
sincos_ps(__m128 x, __m128 *s, __m128 *c) {
    __m128 xmm1, xmm2, xmm3 = _mm_setzero_ps(), sign_bit_sin, y;
    __m128i emm0, emm2, emm4;
    sign_bit_sin = x;
    /* take the absolute value */
    x = _mm_and_ps(x, _ps_inv_sign_mask);
    /* extract the sign bit (upper one) */
    sign_bit_sin = _mm_and_ps(sign_bit_sin, _ps_sign_mask);

    /* scale by 4/Pi */
    y = _mm_mul_ps(x, _ps_cephes_FOPI);

    /* store the integer part of y in emm2 */
    emm2 = _mm_cvttps_epi32(y);

    /* j=(j+1) & (~1) (see the cephes sources) */
    emm2 = _mm_add_epi32(emm2, _pi32_1);
    emm2 = _mm_and_si128(emm2, _pi32_inv1);
    y    = _mm_cvtepi32_ps(emm2);

    emm4 = emm2;

    /* get the swap sign flag for the sine */
    emm0                     = _mm_and_si128(emm2, _pi32_4);
    emm0                     = _mm_slli_epi32(emm0, 29);
    __m128 swap_sign_bit_sin = _mm_castsi128_ps(emm0);

    /* get the polynom selection mask for the sine*/
    emm2             = _mm_and_si128(emm2, _pi32_2);
    emm2             = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());
    __m128 poly_mask = _mm_castsi128_ps(emm2);
    /* The magic pass: "Extended precision modular arithmetic"
       x = ((x - y * DP1) - y * DP2) - y * DP3; */
    xmm1 = _ps_minus_cephes_DP1;
    xmm2 = _ps_minus_cephes_DP2;
    xmm3 = _ps_minus_cephes_DP3;
    xmm1 = _mm_mul_ps(y, xmm1);
    xmm2 = _mm_mul_ps(y, xmm2);
    xmm3 = _mm_mul_ps(y, xmm3);
    x    = _mm_add_ps(x, xmm1);
    x    = _mm_add_ps(x, xmm2);
    x    = _mm_add_ps(x, xmm3);

    emm4                = _mm_sub_epi32(emm4, _pi32_2);
    emm4                = _mm_andnot_si128(emm4, _pi32_4);
    emm4                = _mm_slli_epi32(emm4, 29);
    __m128 sign_bit_cos = _mm_castsi128_ps(emm4);

    sign_bit_sin = _mm_xor_ps(sign_bit_sin, swap_sign_bit_sin);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    __m128 z = _mm_mul_ps(x, x);
    y        = _ps_coscof_p0;

    y          = _mm_mul_ps(y, z);
    y          = _mm_add_ps(y, _ps_coscof_p1);
    y          = _mm_mul_ps(y, z);
    y          = _mm_add_ps(y, _ps_coscof_p2);
    y          = _mm_mul_ps(y, z);
    y          = _mm_mul_ps(y, z);
    __m128 tmp = _mm_mul_ps(z, _ps_0p5);
    y          = _mm_sub_ps(y, tmp);
    y          = _mm_add_ps(y, _ps_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    __m128 y2 = _ps_sincof_p0;
    y2        = _mm_mul_ps(y2, z);
    y2        = _mm_add_ps(y2, _ps_sincof_p1);
    y2        = _mm_mul_ps(y2, z);
    y2        = _mm_add_ps(y2, _ps_sincof_p2);
    y2        = _mm_mul_ps(y2, z);
    y2        = _mm_mul_ps(y2, x);
    y2        = _mm_add_ps(y2, x);

    /* select the correct result from the two polynoms */
    xmm3         = poly_mask;
    __m128 ysin2 = _mm_and_ps(xmm3, y2);
    __m128 ysin1 = _mm_andnot_ps(xmm3, y);
    y2           = _mm_sub_ps(y2, ysin2);
    y            = _mm_sub_ps(y, ysin1);

    xmm1 = _mm_add_ps(ysin1, ysin2);
    xmm2 = _mm_add_ps(y, y2);

    /* update the sign */
    *s = _mm_xor_ps(xmm1, sign_bit_sin);
    *c = _mm_xor_ps(xmm2, sign_bit_cos);
}

__m256
sin256_ps(__m256 x) {
    __m256 ymm1;
    __m256 ymm2 = _mm256_setzero_ps();
    __m256 ymm3;
    __m256 sign_bit = x;
    __m256 y;

    /* take the absolute value */
    x = _mm256_and_ps(x, _256_ps_inv_sign_mask);
    /* extract the sign bit (upper one) */
    sign_bit = _mm256_and_ps(sign_bit, _256_ps_sign_mask);

    /* scale by 4/Pi */
    y = _mm256_mul_ps(x, _256_ps_cephes_FOPI);

    __m256i emm2 = _mm256_cvttps_epi32(y);
    emm2         = _mm256_add_epi32(emm2, _256_pi32_1);
    emm2         = _mm256_and_si256(emm2, _256_pi32_inv1);
    y            = _mm256_cvtepi32_ps(emm2);

    __m256i emm0;
    /* get the swap sign flag */
    emm0 = _mm256_and_si256(emm2, _256_pi32_4);
    emm0 = _mm256_slli_epi32(emm0, 29);
    /* get the polynom selection mask
       there is one polynom for 0 <= x <= Pi/4
       and another one for Pi/4<x<=Pi/2

       Both branches will be computed.
    */
    emm2 = _mm256_and_si256(emm2, _256_pi32_2);
    emm2 = _mm256_cmpeq_epi32(emm2, _mm256_setzero_si256());

    __m256 swap_sign_bit = _mm256_castsi256_ps(emm0);
    __m256 poly_mask     = _mm256_castsi256_ps(emm2);
    sign_bit             = _mm256_xor_ps(sign_bit, swap_sign_bit);

    /* The magic pass: "Extended precision modular arithmetic"
       x = ((x - y * DP1) - y * DP2) - y * DP3; */
    ymm1 = _256_ps_minus_cephes_DP1;
    ymm2 = _256_ps_minus_cephes_DP2;
    ymm3 = _256_ps_minus_cephes_DP3;
    ymm1 = _mm256_mul_ps(y, ymm1);
    ymm2 = _mm256_mul_ps(y, ymm2);
    ymm3 = _mm256_mul_ps(y, ymm3);
    x    = _mm256_add_ps(x, ymm1);
    x    = _mm256_add_ps(x, ymm2);
    x    = _mm256_add_ps(x, ymm3);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    y        = _256_ps_coscof_p0;
    __m256 z = _mm256_mul_ps(x, x);
    y          = _mm256_mul_ps(y, z);
    y          = _mm256_add_ps(y, _256_ps_coscof_p1);
    y          = _mm256_mul_ps(y, z);
    y          = _mm256_add_ps(y, _256_ps_coscof_p2);
    y          = _mm256_mul_ps(y, z);
    y          = _mm256_mul_ps(y, z);
    __m256 tmp = _mm256_mul_ps(z, _256_ps_0p5);
    y          = _mm256_sub_ps(y, tmp);
    y          = _mm256_add_ps(y, _256_ps_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */
	/* Note: this code assumes fmadd is available on the cpu */
    __m256 y2 = _256_ps_sincof_p0;
    y2 = _mm256_fmadd_ps(y2, z, _256_ps_sincof_p1);
    y2 = _mm256_fmadd_ps(y2, z, _256_ps_sincof_p2);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_fmadd_ps(y2, x, x);

    /* select the correct result from the two polynoms */
    ymm3 = poly_mask;
    y2   = _mm256_and_ps(ymm3, y2); //, xmm3);
    y    = _mm256_andnot_ps(ymm3, y);
    y    = _mm256_add_ps(y, y2);
    /* update the sign */
    y = _mm256_xor_ps(y, sign_bit);
    return y;
}
