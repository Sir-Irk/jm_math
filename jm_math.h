/*
Copyright (c) 2016 Jeremy F. Montgomery (jeremyfmontgomery@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated
documentation files (the "Software"), to deal in the Software without
restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of
the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
*/

#ifndef JM_MATH_H
#define JM_MATH_H

#ifndef NULL
#define NULL 0
#endif

#include <math.h>

// NOTE: You can Use the following defines _before_ include this file
//      to change what is included
//
//      1. JM_MATH_IMPLEMENTATION(will include the implementation. Otherwise
//      this file acts like a regular
//      header)
//      1. JM_MATH_USE_SSE (will include sse functions)
//      2. JM_MATH_USE_AVX (will include avx functions)
//      3. JM_MATH_USE_SSE_TRANSCENDENTALS (Needed for some sse functions.
//      Currently requires and external file
//         for sin_ps and cos_ps implementations)
//      4. JM_USE_SIMD_ALL (include all simd related functions)

#ifndef JM_TYPES
#define JM_TYPES

#include <stdint.h>

#define internal static
#define local_persist static
#define global static
#define global_variable static

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef i32 b32;

typedef float r32;
typedef double r64;

#endif // JM_TYPES

#ifdef JM_MATH_USE_SIMD_ALL
#define JM_MATH_USE_SSE
#define JM_MATH_USE_AVX
#define JM_MATH_USE_SSE_TRANSCENDENTALS
#define JM_MATH_USE_STDLIB
#endif // JM_MATH_USE_SIMD_ALL

#ifdef JM_MATH_USE_SSE
#include <emmintrin.h>
#include <immintrin.h>
#include <xmmintrin.h>
#endif // JM_MATH_USE_SSE

#ifdef JM_MATH_USE_SSE_TRANSCENDENTALS
#define USE_SSE2
#include "simd_sincos.h"
#endif // JM_MATH_USE_SSE_TRANSCENDENTALS

#define PI_32 3.14159265358979323846f

#define JM_SWAP(x, y, type)                                                                                     \
    do {                                                                                                        \
        type temp = x;                                                                                          \
        x         = y;                                                                                          \
        y         = temp;                                                                                       \
    } while (0)

#define JM_EPSILON 1e-4
#define JM_APPROX(x, y) (fabs((x) - (y)) <= JM_EPSILON)
#define JM_APPROX_EP(x, y, v) (JM_ABSOLUTE((x) - (y)) < (v))
#define JM_APPROX_V2(a, b) (JM_APPROX((a).x, (b).x) && JM_APPROX((a).y, (b).y))

#define JM_SQR(a) ((a) * (a))

union v2;
union v3;
union v4;
struct mat4x4;
struct aabb_2D;
struct aabb_3D;

// clang-format off

//==========================================================================================================
//
//NOTE: Scalar Declarations
//
//==========================================================================================================

//NOTE: degrees to radians
r32 jmToRadians(r32 angle);

// NOTE(Jeremy): These could be macros but that would present other issues like double eval.
inline i32 jmMin(i32 x, i32 y);
inline i32 jmMax(i32 x, i32 y);
inline i64 jmMin(i64 x, i64 y);
inline i64 jmMax(i64 x, i64 y);

inline r32 jmMin(r32 x, r32 y);
inline r32 jmMax(r32 x, r32 y);
inline r64 jmMin(r64 x, r64 y);
inline r64 jmMax(r64 x, r64 y);

inline r32 jmApproach(r32 goal, r32 current, r32 deltaTime);
inline r32 jmLerp(r32 start, r32 end, r32 percent);
inline r32 jmSinerp(r32 start, r32 end, r32 percent); //NOTE: Eases in
inline r32 jmCoserp(r32 start, r32 end, r32 percent); //NOTE: Eases out

inline i32 jmClamp(i32 current, i32 min, i32 max);
inline r32 jmClamp01(r32 current);
inline r32 jmClampReal(r32 current, r32 min, r32 max);

//NOTE: rounds 0.5 and greater up.
inline i32 jmRoundRealToInt(r32 num);

//NOTE: Remaps value from the range of min0-max0 to the range of min1-max1
inline r32 jmRemap(r32 value, r32 min0, r32 max0, r32 min1, r32 max1);

//NOTE: Mod that handles negatives correctly
inline i32 jmMod(i32 x, i32 factor);

//NOTE: Round a number based on the mod factor passed in(eg. round to the nearest 50)
inline i32 jmModRound(r32 x, i32 mod);

//NOTE: based on degrees
inline r32 jmRotationClamp(r32 rot);

//=========================================================================================================
//
// NOTE: v2 Declarations
//
//=========================================================================================================

union v2 {
    struct {
        r32 x, y;
    };
    r32 v[2];
};

global_variable v2 v2Down  = {0.0f, -1.0f};
global_variable v2 v2Up    = {0.0f, 1.0f};
global_variable v2 v2Right = {1.0f, 0.0f};
global_variable v2 v2Left  = {-1.0f, 0.0f};

inline v2 V2(r32 x, r32 y);

// NOTE: sets all components to value
inline v2 V2(r32 value);

// NOTE: result uses input's x and y and discards z
inline v2 V2(v3 a);

// NOTE: sets all components to "value"
inline v2 V2(r32 value);

// TODO: find out why MSVC(2013) fails to optimize without passing by reference on v2 but not v3
inline v2 operator+(v2 a, v2 b);
inline v2 operator-(v2 a, v2 b);
inline v2 operator*(v2 a, r32 scalar);
inline v2 operator-(v2 a);

inline v2  &operator+=(v2 &a, v2 b);
inline v2  &operator-=(v2 &a, v2 b);
inline v2  &operator*=(v2 &a, r32 scalar);
inline b32 operator==(v2 a, v2 b);

inline r32 jmLengthSqr(v2 a);
inline r32 jmLengthSqr(r32 x, r32 y);

inline r32 jmLength(v2 a);
inline r32 jmLength(r32 x, r32 y);

inline v2 jmNormalized(v2 a);
inline v2 jmNormalized(r32 x, r32 y);

inline void jmNormalize(v2 *a);

inline r32 jmDot(v2 a, v2 b);
inline r32 jmDot(r32 lhsX, r32 lhsY, r32 rhsX, r32 rhsY);

inline v2 jmLerp(v2 start, v2 end, r32 percent);
inline v2 jmLerp(r32 startX, r32 startY, r32 endX, r32 endY, r32 percent);

inline v2 jmSinerp(v2 start, v2 end, r32 percent);
inline v2 jmSinerp(r32 startX, r32 startY, r32 endX, r32 endY, r32 percent);

inline v2 jmNlerp(v2 start, v2 end, r32 percent);
inline v2 jmNlerp(r32 startX, r32 startY, r32 endX, r32 endY, r32 percent);

inline v2 jmSlerp(v2 start, v2 end, r32 percent);
inline v2 jmSlerp(r32 startX, r32 startY, r32 endX, r32 endY, r32 percent);

inline r32 jmRadiansBetween(v2 a, v2 b);
inline r32 jmRadiansBetween(r32 lhsX, r32 lhsY, r32 rhsX, r32 rhsY);

inline v2 jmRotateVectorRadians(v2 a, r32 rad);
inline v2 jmRotateVectorRadians(r32 x, r32 y, r32 rad);

inline v2 jmReflect(v2 a, v2 normal);
inline v2 jmReflect(r32 x, r32 y, v2 normal);

// NOTE: clamps a vector within an axis-aligned bounding box with the center of the box as the origin.
inline v2 jmClampInRect(v2 current, v2 min, v2 max);

// NOTE: returns a random point within a unit circle. Can also be looked at as a random directon
//       whose magnitude is between 0 and 1.
#ifdef JM_MATH_USE_STDLIB
inline v2 jmRandInUnitCircle(r32 scalar);
#endif

//=========================================================================================================
//
// NOTE: v3 Declarations
//
//=========================================================================================================

union v3 {
    struct {
        r32 x, y, z;
    };
    r32 v[3];
};

global v3 v3Up      = {0.0f, 1.0f, 0.0f};
global v3 v3Right   = {1.0f, 0.0f, 0.0f};
global v3 v3Forward = {0.0f, 0.0f, 1.0f};

inline v3 V3(r32 x, r32 y, r32 z);

//NOTE: copies a's x and y. Sets z to zero
inline v3 V3(v2 a);

//NOTE: sets all components to value
inline v3 V3(r32 value);

//NOTE: copies a's x and y. Sets z to "z" 
inline v3 V3(v2 a, r32 z);

//NOTE: copies a's x, y and z. Discards w
inline v3 V3(v4 a);

inline v3 operator+(v3 a, v3 b);
inline v3 operator-(v3 a, v3 b);
inline v3 operator*(v3 a, r32 scalar);
inline v3 operator-(v3 a);

inline v3 & operator+=(v3 &a, v3 b);
inline v3 & operator-=(v3 &a, v3 b);
inline v3 & operator*=(v3 &a, r32 scalar);

inline r32 jmLength(v3 a);
inline r32 jmLength(r32 x, r32 y, r32 z);

inline r32 jmLengthSqr(v3 a);
inline r32 jmLengthSqr(r32 x, r32 y, r32 z);

inline v3 jmNormalized(v3 a);
inline v3 jmNormalized(r32 x, r32 y, r32 z);

//NOTE: Normalizes in-place instead of copying
inline void jmNormalize(v3 *a);

inline r32 jmDot(v3 lhs, v3 rhs);
inline r32 jmDot(r32 lhsX, r32 lhsY, r32 lhsZ, r32 rhsX, r32 rhsY, r32 rhsZ);

inline v3  jmCross(v3 a, v3 b);
inline v3  jmCross(r32 lhsX, r32 lhsY, r32 lhsZ, r32 rhsX, r32 rhsY, r32 rhsZ);

inline v3  jmReflect(v3 a, v3 normal);
inline v3  jmReflect(r32 lhsX, r32 lhsY, r32 lhsZ, r32 rhsX, r32 rhsY, r32 rhsZ);

inline v3 jmLerp(v3 start, v3 end, r32 percent);
inline v3 jmLerp(r32 startX, r32 startY, r32 startZ, r32 endX, r32 endY, r32 endZ, r32 percent);

//NOTE: Eases in (speeds up as it approaches 1)
inline v3 jmSinerp(v3 start, v3 end, r32 percent);
inline v3 jmSinerp(r32 startX, r32 startY, r32 startZ, r32 endX, r32 endY, r32 endZ, r32 percent);

//NOTE: Eases out (slows down as it approaches 1)
inline v3 jmCoserp(v3 start, v3 end, r32 percent);
inline v3 jmCoserp(r32 startX, r32 startY, r32 startZ, r32 endX, r32 endY, r32 endZ, r32 percent);

inline v3 jmNlerp(v3 start, v3 end, r32 percent);
inline v3 jmNlerp(r32 startX, r32 startY, r32 startZ, r32 endX, r32 endY, r32 endZ, r32 percent);

inline v3 jmSlerp(v3 start, v3 end, r32 percent);
inline v3 jmSlerp(r32 startX, r32 startY, r32 startZ, r32 endX, r32 endY, r32 endZ, r32 percent);

//NOTE: returns a random point within a unit-sphere. Can also be looked at as a random direction whose
//      magnitude is between 0 and 1.
#ifdef JM_MATH_USE_STDLIB
inline v3 jmRandInUnitSphere(r32 scalar);
#endif

//NOTE: checks if a and b are almost equal. Threshold is based on JM_EPSILON
inline b32 jmApproximately(v3 a, v3 b);

//=========================================================================================================
//
//NOTE: v4 Declarations
//
//=========================================================================================================

union v4 {
    struct {
        r32 x, y, z, w;
    };
    struct {
        r32 r, g, b, a;
    };
    r32 v[4];
};

inline v4 V4(r32 x, r32 y, r32 z, r32 w);
inline v4 V4(v3, r32 z);

//NOTE: sets all components to value
inline v4 V4(r32 value);

//NOTE: Extracts color channels from a uint32_t and puts them in a v4
inline v4 V4(uint32_t color);

inline v4 operator*(v4 v, r32 s); 
inline v4 operator+(v4 v0, v4 v1); 
inline v4 operator-(v4 v0, v4 v1); 
inline v4 &operator+=(v4 &v0, v4 v1);
inline v4 &operator-=(v4 &v0, v4 v1);

inline v4 jmLerp(v4 start, v4 end, r32 percent);

//=========================================================================================================
//
//NOTE: mat3x3 Declarations
//
//=========================================================================================================

struct mat3x3 {
	r32 v[3][3];
};

//NOTE: returns upper left 3x3 of the input 4x4 matrix
inline mat3x3 Mat3x3(mat4x4 m);
inline mat3x3 Mat3x3(v3 a, v3 b, v3 c);

inline mat3x3 jmTanspose(mat3x3 m);
inline mat3x3 jmInverse(mat3x3 m);


//=========================================================================================================
//
//NOTE: mat4x4 Declarations
//
//=========================================================================================================

struct mat4x4 {
    r32 v[4][4];
};

inline mat4x4 jmGetUpper3x3(mat4x4 m);

inline v3 jmGetForward(mat4x4 m);
inline v3 jmGetUp(mat4x4 m);
inline v3 jmGetRight(mat4x4 m);

//NOTE: returns normalized vectors
inline v3 jmGetUnitForward(mat4x4 m);
inline v3 jmGetUnitUp(mat4x4 m);
inline v3 jmGetUnitRight(mat4x4 m);

inline v3 jmGetScale(mat4x4 m);
inline v3 jmGetPosition(mat4x4 m);

inline mat4x4 operator*(mat4x4 a, mat4x4 b);
inline v4     operator*(mat4x4 m, v4 v);

inline mat4x4 jmIdentityMatrix();
inline mat4x4 jmTranspose(mat4x4 m);
inline mat4x4 jmInverse(mat4x4 m);

inline mat4x4 jmScale(mat4x4 m, v3 scale);
inline mat4x4 jmScale(v3 scale);
inline mat4x4 jmScale(r32 scale);

inline mat4x4 jmTranslate(mat4x4 mat, v3 trans);
inline mat4x4 jmTranslate(v3 trans);

inline mat4x4 jmRotate(mat4x4 mat, r32 radians, v3 axis);
inline mat4x4 jmRotate(r32 radians, v3 axis);

inline mat4x4 jmPerspective(r32 fov, r32 aspectRatio, r32 near, r32 far);
inline mat4x4 jmOrthographic(r32 left, r32 right, r32 bottom, r32 top, r32 near, r32 far);

inline mat4x4 jmLookAt(v3 position, v3 target, v3 up);

//=========================================================================================================
//
//NOTE: AABB Declarations (Axis-Aligned Bounding Box)
//
//=========================================================================================================

struct aabb_2D {
    v2 min;
    v2 max;
};

inline aabb_2D operator+(aabb_2D a, v2 b);

// NOTE: Checks if A contains B
inline b32 jmAABBContains(aabb_2D a, v2 b);
inline b32 jmAABBContains(aabb_2D a, aabb_2D b);
inline b32 jmAABBIntersects(aabb_2D a, aabb_2D b);

internal b32 jmAABBClipLine(i32 d, aabb_2D aabb_2D, v2 v0, v2 v1, r32 *outLow, r32 high);
internal b32 jmAABBLineIntersection(aabb_2D aabb_2D, v2 v0, v2 v1, v2 *outIntersection, r32 *outFraction);
internal b32 jmAABBTraceLine(aabb_2D box, v2 boxPosition, v2 v0, v2 v1, v2 *intersection);

inline v2 jmAABBGetNormalFromIntersection(aabb_2D bounds, v2 worldPosition, v2 intersection);

//NOTE: ensures that min and max points are actually min and max
inline aabb_2D jmAABBCorrect(aabb_2D a);

inline aabb_2D jmAABBRotate90(aabb_2D a);

//TODO: create separate section
struct aabb_3D {
    v3 min;
    v3 max;
};

//=========================================================================================================
//
//NOTE: SSE Declarations 
//
//=========================================================================================================

#ifdef JM_USE_SSE
inline __m128 _vectorcall jmLengthSqrSSE(__m128 x, __m128 y);
inline __m128 _vectorcall jmLengthSSE(__m128 x, __m128 y);

inline __m128 _vectorcall jmLerpSSE(__m128 start, __m128 end, __m128 percent);
inline __m128 _vectorcall jmSinerpSSE(__m128 start, __m128 end, __m128 percent);
inline __m128 _vectorcall jmCoserpSSE(__m128 start, __m128 end, __m128 percent);

inline __m128 _vectorcall jmDotSSE(__m128 lhsX, __m128 lhsY, __m128 rhsX, __m128 rhsY);
inline __m128 _vectorcall jmDotSSE(__m128 lhsX, __m128 lhsY, __m128 lhsZ, __m128 rhsX, __m128 rhsY, __m128 rhsZ);

inline __m128 _vectorcall ZInTriangleSSE(__m128 px1, __m128 py1, __m128 pz1,
		                                 __m128 px2, __m128 py2, __m128 pz2,
			                             __m128 px3, __m128 py3, __m128 pz3,
			                             __m128 x, __m128 y);

#ifdef JM_USE_SSE_TRANSCENDENTALS 
inline __m128 _vectorcall jmSinerpSSE(__m128 start, __m128 end, __m128 percent);
inline __m128 _vectorcall jmCoserpSSE(__m128 start, __m128 end, __m128 percent);
#endif //JM_USE_SSE_TRANSCENDENTALS


#endif

//=========================================================================================================
//
//NOTE: AVX Declarations 
//
//=========================================================================================================

#ifdef JM_MATH_USE_AVX
inline __m256 _vectorcall jmNewtonRaphsonSqrtAVX(__m256 x);
inline __m256 _vectorcall jmLengthNewtRaphAVX(__m256 x, __m256 y);
inline __m256 _vectorcall jmLengthSqrAVX(__m256 x, __m256 y);
inline __m256 _vectorcall jmLengthAVX(__m256 x, __m256 y);
#endif

#ifdef JM_MATH_IMPLEMENTATION
//==========================================================================================================
//
// NOTE: Scalar Implementation
//
//==========================================================================================================

r32
jmToRadians(r32 angle) {
    r32 result = (PI_32 / 180) * angle;
    return result;
}

// NOTE(Jeremy): These could be macros but that would present other issues like
// double eval
inline i32
jmMin(i32 x, i32 y) {
    return (x < y ? x : y);
}

inline i32
jmMax(i32 x, i32 y) {
    return (x > y ? x : y);
}

inline i64
jmMin(i64 x, i64 y) {
    return (x < y ? x : y);
}

inline i64
jmMax(i64 x, i64 y) {
    return (x > y ? x : y);
}

inline r32
jmMin(r32 x, r32 y) {
    return fmin(x, y);
}

inline r32
jmMax(r32 x, r32 y) {
    return fmax(x, y);
}

inline r64
jmMin(r64 x, r64 y) {
    return fmin(x, y);
}

inline r64
jmMax(r64 x, r64 y) {
    return fmax(x, y);
}

inline r32
jmApproach(r32 goal, r32 current, r32 deltaTime) {
    r32 difference = goal - current;
    if (difference > deltaTime) {
        return current + deltaTime;
    }
    if (difference < -deltaTime) {
        return current - deltaTime;
    }
    return goal;
}

inline r32
jmLerp(r32 start, r32 end, r32 percent) {
    return start + ((end - start) * percent);
}

inline r32
jmSinerp(r32 start, r32 end, r32 percent) {
    return jmLerp(start, end, sinf(percent * PI_32 * 0.5f));
}

inline r32
jmCoserp(r32 start, r32 end, r32 percent) {
    return jmLerp(start, end, 1.0f - cosf(percent * PI_32 * 0.5f));
}

inline i32
jmClamp(i32 current, i32 min, i32 max) {
    return jmMax(min, jmMin(current, max));
}

inline r32
jmClamp01(r32 current) {
    if (current > 1.0f) return 1.0f;
    if (current < 0.0f) return 0.0f;
    return current;
}

inline r32
jmClampReal(r32 current, r32 min, r32 max) {
    return jmMax(min, jmMin(current, max));
}

// TODO: Fix this round function. Seems to give inconsistent results
inline i32
jmRoundRealToInt(r32 num) {
    // TODO: replace this with better round.
    i32 truncation = (i32)num;
    return (num - truncation >= 0.5f) ? truncation + 1 : truncation;
}

inline r32
jmRemap(r32 value, r32 min0, r32 max0, r32 min1, r32 max1) {
    return min1 + ((max1 - min1) * ((value - min0) / (max0 - min0)));
}

inline i32
jmMod(i32 x, i32 factor) {
    i32 result = x % factor;
    if (result < 0) result += factor;
    return result;
}

inline i32
jmModRound(r32 x, i32 mod) {
    i32 result = (i32)x;
    i32 modded = jmMod((i32)x, mod);
    result += (modded >= mod / 2) ? (mod - modded) : -modded;
    return result;
}

inline r32
jmRotationClamp(r32 rot) {
    r32 result = rot;
    if (rot > 360.0f)
        result -= 360.0f;
    else if (rot < 0.0f)
        result += 360.0f;
    return result;
}

//=========================================================================================================
//
// NOTE: V2 Implementation
//
//=========================================================================================================

inline v2
V2(r32 x, r32 y) {
    v2 result = {x, y};
    return result;
}

inline v2
V2(r32 a) {
    v2 result = {a, a};
    return result;
}

inline v2
V2(v3 a) {
    return V2(a.x, a.y);
}

inline v2
operator+(v2 a, v2 b) {
    v2 result = {a.x + b.x, a.y + b.y};
    return result;
}

inline v2
operator-(v2 a, v2 b) {
    v2 result = {a.x - b.x, a.y - b.y};
    return result;
}

inline v2 operator*(v2 a, r32 scalar) {
    v2 result = {a.x * scalar, a.y * scalar};
    return result;
}

inline v2
operator-(v2 a) {
    v2 result = {-a.x, -a.y};
    return result;
}

inline v2 &
operator+=(v2 &a, v2 b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}

inline v2 &
operator-=(v2 &a, v2 b) {
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

inline v2 &
operator*=(v2 &a, r32 scalar) {
    a.x *= scalar;
    a.y *= scalar;
    return a;
}

inline b32
operator==(v2 a, v2 b) {
    return (a.x == b.x && a.y == b.y);
}

inline r32
jmLengthSqr(v2 a) {
    return a.x * a.x + a.y * a.y;
}

inline r32
jmLengthSqr(r32 x, r32 y) {
    return x * x + y * y;
}

inline r32
jmLength(v2 a) {
    return sqrtf(jmLengthSqr(a));
}

inline r32
jmLength(r32 x, r32 y) {
    return sqrtf(jmLengthSqr(x, y));
}

inline v2
jmNormalized(r32 x, r32 y) {
    r32 len = jmLength(x, y);
    if (len == 0.0f) return {x, y};
    r32 invLen = 1.0f / len;
    v2 result  = {x * invLen, y * invLen};
    return result;
}

inline v2
jmNormalized(v2 a) {
    return jmNormalized(a.x, a.y);
}

inline void
jmNormalize(v2 *a) {
    *a = jmNormalized(*a);
}

inline r32
jmDot(v2 a, v2 b) {
    return a.x * b.x + a.y * b.y;
}

inline r32
jmDot(r32 lhsX, r32 lhsY, r32 rhsX, r32 rhsY) {
    return lhsX * rhsX + lhsY * rhsY;
}

inline v2
jmLerp(v2 start, v2 end, r32 percent) {
    return start + ((end - start) * percent);
}

inline v2
jmLerp(r32 startX, r32 startY, r32 endX, r32 endY, r32 percent) {
    v2 result = {jmLerp(startX, endX, percent), jmLerp(startY, endY, percent)};
    return result;
}

inline v2
jmSinerp(v2 start, v2 end, r32 percent) {
    return jmLerp(start, end, sinf(percent * PI_32 * 0.5f));
}

inline v2
jmSinerp(r32 startX, r32 startY, r32 endX, r32 endY, r32 percent) {
    v2 result = {jmSinerp(startX, endX, percent), jmSinerp(startY, endY, percent)};
    return result;
}

inline v2
jmCoserp(v2 start, v2 end, r32 percent) {
    return jmLerp(start, end, 1.0f - cosf(percent * PI_32 * 0.5f));
}

inline v2
jmCoserp(r32 startX, r32 startY, r32 endX, r32 endY, r32 percent) {
    v2 result = {jmCoserp(startX, endX, percent), jmCoserp(startY, endY, percent)};
    return result;
}

inline v2
jmNlerp(v2 start, v2 end, r32 percent) {
    return jmNormalized((start + ((end - start) * percent)));
}

inline v2
jmNlerp(r32 startX, r32 startY, r32 endX, r32 endY, r32 percent) {
    r32 x = jmLerp(startX, endX, percent);
    r32 y = jmLerp(startY, endY, percent);
    return jmNormalized(x, y);
}

inline v2
jmSlerp(v2 start, v2 end, r32 percent) {
    r32 dotP  = jmClampReal(jmDot(jmNormalized(start), jmNormalized(end)), -1.0f, 1.0f);
    r32 theta = acos(dotP) * percent;
    // TODO: is this supposed to be ((end - start) * dotP) or (end - (start *
    // dotP))
    v2 relative = jmNormalized(end - (start * dotP));
    return (start * cos(theta)) + (relative * sin(theta));
}

inline v2
jmSlerp(r32 startX, r32 startY, r32 endX, r32 endY, r32 percent) {
    r32 dotP  = jmClampReal(jmDot(jmNormalized(startX, startY), jmNormalized(endX, endY)), -1.0f, 1.0f);
    r32 theta = acos(dotP) * percent;
    // TODO: is this supposed to be ((end - start) * dotP) or (end - (start *
    // dotP))
    v2 relative = jmNormalized(endX - (startX * dotP), endY - (startY * dotP));

    r32 cosine = cos(theta);
    r32 sine   = sin(theta);
    return {startX * cosine + relative.x * sine, startY * cosine + relative.y * sine};
}

inline r32
jmRadiansBetween(v2 a, v2 b) {
    r32 result = atan2(b.y, b.x) - atan2(a.y, a.x);
    return result;
}

inline r32
jmRadiansBetween(r32 lhsX, r32 lhsY, r32 rhsX, r32 rhsY) {
    r32 result = atan2(rhsY, rhsX) - atan2(lhsY, lhsX);
    return result;
}

inline v2
jmRotateVectorRadians(v2 a, r32 rad) {
    v2 result;
    result.x = (a.x * cos(rad)) - (a.y * sin(rad));
    result.y = (a.y * cos(rad)) + (a.x * sin(rad));
    return result;
}

inline v2
jmRotateVectorRadians(r32 x, r32 y, r32 rad) {
    v2 result;
    result.x = (x * cos(rad)) - (y * sin(rad));
    result.y = (y * cos(rad)) + (x * sin(rad));
    return result;
}

inline v2
jmReflect(v2 a, v2 normal) {
    v2 result = a + (normal * (-2 * jmDot(a, normal)));
    return result;
}

inline v2
jmReflect(r32 x, r32 y, r32 normX, r32 normY) {
    r32 reflectionScale = -2 * jmDot(x, y, normX, normY);
    v2 result{x + (normX * reflectionScale), y + (normY * reflectionScale)};
    return result;
}

inline v2
jmClampInRect(v2 current, v2 min, v2 max) {
    v2 result;
    result.x = jmClampReal(current.x, min.x, max.x);
    result.y = jmClampReal(current.y, min.y, max.y);
    return result;
}

#define RAND48_SEED_0 (0x330e)
#define RAND48_SEED_1 (0xabcd)
#define RAND48_SEED_2 (0x1234)
#define RAND48_MULT_0 (0xe66d)
#define RAND48_MULT_1 (0xdeec)
#define RAND48_MULT_2 (0x0005)
#define RAND48_ADD (0x000b)

u16 _rand48_seed[3] = {RAND48_SEED_0, RAND48_SEED_1, RAND48_SEED_2};
u16 _rand48_mult[3] = {RAND48_MULT_0, RAND48_MULT_1, RAND48_MULT_2};
u16 _rand48_add     = RAND48_ADD;

void
_dorand48(u16 xseed[3]) {
    u64 accu;
    u16 temp[2];

    accu    = (u64)_rand48_mult[0] * (u64)xseed[0] + (u64)_rand48_add;
    temp[0] = (u16)accu; /* lower 16 bits */
    accu >>= sizeof(u16) * 8;
    accu += (u64)_rand48_mult[0] * (u64)xseed[1] + (u64)_rand48_mult[1] * (u64)xseed[0];
    temp[1] = (u16)accu; /* middle 16 bits */
    accu >>= sizeof(u16) * 8;
    accu += _rand48_mult[0] * xseed[2] + _rand48_mult[1] * xseed[1] + _rand48_mult[2] * xseed[0];
    xseed[0] = temp[0];
    xseed[1] = temp[1];
    xseed[2] = (u16)accu;
}

double
erand48(u16 xseed[3]) {
    _dorand48(xseed);
    return ldexp((double)xseed[0], -48) + ldexp((double)xseed[1], -32) + ldexp((double)xseed[2], -16);
}

double
drand48() {
    return erand48(_rand48_seed);
}

void
srand48(long seed) {
    _rand48_seed[0] = RAND48_SEED_0;
    _rand48_seed[1] = (u16)seed;
    _rand48_seed[2] = (u16)(seed >> 16);
    _rand48_mult[0] = RAND48_MULT_0;
    _rand48_mult[1] = RAND48_MULT_1;
    _rand48_mult[2] = RAND48_MULT_2;
    _rand48_add     = RAND48_ADD;
}

#ifdef JM_MATH_USE_STDLIB
inline v2
jmRandInUnitCircle(r32 scalar = 1.0f) {
    v2 result = {
        (r32)drand48() * 2.0f - 1.0f, (r32)drand48() * 2.0f - 1.0f,
    };
    return jmNormalized(result) * scalar * (r32)drand48();
}
#endif

//=========================================================================================================
//
// NOTE: v3 Implementation
//
//=========================================================================================================

inline v3
V3(r32 x, r32 y, r32 z) {
    return {x, y, z};
}

inline v3
V3(r32 value) {
    return {value, value, value};
}

inline v3
V3(v2 a) {
    return {a.x, a.y, 0.0f};
}

inline v3
V3(v2 a, r32 z) {
    return {a.x, a.y, z};
}

inline v3
V3(v4 a) {
    return {a.x, a.y, a.z};
}

inline v3
operator+(v3 a, v3 b) {
    v3 result = {a.x + b.x, a.y + b.y, a.z + b.z};
    return result;
}

inline v3
operator-(v3 a, v3 b) {
    v3 result = {a.x - b.x, a.y - b.y, a.z - b.z};
    return result;
}

inline v3 operator*(v3 a, r32 scalar) {
    v3 result = {a.x * scalar, a.y * scalar, a.z * scalar};
    return result;
}

inline v3
operator-(v3 a) {
    v3 result = {-a.x, -a.y, -a.z};
    return result;
}

inline v3 &
operator+=(v3 &a, v3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

inline v3 &
operator-=(v3 &a, v3 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

inline v3 &
operator*=(v3 &a, r32 scalar) {
    a.x *= scalar;
    a.y *= scalar;
    a.z *= scalar;
    return a;
}

inline r32
jmLength(v3 a) {
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

inline r32
jmLength(r32 x, r32 y, r32 z) {
    return sqrtf(x * x + y * y + z * z);
}

inline r32
jmLengthSqr(v3 a) {
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

inline r32
jmLengthSqr(r32 x, r32 y, r32 z) {
    return x * x + y * y + z * z;
}

inline v3
jmNormalized(v3 a) {
    r32 len = jmLength(a);
    if (len == 0.0f) return a;
    r32 invLen = 1.0f / len;
    v3 result  = {a.x * invLen, a.y * invLen, a.z * invLen};
    return result;
}

inline v3
jmNormalized(r32 x, r32 y, r32 z) {
    r32 len = jmLength(x, y, z);
    if (len == 0.0f) return {x, y, z};
    r32 invLen = 1.0f / len;
    v3 result  = {x * invLen, y * invLen, z * invLen};
    return result;
}

inline void
jmNormalize(v3 *a) {
    *a = jmNormalized(*a);
}

inline r32
jmDot(v3 lhs, v3 rhs) {
    return (lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z);
}

inline r32
jmDot(r32 lhsX, r32 lhsY, r32 lhsZ, r32 rhsX, r32 rhsY, r32 rhsZ) {
    return (lhsX * rhsX) + (lhsY * rhsY) + (lhsZ * rhsZ);
}

inline v3
jmLerp(v3 start, v3 end, r32 percent) {
    return start + ((end - start) * percent);
}

inline v3
jmLerp(r32 startX, r32 startY, r32 startZ, r32 endX, r32 endY, r32 endZ, r32 percent) {
    r32 x = jmLerp(startX, endX, percent);
    r32 y = jmLerp(startY, endY, percent);
    r32 z = jmLerp(startZ, endZ, percent);
    return {x, y, z};
}

inline v3
jmSinerp(v3 start, v3 end, r32 percent) {
    return jmLerp(start, end, (r32)sinf(percent * (r32)PI_32 * 0.5f));
}

inline v3
jmSinerp(r32 startX, r32 startY, r32 startZ, r32 endX, r32 endY, r32 endZ, r32 percent) {
    r32 x = jmSinerp(startX, endX, percent);
    r32 y = jmSinerp(startY, endY, percent);
    r32 z = jmSinerp(startZ, endZ, percent);
    return {x, y, z};
}

inline v3
jmCoserp(v3 start, v3 end, r32 percent) {
    return jmLerp(start, end, 1.0f - (r32)cosf(percent * (r32)PI_32 * 0.5f));
}

inline v3
jmCoserp(r32 startX, r32 startY, r32 startZ, r32 endX, r32 endY, r32 endZ, r32 percent) {
    r32 x = jmCoserp(startX, endX, percent);
    r32 y = jmCoserp(startY, endY, percent);
    r32 z = jmCoserp(startZ, endZ, percent);
    return {x, y, z};
}

inline v3
jmNlerp(v3 start, v3 end, r32 percent) {
    return jmNormalized(start + ((end - start) * percent));
}

inline v3
jmNlerp(r32 startX, r32 startY, r32 startZ, r32 endX, r32 endY, r32 endZ, r32 percent) {
    r32 x = jmLerp(startX, endX, percent);
    r32 y = jmLerp(startY, endY, percent);
    r32 z = jmLerp(startZ, endZ, percent);
    return jmNormalized(x, y, z);
}

inline v3
jmSlerp(v3 start, v3 end, r32 percent) {
    r32 dotP    = jmClampReal(jmDot(start, end), -1.0f, 1.0f);
    r32 theta   = acos(dotP) * percent;
    v3 relative = jmNormalized(end - (start * dotP));
    return ((start * cos(theta)) + (relative * sin(theta)));
}

inline v3
jmSlerp(r32 startX, r32 startY, r32 startZ, r32 endX, r32 endY, r32 endZ, r32 percent) {
    r32 dotP  = jmClampReal(jmDot(startX, startY, startZ, endX, endY, endZ), -1.0f, 1.0f);
    r32 theta = acos(dotP) * percent;

    r32 x  = endX - (startX * dotP);
    r32 y  = endY - (startY * dotP);
    r32 z  = endZ - (startZ * dotP);
    v3 rel = jmNormalized(endX, endY, endZ) * sinf(theta);

    r32 cosine = cosf(theta);
    r32 cX     = startX * cosine;
    r32 cY     = startY * cosine;
    r32 cZ     = startZ * cosine;

    return {cX + rel.x, cY + rel.y, cZ + rel.z};
}

inline v3
jmRandInUnitSphere(r32 scalar = 1.0f) {
    v3 result = {
        (r32)drand48() * 2.0f - 1.0f, (r32)drand48() * 2.0f - 1.0f, (r32)drand48() * 2.0f - 1.0f,
    };
    return jmNormalized(result) * scalar * (r32)drand48();
}

// clang-format off
inline v3
jmCross(v3 a, v3 b) {
    v3 result = { 
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x 
	};

	return result;
}
	
inline v3
jmCross(r32 lhsX, r32 lhsY, r32 lhsZ, r32 rhsX, r32 rhsY, r32 rhsZ) {
    v3 result = { 
		lhsY * rhsZ - lhsZ * rhsY,
		lhsZ * rhsX - lhsX * rhsZ,
		lhsX * rhsY - lhsY * rhsX 
	};

	return result;
}
// clang-format on

inline v3
jmReflect(v3 a, v3 normal) {
    v3 result = a + (normal * (-2 * jmDot(a, normal)));
    return result;
}

inline v3
jmReflect(r32 x, r32 y, r32 z, r32 normX, r32 normY, r32 normZ) {
    r32 reflectScale = -2 * jmDot(x, y, normX, normY);
    v3 result{x + (normX * reflectScale), y + (normY * reflectScale), z + (normZ * reflectScale)};
    return result;
}

inline b32
jmApproximately(v3 a, v3 b) {
    return (JM_APPROX(a.x, b.x) && JM_APPROX(a.y, b.y) && JM_APPROX(a.z, b.z));
}

//=========================================================================================================
//
// NOTE: v4 Implementation
//
//=========================================================================================================

inline v4
V4(r32 x, r32 y, r32 z, r32 w) {
    return {x, y, z, w};
}

inline v4
V4(v3 v, r32 z) {
    return {v.x, v.y, v.z, z};
}

inline v4
V4(r32 value) {
    return {value, value, value, value};
}

// clang-format off
inline v4
V4(uint32_t color)
{
    v4 result = {
		(r32)((color & 0x000000FF) >>  0), //R
		(r32)((color & 0x0000FF00) >>  8), //G
		(r32)((color & 0x00FF0000) >> 16), //B
		(r32)((color & 0xFF000000) >> 24), //A
	};

    return result;
}
// clang-format on

// TODO: change these to not use V4() call
#define jmWHITE (V4(1.0f, 1.0f, 1.0f, 1.0f))
#define jmBLACK (V4(0.0f, 0.0f, 0.0f, 1.0f))
#define jmRED (V4(1.0f, 0.0f, 0.0f, 1.0f))
#define jmGREEN (V4(0.0f, 1.0f, 0.0f, 1.0f))
#define jmBLUE (V4(0.0f, 0.0f, 1.0f, 1.0f))
#define jmYELLOW (V4(1.0f, 1.0f, 0.0f, 1.0f))
#define jmCYAN (V4(0.0f, 1.0f, 1.0f, 1.0f))
#define jmPINK (V4(1.0f, 0.0f, 1.0f, 1.0f))

inline v4
jmLerp(v4 start, v4 end, r32 percent) {
    v4 result = {};
    result.r  = jmLerp(start.r, end.r, percent);
    result.g  = jmLerp(start.g, end.g, percent);
    result.b  = jmLerp(start.b, end.b, percent);
    result.a  = jmLerp(start.a, end.a, percent);
    return result;
}

inline v4
jmSinerp(v4 start, v4 end, r32 percent) {
    return jmLerp(start, end, sinf(percent * PI_32 * 0.5f));
}

inline v4 operator*(v4 v, r32 s) { return V4(v.x * s, v.y * s, v.z * s, v.w * s); }

inline v4
operator+(v4 v0, v4 v1) {
    v4 result;
    result.x = v0.x + v1.x;
    result.y = v0.y + v1.y;
    result.z = v0.z + v1.z;
    result.a = v0.a + v1.a;
    return result;
}

inline v4
operator-(v4 v0, v4 v1) {
    v4 result;
    result.x = v0.x - v1.x;
    result.y = v0.y - v1.y;
    result.z = v0.z - v1.z;
    result.a = v0.a - v1.a;
    return result;
}

inline v4 &
operator+=(v4 &v0, v4 v1) {
    v0 = v0 + v1;
    return v0;
}

inline v4 &
operator-=(v4 &v0, v4 v1) {
    v0 = v0 - v1;
    return v0;
}

//=========================================================================================================
//
// NOTE: mat3x3 Implementation
//
//=========================================================================================================

// NOTE: returns upper left 3x3 of the input 4x4 matrix
inline mat3x3
Mat3x3(mat4x4 m) {
    mat3x3 result;

    result.v[0][0] = m.v[0][0];
    result.v[0][1] = m.v[0][1];
    result.v[0][2] = m.v[0][2];

    result.v[1][0] = m.v[1][0];
    result.v[1][1] = m.v[1][1];
    result.v[1][2] = m.v[1][2];

    result.v[2][0] = m.v[2][0];
    result.v[2][1] = m.v[2][1];
    result.v[2][2] = m.v[2][2];

    return result;
}

inline mat3x3
Mat3x3(v3 a, v3 b, v3 c) {
    mat3x3 result = {
        a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z,
    };

    return result;
}

inline v3 operator*(mat3x3 m, v3 v) {
    v3 result = {};
    for (i32 x = 0; x < 3; ++x) {
        r32 sum = 0;
        for (i32 y = 0; y < 3; ++y) {
            sum += m.v[y][x] * v.v[y];
        }
        result.v[x] = sum;
    }
    return result;
}

inline mat3x3
jmTranspose(mat3x3 m) {
    mat3x3 result;

    for (i32 x = 0; x < 3; ++x) {
        for (i32 y = 0; y < 3; ++y) {
            result.v[x][y] = m.v[y][x];
        }
    }

    return result;
}

inline mat3x3
jmInverse(mat3x3 m) {
    v3 a = {m.v[0][0], m.v[1][0], m.v[2][0]};
    v3 b = {m.v[0][1], m.v[1][1], m.v[2][1]};
    v3 c = {m.v[0][2], m.v[1][2], m.v[2][2]};

    v3 r0 = jmCross(b, c);
    v3 r1 = jmCross(c, a);
    v3 r2 = jmCross(a, b);

    r32 invDet = 1.0f / jmDot(r2, c);

    return Mat3x3(r0 * invDet, r1 * invDet, r2 * invDet);
}

//=========================================================================================================
//
// NOTE: mat4x4 Implementation
//
//=========================================================================================================

inline mat4x4
jmGetUpper3x3(mat4x4 m) {
    mat4x4 result  = m;
    result.v[0][3] = 0.0f;
    result.v[1][3] = 0.0f;
    result.v[2][3] = 0.0f;

    result.v[3][0] = 0.0f;
    result.v[3][1] = 0.0f;
    result.v[3][2] = 0.0f;
    result.v[3][3] = 0.0f;
    return result;
}

inline v3
jmGetUnitForward(mat4x4 m) {
    return jmNormalized(m.v[0][2], m.v[1][2], m.v[2][2]);
}

inline v3
jmGetUnitUp(mat4x4 m) {
    return jmNormalized(m.v[0][1], m.v[1][1], m.v[2][1]);
}

inline v3
jmGetUnitRight(mat4x4 m) {
    return jmNormalized(m.v[0][0], m.v[1][0], m.v[2][0]);
}

inline v3
jmGetForward(mat4x4 m) {
    return {m.v[0][2], m.v[1][2], m.v[2][2]};
}

inline v3
jmGetUp(mat4x4 m) {
    return {m.v[0][1], m.v[1][1], m.v[2][1]};
}

inline v3
jmGetRight(mat4x4 m) {
    return {m.v[0][0], m.v[1][0], m.v[2][0]};
}

inline v3
jmGetScale(mat4x4 m) {
    v3 result = {};
    result.x  = jmLength(jmGetForward(m));
    result.y  = jmLength(jmGetUp(m));
    result.z  = jmLength(jmGetForward(m));
    return result;
}

inline v3
jmGetPosition(mat4x4 m) {
    return {m.v[3][0], m.v[3][1], m.v[3][2]};
}

inline mat4x4 operator*(mat4x4 a, mat4x4 b) {
    mat4x4 result = {};
    for (i32 x = 0; x < 4; ++x) {
        for (i32 y = 0; y < 4; ++y) {
            r32 sum = 0;
            for (i32 i = 0; i < 4; ++i) {
                sum += a.v[i][y] * b.v[x][i];
            }
            result.v[x][y] = sum;
        }
    }
    return result;
}

inline v4 operator*(mat4x4 m, v4 v) {
    v4 result = {};
    for (i32 x = 0; x < 4; ++x) {
        r32 sum = 0;
        for (i32 y = 0; y < 4; ++y) {
            sum += m.v[y][x] * v.v[y];
        }
        result.v[x] = sum;
    }
    return result;
}

inline mat4x4
jmIdentityMatrix() {
    mat4x4 result = {};

    result.v[0][0] = 1.0f;
    result.v[1][1] = 1.0f;
    result.v[2][2] = 1.0f;
    result.v[3][3] = 1.0f;

    return result;
}

inline mat4x4
jmTranspose(mat4x4 m) {
    mat4x4 result;
    for (i32 y = 0; y < 4; ++y) {
        for (i32 x = 0; x < 4; ++x) {
            result.v[y][x] = m.v[x][y];
        }
    }
    return result;
}

// clang-format off
inline mat4x4
jmInverse(mat4x4 m) {
    mat4x4 r;

    r.v[0][0] = m.v[1][1] * m.v[2][2] * m.v[3][3] + m.v[1][2] * m.v[2][3] * m.v[3][1] +
                m.v[1][3] * m.v[2][1] * m.v[3][2] - m.v[1][1] * m.v[2][3] * m.v[3][2] -
                m.v[1][2] * m.v[2][1] * m.v[3][3] - m.v[1][3] * m.v[2][2] * m.v[3][1];
    r.v[0][1] = m.v[0][1] * m.v[2][3] * m.v[3][2] + m.v[0][2] * m.v[2][1] * m.v[3][3] +
                m.v[0][3] * m.v[2][2] * m.v[3][1] - m.v[0][1] * m.v[2][2] * m.v[3][3] -
                m.v[0][2] * m.v[2][3] * m.v[3][1] - m.v[0][3] * m.v[2][1] * m.v[3][2];
    r.v[0][2] = m.v[0][1] * m.v[1][2] * m.v[3][3] + m.v[0][2] * m.v[1][3] * m.v[3][1] +
                m.v[0][3] * m.v[1][1] * m.v[3][2] - m.v[0][1] * m.v[1][3] * m.v[3][2] -
                m.v[0][2] * m.v[1][1] * m.v[3][3] - m.v[0][3] * m.v[1][2] * m.v[3][1];
    r.v[0][3] = m.v[0][1] * m.v[1][3] * m.v[2][2] + m.v[0][2] * m.v[1][1] * m.v[2][3] +
                m.v[0][3] * m.v[1][2] * m.v[2][1] - m.v[0][1] * m.v[1][2] * m.v[2][3] -
                m.v[0][2] * m.v[1][3] * m.v[2][1] - m.v[0][3] * m.v[1][1] * m.v[2][2];

    r.v[1][0] = m.v[1][0] * m.v[2][3] * m.v[3][2] + m.v[1][2] * m.v[2][0] * m.v[3][3] +
                m.v[1][3] * m.v[2][2] * m.v[3][0] - m.v[1][0] * m.v[2][2] * m.v[3][3] -
                m.v[1][2] * m.v[2][3] * m.v[3][0] - m.v[1][3] * m.v[2][0] * m.v[3][2];
    r.v[1][1] = m.v[0][0] * m.v[2][2] * m.v[3][3] + m.v[0][2] * m.v[2][3] * m.v[3][0] +
                m.v[0][3] * m.v[2][0] * m.v[3][2] - m.v[0][0] * m.v[2][3] * m.v[3][2] -
                m.v[0][2] * m.v[2][0] * m.v[3][3] - m.v[0][3] * m.v[2][2] * m.v[3][0];
    r.v[1][2] = m.v[0][0] * m.v[1][3] * m.v[3][2] + m.v[0][2] * m.v[1][0] * m.v[3][3] +
                m.v[0][3] * m.v[1][2] * m.v[3][0] - m.v[0][0] * m.v[1][2] * m.v[3][3] -
                m.v[0][2] * m.v[1][3] * m.v[3][0] - m.v[0][3] * m.v[1][0] * m.v[3][2];
    r.v[1][3] = m.v[0][0] * m.v[1][2] * m.v[2][3] + m.v[0][2] * m.v[1][3] * m.v[2][0] +
                m.v[0][3] * m.v[1][0] * m.v[2][2] - m.v[0][0] * m.v[1][3] * m.v[2][2] -
                m.v[0][2] * m.v[1][0] * m.v[2][3] - m.v[0][3] * m.v[1][2] * m.v[2][0];

    r.v[2][0] = m.v[1][0] * m.v[2][1] * m.v[3][3] + m.v[1][1] * m.v[2][3] * m.v[3][0] +
                m.v[1][3] * m.v[2][0] * m.v[3][1] - m.v[1][0] * m.v[2][3] * m.v[3][1] -
                m.v[1][1] * m.v[2][0] * m.v[3][3] - m.v[1][3] * m.v[2][1] * m.v[3][0];
    r.v[2][1] = m.v[0][0] * m.v[2][3] * m.v[3][1] + m.v[0][1] * m.v[2][0] * m.v[3][3] +
                m.v[0][3] * m.v[2][1] * m.v[3][0] - m.v[0][0] * m.v[2][1] * m.v[3][3] -
                m.v[0][1] * m.v[2][3] * m.v[3][0] - m.v[0][3] * m.v[2][0] * m.v[3][1];
    r.v[2][2] = m.v[0][0] * m.v[1][1] * m.v[3][3] + m.v[0][1] * m.v[1][3] * m.v[3][0] +
                m.v[0][3] * m.v[1][0] * m.v[3][1] - m.v[0][0] * m.v[1][3] * m.v[3][1] -
                m.v[0][1] * m.v[1][0] * m.v[3][3] - m.v[0][3] * m.v[1][1] * m.v[3][0];
    r.v[2][3] = m.v[0][0] * m.v[1][3] * m.v[2][1] + m.v[0][1] * m.v[1][0] * m.v[2][3] +
                m.v[0][3] * m.v[1][1] * m.v[2][0] - m.v[0][0] * m.v[1][1] * m.v[2][3] -
                m.v[0][1] * m.v[1][3] * m.v[2][0] - m.v[0][3] * m.v[1][0] * m.v[2][1];

    r.v[3][0] = m.v[1][0] * m.v[2][2] * m.v[3][1] + m.v[1][1] * m.v[2][0] * m.v[3][2] +
                m.v[1][2] * m.v[2][1] * m.v[3][0] - m.v[1][0] * m.v[2][1] * m.v[3][2] -
                m.v[1][1] * m.v[2][2] * m.v[3][0] - m.v[1][2] * m.v[2][0] * m.v[3][1];
    r.v[3][1] = m.v[0][0] * m.v[2][1] * m.v[3][2] + m.v[0][1] * m.v[2][2] * m.v[3][0] +
                m.v[0][2] * m.v[2][0] * m.v[3][1] - m.v[0][0] * m.v[2][2] * m.v[3][1] -
                m.v[0][1] * m.v[2][0] * m.v[3][2] - m.v[0][2] * m.v[2][1] * m.v[3][0];
    r.v[3][2] = m.v[0][0] * m.v[1][2] * m.v[3][1] + m.v[0][1] * m.v[1][0] * m.v[3][2] +
                m.v[0][2] * m.v[1][1] * m.v[3][0] - m.v[0][0] * m.v[1][1] * m.v[3][0] -
                m.v[0][1] * m.v[1][2] * m.v[3][0] - m.v[0][2] * m.v[1][0] * m.v[3][1];
    r.v[3][3] = m.v[0][0] * m.v[1][1] * m.v[2][2] + m.v[0][1] * m.v[1][2] * m.v[2][0] +
                m.v[0][2] * m.v[1][0] * m.v[2][1] - m.v[0][0] * m.v[1][2] * m.v[2][1] -
                m.v[0][1] * m.v[1][0] * m.v[2][2] - m.v[0][2] * m.v[1][1] * m.v[2][0];

    return r;
}
// clang-format on

inline mat4x4
jmScale(mat4x4 m, v3 scale) {
    m.v[0][0] = scale.x;
    m.v[1][1] = scale.y;
    m.v[2][2] = scale.z;
    return m;
}

inline mat4x4
jmScale(v3 scale) {
    mat4x4 result  = jmIdentityMatrix();
    result.v[0][0] = scale.x;
    result.v[1][1] = scale.y;
    result.v[2][2] = scale.z;
    return result;
}

inline mat4x4
jmScale(r32 scale) {
    mat4x4 result  = jmIdentityMatrix();
    result.v[0][0] = scale;
    result.v[1][1] = scale;
    result.v[2][2] = scale;
    return result;
}

inline mat4x4
jmPerspective(r32 fov, r32 aspectRatio, r32 near, r32 far) {
    mat4x4 result = jmIdentityMatrix();

    r32 tanThetaOver2 = tanf(fov * ((r32)PI_32 / 360.0f));

    result.v[0][0] = 1.0f / tanThetaOver2;
    result.v[1][1] = (aspectRatio / tanThetaOver2);
    result.v[2][3] = -1.0f;
    result.v[2][2] = (near + far) / (near - far);
    result.v[3][2] = (2.0f * near * far) / (near - far);
    result.v[3][3] = 0.0f;

    return result;
}

inline mat4x4
jmOrthographic(r32 left, r32 right, r32 bottom, r32 top, r32 near, r32 far) {
    mat4x4 result = jmIdentityMatrix();

    result.v[0][0] = 2.0f / (right - left);
    result.v[1][1] = 2.0f / (top - bottom);
    result.v[2][2] = 2.0f / (near - far);

    result.v[3][0] = (left + right) / (left - right);
    result.v[3][1] = (bottom + top) / (bottom - top);
    result.v[3][2] = (far + near) / (near - far);

    return result;
}

inline mat4x4
jmLookAt(v3 position, v3 target, v3 up) {
    mat4x4 result = {0};

    v3 f = jmNormalized(target - position);
    v3 r = jmNormalized(jmCross(f, jmNormalized(up)));
    // v3 r = (jmCross(f, jmNormalized(up)));
    v3 u = jmCross(r, f);

    result.v[0][0] = r.x;
    result.v[0][1] = u.x;
    result.v[0][2] = -f.x;

    result.v[1][0] = r.y;
    result.v[1][1] = u.y;
    result.v[1][2] = -f.y;

    result.v[2][0] = r.z;
    result.v[2][1] = u.z;
    result.v[2][2] = -f.z;

    // TODO: Study this
    // result.v[3][0] = -jmDot(r, f);
    // result.v[3][1] = -jmDot(u, f);
    // result.v[3][2] = jmDot(f, f);

    result.v[3][0] = -jmDot(r, position);
    result.v[3][1] = -jmDot(u, position);
    result.v[3][2] = jmDot(f, position);

    result.v[3][3] = 1.0f;

    return result;
}

inline mat4x4
jmRotate(mat4x4 mat, r32 radians, v3 axis) {
    axis     = jmNormalized(axis);
    r32 sinT = sinf(radians);
    r32 cosT = cosf(radians);
    r32 cos  = 1.0f - cosT;

    mat.v[0][0] = (axis.x * axis.x * cos) + cosT;
    mat.v[0][1] = (axis.x * axis.y * cos) + (axis.z * sinT);
    mat.v[0][2] = (axis.x * axis.z * cos) - (axis.y * sinT);

    mat.v[1][0] = (axis.y * axis.x * cos) - (axis.z * sinT);
    mat.v[1][1] = (axis.y * axis.y * cos) + cosT;
    mat.v[1][2] = (axis.y * axis.z * cos) + (axis.x * sinT);

    mat.v[2][0] = (axis.z * axis.x * cos) + (axis.y * sinT);
    mat.v[2][1] = (axis.z * axis.y * cos) - (axis.x * sinT);
    mat.v[2][2] = (axis.z * axis.z * cos) + cosT;

    return mat;
}

inline mat4x4
jmRotate(r32 radians, v3 axis) {
    return jmRotate(jmIdentityMatrix(), radians, axis);
}

inline mat4x4
jmTranslate(mat4x4 mat, v3 trans) {
    mat.v[3][0] = trans.x;
    mat.v[3][1] = trans.y;
    mat.v[3][2] = trans.z;
    return mat;
}

inline mat4x4
jmTranslate(v3 trans) {
    return jmTranslate(jmIdentityMatrix(), trans);
}

//=========================================================================================================
//
// NOTE: AABB Implementation (Axis-Aligned Bounding Box)
//
//=========================================================================================================

inline aabb_2D
AABB(v2 min, v2 max) {
    aabb_2D result;
    result.min = min;
    result.max = max;
    return result;
}

inline aabb_2D
operator+(aabb_2D a, v2 b) {
    aabb_2D result = a;
    result.min     = b + a.min;
    result.max     = b + a.max;
    return result;
}

// NOTE: Checks if A contains B
inline b32
jmAABBContains(aabb_2D bounds, v2 point) {
    if (point.x > bounds.max.x || point.x < bounds.min.x) return false;
    if (point.y > bounds.max.y || point.y < bounds.min.y) return false;
    return true;
}

// NOTE: Checks to see if A contains B
inline b32
jmAABBContains(aabb_2D a, aabb_2D b) {
    for (i32 i = 0; i < 2; ++i) {
        if (b.min.v[i] < a.min.v[i]) return false;
        if (b.max.v[i] > a.max.v[i]) return false;
    }
    return true;
}

inline b32
jmAABBIntersects(aabb_2D a, aabb_2D b) {
    for (i32 i = 0; i < 2; i++) {
        if (a.min.v[i] > b.max.v[i]) return false;
        if (a.max.v[i] < b.min.v[i]) return false;
    }
    return true;
}

internal b32
jmAABBClipLine(i32 d, aabb_2D aabb, v2 v0, v2 v1, r32 *outLow, r32 high) {
    r32 low     = *outLow;
    r32 dimLow  = (aabb.min.v[d] - v0.v[d]) / (v1.v[d] - v0.v[d]);
    r32 dimHigh = (aabb.max.v[d] - v0.v[d]) / (v1.v[d] - v0.v[d]);
    if (dimHigh < dimLow) JM_SWAP(dimHigh, dimLow, r32);
    if (dimHigh < low) return false;
    if (dimLow > high) return false;

    low  = jmMax(dimLow, low);
    high = jmMin(dimHigh, high);

    if (low > high) return false;
    *outLow = low;
    return true;
}

internal b32
jmAABBLineIntersection(aabb_2D aabb, v2 v0, v2 v1, v2 *outIntersection, r32 *outFraction) {
    r32 low  = 0;
    r32 high = 1;

    if (!jmAABBClipLine(0, aabb, v0, v1, &low, high)) return false;
    if (!jmAABBClipLine(1, aabb, v0, v1, &low, high)) return false;

    v2 b             = v1 - v0;
    *outIntersection = v0 + b * low;
    *outFraction     = low;
    return true;
}

internal b32
jmAABBTraceLine(aabb_2D box, v2 boxPosition, v2 v0, v2 v1, v2 *intersection) {
    r32 lowestFraction = 1.0f;
    r32 testFraction;
    v2 testInter;

    if (jmAABBLineIntersection((box + boxPosition), v0, v1, &testInter, &testFraction) &&
        testFraction < lowestFraction) {
        *intersection  = testInter;
        lowestFraction = testFraction;
    }

    if (lowestFraction < 1) return true;

    return false;
}

#define IS_NEAR_REAL32(x, y) (fabsf((x) - (y)) < 0.00000001f)
#define IS_NEAR_ROUGH_REAL32(x, y) (fabsf((x) - (y)) < 1.0f)

inline v2
JmAABBGetNormalFromIntersection(aabb_2D bounds, v2 worldPosition, v2 intersection) {
    v2 normal = intersection - worldPosition;
    if (!IS_NEAR_ROUGH_REAL32(normal.x, bounds.min.x) && !IS_NEAR_ROUGH_REAL32(normal.x, bounds.max.x))
        normal.x = 0;
    if (!IS_NEAR_ROUGH_REAL32(normal.y, bounds.min.y) && !IS_NEAR_ROUGH_REAL32(normal.y, bounds.max.y))
        normal.y = 0;

#if 0
	if(normal.x > bounds.min.x && normal.x < bounds.max.x)
	{
		if(IS_NEAR(normal.y, bounds.min.y)) normal.y = -1;
		else normal.y = 1;
		normal .x = 0;
	}
	else
	{
		if(IS_NEAR(normal.x, bounds.min.x)) normal.x = -1;
		else normal.x = 1;
		normal.y = 0;
	}
#endif

    return jmNormalized(normal);
}

inline aabb_2D
jmAABBCorrect(aabb_2D a) {
    aabb_2D result;
    result.min = V2(-fabsf(a.min.x), -fabsf(a.min.y));
    result.max = V2(fabsf(a.max.x), fabsf(a.max.y));
    return result;
}

inline aabb_2D
jmAABBRotate90(aabb_2D a) {
    aabb_2D result;
    result.min = V2(-a.min.y, a.min.x);
    result.max = V2(-a.max.y, a.max.x);
    return jmAABBCorrect(result);
}

// TODO: move to own section
inline aabb_3D
operator+(aabb_3D a, v3 v) {
    aabb_3D result = {a.min + v, a.max + v};

    return result;
}

inline aabb_3D
jmAABB3DScale(aabb_3D a, v3 scale) {
    aabb_3D result = {
        {a.min.x * scale.x, a.min.y * scale.y, a.min.z * scale.z},
        {a.max.x * scale.x, a.max.y * scale.y, a.max.z * scale.z},
    };

    return result;
}

internal b32
jmAABB3DClipLine(i32 d, aabb_3D aabb, v3 v0, v3 v1, r32 *outLow, r32 high) {
    r32 low     = *outLow;
    r32 dimLow  = (aabb.min.v[d] - v0.v[d]) / (v1.v[d] - v0.v[d]);
    r32 dimHigh = (aabb.max.v[d] - v0.v[d]) / (v1.v[d] - v0.v[d]);
    if (dimHigh < dimLow) JM_SWAP(dimHigh, dimLow, r32);
    if (dimHigh < low) return false;
    if (dimLow > high) return false;

    low  = jmMax(dimLow, low);
    high = jmMin(dimHigh, high);

    if (low > high) return false;
    *outLow = low;
    return true;
}

internal b32
jmAABB3DLineIntersection(aabb_3D aabb, v3 v0, v3 v1, v3 *outIntersection, r32 *outFraction) {
    r32 low  = 0;
    r32 high = 1;

    if (!jmAABB3DClipLine(0, aabb, v0, v1, &low, high)) return false;
    if (!jmAABB3DClipLine(1, aabb, v0, v1, &low, high)) return false;
    if (!jmAABB3DClipLine(2, aabb, v0, v1, &low, high)) return false;

    v3 b             = v1 - v0;
    *outIntersection = v0 + b * low;
    *outFraction     = low;
    return true;
}

internal b32
jmAABB3DTraceLine(aabb_3D box, v3 boxPosition, v3 v0, v3 v1, v3 *intersection = NULL) {
    r32 lowestFraction = 1.0f;
    r32 testFraction;
    v3 testInter;

    if (jmAABB3DLineIntersection((box + boxPosition), v0, v1, &testInter, &testFraction) &&
        testFraction < lowestFraction) {
        if (intersection) *intersection = testInter;
        lowestFraction                  = testFraction;
    }

    if (lowestFraction < 1) return true;

    return false;
}

//=========================================================================================================
//
// NOTE: SSE Implementation
//
//=========================================================================================================

#ifdef JM_MATH_USE_SSE

#define MM_V2_LOAD_X(m1, m2) _mm_shuffle_ps((m1), (m2), _MM_SHUFFLE(2, 0, 2, 0))
#define MM_V2_LOAD_Y(m1, m2) _mm_shuffle_ps((m1), (m2), _MM_SHUFFLE(3, 1, 3, 1))

#define MM256_V2_LOAD_X(m1, m2) _mm_shuffle_ps((m1), (m2), _MM256_SHUFFLE(2, 0, 2, 0, 2, 0, 2, 0))
#define MM256_V2_LOAD_Y(m1, m2) _mm_shuffle_ps((m1), (m2), _MM256_SHUFFLE(3, 1, 3, 1, 3, 1, 3, 1))

#define MM_CLEAR_INVALIDS(x)                                                                                    \
    do {                                                                                                        \
        __m128 _test = _mm_mul_ps(x, _mm_setzero_ps());                                                         \
        _test        = _mm_cmpeq_ps(_test, _mm_setzero_ps());                                                   \
        x            = _mm_and_ps(x, _test);                                                                    \
    } while (0)

#define MM_V2_AOS_TO_SOA(p1, x, y, ptr)                                                                         \
    p1 = _mm_load_ps(ptr);                                                                                      \
    x  = MM_V2_LOAD_X(p1, p1);                                                                                  \
    y  = MM_V2_LOAD_Y(p1, p1);

#define MM_V2_AOS_TO_SOA_2(p1, p2, x, y, ptr)                                                                   \
    p1 = _mm_load_ps(ptr);                                                                                      \
    p2 = _mm_load_ps((r32 *)((v2 *)ptr + 2));                                                                   \
    x  = MM_V2_LOAD_X(p1, p2);                                                                                  \
    y  = MM_V2_LOAD_Y(p1, p2);

#define MM_V2_SOA_TO_AOS_2(p1, p2, x, y)                                                                        \
    p1 = _mm_shuffle_ps(x, y, _MM_SHUFFLE(1, 0, 1, 0));                                                         \
    p1 = _mm_shuffle_ps(p1, p1, _MM_SHUFFLE(3, 1, 2, 0));                                                       \
    p2 = _mm_shuffle_ps(x, y, _MM_SHUFFLE(3, 2, 3, 2));                                                         \
    p2 = _mm_shuffle_ps(p2, p2, _MM_SHUFFLE(3, 1, 2, 0));

#define MM_V2_NORMALIZE(x, y, len)                                                                              \
    {                                                                                                           \
        __m128 mask = _mm_cmpeq_ps(_mm_set_ps1(0.0f), len);                                                     \
        x           = _mm_div_ps(x, len);                                                                       \
        x           = _mm_andnot_ps(mask, x);                                                                   \
        y           = _mm_div_ps(y, len);                                                                       \
        y           = _mm_andnot_ps(mask, y);                                                                   \
    }

#define MM256_V2_NORMALIZE(x, y, len)                                                                           \
    {                                                                                                           \
        __m256 mask = _mm256_cmp_ps(_mm256_setzero_ps(), len, _CMP_EQ_OQ);                                      \
        x           = _mm256_div_ps(x, len);                                                                    \
        x           = _mm256_andnot_ps(mask, x);                                                                \
        y           = _mm256_div_ps(y, len);                                                                    \
        y           = _mm256_andnot_ps(mask, y);                                                                \
    }

// clang-format off
inline __m128 _vectorcall
jmLengthSqrSSE(__m128 x, __m128 y) {
     return _mm_add_ps(_mm_mul_ps(x, x), _mm_mul_ps(y, y));
}

inline __m128 _vectorcall
jmLengthSSE(__m128 x, __m128 y) {
    return _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(x, x), _mm_mul_ps(y, y)));
}

inline __m128 _vectorcall 
jmLerpSSE(__m128 start, __m128 end, __m128 percent) {
    return _mm_add_ps(start, _mm_mul_ps(_mm_sub_ps(end, start), percent));
}

#ifdef JM_USE_SSE_TRANSCENDENTALS
inline __m128 _vectorcall 
jmSinerpSSE(__m128 start, __m128 end, __m128 percent) {
    __m128 piFactor = _mm_set_ps1(PI_32 * 0.5f);
    __m128 sine     = sin_ps(_mm_mul_ps(percent, piFactor));
    __m128 result   = jmLerpSSE(start, end, sine);
    return result;
}
inline __m128 _vectorcall 
jmCoserpSSE(__m128 start, __m128 end, __m128 percent) {
    __m128 piFactor = _mm_set_ps1(PI_32 * 0.5f);
    __m128 cosine   = cos_ps(_mm_mul_ps(piFactor, percent));
    __m128 result   = jmLerpSSE(start, end, cosine);
    return result;
}
#endif JM_USE_SIMD_TRANS

inline __m128 _vectorcall 
jmDotSSE(__m128 lhsX, __m128 lhsY, __m128 rhsX, __m128 rhsY) {
	__m128 x = _mm_mul_ps(lhsX, rhsX);
	__m128 y = _mm_mul_ps(lhsY, rhsY);
	__m128 result = _mm_add_ps(x, y);
	return result;
}
	
inline __m128 _vectorcall 
jmDotSSE(__m128 lhsX, __m128 lhsY, __m128 lhsZ, __m128 rhsX, __m128 rhsY, __m128 rhsZ) {
	__m128 x = _mm_mul_ps(lhsX, rhsX);
	__m128 y = _mm_mul_ps(lhsY, rhsY);
	__m128 z = _mm_mul_ps(lhsZ, rhsZ);
	__m128 result = _mm_add_ps(_mm_add_ps(x, y), z);
	return result;
}

inline __m128 _vectorcall
ZInTriangleSSE(__m128 px1, __m128 py1, __m128 pz1,
		       __m128 px2, __m128 py2, __m128 pz2,
			   __m128 px3, __m128 py3, __m128 pz3,
			   __m128 x, __m128 y) 
{
	__m128 prod1  = _mm_mul_ps(_mm_sub_ps(py2, py3), _mm_sub_ps(px1, px3));
	__m128 prod2  = _mm_mul_ps(_mm_sub_ps(px3, px2), _mm_sub_ps(py1, py3));
	__m128 invDet = _mm_div_ps(_mm_set1_ps(1.0f), _mm_add_ps(prod1, prod2));

	__m128 l1prod1 = _mm_mul_ps(_mm_sub_ps(py2, py3), _mm_sub_ps(x, px3));
	__m128 l1prod2 = _mm_mul_ps(_mm_sub_ps(px3, px2), _mm_sub_ps(y, py3));
	__m128 l1      = _mm_mul_ps(_mm_add_ps(l1prod1, l1prod2), invDet);

	__m128 l2prod1 = _mm_mul_ps(_mm_sub_ps(py3, py1), _mm_sub_ps(x, px3));
	__m128 l2prod2 = _mm_mul_ps(_mm_sub_ps(px1, px3), _mm_sub_ps(y, py3));
	__m128 l2      = _mm_mul_ps(_mm_add_ps(l2prod1, l2prod2), invDet);
	__m128 l3      = _mm_sub_ps(_mm_sub_ps(_mm_set1_ps(1.0f), l1), l2); 

	__m128 result = _mm_add_ps(_mm_add_ps(_mm_mul_ps(l1, pz1), _mm_mul_ps(l2, pz2)), _mm_mul_ps(l3, pz3));
	return result;
}

#endif //JM_USE_SSE
// clang-format on

//=============================================================================================================
// 
//NOTE: AVX Implementation
// 
//=============================================================================================================

#ifdef JM_MATH_USE_AVX
inline __m256 _vectorcall
jmNewtonRaphsonSqrtAVX(__m256 x) {
	__m256 nr = _mm256_rsqrt_ps(x);
	__m256 xnr = _mm256_mul_ps(x, nr);
	__m256 hnr  = _mm256_mul_ps(_mm256_set1_ps(0.5f), nr);
	__m256 result  = _mm256_fnmadd_ps(xnr, nr, _mm256_set1_ps(3.0f));
	result = _mm256_mul_ps(hnr, result);
	return result;
}

inline __m256 _vectorcall jmLengthNewtRaphAVX(__m256 x, __m256 y) {
    return jmNewtonRaphsonSqrtAVX(_mm256_add_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y)));
}

inline __m256 _vectorcall
jmLengthSqrAVX(__m256 x, __m256 y) {
     //return _mm256_add_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y));
     return _mm256_fmadd_ps(x, x, _mm256_mul_ps(y, y));
}

inline __m256 _vectorcall
jmLengthAVX(__m256 x, __m256 y) {
    return _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y)));
}

#define MM256_CLEAR_INVALIDS(x)                                                                                 \
    do {                                                                                                        \
        __m256 _test = _mm256_mul_ps(x, _mm256_setzero_ps());                                                   \
        _test        = _mm256_cmp_ps(_test, _mm256_setzero_ps(), _CMP_EQ_OQ);                                   \
        x            = _mm256_and_ps(x, _test);                                                                 \
    } while (0)

#define MM256_V2_NORMALIZE(x, y, len)                                                                           \
    {                                                                                                           \
        __m256 mask = _mm256_cmp_ps(_mm256_setzero_ps(), len, _CMP_EQ_OQ);                                      \
        x           = _mm256_div_ps(x, len);                                                                    \
        x           = _mm256_andnot_ps(mask, x);                                                                \
        y           = _mm256_div_ps(y, len);                                                                    \
        y           = _mm256_andnot_ps(mask, y);                                                                \
    }

#endif //JM_USE_AVX


#if 0 
inline b32
jmResolveCollision(v2 posA, v2 posB, r32 massA, r32 massB, v2 *velA, v2 *velB, r32 width) {
    v2 between      = posA - posB;
    r32 distSquared = jmLengthSqr(between);
    if (distSquared <= width * width) {
        v2 newVel = *velB - *velA;
        r32 dotP  = jmDot(between, newVel);
        if (dotP >= 0.0f) {
            r32 scale     = dotP / distSquared;
            v2 collision  = between * scale;
            r32 totalMass = massA + massB;
            r32 weightA   = 2 * massB / totalMass;
            r32 weightB   = 2 * massA / totalMass;

            *velA += (collision * weightA);
            *velB -= (collision * weightB);
            return true;
        }
    }
    return false;
}
#endif

#endif // JM_MATH_IMPLEMENTATION

#endif // JM_MATH_H
