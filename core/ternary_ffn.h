#pragma once
#include <cstdint>
#include <cmath>
#include <immintrin.h>

// -------------------------------------------------------
// Ternary matmul  y[M] = W_ternary[M×N] · x[N] × alpha
//
// W values are {-1, 0, +1} stored as int8_t.
// We exploit the ternary structure: multiply is just
// conditional-add, which maps cleanly to AVX2 masking.
// -------------------------------------------------------
inline void ternary_matmul_avx2(const float* __restrict x,
                                 const int8_t* __restrict W,
                                 float* __restrict y,
                                 int M, int N, float alpha = 1.0f) {
    const __m256i one  = _mm256_set1_epi32( 1);
    const __m256i mone = _mm256_set1_epi32(-1);

    for (int i = 0; i < M; ++i) {
        const int8_t* row = W + i * N;
        __m256 sum_pos = _mm256_setzero_ps();
        __m256 sum_neg = _mm256_setzero_ps();
        int j = 0;

        for (; j <= N - 8; j += 8) {
            __m256  vx  = _mm256_loadu_ps(x + j);
            __m256i w32 = _mm256_cvtepi8_epi32(
                              _mm_loadl_epi64((const __m128i*)(row + j)));

            __m256 mask_p = _mm256_castsi256_ps(_mm256_cmpeq_epi32(w32, one));
            __m256 mask_n = _mm256_castsi256_ps(_mm256_cmpeq_epi32(w32, mone));

            sum_pos = _mm256_add_ps(sum_pos, _mm256_and_ps(vx, mask_p));
            sum_neg = _mm256_add_ps(sum_neg, _mm256_and_ps(vx, mask_n));
        }

        float p[8], n[8];
        _mm256_storeu_ps(p, sum_pos);
        _mm256_storeu_ps(n, sum_neg);
        float s = 0.0f;
        for (int k = 0; k < 8; ++k) s += p[k] - n[k];

        // Scalar tail
        for (; j < N; ++j) {
            if      (row[j] ==  1) s += x[j];
            else if (row[j] == -1) s -= x[j];
        }

        y[i] = s * alpha;
    }
}

inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}
