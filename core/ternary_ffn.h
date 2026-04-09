#pragma once
#include <cstdint>
#include <cmath>
#include <immintrin.h>

inline void ternary_matmul_avx2(const float* __restrict x, const int8_t* __restrict W, float* __restrict y, int M, int N, float alpha = 1.0f) {
    for (int i = 0; i < M; ++i) {
        const int8_t* w_row = W + i * N;
        __m256 sum_pos = _mm256_setzero_ps();
        __m256 sum_neg = _mm256_setzero_ps();
        int j = 0;
        
        for (; j <= N - 8; j += 8) {
            __m256 vx = _mm256_loadu_ps(x + j);
            __m128i w8 = _mm_loadl_epi64((const __m128i*)(w_row + j));
            __m256i w32 = _mm256_cvtepi8_epi32(w8);

            __m256i mask_pos = _mm256_cmpeq_epi32(w32, _mm256_set1_epi32(1));
            __m256i mask_neg = _mm256_cmpeq_epi32(w32, _mm256_set1_epi32(-1));

            sum_pos = _mm256_add_ps(sum_pos, _mm256_and_ps(vx, _mm256_castsi256_ps(mask_pos)));
            sum_neg = _mm256_add_ps(sum_neg, _mm256_and_ps(vx, _mm256_castsi256_ps(mask_neg)));
        }

        float p[8], n[8];
        _mm256_storeu_ps(p, sum_pos);
        _mm256_storeu_ps(n, sum_neg);
        float s_pos = 0.0f, s_neg = 0.0f;
        for (int k = 0; k < 8; ++k) {
            s_pos += p[k];
            s_neg += n[k];
        }

        float remainder_s = 0.0f;
        for (; j < N; ++j) {
            int8_t weight = w_row[j];
            float val = x[j];
            float mask_p = (weight == 1) ? val : 0.0f;
            float mask_n = (weight == -1) ? val : 0.0f;
            remainder_s += (mask_p - mask_n);
        }
        
        y[i] = ((s_pos - s_neg) + remainder_s) * alpha;
    }
}

inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}