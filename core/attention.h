#pragma once
#include <cmath>
#include <immintrin.h>
#include "kv_cache.h"

inline void fp32_matmul(const float* x, const float* W, float* y, int M, int N) {
    for (int i = 0; i < M; ++i) {
        float sum = 0.0f;
        const float* w_row = W + i * N;
        int j = 0;
        __m256 v_sum = _mm256_setzero_ps();
        for (; j <= N - 8; j += 8) {
            __m256 vx = _mm256_loadu_ps(x + j);
            __m256 vw = _mm256_loadu_ps(w_row + j);
            v_sum = _mm256_fmadd_ps(vx, vw, v_sum);
        }
        float buf[8];
        _mm256_storeu_ps(buf, v_sum);
        for (int k = 0; k < 8; ++k) sum += buf[k];
        for (; j < N; ++j) sum += x[j] * w_row[j];
        y[i] = sum;
    }
}

inline float quantized_dot_product(const float* q, const int8_t* k, float scale, int head_dim) {
    __m256 v_sum = _mm256_setzero_ps();
    int j = 0;
    for (; j <= head_dim - 8; j += 8) {
        __m256 vq = _mm256_loadu_ps(q + j);
        __m128i k8 = _mm_loadl_epi64((const __m128i*)(k + j));
        __m256i k32 = _mm256_cvtepi8_epi32(k8);
        __m256 vk = _mm256_cvtepi32_ps(k32);
        v_sum = _mm256_fmadd_ps(vq, vk, v_sum);
    }
    float buf[8];
    _mm256_storeu_ps(buf, v_sum);
    float sum = 0.0f;
    for (int i = 0; i < 8; ++i) sum += buf[i];
    for (; j < head_dim; ++j) sum += q[j] * static_cast<float>(k[j]);
    return sum * scale;
}