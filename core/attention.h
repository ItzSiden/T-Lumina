#pragma once
#include <cmath>
#include <immintrin.h>
#include "kv_cache.h"

// ⚡ FIXED: HuggingFace Llama Exact RoPE Implementation
inline void apply_rope(float* q, float* k, int pos, int head_dim, int n_heads, int n_kv_heads) {
    int half_dim = head_dim / 2;
    for (int i = 0; i < half_dim; ++i) {
        float freq = 1.0f / std::pow(10000.0f, (float)(2 * i) / head_dim);
        float val = pos * freq;
        float cos_val = std::cos(val);
        float sin_val = std::sin(val);
        
        for (int h = 0; h < n_heads; ++h) {
            float q0 = q[h * head_dim + i];
            float q1 = q[h * head_dim + i + half_dim];
            q[h * head_dim + i] = q0 * cos_val - q1 * sin_val;
            q[h * head_dim + i + half_dim] = q0 * sin_val + q1 * cos_val;
        }
        for (int h = 0; h < n_kv_heads; ++h) {
            float k0 = k[h * head_dim + i];
            float k1 = k[h * head_dim + i + half_dim];
            k[h * head_dim + i] = k0 * cos_val - k1 * sin_val;
            k[h * head_dim + i + half_dim] = k0 * sin_val + k1 * cos_val;
        }
    }
}

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