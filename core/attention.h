#pragma once
#include <cmath>
#include <immintrin.h>
#include "kv_cache.h"

// -------------------------------------------------------
// RoPE — HuggingFace Llama exact variant
// Operates in-place on q[n_heads × head_dim]
//                    and k[n_kv_heads × head_dim]
// -------------------------------------------------------
inline void apply_rope(float* q, float* k, int pos,
                       int head_dim, int n_heads, int n_kv_heads) {
    const int half = head_dim / 2;
    for (int i = 0; i < half; ++i) {
        float freq    = 1.0f / std::pow(10000.0f, (2.0f * i) / head_dim);
        float theta   = pos * freq;
        float cos_val = std::cos(theta);
        float sin_val = std::sin(theta);

        for (int h = 0; h < n_heads; ++h) {
            float* qh = q + h * head_dim;
            float q0 = qh[i], q1 = qh[i + half];
            qh[i]        = q0 * cos_val - q1 * sin_val;
            qh[i + half] = q0 * sin_val + q1 * cos_val;
        }
        for (int h = 0; h < n_kv_heads; ++h) {
            float* kh = k + h * head_dim;
            float k0 = kh[i], k1 = kh[i + half];
            kh[i]        = k0 * cos_val - k1 * sin_val;
            kh[i + half] = k0 * sin_val + k1 * cos_val;
        }
    }
}

// -------------------------------------------------------
// Dense FP32 matmul  y[M] = W[M×N] · x[N]  (AVX2)
// -------------------------------------------------------
inline void fp32_matmul(const float* __restrict x,
                        const float* __restrict W,
                        float* __restrict y, int M, int N) {
    for (int i = 0; i < M; ++i) {
        const float* row = W + i * N;
        __m256 acc = _mm256_setzero_ps();
        int j = 0;
        for (; j <= N - 8; j += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + j),
                                  _mm256_loadu_ps(row + j), acc);
        float buf[8];
        _mm256_storeu_ps(buf, acc);
        float s = buf[0]+buf[1]+buf[2]+buf[3]+buf[4]+buf[5]+buf[6]+buf[7];
        for (; j < N; ++j) s += x[j] * row[j];
        y[i] = s;
    }
}

// -------------------------------------------------------
// Quantised dot-product: q[head_dim] · k_int8[head_dim]
// Returns the de-quantised, scaled score.
// -------------------------------------------------------
inline float quant_dot(const float* __restrict q,
                       const int8_t* __restrict k,
                       float scale, int head_dim) {
    __m256 acc = _mm256_setzero_ps();
    int j = 0;
    for (; j <= head_dim - 8; j += 8) {
        __m256  vq  = _mm256_loadu_ps(q + j);
        __m128i k8  = _mm_loadl_epi64((const __m128i*)(k + j));
        __m256  vk  = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(k8));
        acc = _mm256_fmadd_ps(vq, vk, acc);
    }
    float buf[8];
    _mm256_storeu_ps(buf, acc);
    float s = buf[0]+buf[1]+buf[2]+buf[3]+buf[4]+buf[5]+buf[6]+buf[7];
    for (; j < head_dim; ++j) s += q[j] * static_cast<float>(k[j]);
    return s * scale;
}
