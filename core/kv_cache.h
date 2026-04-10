#pragma once
#include <cstdint>
#include <cmath>
#include <algorithm>

// -------------------------------------------------------
// Per-layer KV cache (INT8-quantised, per-position scale)
// -------------------------------------------------------
struct LayerKVCache {
    int8_t* k_cache;   // [max_len × kv_dim]
    int8_t* v_cache;
    float*  k_scales;  // [max_len]
    float*  v_scales;

    int max_len;
    int kv_dim;

    LayerKVCache(int max_l, int kv_d) : max_len(max_l), kv_dim(kv_d) {
        k_cache  = new int8_t[max_len * kv_dim]();
        v_cache  = new int8_t[max_len * kv_dim]();
        k_scales = new float[max_len]();
        v_scales = new float[max_len]();
    }

    ~LayerKVCache() {
        delete[] k_cache;
        delete[] v_cache;
        delete[] k_scales;
        delete[] v_scales;
    }

    // Quantise and store a (k, v) pair at position pos
    void update(int pos, const float* k, const float* v) {
        auto quantise = [&](const float* src, int8_t* dst, float& scale_out) {
            float max_abs = 1e-8f;
            for (int i = 0; i < kv_dim; ++i)
                max_abs = std::max(max_abs, std::abs(src[i]));
            scale_out = max_abs / 127.0f;
            float inv = 1.0f / scale_out;
            for (int i = 0; i < kv_dim; ++i)
                dst[i] = static_cast<int8_t>(std::lroundf(src[i] * inv));
        };

        quantise(k, k_cache + pos * kv_dim, k_scales[pos]);
        quantise(v, v_cache + pos * kv_dim, v_scales[pos]);
    }
};
