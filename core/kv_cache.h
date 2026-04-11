#pragma once
#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <immintrin.h>

struct LayerKVCache {
    int8_t* k_cache;
    int8_t* v_cache;
    float*  k_scales;
    float*  v_scales;

    int max_len;
    int kv_dim;   // n_kv_heads * head_dim (GQA এর জন্য d_model এর চেয়ে ছোট হতে পারে)

    LayerKVCache(int max_l, int kv_d)
        : max_len(max_l), kv_dim(kv_d) {
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

    // Copy/assign নিষিদ্ধ
    LayerKVCache(const LayerKVCache&) = delete;
    LayerKVCache& operator=(const LayerKVCache&) = delete;

    void reset() {
        std::memset(k_cache,  0, max_len * kv_dim * sizeof(int8_t));
        std::memset(v_cache,  0, max_len * kv_dim * sizeof(int8_t));
        std::memset(k_scales, 0, max_len * sizeof(float));
        std::memset(v_scales, 0, max_len * sizeof(float));
    }

    void update_cache(int pos, const float* k, const float* v) {
        float k_max = 1e-8f, v_max = 1e-8f;
        for (int i = 0; i < kv_dim; ++i) {
            k_max = std::max(k_max, std::abs(k[i]));
            v_max = std::max(v_max, std::abs(v[i]));
        }

        k_scales[pos] = k_max / 127.0f;
        v_scales[pos] = v_max / 127.0f;

        float inv_k = 1.0f / k_scales[pos];
        float inv_v = 1.0f / v_scales[pos];

        int8_t* k_row = k_cache + pos * kv_dim;
        int8_t* v_row = v_cache + pos * kv_dim;

        for (int i = 0; i < kv_dim; ++i) {
            k_row[i] = static_cast<int8_t>(std::round(k[i] * inv_k));
            v_row[i] = static_cast<int8_t>(std::round(v[i] * inv_v));
        }
    }
};
