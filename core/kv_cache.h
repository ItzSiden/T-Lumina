#pragma once
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <immintrin.h>

struct LayerKVCache {
    int8_t* k_cache;   
    int8_t* v_cache;   
    float* k_scales;   
    float* v_scales;   
    int max_len;
    int d_model;

    LayerKVCache(int max_l, int d_mod) : max_len(max_l), d_model(d_mod) {
        k_cache = new int8_t[max_len * d_model]();
        v_cache = new int8_t[max_len * d_model]();
        k_scales = new float[max_len]();
        v_scales = new float[max_len]();
    }

    ~LayerKVCache() {
        delete[] k_cache;
        delete[] v_cache;
        delete[] k_scales;
        delete[] v_scales;
    }

    void update_cache(int pos, const float* k, const float* v) {
        float k_max = 1e-8f, v_max = 1e-8f;
        for (int i = 0; i < d_model; ++i) {
            k_max = std::max(k_max, std::abs(k[i]));
            v_max = std::max(v_max, std::abs(v[i]));
        }
        
        k_scales[pos] = k_max / 127.0f;
        v_scales[pos] = v_max / 127.0f;
        
        float inv_k = 1.0f / k_scales[pos];
        float inv_v = 1.0f / v_scales[pos];

        int8_t* k_row = k_cache + pos * d_model;
        int8_t* v_row = v_cache + pos * d_model;
        
        for (int i = 0; i < d_model; ++i) {
            k_row[i] = static_cast<int8_t>(std::round(k[i] * inv_k));
            v_row[i] = static_cast<int8_t>(std::round(v[i] * inv_v));
        }
    }
};