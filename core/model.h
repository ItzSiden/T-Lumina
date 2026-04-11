#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "tensor.h"
#include "kv_cache.h"

struct TLuminaBlock {
    Tensor wq, wk, wv, wo; // Custom Attention Weights
    Tensor norm1_w, norm2_w; // RMSNorm has no bias
    Tensor ffn_gate_w, ffn_up_w, ffn_down_w;
    
    float ffn_gate_alpha = 1.0f;
    float ffn_up_alpha = 1.0f;
    float ffn_down_alpha = 1.0f;
};

class TLuminaModel {
public:
    int vocab_size = 50257;
    int d_model = 256;
    int n_heads = 8;
    int d_ffn = 1024;
    int n_layers = 8;
    int max_len = 256;
    int head_dim = 32;

    Tensor embed_w;
    Tensor norm_w;
    Tensor head_w;
    std::vector<TLuminaBlock> blocks;
    std::vector<std::unique_ptr<LayerKVCache>> kv_caches;

    float* hidden;
    float* hidden_norm;
    float* q_buf;
    float* k_buf;
    float* v_buf;
    float* att_scores;
    float* ffn_gate_buf;
    float* ffn_up_buf;
    float* logits;

    TLuminaModel();
    ~TLuminaModel();

    void load(const std::string& path);
    void reset_cache();
    float* forward(int token, int pos);
};
