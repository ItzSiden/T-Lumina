#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "tensor.h"
#include "kv_cache.h"

struct TLuminaBlock {
    Tensor attn_in_w, attn_in_b;
    Tensor attn_out_w, attn_out_b;
    Tensor norm1_w, norm1_b;
    Tensor norm2_w, norm2_b;
    Tensor ffn_gate_w, ffn_up_w, ffn_down_w;
    
    // নতুন Alpha ভেরিয়েবল
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
    Tensor pos_emb;
    Tensor norm_w, norm_b;
    Tensor head_w;
    std::vector<TLuminaBlock> blocks;
    std::vector<std::unique_ptr<LayerKVCache>> kv_caches;

    float* hidden;
    float* hidden_norm;
    float* qkv_buf;
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