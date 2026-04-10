#pragma once
#include <string>
#include <vector>
#include <memory>
#include "tensor.h"
#include "kv_cache.h"

// -------------------------------------------------------
// One transformer block (attention + FFN)
// -------------------------------------------------------
struct TLuminaBlock {
    // Attention projections (FP32)
    Tensor q_proj, k_proj, v_proj, o_proj;
    // RMSNorm weights (FP32)
    Tensor norm1_w, norm2_w;
    // FFN projections (ternary INT8 + per-layer alpha)
    Tensor ffn_gate_w, ffn_up_w, ffn_down_w;
    float  ffn_gate_alpha = 1.0f;
    float  ffn_up_alpha   = 1.0f;
    float  ffn_down_alpha = 1.0f;
};

// -------------------------------------------------------
// TinyLlama 1.1B architecture constants
// -------------------------------------------------------
class TLuminaModel {
public:
    // Architecture (matches TinyLlama 1.1B)
    static constexpr int VOCAB_SIZE  = 32000;
    static constexpr int D_MODEL     = 2048;
    static constexpr int N_HEADS     = 32;
    static constexpr int N_KV_HEADS  = 4;     // GQA
    static constexpr int D_FFN       = 5632;
    static constexpr int N_LAYERS    = 22;
    static constexpr int MAX_LEN     = 2048;
    static constexpr int HEAD_DIM    = D_MODEL / N_HEADS;  // 64
    static constexpr int KV_DIM      = N_KV_HEADS * HEAD_DIM; // 256

    // Kept public for main.cpp compat
    int vocab_size = VOCAB_SIZE;
    int max_len    = MAX_LEN;

    // Model weights
    Tensor embed_w;   // [vocab_size × d_model]
    Tensor norm_w;    // [d_model]
    Tensor head_w;    // [vocab_size × d_model]

    std::vector<TLuminaBlock>                  blocks;
    std::vector<std::unique_ptr<LayerKVCache>> kv_caches;

    // Persistent working buffers (heap-allocated once)
    float* hidden      = nullptr;
    float* hidden_norm = nullptr;
    float* q_buf       = nullptr;
    float* k_buf       = nullptr;
    float* v_buf       = nullptr;
    float* att_scores  = nullptr;
    float* ffn_gate_buf= nullptr;
    float* ffn_up_buf  = nullptr;
    float* attn_out_buf= nullptr;  // [d_model]  — replaces stack array
    float* ffn_out_buf = nullptr;  // [d_model]  — replaces stack array
    float* logits      = nullptr;

    TLuminaModel();
    ~TLuminaModel();

    void   load(const std::string& path);
    float* forward(int token, int pos);
    void   reset_cache();   // resets KV cache (for new conversation)
};
