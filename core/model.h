#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "tensor.h"
#include "kv_cache.h"
#include "model_config.h"

struct TLuminaBlock {
    Tensor wq, wk, wv, wo;
    Tensor norm1_w, norm2_w;
    Tensor ffn_gate_w, ffn_up_w, ffn_down_w;

    float ffn_gate_alpha = 1.0f;
    float ffn_up_alpha   = 1.0f;
    float ffn_down_alpha = 1.0f;
};

class TLuminaModel {
public:
    // Config — সব dimensions এখান থেকে আসে, hardcoded নয়
    ModelConfig cfg;

    // Shorthand getters (পুরনো code এর compatibility এর জন্য)
    int& vocab_size() { return cfg.vocab_size; }
    int& d_model()    { return cfg.d_model; }
    int& n_heads()    { return cfg.n_heads; }
    int& n_kv_heads() { return cfg.n_kv_heads; }
    int& d_ffn()      { return cfg.d_ffn; }
    int& n_layers()   { return cfg.n_layers; }
    int& max_len()    { return cfg.max_len; }
    int& head_dim()   { return cfg.head_dim; }
    int& n_groups()   { return cfg.n_groups; }

    // Weights
    Tensor embed_w, norm_w, head_w;
    std::vector<TLuminaBlock> blocks;
    std::vector<std::unique_ptr<LayerKVCache>> kv_caches;

    // Runtime buffers — dynamically allocated after config load
    float* hidden       = nullptr;
    float* hidden_norm  = nullptr;
    float* q_buf        = nullptr;
    float* k_buf        = nullptr;  // size: n_kv_heads * head_dim
    float* v_buf        = nullptr;  // size: n_kv_heads * head_dim
    float* att_scores   = nullptr;
    float* ffn_gate_buf = nullptr;
    float* ffn_up_buf   = nullptr;
    float* out_attn_buf = nullptr;  // d_model (আর hardcoded [256] নয়)
    float* ffn_down_buf = nullptr;  // d_model
    float* logits_buf   = nullptr;

    TLuminaModel() = default;
    ~TLuminaModel();

    // config.json load করে তারপর model.bin load করে
    void load(const std::string& bin_path, const std::string& config_path = "config.json");
    void reset_cache();
    float* forward(int token, int pos);

private:
    void allocate_buffers();
    void free_buffers();

    // Architecture-specific layer name mapping
    std::string map_name(const std::string& canonical) const;
};
