#include "model.h"
#include "packing.h"
#include "ternary_ffn.h"
#include "attention.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <cmath>
#include <regex>

// ─────────────────────────────────────────────
// RMSNorm
// ─────────────────────────────────────────────
static void rms_norm(const float* x, float* out, const float* w, int d) {
    float ss = 0.0f;
    for (int i = 0; i < d; ++i) ss += x[i] * x[i];
    ss = 1.0f / std::sqrt(ss / d + 1e-5f);
    for (int i = 0; i < d; ++i) out[i] = x[i] * ss * w[i];
}

// ─────────────────────────────────────────────
// Tensor extract helpers
// ─────────────────────────────────────────────
static void extract_fp32(Tensor& t,
                          const std::string& name,
                          std::unordered_map<std::string, TensorRaw>& raw) {
    auto it = raw.find(name);
    if (it == raw.end() || it->second.data.empty()) {
        std::cerr << "\n[CRITICAL ERROR] Missing layer: " << name << std::endl;
        exit(1);
    }
    t.size = it->second.data.size() / 4;
    t.data = new float[t.size]();
    std::memcpy(t.data, it->second.data.data(), it->second.data.size());
}

static void extract_ternary(Tensor& t,
                             const std::string& name,
                             std::unordered_map<std::string, TensorRaw>& raw) {
    std::string packed_name = name + "_packed";
    std::string size_name   = name + "_size";

    auto it = raw.find(packed_name);
    if (it == raw.end() || it->second.data.empty()) {
        std::cerr << "\n[CRITICAL ERROR] Missing ternary layer: " << packed_name << std::endl;
        exit(1);
    }
    int orig_size = raw[size_name].int_val;
    t.size     = orig_size;
    t.data_i8  = new int8_t[orig_size]();
    unpack_5in8(it->second.data.data(), t.data_i8, orig_size);
}

// ─────────────────────────────────────────────
// Architecture name mapping
// canonical name → actual name in binary file
// ─────────────────────────────────────────────
std::string TLuminaModel::map_name(const std::string& s) const {
    if (cfg.arch == ArchType::TLUMINA) return s;  // তোমার নিজের model — কোনো mapping নেই

    // LLaMA / TinyLlama / Mistral
    std::string r = s;

    // Top-level
    if (r == "embed.weight")  return "model.embed_tokens.weight";
    if (r == "norm.weight")   return "model.norm.weight";
    if (r == "head.weight")   return "lm_head.weight";

    // blocks.X. → model.layers.X.
    r = std::regex_replace(r, std::regex("^blocks\\."), "model.layers.");

    // Attention
    r = std::regex_replace(r, std::regex("\\.attn\\.wq\\.weight"), ".self_attn.q_proj.weight");
    r = std::regex_replace(r, std::regex("\\.attn\\.wk\\.weight"), ".self_attn.k_proj.weight");
    r = std::regex_replace(r, std::regex("\\.attn\\.wv\\.weight"), ".self_attn.v_proj.weight");
    r = std::regex_replace(r, std::regex("\\.attn\\.wo\\.weight"), ".self_attn.o_proj.weight");

    // Norms
    r = std::regex_replace(r, std::regex("\\.norm1\\.weight"), ".input_layernorm.weight");
    r = std::regex_replace(r, std::regex("\\.norm2\\.weight"), ".post_attention_layernorm.weight");

    // FFN (ternary packed names)
    r = std::regex_replace(r, std::regex("\\.ffn\\.gate\\.weight_fp"), ".mlp.gate_proj.weight_fp");
    r = std::regex_replace(r, std::regex("\\.ffn\\.up\\.weight_fp"),   ".mlp.up_proj.weight_fp");
    r = std::regex_replace(r, std::regex("\\.ffn\\.down\\.weight_fp"), ".mlp.down_proj.weight_fp");

    return r;
}

// ─────────────────────────────────────────────
// Buffer allocation (config load হওয়ার পরে)
// ─────────────────────────────────────────────
void TLuminaModel::allocate_buffers() {
    int dm  = cfg.d_model;
    int kv  = cfg.n_kv_heads * cfg.head_dim;
    int df  = cfg.d_ffn;
    int vs  = cfg.vocab_size;
    int ml  = cfg.max_len;

    hidden       = new float[dm]();
    hidden_norm  = new float[dm]();
    q_buf        = new float[dm]();      // n_heads * head_dim
    k_buf        = new float[kv]();      // n_kv_heads * head_dim (GQA!)
    v_buf        = new float[kv]();
    att_scores   = new float[ml]();
    ffn_gate_buf = new float[df]();
    ffn_up_buf   = new float[df]();
    out_attn_buf = new float[dm]();      // আর hardcoded [256] নয়!
    ffn_down_buf = new float[dm]();
    logits_buf   = new float[vs]();

    // KV cache: kv_dim = n_kv_heads * head_dim
    for (int i = 0; i < cfg.n_layers; ++i)
        kv_caches.push_back(std::make_unique<LayerKVCache>(ml, kv));
}

void TLuminaModel::free_buffers() {
    delete[] hidden;       delete[] hidden_norm;
    delete[] q_buf;        delete[] k_buf;        delete[] v_buf;
    delete[] att_scores;   delete[] ffn_gate_buf;  delete[] ffn_up_buf;
    delete[] out_attn_buf; delete[] ffn_down_buf;  delete[] logits_buf;
}

TLuminaModel::~TLuminaModel() { free_buffers(); }

// ─────────────────────────────────────────────
// Load: config → binary
// ─────────────────────────────────────────────
void TLuminaModel::load(const std::string& bin_path, const std::string& config_path) {
    // 1. Config load
    if (!cfg.load(config_path)) {
        throw std::runtime_error("Failed to load config: " + config_path);
    }
    cfg.print();

    // 2. Buffers allocate
    allocate_buffers();

    // 3. Binary file পড়ো
    std::ifstream f(bin_path, std::ios::binary);
    if (!f) throw std::runtime_error("Could not open: " + bin_path);

    std::unordered_map<std::string, TensorRaw> raw;
    uint32_t name_len;

    while (f.read(reinterpret_cast<char*>(&name_len), 4)) {
        std::string name(name_len, ' ');
        f.read(&name[0], name_len);

        uint32_t type;
        f.read(reinterpret_cast<char*>(&type), 4);

        TensorRaw tr;
        tr.type = type;

        if (type == 1 || type == 2) {
            uint32_t len;
            f.read(reinterpret_cast<char*>(&len), 4);
            tr.data.resize(len);
            f.read(reinterpret_cast<char*>(tr.data.data()), len);
        } else if (type == 4) {
            uint32_t len;
            f.read(reinterpret_cast<char*>(&len), 4);
            f.read(reinterpret_cast<char*>(&tr.int_val), 4);
        } else if (type == 5) {
            uint32_t len;
            f.read(reinterpret_cast<char*>(&len), 4);
            f.read(reinterpret_cast<char*>(&tr.float_val), 4);
        }
        raw[name] = std::move(tr);
    }

    // 4. Weights extract করো (name mapping দিয়ে)
    auto fp  = [&](Tensor& t, const std::string& n) { extract_fp32(t, map_name(n), raw); };
    auto ter = [&](Tensor& t, const std::string& n) { extract_ternary(t, map_name(n), raw); };

    fp(embed_w, "embed.weight");
    fp(norm_w,  "norm.weight");
    fp(head_w,  "head.weight");

    blocks.resize(cfg.n_layers);
    for (int i = 0; i < cfg.n_layers; ++i) {
        std::string p = "blocks." + std::to_string(i) + ".";

        fp(blocks[i].wq,     p + "attn.wq.weight");
        fp(blocks[i].wk,     p + "attn.wk.weight");
        fp(blocks[i].wv,     p + "attn.wv.weight");
        fp(blocks[i].wo,     p + "attn.wo.weight");
        fp(blocks[i].norm1_w, p + "norm1.weight");
        fp(blocks[i].norm2_w, p + "norm2.weight");

        ter(blocks[i].ffn_gate_w, p + "ffn.gate.weight_fp");
        ter(blocks[i].ffn_up_w,   p + "ffn.up.weight_fp");
        ter(blocks[i].ffn_down_w, p + "ffn.down.weight_fp");

        std::string alpha_key_gate = map_name(p + "ffn.gate.weight_fp") + "_alpha";
        std::string alpha_key_up   = map_name(p + "ffn.up.weight_fp")   + "_alpha";
        std::string alpha_key_down = map_name(p + "ffn.down.weight_fp") + "_alpha";

        blocks[i].ffn_gate_alpha = raw[alpha_key_gate].float_val;
        blocks[i].ffn_up_alpha   = raw[alpha_key_up].float_val;
        blocks[i].ffn_down_alpha = raw[alpha_key_down].float_val;
    }

    std::cout << "[Model] Loaded successfully!\n";
}

// ─────────────────────────────────────────────
// Reset KV cache
// ─────────────────────────────────────────────
void TLuminaModel::reset_cache() {
    for (auto& cache : kv_caches) cache->reset();
}

// ─────────────────────────────────────────────
// Forward pass (GQA-aware)
// ─────────────────────────────────────────────
float* TLuminaModel::forward(int token, int pos) {
    if (token < 0 || token >= cfg.vocab_size) return logits_buf;

    const int dm       = cfg.d_model;
    const int hd       = cfg.head_dim;
    const int nh       = cfg.n_heads;
    const int nkv      = cfg.n_kv_heads;
    const int ng       = cfg.n_groups;   // nh / nkv
    const int kv_dim   = nkv * hd;
    const float s_attn = 1.0f / std::sqrt(static_cast<float>(hd));

    // Embedding lookup
    const float* emb = embed_w.data + token * dm;
    std::memcpy(hidden, emb, dm * sizeof(float));

    for (int l = 0; l < cfg.n_layers; ++l) {
        // ── Attention ──────────────────────────────
        rms_norm(hidden, hidden_norm, blocks[l].norm1_w.data, dm);

        // Q: dm×dm, K/V: kv_dim×dm
        fp32_matmul(hidden_norm, blocks[l].wq.data, q_buf, dm,     dm);
        fp32_matmul(hidden_norm, blocks[l].wk.data, k_buf, kv_dim, dm);
        fp32_matmul(hidden_norm, blocks[l].wv.data, v_buf, kv_dim, dm);

        // RoPE: Q সব heads এ, K শুধু KV heads এ
        apply_rope(q_buf, k_buf, pos, hd, nh, nkv, cfg.rope_theta);

        // KV cache update (kv_dim size দিয়ে)
        kv_caches[l]->update_cache(pos, k_buf, v_buf);

        // Attention output buffer clear
        std::memset(out_attn_buf, 0, dm * sizeof(float));

        for (int h = 0; h < nh; ++h) {
            // GQA: কোন KV head?
            int kv_h = h / ng;

            float* q_h = q_buf + h * hd;

            // Scores
            float max_score = -1e9f;
            for (int p = 0; p <= pos; ++p) {
                int8_t* k_p = kv_caches[l]->k_cache + p * kv_dim + kv_h * hd;
                float score = quantized_dot_product(q_h, k_p,
                              kv_caches[l]->k_scales[p], hd) * s_attn;
                att_scores[p] = score;
                if (score > max_score) max_score = score;
            }

            // Softmax
            float sum_exp = 0.0f;
            for (int p = 0; p <= pos; ++p) {
                att_scores[p] = std::exp(att_scores[p] - max_score);
                sum_exp += att_scores[p];
            }
            for (int p = 0; p <= pos; ++p) att_scores[p] /= sum_exp;

            // Value aggregation
            for (int p = 0; p <= pos; ++p) {
                float  sc    = att_scores[p];
                int8_t* v_p  = kv_caches[l]->v_cache + p * kv_dim + kv_h * hd;
                float  vsc   = kv_caches[l]->v_scales[p];
                float* out_h = out_attn_buf + h * hd;
                for (int i = 0; i < hd; ++i)
                    out_h[i] += sc * v_p[i] * vsc;
            }
        }

        // Projection + residual
        std::memset(ffn_down_buf, 0, dm * sizeof(float));
        fp32_matmul(out_attn_buf, blocks[l].wo.data, ffn_down_buf, dm, dm);
        for (int i = 0; i < dm; ++i) hidden[i] += ffn_down_buf[i];

        // ── FFN ────────────────────────────────────
        rms_norm(hidden, hidden_norm, blocks[l].norm2_w.data, dm);

        ternary_matmul_avx2(hidden_norm, blocks[l].ffn_gate_w.data_i8,
                            ffn_gate_buf, cfg.d_ffn, dm, blocks[l].ffn_gate_alpha);
        ternary_matmul_avx2(hidden_norm, blocks[l].ffn_up_w.data_i8,
                            ffn_up_buf,   cfg.d_ffn, dm, blocks[l].ffn_up_alpha);

        for (int i = 0; i < cfg.d_ffn; ++i)
            ffn_gate_buf[i] = silu(ffn_gate_buf[i]) * ffn_up_buf[i];

        std::memset(ffn_down_buf, 0, dm * sizeof(float));
        ternary_matmul_avx2(ffn_gate_buf, blocks[l].ffn_down_w.data_i8,
                            ffn_down_buf, dm, cfg.d_ffn, blocks[l].ffn_down_alpha);

        for (int i = 0; i < dm; ++i) hidden[i] += ffn_down_buf[i];
    }

    // Final norm + lm_head
    rms_norm(hidden, hidden_norm, norm_w.data, dm);
    fp32_matmul(hidden_norm, head_w.data, logits_buf, cfg.vocab_size, dm);

    return logits_buf;
}
