#include "model.h"
#include "packing.h"
#include "ternary_ffn.h"
#include "attention.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <cmath>

TLuminaModel::TLuminaModel() {
    head_dim = d_model / n_heads;
    int kv_dim = n_kv_heads * head_dim; 
    
    for (int i = 0; i < n_layers; ++i) kv_caches.push_back(std::make_unique<LayerKVCache>(max_len, kv_dim));
    
    hidden = new float[d_model](); hidden_norm = new float[d_model]();
    q_buf = new float[d_model](); k_buf = new float[kv_dim](); v_buf = new float[kv_dim]();
    att_scores = new float[max_len]();
    ffn_gate_buf = new float[d_ffn](); ffn_up_buf = new float[d_ffn](); logits = new float[vocab_size]();
}

TLuminaModel::~TLuminaModel() {
    delete[] hidden; delete[] hidden_norm; delete[] q_buf; delete[] k_buf; delete[] v_buf;
    delete[] att_scores; delete[] ffn_gate_buf; delete[] ffn_up_buf; delete[] logits;
}

void rms_norm(const float* x, float* out, const float* w, int d) {
    float ss = 0.0f;
    for (int i = 0; i < d; ++i) ss += x[i] * x[i];
    ss /= d;
    ss += 1e-5f;
    ss = 1.0f / std::sqrt(ss);
    for (int i = 0; i < d; ++i) out[i] = x[i] * ss * w[i];
}

// ⚡ FIXED: Extract FP32 with FP16 conversion support (uses fp16_to_fp32 from packing.h)
void extract_fp32(Tensor& t, const std::string& name, std::unordered_map<std::string, TensorRaw>& raw_tensors) {
    if (raw_tensors.find(name) == raw_tensors.end() || raw_tensors[name].data.empty()) {
        std::cerr << "\n[CRITICAL ERROR] Missing Layer in binary file: " << name << std::endl;
        exit(1);
    }
    const TensorRaw& raw = raw_tensors[name];
    
    // Type 1 = FP16 Tensor, so divide by 2 for element count
    if (raw.type == 1) {
        t.size = raw.data.size() / 2;
        t.data = new float[t.size]();
        const uint16_t* fp16_data = reinterpret_cast<const uint16_t*>(raw.data.data());
        for (int i = 0; i < t.size; ++i) {
            t.data[i] = fp16_to_fp32(fp16_data[i]);
        }
    } else {
        // Assume FP32 (type 0 or unspecified)
        t.size = raw.data.size() / 4;
        t.data = new float[t.size]();
        std::memcpy(t.data, raw.data.data(), raw.data.size());
    }
}

// ⚡ OPTIONAL: Extract FP32 with fallback initialization
void extract_fp32_optional(Tensor& t, const std::string& name, int expected_size, 
                           std::unordered_map<std::string, TensorRaw>& raw_tensors, 
                           bool use_identity = false) {
    if (raw_tensors.find(name) != raw_tensors.end() && !raw_tensors[name].data.empty()) {
        const TensorRaw& raw = raw_tensors[name];
        
        if (raw.type == 1) {
            t.size = raw.data.size() / 2;
            t.data = new float[t.size]();
            const uint16_t* fp16_data = reinterpret_cast<const uint16_t*>(raw.data.data());
            for (int i = 0; i < t.size; ++i) {
                t.data[i] = fp16_to_fp32(fp16_data[i]);
            }
        } else {
            t.size = raw.data.size() / 4;
            t.data = new float[t.size]();
            std::memcpy(t.data, raw.data.data(), raw.data.size());
        }
        std::cout << "  ✓ Loaded: " << name << " (size=" << t.size << ")" << std::endl;
    } else {
        // Initialize with safe defaults
        t.size = expected_size;
        t.data = new float[t.size]();
        if (use_identity) {
            for (int i = 0; i < t.size; ++i) t.data[i] = 1.0f; // Identity for norm weights
        }
        std::cout << "  ⚠ Missing: " << name << " (initialized with " 
                  << (use_identity ? "identity" : "zeros") << ")" << std::endl;
    }
}

void extract_ternary(Tensor& t, const std::string& name, std::unordered_map<std::string, TensorRaw>& raw_tensors) {
    std::string packed_name = name + "_packed";
    std::string size_name = name + "_size";
    if (raw_tensors.find(packed_name) == raw_tensors.end() || raw_tensors[packed_name].data.empty()) {
        std::cerr << "\n[CRITICAL ERROR] Missing Ternary Layer: " << packed_name << std::endl;
        exit(1);
    }
    int orig_size = raw_tensors[size_name].int_val;
    t.size = orig_size;
    t.data_i8 = new int8_t[orig_size]();
    unpack_5in8(raw_tensors[packed_name].data.data(), t.data_i8, orig_size);
}

void TLuminaModel::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Could not open binary model file.");
    
    std::cout << "\n📂 Parsing binary model file..." << std::endl;
    
    std::unordered_map<std::string, TensorRaw> raw_tensors;
    uint32_t name_len;
    
    while (f.read(reinterpret_cast<char*>(&name_len), 4)) {
        std::string name(name_len, ' '); f.read(&name[0], name_len);
        uint32_t type; f.read(reinterpret_cast<char*>(&type), 4);
        
        TensorRaw tr; tr.type = type;
        if (type == 1 || type == 2) {
            uint32_t len; f.read(reinterpret_cast<char*>(&len), 4);
            tr.data.resize(len); f.read(reinterpret_cast<char*>(tr.data.data()), len);
        } else if (type == 4) {
            uint32_t len; f.read(reinterpret_cast<char*>(&len), 4);
            f.read(reinterpret_cast<char*>(&tr.int_val), 4);
        } else if (type == 5) { 
            uint32_t len; f.read(reinterpret_cast<char*>(&len), 4);
            f.read(reinterpret_cast<char*>(&tr.float_val), 4);
        }
        raw_tensors[name] = std::move(tr);
    }
    
    std::cout << "✓ Loaded " << raw_tensors.size() << " tensor entries from file\n" << std::endl;

    extract_fp32(embed_w, "model.embed_tokens.weight", raw_tensors);
    
    // Optional layers with fallback initialization
    extract_fp32_optional(norm_w, "model.norm.weight", d_model, raw_tensors, true);  // Identity for norm
    extract_fp32_optional(head_w, "lm_head.weight", vocab_size * d_model, raw_tensors, false); // Zeros for head

    blocks.resize(n_layers);
    for (int i = 0; i < n_layers; ++i) {
        std::string prefix = "model.layers." + std::to_string(i) + ".";
        
        // Try to load all layer components, skip if not available
        try {
            extract_fp32(blocks[i].q_proj, prefix + "self_attn.q_proj.weight", raw_tensors);
            extract_fp32(blocks[i].k_proj, prefix + "self_attn.k_proj.weight", raw_tensors);
            extract_fp32(blocks[i].v_proj, prefix + "self_attn.v_proj.weight", raw_tensors);
            extract_fp32(blocks[i].o_proj, prefix + "self_attn.o_proj.weight", raw_tensors);
            
            extract_fp32(blocks[i].norm1_w, prefix + "input_layernorm.weight", raw_tensors);
            extract_fp32(blocks[i].norm2_w, prefix + "post_attention_layernorm.weight", raw_tensors);

            extract_ternary(blocks[i].ffn_gate_w, prefix + "mlp.gate_proj.weight_fp", raw_tensors);
            extract_ternary(blocks[i].ffn_up_w, prefix + "mlp.up_proj.weight_fp", raw_tensors);
            extract_ternary(blocks[i].ffn_down_w, prefix + "mlp.down_proj.weight_fp", raw_tensors);

            blocks[i].ffn_gate_alpha = raw_tensors[prefix + "mlp.gate_proj.weight_fp_alpha"].float_val;
            blocks[i].ffn_up_alpha = raw_tensors[prefix + "mlp.up_proj.weight_fp_alpha"].float_val;
            blocks[i].ffn_down_alpha = raw_tensors[prefix + "mlp.down_proj.weight_fp_alpha"].float_val;
            
            if (i % 5 == 0) std::cout << "✓ Loaded layer " << i << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "⚠ Warning: Failed to load layer " << i << ": " << e.what() << std::endl;
            throw; // Re-throw to stop loading if layer structure is critical
        }
    }
}

float* TLuminaModel::forward(int token, int pos) {
    if (token < 0 || token >= vocab_size) return logits; // Safe bound check

    const float* emb_row = embed_w.data + token * d_model;
    for (int i = 0; i < d_model; ++i) hidden[i] = emb_row[i];

    float scale_attn = 1.0f / std::sqrt(static_cast<float>(head_dim));
    int kv_dim = n_kv_heads * head_dim;

    for (int l = 0; l < n_layers; ++l) {
        rms_norm(hidden, hidden_norm, blocks[l].norm1_w.data, d_model);

        fp32_matmul(hidden_norm, blocks[l].q_proj.data, q_buf, d_model, d_model);
        fp32_matmul(hidden_norm, blocks[l].k_proj.data, k_buf, kv_dim, d_model);
        fp32_matmul(hidden_norm, blocks[l].v_proj.data, v_buf, kv_dim, d_model);

        apply_rope(q_buf, k_buf, pos, head_dim, n_heads, n_kv_heads);
        kv_caches[l]->update_cache(pos, k_buf, v_buf);

        float out_attn[2048] = {0}; 
        for (int h = 0; h < n_heads; ++h) {
            float* q_h = q_buf + h * head_dim;
            int kv_h = h / (n_heads / n_kv_heads);
            
            float max_score = -1e9f;
            for (int p = 0; p <= pos; ++p) {
                int8_t* k_cached = kv_caches[l]->k_cache + p * kv_dim + kv_h * head_dim;
                float score = quantized_dot_product(q_h, k_cached, kv_caches[l]->k_scales[p], head_dim) * scale_attn;
                att_scores[p] = score;
                if (score > max_score) max_score = score;
            }

            float sum_exp = 0.0f;
            for (int p = 0; p <= pos; ++p) {
                att_scores[p] = std::exp(att_scores[p] - max_score);
                sum_exp += att_scores[p];
            }
            for (int p = 0; p <= pos; ++p) att_scores[p] /= sum_exp;

            for (int p = 0; p <= pos; ++p) {
                float score = att_scores[p];
                int8_t* v_cached = kv_caches[l]->v_cache + p * kv_dim + kv_h * head_dim;
                float v_scale = kv_caches[l]->v_scales[p];
                for (int i = 0; i < head_dim; ++i) out_attn[h * head_dim + i] += score * v_cached[i] * v_scale;
            }
        }

        float attn_proj_out[2048] = {0};
        fp32_matmul(out_attn, blocks[l].o_proj.data, attn_proj_out, d_model, d_model);
        for (int i = 0; i < d_model; ++i) hidden[i] += attn_proj_out[i];

        rms_norm(hidden, hidden_norm, blocks[l].norm2_w.data, d_model);

        ternary_matmul_avx2(hidden_norm, blocks[l].ffn_gate_w.data_i8, ffn_gate_buf, d_ffn, d_model, blocks[l].ffn_gate_alpha);
        ternary_matmul_avx2(hidden_norm, blocks[l].ffn_up_w.data_i8, ffn_up_buf, d_ffn, d_model, blocks[l].ffn_up_alpha);

        for (int i = 0; i < d_ffn; ++i) ffn_gate_buf[i] = silu(ffn_gate_buf[i]) * ffn_up_buf[i];

        float ffn_down_out[2048] = {0}; 
        ternary_matmul_avx2(ffn_gate_buf, blocks[l].ffn_down_w.data_i8, ffn_down_out, d_model, d_ffn, blocks[l].ffn_down_alpha);

        for (int i = 0; i < d_model; ++i) hidden[i] += ffn_down_out[i];
    }

    rms_norm(hidden, hidden_norm, norm_w.data, d_model);
    fp32_matmul(hidden_norm, head_w.data, logits, vocab_size, d_model);
    
    return logits;
}
void TLuminaModel::reset_cache() {}
