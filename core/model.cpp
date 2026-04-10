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
    for (int i = 0; i < n_layers; ++i) kv_caches.push_back(std::make_unique<LayerKVCache>(max_len, d_model));
    hidden = new float[d_model]; hidden_norm = new float[d_model];
    qkv_buf = new float[3 * d_model]; att_scores = new float[max_len];
    ffn_gate_buf = new float[d_ffn]; ffn_up_buf = new float[d_ffn]; logits = new float[vocab_size];
}

TLuminaModel::~TLuminaModel() {
    delete[] hidden; delete[] hidden_norm; delete[] qkv_buf;
    delete[] att_scores; delete[] ffn_gate_buf; delete[] ffn_up_buf; delete[] logits;
}

void layer_norm(const float* x, float* out, const float* w, const float* b, int d) {
    float mean = 0, var = 0;
    for (int i = 0; i < d; ++i) mean += x[i];
    mean /= d;
    for (int i = 0; i < d; ++i) var += (x[i] - mean) * (x[i] - mean);
    var /= d;
    float inv_std = 1.0f / std::sqrt(var + 1e-5f);
    for (int i = 0; i < d; ++i) out[i] = (x[i] - mean) * inv_std * w[i] + b[i];
}

// ⚡ 100% Safe IEEE-754 Compliant FP16 to FP32 Decoder
inline float fp16_to_fp32(uint16_t x) {
    uint32_t e = (x >> 10) & 0x1f;
    uint32_t m = x & 0x3ff;
    uint32_t v = (uint32_t)(x & 0x8000) << 16;
    if (e == 0) {
        if (m != 0) { // denormalized
            while ((m & 0x400) == 0) { m <<= 1; e--; }
            e++; m &= ~0x400;
            v |= ((e + 112) << 23) | (m << 13);
        }
    } else if (e == 31) { // inf/nan
        v |= 0x7f800000 | (m << 13);
    } else { // normalized
        v |= ((e + 112) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &v, 4);
    return f;
}

void extract_fp32(Tensor& t, const TensorRaw& raw) {
    t.size = raw.data.size() / 2; // 2 bytes per float16
    t.data = new float[t.size];
    for (size_t i = 0; i < t.size; ++i) {
        uint16_t h;
        std::memcpy(&h, &raw.data[i * 2], 2);
        t.data[i] = fp16_to_fp32(h);
    }
}

void extract_ternary(Tensor& t, const TensorRaw& raw_packed, int orig_size) {
    t.size = orig_size;
    t.data_i8 = new int8_t[orig_size];
    unpack_5in8(raw_packed.data.data(), t.data_i8, orig_size);
}

void TLuminaModel::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Could not open binary model file.");
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

    extract_fp32(embed_w, raw_tensors["embed.weight"]);
    extract_fp32(pos_emb, raw_tensors["pos_emb"]);
    extract_fp32(norm_w, raw_tensors["norm.weight"]);
    extract_fp32(norm_b, raw_tensors["norm.bias"]);
    extract_fp32(head_w, raw_tensors["head.weight"]);

    blocks.resize(n_layers);
    for (int i = 0; i < n_layers; ++i) {
        std::string prefix = "blocks." + std::to_string(i) + ".";
        
        extract_fp32(blocks[i].attn_in_w, raw_tensors[prefix + "attn.in_proj_weight"]);
        extract_fp32(blocks[i].attn_in_b, raw_tensors[prefix + "attn.in_proj_bias"]);
        extract_fp32(blocks[i].attn_out_w, raw_tensors[prefix + "attn.out_proj.weight"]);
        extract_fp32(blocks[i].attn_out_b, raw_tensors[prefix + "attn.out_proj.bias"]);
        
        extract_fp32(blocks[i].norm1_w, raw_tensors[prefix + "norm1.weight"]);
        extract_fp32(blocks[i].norm1_b, raw_tensors[prefix + "norm1.bias"]);
        extract_fp32(blocks[i].norm2_w, raw_tensors[prefix + "norm2.weight"]);
        extract_fp32(blocks[i].norm2_b, raw_tensors[prefix + "norm2.bias"]);

        extract_ternary(blocks[i].ffn_gate_w, raw_tensors[prefix + "ffn.gate.weight_fp_packed"], raw_tensors[prefix + "ffn.gate.weight_fp_size"].int_val);
        extract_ternary(blocks[i].ffn_up_w, raw_tensors[prefix + "ffn.up.weight_fp_packed"], raw_tensors[prefix + "ffn.up.weight_fp_size"].int_val);
        extract_ternary(blocks[i].ffn_down_w, raw_tensors[prefix + "ffn.down.weight_fp_packed"], raw_tensors[prefix + "ffn.down.weight_fp_size"].int_val);

        blocks[i].ffn_gate_alpha = raw_tensors[prefix + "ffn.gate.weight_fp_alpha"].float_val;
        blocks[i].ffn_up_alpha = raw_tensors[prefix + "ffn.up.weight_fp_alpha"].float_val;
        blocks[i].ffn_down_alpha = raw_tensors[prefix + "ffn.down.weight_fp_alpha"].float_val;
    }
}

float* TLuminaModel::forward(int token, int pos) {
    const float* emb_row = embed_w.data + token * d_model;
    const float* pos_row = pos_emb.data + pos * d_model;
    
    for (int i = 0; i < d_model; ++i) hidden[i] = emb_row[i] + pos_row[i];
    float scale_attn = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int l = 0; l < n_layers; ++l) {
        fp32_matmul(hidden, blocks[l].attn_in_w.data, qkv_buf, 3 * d_model, d_model);
        for(int i = 0; i < 3 * d_model; ++i) qkv_buf[i] += blocks[l].attn_in_b.data[i];

        float* q = qkv_buf; float* k = qkv_buf + d_model; float* v = qkv_buf + 2 * d_model;
        kv_caches[l]->update_cache(pos, k, v);

        float out_attn[256] = {0}; 
        for (int h = 0; h < n_heads; ++h) {
            float* q_h = q + h * head_dim;
            float max_score = -1e9f;
            for (int p = 0; p <= pos; ++p) {
                int8_t* k_cached = kv_caches[l]->k_cache + p * d_model + h * head_dim;
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
                int8_t* v_cached = kv_caches[l]->v_cache + p * d_model + h * head_dim;
                float v_scale = kv_caches[l]->v_scales[p];
                for (int i = 0; i < head_dim; ++i) out_attn[h * head_dim + i] += score * v_cached[i] * v_scale;
            }
        }

        float attn_proj_out[256];
        fp32_matmul(out_attn, blocks[l].attn_out_w.data, attn_proj_out, d_model, d_model);
        
        for (int i = 0; i < d_model; ++i) hidden[i] += attn_proj_out[i] + blocks[l].attn_out_b.data[i];
        layer_norm(hidden, hidden_norm, blocks[l].norm1_w.data, blocks[l].norm1_b.data, d_model);
        for (int i = 0; i < d_model; ++i) hidden[i] = hidden_norm[i];

        ternary_matmul_avx2(hidden, blocks[l].ffn_gate_w.data_i8, ffn_gate_buf, d_ffn, d_model, blocks[l].ffn_gate_alpha);
        ternary_matmul_avx2(hidden, blocks[l].ffn_up_w.data_i8, ffn_up_buf, d_ffn, d_model, blocks[l].ffn_up_alpha);

        for (int i = 0; i < d_ffn; ++i) ffn_gate_buf[i] = silu(ffn_gate_buf[i]) * ffn_up_buf[i];

        float ffn_down_out[256]; 
        ternary_matmul_avx2(ffn_gate_buf, blocks[l].ffn_down_w.data_i8, ffn_down_out, d_model, d_ffn, blocks[l].ffn_down_alpha);

        for (int i = 0; i < d_model; ++i) hidden[i] += ffn_down_out[i];
        layer_norm(hidden, hidden_norm, blocks[l].norm2_w.data, blocks[l].norm2_b.data, d_model);
        for (int i = 0; i < d_model; ++i) hidden[i] = hidden_norm[i];
    }

    layer_norm(hidden, hidden_norm, norm_w.data, norm_b.data, d_model);
    fp32_matmul(hidden_norm, head_w.data, logits, vocab_size, d_model);
    
    return logits;
}
void TLuminaModel::reset_cache() {}
