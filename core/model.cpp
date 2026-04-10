#include "model.h"
#include "packing.h"
#include "ternary_ffn.h"
#include "attention.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <limits>

namespace {
constexpr uint32_t kMaxTensorNameLen = 4096;

std::string find_existing_name(
    const std::unordered_map<std::string, TensorRaw>& raw_tensors,
    const std::vector<std::string>& candidates
) {
    for (const auto& candidate : candidates) {
        auto it = raw_tensors.find(candidate);
        if (it != raw_tensors.end() && !it->second.data.empty()) {
            return candidate;
        }
    }
    return "";
}

template <typename T>
void read_exact(std::ifstream& f, T& out, const std::string& what) {
    if (!f.read(reinterpret_cast<char*>(&out), sizeof(T))) {
        throw std::runtime_error("Corrupted binary model file: failed to read " + what);
    }
}

void read_exact_bytes(std::ifstream& f, char* out, size_t len, const std::string& what) {
    if (!f.read(out, static_cast<std::streamsize>(len))) {
        throw std::runtime_error("Corrupted binary model file: failed to read " + what);
    }
}

void decode_float_tensor(
    Tensor& t,
    const TensorRaw& raw,
    size_t expected_size = 0
) {
    const size_t bytes = raw.data.size();
    const bool fp16_matches_expected = expected_size > 0 && bytes == expected_size * 2;
    const bool fp32_matches_expected = expected_size > 0 && bytes == expected_size * 4;

    // Some exporters mistakenly tag FP32 tensors as type=1.
    const bool treat_as_fp32 =
        (raw.type != 1) ||
        fp32_matches_expected ||
        (!fp16_matches_expected && bytes % 4 == 0 && expected_size > 0);

    if (bytes == 0) {
        throw std::runtime_error("Invalid empty tensor payload");
    }

    if (treat_as_fp32) {
        if (bytes % 4 != 0) {
            throw std::runtime_error("Invalid FP32 tensor byte length: " + std::to_string(bytes));
        }
        t.size = bytes / 4;
        t.data = new float[t.size]();
        std::memcpy(t.data, raw.data.data(), t.size * sizeof(float));
    } else {
        if (bytes % 2 != 0) {
            throw std::runtime_error("Invalid FP16 tensor byte length: " + std::to_string(bytes));
        }
        t.size = bytes / 2;
        t.data = new float[t.size]();
        const uint16_t* fp16_data = reinterpret_cast<const uint16_t*>(raw.data.data());
        for (size_t i = 0; i < t.size; ++i) {
            t.data[i] = fp16_to_fp32(fp16_data[i]);
        }
    }

    if (expected_size > 0 && t.size != expected_size) {
        throw std::runtime_error(
            "Tensor shape/size mismatch. Expected elements=" + std::to_string(expected_size) +
            ", loaded=" + std::to_string(t.size)
        );
    }
}
}

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
void extract_fp32(Tensor& t, const std::string& name, std::unordered_map<std::string, TensorRaw>& raw_tensors, size_t expected_size = 0) {
    if (raw_tensors.find(name) == raw_tensors.end() || raw_tensors[name].data.empty()) {
        throw std::runtime_error("Missing layer in binary file: " + name);
    }
    const TensorRaw& raw = raw_tensors[name];
    decode_float_tensor(t, raw, expected_size);
}

// ⚡ OPTIONAL: Extract FP32 with fallback initialization
void extract_fp32_optional(Tensor& t, const std::string& name, size_t expected_size, 
                           std::unordered_map<std::string, TensorRaw>& raw_tensors, 
                           bool use_identity = false) {
    if (raw_tensors.find(name) != raw_tensors.end() && !raw_tensors[name].data.empty()) {
        const TensorRaw& raw = raw_tensors[name];
        decode_float_tensor(t, raw, expected_size);
        std::cout << "  ✓ Loaded: " << name << " (size=" << t.size << ")" << std::endl;
    } else {
        // Initialize with safe defaults
        t.size = expected_size;
        t.data = new float[t.size]();
        if (use_identity) {
            for (size_t i = 0; i < t.size; ++i) t.data[i] = 1.0f; // Identity for norm weights
        }
        std::cout << "  ⚠ Missing: " << name << " (initialized with " 
                  << (use_identity ? "identity" : "zeros") << ")" << std::endl;
    }
}

void extract_ternary(Tensor& t, const std::string& name, std::unordered_map<std::string, TensorRaw>& raw_tensors) {
    std::string packed_name = name + "_packed";
    std::string size_name = name + "_size";
    if (raw_tensors.find(packed_name) == raw_tensors.end() || raw_tensors[packed_name].data.empty()) {
        throw std::runtime_error("Missing ternary layer in binary file: " + packed_name);
    }
    if (raw_tensors.find(size_name) == raw_tensors.end()) {
        throw std::runtime_error("Missing ternary size metadata in binary file: " + size_name);
    }
    int orig_size = raw_tensors[size_name].int_val;
    if (orig_size <= 0) {
        throw std::runtime_error("Invalid ternary size for " + size_name + ": " + std::to_string(orig_size));
    }
    t.size = orig_size;
    t.data_i8 = new int8_t[orig_size]();
    unpack_5in8(raw_tensors[packed_name].data.data(), t.data_i8, orig_size);
}

void TLuminaModel::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Could not open binary model file.");
    
    std::cout << "\n📂 Parsing binary model file..." << std::endl;
    
    std::unordered_map<std::string, TensorRaw> raw_tensors;
    while (true) {
        uint32_t name_len = 0;
        if (!f.read(reinterpret_cast<char*>(&name_len), sizeof(name_len))) {
            if (f.eof()) break;
            throw std::runtime_error("Corrupted binary model file: failed to read tensor name length");
        }
        if (name_len == 0 || name_len > kMaxTensorNameLen) {
            throw std::runtime_error("Corrupted binary model file: invalid tensor name length " + std::to_string(name_len));
        }

        std::string name(name_len, ' ');
        read_exact_bytes(f, &name[0], name_len, "tensor name");

        uint32_t type = 0;
        read_exact(f, type, "tensor type");
        
        TensorRaw tr; tr.type = type;
        if (type == 1 || type == 2) {
            uint32_t len = 0;
            read_exact(f, len, "tensor payload length");
            if (len == 0 || len > std::numeric_limits<uint32_t>::max() / 2) {
                throw std::runtime_error("Corrupted binary model file: invalid payload length for " + name);
            }
            tr.data.resize(len);
            read_exact_bytes(f, reinterpret_cast<char*>(tr.data.data()), len, "tensor payload");
        } else if (type == 4) {
            // Backward-compatible parser:
            // - old format: [len=4][int32 value]
            // - some exporters: [int32 value] (no len field)
            uint32_t first = 0;
            read_exact(f, first, "int32 scalar payload");
            if (first == 4) {
                read_exact(f, tr.int_val, "int32 scalar value");
            } else {
                tr.int_val = static_cast<int>(first);
            }
        } else if (type == 5) { 
            // Backward-compatible parser:
            // - old format: [len=4][float32 value]
            // - some exporters: [float32 value] (no len field)
            uint32_t first = 0;
            read_exact(f, first, "float32 scalar payload");
            if (first == 4) {
                read_exact(f, tr.float_val, "float32 scalar value");
            } else {
                std::memcpy(&tr.float_val, &first, sizeof(float));
            }
        } else {
            throw std::runtime_error("Unsupported tensor type " + std::to_string(type) + " for entry: " + name);
        }
        raw_tensors[name] = std::move(tr);
    }
    
    std::cout << "✓ Loaded " << raw_tensors.size() << " tensor entries from file\n" << std::endl;

    auto resolve_and_extract_fp32 = [&](Tensor& t, const std::vector<std::string>& candidates, size_t expected_size) {
        std::string found = find_existing_name(raw_tensors, candidates);
        if (found.empty()) {
            throw std::runtime_error("Missing layer in binary file. Tried names: " + candidates.front());
        }
        if (found != candidates.front()) {
            std::cout << "  ↪ Alias matched: " << candidates.front() << " <- " << found << std::endl;
        }
        extract_fp32(t, found, raw_tensors, expected_size);
    };

    auto resolve_and_extract_ternary = [&](Tensor& t, const std::vector<std::string>& candidates, float& alpha_out) {
        std::string found;
        for (const auto& candidate : candidates) {
            std::string packed_name = candidate + "_packed";
            if (raw_tensors.find(packed_name) != raw_tensors.end() && !raw_tensors[packed_name].data.empty()) {
                found = candidate;
                break;
            }
        }
        if (found.empty()) {
            throw std::runtime_error("Missing ternary layer in binary file. Tried names: " + candidates.front() + "_packed");
        }
        if (found != candidates.front()) {
            std::cout << "  ↪ Alias matched: " << candidates.front() << " <- " << found << std::endl;
        }
        extract_ternary(t, found, raw_tensors);

        std::string alpha_name = found + "_alpha";
        if (raw_tensors.find(alpha_name) != raw_tensors.end()) {
            alpha_out = raw_tensors[alpha_name].float_val;
        } else {
            alpha_out = 1.0f;
            std::cout << "  ⚠ Missing: " << alpha_name << " (using alpha=1.0)" << std::endl;
        }
    };

    resolve_and_extract_fp32(embed_w, {
        "model.embed_tokens.weight",
        "embed_tokens.weight",
        "tok_embeddings.weight"
    }, static_cast<size_t>(vocab_size) * d_model);
    
    // Optional layers with fallback initialization
    extract_fp32_optional(norm_w, "model.norm.weight", d_model, raw_tensors, true);  // Identity for norm
    extract_fp32_optional(head_w, "lm_head.weight", static_cast<size_t>(vocab_size) * d_model, raw_tensors, false); // Zeros for head

    blocks.resize(n_layers);
    for (int i = 0; i < n_layers; ++i) {
        std::string model_prefix = "model.layers." + std::to_string(i) + ".";
        std::string plain_prefix = "layers." + std::to_string(i) + ".";
        
        // Try to load all layer components, skip if not available
        try {
            resolve_and_extract_fp32(blocks[i].q_proj, {
                model_prefix + "self_attn.q_proj.weight",
                plain_prefix + "self_attn.q_proj.weight",
                model_prefix + "attn.q_proj.weight",
                plain_prefix + "attn.q_proj.weight"
            }, static_cast<size_t>(d_model) * d_model);
            resolve_and_extract_fp32(blocks[i].k_proj, {
                model_prefix + "self_attn.k_proj.weight",
                plain_prefix + "self_attn.k_proj.weight",
                model_prefix + "attn.k_proj.weight",
                plain_prefix + "attn.k_proj.weight"
            }, static_cast<size_t>(n_kv_heads * head_dim) * d_model);
            resolve_and_extract_fp32(blocks[i].v_proj, {
                model_prefix + "self_attn.v_proj.weight",
                plain_prefix + "self_attn.v_proj.weight",
                model_prefix + "attn.v_proj.weight",
                plain_prefix + "attn.v_proj.weight"
            }, static_cast<size_t>(n_kv_heads * head_dim) * d_model);
            resolve_and_extract_fp32(blocks[i].o_proj, {
                model_prefix + "self_attn.o_proj.weight",
                plain_prefix + "self_attn.o_proj.weight",
                model_prefix + "attn.o_proj.weight",
                plain_prefix + "attn.o_proj.weight"
            }, static_cast<size_t>(d_model) * d_model);
            
            resolve_and_extract_fp32(blocks[i].norm1_w, {
                model_prefix + "input_layernorm.weight",
                plain_prefix + "input_layernorm.weight",
                model_prefix + "attention_norm.weight",
                plain_prefix + "attention_norm.weight",
                model_prefix + "self_attn_layer_norm.weight",
                plain_prefix + "self_attn_layer_norm.weight"
            }, d_model);
            resolve_and_extract_fp32(blocks[i].norm2_w, {
                model_prefix + "post_attention_layernorm.weight",
                plain_prefix + "post_attention_layernorm.weight",
                model_prefix + "ffn_norm.weight",
                plain_prefix + "ffn_norm.weight",
                model_prefix + "mlp_layernorm.weight",
                plain_prefix + "mlp_layernorm.weight"
            }, d_model);

            resolve_and_extract_ternary(blocks[i].ffn_gate_w, {
                model_prefix + "mlp.gate_proj.weight_fp",
                plain_prefix + "mlp.gate_proj.weight_fp",
                model_prefix + "ffn.gate.weight_fp",
                plain_prefix + "ffn.gate.weight_fp"
            }, blocks[i].ffn_gate_alpha);
            resolve_and_extract_ternary(blocks[i].ffn_up_w, {
                model_prefix + "mlp.up_proj.weight_fp",
                plain_prefix + "mlp.up_proj.weight_fp",
                model_prefix + "ffn.up.weight_fp",
                plain_prefix + "ffn.up.weight_fp"
            }, blocks[i].ffn_up_alpha);
            resolve_and_extract_ternary(blocks[i].ffn_down_w, {
                model_prefix + "mlp.down_proj.weight_fp",
                plain_prefix + "mlp.down_proj.weight_fp",
                model_prefix + "ffn.down.weight_fp",
                plain_prefix + "ffn.down.weight_fp"
            }, blocks[i].ffn_down_alpha);
            
            if (i % 5 == 0) std::cout << "✓ Loaded layer " << i << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "⚠ Error loading layer " << i << ": " << e.what() << std::endl;
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
