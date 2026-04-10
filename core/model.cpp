#include "model.h"
#include "packing.h"
#include "ternary_ffn.h"
#include "attention.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

// ================================================================
//  HELPERS: binary file parsing
// ================================================================

// Try a list of candidate names; return first that exists in map.
static std::string resolve_name(
    const std::unordered_map<std::string, TensorRaw>& m,
    const std::vector<std::string>& candidates)
{
    for (auto& c : candidates) {
        auto it = m.find(c);
        if (it != m.end() && !it->second.data.empty())
            return c;
    }
    return "";
}

// ----------------------------------------------------------------
// THE CRITICAL FIX:
//   Python exports FP32 data with type=1.
//   Previous code treated type==1 as FP16 (÷2) — WRONG.
//   Correct mapping: type 1 → raw FP32 bytes (÷4 per element).
// ----------------------------------------------------------------
static void load_fp32(Tensor& t, const std::string& name,
                      std::unordered_map<std::string, TensorRaw>& raw)
{
    auto it = raw.find(name);
    if (it == raw.end() || it->second.data.empty())
        throw std::runtime_error("Missing tensor in .bin file: " + name);

    const TensorRaw& r = it->second;

    if (r.type == 0) {
        // Legacy: raw FP16 bytes (not used by current exporter but kept for compat)
        t.size = r.data.size() / 2;
        t.data = new float[t.size]();
        const uint16_t* src = reinterpret_cast<const uint16_t*>(r.data.data());
        for (size_t i = 0; i < t.size; ++i)
            t.data[i] = fp16_to_fp32(src[i]);
    } else {
        // type 1 (and any other) → FP32  (Python: param.to(float32).numpy())
        if (r.data.size() % 4 != 0)
            throw std::runtime_error("FP32 tensor has non-multiple-of-4 byte count: " + name);
        t.size = r.data.size() / 4;
        t.data = new float[t.size]();
        std::memcpy(t.data, r.data.data(), r.data.size());
    }
}

static void load_ternary(Tensor& t, float& alpha_out,
                         const std::string& base_name,
                         std::unordered_map<std::string, TensorRaw>& raw)
{
    std::string packed_key = base_name + "_packed";
    std::string size_key   = base_name + "_size";
    std::string alpha_key  = base_name + "_alpha";

    auto it_p = raw.find(packed_key);
    auto it_s = raw.find(size_key);
    if (it_p == raw.end() || it_p->second.data.empty())
        throw std::runtime_error("Missing packed ternary: " + packed_key);
    if (it_s == raw.end())
        throw std::runtime_error("Missing ternary size: " + size_key);

    int orig = it_s->second.int_val;
    t.size    = static_cast<size_t>(orig);
    t.data_i8 = new int8_t[orig]();
    unpack_5in8(it_p->second.data.data(), t.data_i8, orig);

    auto it_a = raw.find(alpha_key);
    alpha_out = (it_a != raw.end()) ? it_a->second.float_val : 1.0f;
}

// ================================================================
//  RMSNorm  (inline, used only by forward())
// ================================================================
static inline void rms_norm(const float* x, float* out,
                            const float* w, int d)
{
    float ss = 0.0f;
    for (int i = 0; i < d; ++i) ss += x[i] * x[i];
    float inv = 1.0f / std::sqrt(ss / d + 1e-5f);
    for (int i = 0; i < d; ++i) out[i] = x[i] * inv * w[i];
}

// ================================================================
//  TLuminaModel  constructor / destructor
// ================================================================
TLuminaModel::TLuminaModel() {
    for (int i = 0; i < N_LAYERS; ++i)
        kv_caches.push_back(std::make_unique<LayerKVCache>(MAX_LEN, KV_DIM));

    hidden       = new float[D_MODEL]();
    hidden_norm  = new float[D_MODEL]();
    q_buf        = new float[D_MODEL]();
    k_buf        = new float[KV_DIM]();
    v_buf        = new float[KV_DIM]();
    att_scores   = new float[MAX_LEN]();
    ffn_gate_buf = new float[D_FFN]();
    ffn_up_buf   = new float[D_FFN]();
    attn_out_buf = new float[D_MODEL]();  // heap — no more stack VLA
    ffn_out_buf  = new float[D_MODEL]();  // heap — no more stack VLA
    logits       = new float[VOCAB_SIZE]();
}

TLuminaModel::~TLuminaModel() {
    delete[] hidden;    delete[] hidden_norm;
    delete[] q_buf;     delete[] k_buf;     delete[] v_buf;
    delete[] att_scores;
    delete[] ffn_gate_buf; delete[] ffn_up_buf;
    delete[] attn_out_buf; delete[] ffn_out_buf;
    delete[] logits;
}

// ================================================================
//  load()  — parse custom .bin format and fill tensors
// ================================================================
void TLuminaModel::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open model file: " + path);

    std::cout << "\n  Parsing " << path << " ..." << std::endl;

    // ---- Read all raw entries from the file ----
    std::unordered_map<std::string, TensorRaw> raw;
    uint32_t name_len;
    while (f.read(reinterpret_cast<char*>(&name_len), 4)) {
        std::string name(name_len, '\0');
        f.read(&name[0], name_len);

        uint32_t type;
        f.read(reinterpret_cast<char*>(&type), 4);

        TensorRaw tr;
        tr.type = static_cast<int>(type);

        if (type == 1 || type == 2) {          // FP32 blob / packed ternary
            uint32_t len;
            f.read(reinterpret_cast<char*>(&len), 4);
            tr.data.resize(len);
            f.read(reinterpret_cast<char*>(tr.data.data()), len);
        } else if (type == 4) {                // int32 scalar
            uint32_t dummy; f.read(reinterpret_cast<char*>(&dummy), 4); // length field
            f.read(reinterpret_cast<char*>(&tr.int_val), 4);
        } else if (type == 5) {                // float32 scalar
            uint32_t dummy; f.read(reinterpret_cast<char*>(&dummy), 4);
            f.read(reinterpret_cast<char*>(&tr.float_val), 4);
        }
        raw[std::move(name)] = std::move(tr);
    }
    std::cout << "  Parsed " << raw.size() << " entries.\n";

    // ---- Helper lambdas ----
    auto fp32_resolve = [&](Tensor& t, const std::vector<std::string>& cands) {
        std::string found = resolve_name(raw, cands);
        if (found.empty())
            throw std::runtime_error("Missing tensor, tried: " + cands.front());
        load_fp32(t, found, raw);
    };

    auto tern_resolve = [&](Tensor& t, float& alpha,
                            const std::vector<std::string>& cands) {
        std::string found;
        for (auto& c : cands) {
            if (raw.count(c + "_packed") && !raw[c + "_packed"].data.empty()) {
                found = c; break;
            }
        }
        if (found.empty())
            throw std::runtime_error("Missing ternary tensor, tried: " + cands.front());
        load_ternary(t, alpha, found, raw);
    };

    // ---- Global weights ----
    fp32_resolve(embed_w, {
        "model.embed_tokens.weight",
        "embed_tokens.weight",
        "tok_embeddings.weight"
    });
    fp32_resolve(norm_w, {
        "model.norm.weight",
        "norm.weight",
        "final_norm.weight"
    });
    fp32_resolve(head_w, {
        "lm_head.weight",
        "output.weight"
    });

    // ---- Per-layer weights ----
    blocks.resize(N_LAYERS);
    for (int i = 0; i < N_LAYERS; ++i) {
        std::string mp = "model.layers." + std::to_string(i) + ".";
        std::string pp = "layers."       + std::to_string(i) + ".";

        fp32_resolve(blocks[i].q_proj, {
            mp+"self_attn.q_proj.weight", pp+"self_attn.q_proj.weight"});
        fp32_resolve(blocks[i].k_proj, {
            mp+"self_attn.k_proj.weight", pp+"self_attn.k_proj.weight"});
        fp32_resolve(blocks[i].v_proj, {
            mp+"self_attn.v_proj.weight", pp+"self_attn.v_proj.weight"});
        fp32_resolve(blocks[i].o_proj, {
            mp+"self_attn.o_proj.weight", pp+"self_attn.o_proj.weight"});

        fp32_resolve(blocks[i].norm1_w, {
            mp+"input_layernorm.weight",        pp+"input_layernorm.weight",
            mp+"attention_norm.weight",         pp+"attention_norm.weight"});
        fp32_resolve(blocks[i].norm2_w, {
            mp+"post_attention_layernorm.weight",pp+"post_attention_layernorm.weight",
            mp+"ffn_norm.weight",               pp+"ffn_norm.weight"});

        tern_resolve(blocks[i].ffn_gate_w, blocks[i].ffn_gate_alpha, {
            mp+"mlp.gate_proj.weight_fp", pp+"mlp.gate_proj.weight_fp"});
        tern_resolve(blocks[i].ffn_up_w,   blocks[i].ffn_up_alpha, {
            mp+"mlp.up_proj.weight_fp",   pp+"mlp.up_proj.weight_fp"});
        tern_resolve(blocks[i].ffn_down_w, blocks[i].ffn_down_alpha, {
            mp+"mlp.down_proj.weight_fp", pp+"mlp.down_proj.weight_fp"});

        if (i % 5 == 0 || i == N_LAYERS - 1)
            std::cout << "  Loaded layer " << i << "\n";
    }
    std::cout << "  All weights loaded OK.\n";
}

// ================================================================
//  forward()  — one token, one step
// ================================================================
float* TLuminaModel::forward(int token, int pos) {
    if (token < 0 || token >= VOCAB_SIZE) return logits;

    // Embedding lookup
    const float* emb = embed_w.data + token * D_MODEL;
    std::memcpy(hidden, emb, D_MODEL * sizeof(float));

    const float scale_attn = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    const int   gqa_ratio  = N_HEADS / N_KV_HEADS;  // 8

    for (int l = 0; l < N_LAYERS; ++l) {
        // ---- Pre-attention RMSNorm ----
        rms_norm(hidden, hidden_norm, blocks[l].norm1_w.data, D_MODEL);

        // ---- QKV projections ----
        fp32_matmul(hidden_norm, blocks[l].q_proj.data, q_buf, D_MODEL, D_MODEL);
        fp32_matmul(hidden_norm, blocks[l].k_proj.data, k_buf, KV_DIM,  D_MODEL);
        fp32_matmul(hidden_norm, blocks[l].v_proj.data, v_buf, KV_DIM,  D_MODEL);

        // ---- RoPE + KV cache update ----
        apply_rope(q_buf, k_buf, pos, HEAD_DIM, N_HEADS, N_KV_HEADS);
        kv_caches[l]->update(pos, k_buf, v_buf);

        // ---- Multi-head attention ----
        //  attn_out_buf is zeroed before each use (heap buffer, not stack VLA)
        std::memset(attn_out_buf, 0, D_MODEL * sizeof(float));

        for (int h = 0; h < N_HEADS; ++h) {
            float*        q_h  = q_buf + h * HEAD_DIM;
            int           kv_h = h / gqa_ratio;
            const int8_t* k_base = kv_caches[l]->k_cache + kv_h * HEAD_DIM;
            const int8_t* v_base = kv_caches[l]->v_cache + kv_h * HEAD_DIM;

            // Compute attention scores
            float max_s = -1e30f;
            for (int p = 0; p <= pos; ++p) {
                float s = quant_dot(q_h,
                                    k_base + p * KV_DIM,
                                    kv_caches[l]->k_scales[p],
                                    HEAD_DIM) * scale_attn;
                att_scores[p] = s;
                if (s > max_s) max_s = s;
            }

            // Softmax
            float sum_e = 0.0f;
            for (int p = 0; p <= pos; ++p) {
                att_scores[p] = std::exp(att_scores[p] - max_s);
                sum_e += att_scores[p];
            }
            float inv_sum = 1.0f / sum_e;
            for (int p = 0; p <= pos; ++p) att_scores[p] *= inv_sum;

            // Weighted sum of values
            float* out_h = attn_out_buf + h * HEAD_DIM;
            for (int p = 0; p <= pos; ++p) {
                float        w      = att_scores[p];
                float        vscale = kv_caches[l]->v_scales[p];
                const int8_t* vp    = v_base + p * KV_DIM;
                for (int d = 0; d < HEAD_DIM; ++d)
                    out_h[d] += w * vp[d] * vscale;
            }
        }

        // ---- Output projection + residual ----
        fp32_matmul(attn_out_buf, blocks[l].o_proj.data,
                    ffn_out_buf,  D_MODEL, D_MODEL);  // reuse ffn_out_buf temporarily
        for (int i = 0; i < D_MODEL; ++i) hidden[i] += ffn_out_buf[i];

        // ---- Pre-FFN RMSNorm ----
        rms_norm(hidden, hidden_norm, blocks[l].norm2_w.data, D_MODEL);

        // ---- SwiGLU FFN ----
        ternary_matmul_avx2(hidden_norm, blocks[l].ffn_gate_w.data_i8,
                            ffn_gate_buf, D_FFN, D_MODEL, blocks[l].ffn_gate_alpha);
        ternary_matmul_avx2(hidden_norm, blocks[l].ffn_up_w.data_i8,
                            ffn_up_buf,   D_FFN, D_MODEL, blocks[l].ffn_up_alpha);

        for (int i = 0; i < D_FFN; ++i)
            ffn_gate_buf[i] = silu(ffn_gate_buf[i]) * ffn_up_buf[i];

        // ffn_out_buf is D_MODEL (down-proj: D_MODEL × D_FFN)
        std::memset(ffn_out_buf, 0, D_MODEL * sizeof(float));
        ternary_matmul_avx2(ffn_gate_buf, blocks[l].ffn_down_w.data_i8,
                            ffn_out_buf,  D_MODEL, D_FFN, blocks[l].ffn_down_alpha);

        for (int i = 0; i < D_MODEL; ++i) hidden[i] += ffn_out_buf[i];
    }

    // ---- Final norm + LM head ----
    rms_norm(hidden, hidden_norm, norm_w.data, D_MODEL);
    fp32_matmul(hidden_norm, head_w.data, logits, VOCAB_SIZE, D_MODEL);

    return logits;
}

// ================================================================
//  reset_cache()  — call between conversations
// ================================================================
void TLuminaModel::reset_cache() {
    for (auto& c : kv_caches) {
        std::memset(c->k_cache,  0, MAX_LEN * KV_DIM * sizeof(int8_t));
        std::memset(c->v_cache,  0, MAX_LEN * KV_DIM * sizeof(int8_t));
        std::memset(c->k_scales, 0, MAX_LEN * sizeof(float));
        std::memset(c->v_scales, 0, MAX_LEN * sizeof(float));
    }
}
