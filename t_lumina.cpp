// ============================================================
// T-Lumina Raw Inference Engine
// C++17 | SSE2 Compatible | No AVX Required
// ============================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <thread>
#include <unordered_map>
#include <algorithm>
#include <mutex>

// ================= MODEL CONFIG =================

constexpr int D_MODEL = 256;
constexpr int N_HEADS = 8;
constexpr int D_HEAD  = 32;
constexpr int D_FFN   = 1024;
constexpr int N_LAYERS = 8;
constexpr int VOCAB_SIZE = 50048;
constexpr int CONTEXT = 256;

// ============================================================
// FP16 → FP32
// ============================================================

float fp16_to_fp32(uint16_t h)
{
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF);

    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FF;
            exp += (127 - 15);
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        exp = exp + (127 - 15);
        f = sign | (exp << 23) | (mant << 13);
    }

    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}

// ============================================================
// TERNARY UNPACK (5-in-8 Base-3)
// ============================================================

inline void unpack_5in8(uint8_t byte, int8_t out[5])
{
    out[0] =  (byte % 3) - 1;
    out[1] = ((byte / 3) % 3) - 1;
    out[2] = ((byte / 9) % 3) - 1;
    out[3] = ((byte / 27) % 3) - 1;
    out[4] = ((byte / 81) % 3) - 1;
}

// ============================================================
// RAW BINARY LOADER
// ============================================================

struct TensorData {
    uint32_t type;
    std::vector<uint8_t> raw;
};

std::unordered_map<std::string, TensorData>
load_model(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    std::unordered_map<std::string, TensorData> map;

    while (file) {
        uint32_t name_len;
        if (!file.read((char*)&name_len, 4)) break;

        std::string name(name_len, '\0');
        file.read(&name[0], name_len);

        uint32_t type;
        file.read((char*)&type, 4);

        uint32_t data_len;
        file.read((char*)&data_len, 4);

        std::vector<uint8_t> data(data_len);
        file.read((char*)data.data(), data_len);

        map[name] = {type, std::move(data)};
    }

    return map;
}

// ============================================================
// PARAMETER STRUCT
// ============================================================

struct Parameter {
    float* embedding;
    float* rms[N_LAYERS];

    uint8_t* ffn_w1[N_LAYERS];
    uint8_t* ffn_w2[N_LAYERS];
};

// ============================================================
// INITIALIZER
// ============================================================

void initialize(Parameter& p,
    std::unordered_map<std::string, TensorData>& w)
{
    // Embedding
    if (w.find("embed.weight") == w.end()) {
        std::cerr << "Missing tensor: embed.weight" << std::endl;
        exit(1);
    }
    auto& emb = w.at("embed.weight");
    size_t count = emb.raw.size() / 2;

    p.embedding = new float[count];
    uint16_t* src = (uint16_t*)emb.raw.data();
    for (size_t i = 0; i < count; i++)
        p.embedding[i] = fp16_to_fp32(src[i]);

    // RMSNorm + FFN packed
    for (int i = 0; i < N_LAYERS; i++) {

        std::string rms_name =
            "blocks." + std::to_string(i) + ".norm1.weight";

        if (w.find(rms_name) == w.end()) {
            std::cerr << "Missing tensor: " << rms_name << std::endl;
            exit(1);
        }
        auto& r = w.at(rms_name);
        p.rms[i] = new float[D_MODEL];

        uint16_t* rs = (uint16_t*)r.raw.data();
        for (int j = 0; j < D_MODEL; j++)
            p.rms[i][j] = fp16_to_fp32(rs[j]);

        std::string ffn_w1_name = "blocks."+std::to_string(i)+".ffn.gate.weight_fp_packed";
        if (w.find(ffn_w1_name) == w.end()) {
            std::cerr << "Missing tensor: " << ffn_w1_name << std::endl;
            exit(1);
        }
        p.ffn_w1[i] = w.at(ffn_w1_name).raw.data();

        std::string ffn_w2_name = "blocks."+std::to_string(i)+".ffn.down.weight_fp_packed";
        if (w.find(ffn_w2_name) == w.end()) {
            std::cerr << "Missing tensor: " << ffn_w2_name << std::endl;
            exit(1);
        }
        p.ffn_w2[i] = w.at(ffn_w2_name).raw.data();
    }
}

// ============================================================
// RMSNorm
// ============================================================

void rmsnorm(float* x, float* w)
{
    float ss = 0.f;
    for (int i = 0; i < D_MODEL; i++)
        ss += x[i] * x[i];

    float inv = 1.f / std::sqrt(ss / D_MODEL + 1e-6f);

    for (int i = 0; i < D_MODEL; i++)
        x[i] = x[i] * inv * w[i];
}

// ============================================================
// TERNARY GEMM (Multicore)
// ============================================================

void ternary_gemm(
    const float* input,
    const uint8_t* packed,
    float* output,
    int in_dim,
    int out_dim)
{
    unsigned threads = std::thread::hardware_concurrency();
    if (threads == 0) threads = 2;

    int chunk = out_dim / threads;
    std::vector<std::thread> pool;

    for (unsigned t = 0; t < threads; t++) {

        int start = t * chunk;
        int end   = (t == threads-1) ? out_dim : start + chunk;

        pool.emplace_back([=]() {

            int packed_stride = (in_dim + 4) / 5;

            for (int o = start; o < end; o++) {

                float acc = 0.f;
                const uint8_t* row =
                    packed + o * packed_stride;

                for (int p = 0; p < packed_stride; p++) {

                    int8_t w5[5];
                    unpack_5in8(row[p], w5);

                    for (int i = 0; i < 5; i++) {
                        int idx = p*5 + i;
                        if (idx >= in_dim) break;

                        if (w5[i] == 1)
                            acc += input[idx];
                        else if (w5[i] == -1)
                            acc -= input[idx];
                    }
                }

                output[o] = acc;
            }
        });
    }

    for (auto& th : pool) th.join();
}

// ============================================================
// SIMPLE GENERATE
// ============================================================

void generate(Parameter& p)
{
    float x[D_MODEL];

    // fake input token = 1
    std::memcpy(x,
        p.embedding + D_MODEL,
        sizeof(float)*D_MODEL);

    for (int l = 0; l < N_LAYERS; l++) {

        rmsnorm(x, p.rms[l]);

        float f1[D_FFN] = {0};
        ternary_gemm(x, p.ffn_w1[l], f1, D_MODEL, D_FFN);

        for (int i = 0; i < D_FFN; i++)
            f1[i] = std::max(0.f, f1[i]);

        float f2[D_MODEL] = {0};
        ternary_gemm(f1, p.ffn_w2[l], f2, D_FFN, D_MODEL);

        for (int i = 0; i < D_MODEL; i++)
            x[i] += f2[i];
    }

    std::cout << "Forward pass complete.\n";
}

// ============================================================
// MAIN
// ============================================================

int main()
{
    std::cout << "Loading T-Lumina...\n";

    auto weights = load_model("tlumina_model.bin");

    for (auto& kv : weights)
        std::cout << kv.first << std::endl;

    Parameter params{};
    initialize(params, weights);

    std::cout << "Model loaded.\n";

    generate(params);

    return 0;
}
