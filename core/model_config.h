#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

// Supported architecture types
enum class ArchType {
    TLUMINA,      // তোমার নিজের model
    TINYLLAMA,    // TinyLlama 1.1B
    LLAMA,        // LLaMA / LLaMA-2
    MISTRAL,      // Mistral 7B
    UNKNOWN
};

struct ModelConfig {
    // Core dimensions
    int vocab_size  = 0;
    int d_model     = 0;
    int n_heads     = 0;
    int n_kv_heads  = 0;   // GQA: TinyLlama=4, LLaMA=n_heads, তোমার model=n_heads
    int d_ffn       = 0;
    int n_layers    = 0;
    int max_len     = 2048;
    int head_dim    = 0;
    int n_groups    = 1;   // n_heads / n_kv_heads (MQA/GQA এর জন্য)

    // Architecture type
    ArchType arch   = ArchType::TLUMINA;
    std::string arch_str = "tlumina";
    std::string tokenizer_type = "gpt2";

    // RoPE theta (default GPT-NeoX style)
    float rope_theta = 10000.0f;

    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f) {
            std::cerr << "[Config] Could not open: " << path << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(f, line)) {
            auto get_int = [&](const std::string& key, int& val) {
                size_t pos = line.find("\"" + key + "\"");
                if (pos == std::string::npos) return;
                pos = line.find(":", pos);
                if (pos == std::string::npos) return;
                try { val = std::stoi(line.substr(pos + 1)); } catch (...) {}
            };
            auto get_float = [&](const std::string& key, float& val) {
                size_t pos = line.find("\"" + key + "\"");
                if (pos == std::string::npos) return;
                pos = line.find(":", pos);
                if (pos == std::string::npos) return;
                try { val = std::stof(line.substr(pos + 1)); } catch (...) {}
            };
            auto get_str = [&](const std::string& key, std::string& val) {
                size_t pos = line.find("\"" + key + "\"");
                if (pos == std::string::npos) return;
                pos = line.find("\"", pos + key.size() + 2);
                if (pos == std::string::npos) return;
                size_t end = line.find("\"", pos + 1);
                if (end == std::string::npos) return;
                val = line.substr(pos + 1, end - pos - 1);
            };

            get_int("vocab_size",   vocab_size);
            get_int("d_model",      d_model);
            get_int("hidden_size",  d_model);    // HuggingFace name
            get_int("n_heads",      n_heads);
            get_int("num_attention_heads", n_heads);  // HuggingFace name
            get_int("n_kv_heads",   n_kv_heads);
            get_int("num_key_value_heads", n_kv_heads);  // HuggingFace name
            get_int("d_ffn",        d_ffn);
            get_int("intermediate_size", d_ffn);   // HuggingFace name
            get_int("n_layers",     n_layers);
            get_int("num_hidden_layers", n_layers); // HuggingFace name
            get_int("max_len",      max_len);
            get_int("max_position_embeddings", max_len); // HuggingFace name
            get_float("rope_theta", rope_theta);
            get_str("arch",         arch_str);
            get_str("model_type",   arch_str);  // HuggingFace name
            get_str("tokenizer",    tokenizer_type);
        }

        // Defaults ও derived values
        if (n_kv_heads == 0) n_kv_heads = n_heads;  // MHA (no GQA)
        if (d_model > 0 && n_heads > 0) head_dim = d_model / n_heads;
        if (n_heads > 0 && n_kv_heads > 0) n_groups = n_heads / n_kv_heads;

        // Arch type detect করো
        if (arch_str == "tlumina")              arch = ArchType::TLUMINA;
        else if (arch_str == "tinyllama" ||
                 arch_str == "llama")           arch = ArchType::TINYLLAMA;
        else if (arch_str == "mistral")         arch = ArchType::MISTRAL;
        else                                    arch = ArchType::UNKNOWN;

        return validate();
    }

    bool validate() const {
        if (vocab_size <= 0) { std::cerr << "[Config] vocab_size missing!\n"; return false; }
        if (d_model <= 0)    { std::cerr << "[Config] d_model/hidden_size missing!\n"; return false; }
        if (n_heads <= 0)    { std::cerr << "[Config] n_heads missing!\n"; return false; }
        if (d_ffn <= 0)      { std::cerr << "[Config] d_ffn/intermediate_size missing!\n"; return false; }
        if (n_layers <= 0)   { std::cerr << "[Config] n_layers missing!\n"; return false; }
        return true;
    }

    void print() const {
        std::cout << "[Config] arch=" << arch_str
                  << " | tokenizer=" << tokenizer_type
                  << " | vocab=" << vocab_size
                  << " | d_model=" << d_model
                  << " | heads=" << n_heads << "/" << n_kv_heads
                  << " | layers=" << n_layers
                  << " | d_ffn=" << d_ffn
                  << " | max_len=" << max_len
                  << " | rope_theta=" << rope_theta << "\n";
    }
};
