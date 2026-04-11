#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include <sys/resource.h>
#include "core/model.h"
#include "core/tokenizer.h"

double get_ram_usage_mb() {
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
    return r.ru_maxrss / 1024.0;
}

int sample_top_p(float* logits, int size, float temp, float top_p) {
    for (int i = 0; i < size; ++i) logits[i] /= temp;

    float max_l = logits[0];
    for (int i = 1; i < size; ++i) max_l = std::max(max_l, logits[i]);

    float sum_exp = 0.0f;
    for (int i = 0; i < size; ++i) {
        logits[i] = std::exp(logits[i] - max_l);
        sum_exp += logits[i];
    }
    for (int i = 0; i < size; ++i) logits[i] /= sum_exp;

    std::vector<std::pair<float, int>> probs(size);
    for (int i = 0; i < size; ++i) probs[i] = {logits[i], i};
    std::sort(probs.begin(), probs.end(), [](auto& a, auto& b){ return a.first > b.first; });

    float cum = 0.0f;
    std::vector<std::pair<float,int>> filtered;
    for (auto& p : probs) {
        filtered.push_back(p);
        cum += p.first;
        if (cum > top_p) break;
    }

    float fsum = 0.0f;
    for (auto& p : filtered) fsum += p.first;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    float r = dis(gen) * fsum;

    float acc = 0.0f;
    for (auto& p : filtered) {
        acc += p.first;
        if (acc >= r) return p.second;
    }
    return filtered.back().second;
}

int main(int argc, char* argv[]) {
    std::cout << "\033[1;36m====================================================\033[0m\n";
    std::cout << "\033[1;33m  T-LUMINA INFERENCE ENGINE (Ternary-First LLM)\033[0m\n";
    std::cout << "\033[1;32m  Architecture by (C) Abdul Aleem, Dinajpur, BD\033[0m\n";
    std::cout << "\033[1;36m====================================================\033[0m\n\n";

    // Binary ও config path — argument থেকে নেয়, না হলে default
    std::string bin_path    = "tlumina_model.bin";
    std::string config_path = "config.json";

    if (argc >= 2) bin_path    = argv[1];
    if (argc >= 3) config_path = argv[2];

    std::cout << "Loading model: " << bin_path << "\n";
    std::cout << "Config:        " << config_path << "\n\n";

    TLuminaModel model;
    try {
        model.load(bin_path, config_path);
        std::cout << "\033[1;32m[OK] Model loaded!\033[0m\n";
        std::cout << "RAM after load: " << std::fixed << std::setprecision(1)
                  << get_ram_usage_mb() << " MB\n";
        std::cout << "Type 'exit' or 'quit' to stop.\n\n";
    } catch (std::exception& e) {
        std::cerr << "\033[1;31mError: " << e.what() << "\033[0m\n";
        std::cerr << "Usage: ./tlumina [model.bin] [config.json]\n";
        return 1;
    }

    // vocab file — model এর পাশে বা default
    std::string vocab_path = "vocab.bin";
    // config এর tokenizer type দিয়ে tokenizer initialize করো
    Tokenizer tokenizer(vocab_path, model.cfg.tokenizer_type);
    if (tokenizer.id_to_token.empty()) {
        std::cerr << "\033[1;31mError: " << vocab_path << " not found!\033[0m\n";
        std::cerr << "GPT-2 model:  python scripts/export_vocab.py\n";
        std::cerr << "LLaMA model:  python scripts/export_vocab_tinyllama.py\n";
        return 1;
    }

    std::string input;
    while (true) {
        std::cout << "\n\033[1;32mPrompt>\033[0m ";
        if (!std::getline(std::cin, input) || input == "exit" || input == "quit") break;
        if (input.empty()) continue;

        // Chat template wrap করো যদি llama mode হয়
        std::string wrapped_input;
        if (model.cfg.tokenizer_type == "llama") {
            wrapped_input = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n" 
                          + input + "</s>\n<|assistant|>\n";
        } else {
            wrapped_input = input;
        }

        std::vector<int> tokens = tokenizer.encode(wrapped_input);
        if (tokens.empty()) { std::cout << "(empty encode)\n"; continue; }

        if ((int)tokens.size() >= model.cfg.max_len - 10) {
            std::cout << "Prompt too long! Max: " << model.cfg.max_len - 10 << " tokens.\n";
            continue;
        }

        // ✅ Cache reset প্রতিটা নতুন prompt এ
        model.reset_cache();

        std::cout << "\033[1;34mT-Lumina>\033[0m " << input << std::flush;

        auto  t0        = std::chrono::high_resolution_clock::now();
        clock_t cpu0    = clock();
        int gen_tokens  = 0;

        // Prompt prefill
        for (size_t i = 0; i + 1 < tokens.size(); ++i)
            model.forward(tokens[i], (int)i);

        int next_token  = tokens.back();
        int cur_pos     = (int)tokens.size() - 1;
        int max_gen     = model.cfg.max_len - cur_pos - 1;

        // Generate
        for (int i = 0; i < max_gen; ++i, ++cur_pos) {
            float* logits = model.forward(next_token, cur_pos);
            next_token    = sample_top_p(logits, model.cfg.vocab_size, 0.8f, 0.9f);

            if (next_token == tokenizer.eos_token) break;

            std::cout << tokenizer.decode(next_token) << std::flush;
            gen_tokens++;
        }
        std::cout << "\n";

        // Metrics
        auto   t1       = std::chrono::high_resolution_clock::now();
        clock_t cpu1    = clock();
        double wall     = std::chrono::duration<double>(t1 - t0).count();
        double cpu_t    = (double)(cpu1 - cpu0) / CLOCKS_PER_SEC;
        double tps      = gen_tokens / wall;
        double cpu_util = (cpu_t / wall) * 100.0;

        std::cout << "\n\033[90m┌──────────────────────────────────────────────┐\033[0m\n";
        std::cout << "\033[90m│\033[0m \033[1;35m⚡ PERFORMANCE METRICS\033[0m                         \033[90m│\033[0m\n";
        std::cout << "\033[90m├──────────────────────────────────────────────┤\033[0m\n";
        std::cout << "\033[90m│\033[0m Tokens Generated : " << std::left << std::setw(25) << gen_tokens     << " \033[90m│\033[0m\n";
        std::cout << "\033[90m│\033[0m Time Taken       : " << std::left << std::fixed << std::setprecision(3) << std::setw(21) << wall      << " sec \033[90m│\033[0m\n";
        std::cout << "\033[90m│\033[0m Speed            : \033[1;32m" << std::left << std::fixed << std::setprecision(2) << std::setw(21) << tps       << " tok/s\033[0m \033[90m│\033[0m\n";
        std::cout << "\033[90m│\033[0m RAM Usage        : " << std::left << std::fixed << std::setprecision(1) << std::setw(21) << get_ram_usage_mb() << " MB  \033[90m│\033[0m\n";
        std::cout << "\033[90m│\033[0m CPU Utilization  : " << std::left << std::fixed << std::setprecision(1) << std::setw(21) << cpu_util  << " %   \033[90m│\033[0m\n";
        std::cout << "\033[90m└──────────────────────────────────────────────┘\033[0m\n";
    }

    std::cout << "Goodbye!\n";
    return 0;
}