#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <sys/resource.h>
#include "core/model.h"
#include "core/tokenizer.h"

// -------------------------------------------------------
// RAM usage helper (Linux)
// -------------------------------------------------------
static double ram_mb() {
    struct rusage r; getrusage(RUSAGE_SELF, &r);
    return r.ru_maxrss / 1024.0;
}

// -------------------------------------------------------
// Top-p (nucleus) sampling with temperature
// Returns a token id, or EOS (2) on NaN/Inf.
// -------------------------------------------------------
static int sample_top_p(float* logits, int size, float temp, float top_p) {
    // NaN / Inf guard
    for (int i = 0; i < size; ++i) {
        if (!std::isfinite(logits[i])) {
            std::cerr << "\n[WARN] Non-finite logit detected — returning EOS.\n";
            return Tokenizer::EOS;
        }
        logits[i] /= temp;
    }

    // Softmax (numerically stable)
    float max_l = *std::max_element(logits, logits + size);
    float sum_e = 0.0f;
    for (int i = 0; i < size; ++i) { logits[i] = std::exp(logits[i] - max_l); sum_e += logits[i]; }
    for (int i = 0; i < size; ++i) logits[i] /= sum_e;

    // Sort descending
    static std::vector<std::pair<float,int>> probs;
    probs.resize(size);
    for (int i = 0; i < size; ++i) probs[i] = {logits[i], i};
    std::sort(probs.begin(), probs.end(),
              [](const auto& a, const auto& b){ return a.first > b.first; });

    // Nucleus truncation
    float cum = 0.0f;
    int   cut = 0;
    for (; cut < size; ++cut) {
        cum += probs[cut].first;
        if (cum >= top_p) { ++cut; break; }
    }
    if (cut == 0) cut = 1;

    // Re-normalise nucleus
    float nsum = 0.0f;
    for (int i = 0; i < cut; ++i) nsum += probs[i].first;

    std::mt19937& rng = []() -> std::mt19937& {
        static std::mt19937 g(std::random_device{}());
        return g;
    }();
    std::uniform_real_distribution<float> dis(0.0f, nsum);
    float r = dis(rng), acc = 0.0f;
    for (int i = 0; i < cut; ++i) {
        acc += probs[i].first;
        if (acc >= r) return probs[i].second;
    }
    return probs[cut - 1].second;
}

// -------------------------------------------------------
// main
// -------------------------------------------------------
int main(int argc, char* argv[]) {
    std::string model_path = "tlumina_model.bin";
    if (argc > 1) model_path = argv[1];

    std::cout << "\033[1;36m====================================================\033[0m\n"
              << "\033[1;33m  T-LUMINA INFERENCE ENGINE  (Ternary-First LLM)\033[0m\n"
              << "\033[1;32m  Architecture: Abdul Aleem, Dinajpur, Bangladesh\033[0m\n"
              << "\033[1;36m====================================================\033[0m\n\n";

    std::cout << "Loading model from: " << model_path << " ..." << std::flush;
    TLuminaModel model;
    try {
        model.load(model_path);
        std::cout << " \033[1;32m[OK]\033[0m  RAM: "
                  << std::fixed << std::setprecision(1) << ram_mb() << " MB\n";
    } catch (const std::exception& e) {
        std::cerr << "\n\033[1;31m[LOAD ERROR] " << e.what() << "\033[0m\n";
        return 1;
    }

    Tokenizer tok;
    if (tok.id_to_token.empty()) {
        std::cerr << "[ERROR] Vocabulary is empty. Make sure vocab.bin is present.\n";
        return 1;
    }

    std::cout << "Type your prompt. Enter 'exit' or 'quit' to stop.\n";

    std::string line;
    while (true) {
        std::cout << "\n\033[1;32mPrompt>\033[0m ";
        if (!std::getline(std::cin, line)) break;
        if (line == "exit" || line == "quit") break;
        if (line.empty()) continue;

        // Reset KV cache for every new prompt
        model.reset_cache();

        std::vector<int> tokens = tok.encode(line);
        if (tokens.empty()) {
            std::cout << "\033[1;31m[Error] Prompt produced no tokens.\033[0m\n";
            continue;
        }
        if (static_cast<int>(tokens.size()) >= model.max_len - 8) {
            std::cout << "[Error] Prompt too long (max "
                      << model.max_len - 8 << " tokens).\n";
            continue;
        }

        std::cout << "\033[1;34mT-Lumina>\033[0m " << line << std::flush;

        auto t0 = std::chrono::steady_clock::now();

        // Prefill: run all prompt tokens except the last
        for (int i = 0; i < static_cast<int>(tokens.size()) - 1; ++i)
            model.forward(tokens[i], i);

        int next   = tokens.back();
        int cur    = static_cast<int>(tokens.size()) - 1;
        int max_gen = model.max_len - cur - 1;
        int n_gen   = 0;

        for (int step = 0; step < max_gen; ++step, ++cur) {
            float* logits = model.forward(next, cur);
            next = sample_top_p(logits, model.vocab_size, /*temp=*/0.8f, /*top_p=*/0.9f);

            if (next == tok.eos_token) break;

            std::string word = tok.decode(next);
            if (!word.empty()) std::cout << word << std::flush;
            ++n_gen;
        }
        std::cout << '\n';

        auto t1 = std::chrono::steady_clock::now();
        double wall = std::chrono::duration<double>(t1 - t0).count();
        double tps  = (wall > 0) ? n_gen / wall : 0.0;

        std::cout << "\033[90m┌──────────────────────────────────────┐\033[0m\n"
                  << "\033[90m│\033[0m \033[1;35m⚡ Stats\033[0m"
                  << "                               \033[90m│\033[0m\n"
                  << "\033[90m│\033[0m Tokens  : " << std::left << std::setw(27) << n_gen  << " \033[90m│\033[0m\n"
                  << "\033[90m│\033[0m Time    : " << std::left << std::fixed << std::setprecision(2) << std::setw(23) << wall << " sec  \033[90m│\033[0m\n"
                  << "\033[90m│\033[0m Speed   : \033[1;32m" << std::left << std::setprecision(1) << std::setw(22) << tps   << " tok/s\033[0m \033[90m│\033[0m\n"
                  << "\033[90m│\033[0m RAM     : " << std::left << std::setprecision(1) << std::setw(23) << ram_mb() << " MB  \033[90m│\033[0m\n"
                  << "\033[90m└──────────────────────────────────────┘\033[0m\n";
    }

    std::cout << "Goodbye!\n";
    return 0;
}
