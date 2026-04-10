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
    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
    return r_usage.ru_maxrss / 1024.0;
}

int sample_top_p(float* logits, int size, float temp, float top_p) {
    // ⚡ FIXED: NaN Guard to prevent Segmentation Fault!
    for (int i = 0; i < size; ++i) {
        if (std::isnan(logits[i]) || std::isinf(logits[i])) {
            std::cerr << "\n\033[1;31m[MATH ERROR] NaN detected in model calculation! Safely stopping generation.\033[0m" << std::endl;
            return 2; // return EOS token safely
        }
        logits[i] /= temp;
    }

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
    std::sort(probs.begin(), probs.end(), [](auto& a, auto& b) { return a.first > b.first; });
    float cumulative = 0.0f;
    std::vector<std::pair<float, int>> filtered;
    for (auto& p : probs) {
        filtered.push_back(p);
        cumulative += p.first;
        if (cumulative > top_p) break;
    }
    float filtered_sum = 0.0f;
    for (auto& p : filtered) filtered_sum += p.first;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    float r = dis(gen) * filtered_sum;
    float acc = 0.0f;
    for (auto& p : filtered) {
        acc += p.first;
        if (acc >= r) return p.second;
    }
    return filtered.back().second;
}

int main() {
    std::cout << "\033[1;36m====================================================\033[0m\n";
    std::cout << "\033[1;33mT-LUMINA INFERENCE ENGINE (Ternary-First LLM)\033[0m\n";
    std::cout << "\033[1;32mArchitecture by (C) Abdul Aleem, Dinajpur, Bangladesh\033[0m\n";
    std::cout << "\033[1;36m====================================================\033[0m\n\n";

    std::cout << "Loading T-Lumina Model..." << std::flush;
    TLuminaModel model;
    model.vocab_size = 32000;
    
    try {
        model.load("tlumina_model.bin");
        std::cout << " \033[1;32m[OK]\033[0m\n";
        std::cout << "RAM Usage after load: " << std::fixed << std::setprecision(2) << get_ram_usage_mb() << " MB\n";
        std::cout << "Type 'exit' or 'quit' to stop.\n" << std::endl;
    } catch(std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }

    Tokenizer tokenizer;
    std::string input;

    while (true) {
        std::cout << "\n\033[1;32mPrompt>\033[0m ";
        if (!std::getline(std::cin, input) || input == "exit" || input == "quit") {
            break;
        }
        if (input.empty()) continue;

        std::vector<int> tokens = tokenizer.encode(input);
        
        if (tokens.empty()) {
            std::cout << "\033[1;31m[Error]: Unrecognized characters. Please try a different word.\033[0m" << std::endl;
            continue;
        }

        if (tokens.size() >= static_cast<size_t>(model.max_len - 10)) {
            std::cout << "Prompt too long! Max limit is " << model.max_len - 10 << " tokens." << std::endl;
            continue;
        }

        std::cout << "\033[1;34mT-Lumina>\033[0m " << input << std::flush;

        auto start_time = std::chrono::high_resolution_clock::now();
        clock_t cpu_start = clock();
        int generated_tokens = 0;

        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            model.forward(tokens[i], static_cast<int>(i));
        }

        int next_token = tokens.back();
        int current_pos = static_cast<int>(tokens.size()) - 1;
        int max_gen = model.max_len - current_pos - 1;

        for (int i = 0; i < max_gen; ++i, ++current_pos) {
            float* logits = model.forward(next_token, current_pos);
            next_token = sample_top_p(logits, model.vocab_size, 0.8f, 0.9f);
            
            if (next_token == tokenizer.eos_token) break;
            
            std::string word = tokenizer.decode(next_token);
            std::cout << word << std::flush;
            generated_tokens++;
        }
        std::cout << std::endl;

        auto end_time = std::chrono::high_resolution_clock::now();
        clock_t cpu_end = clock();

        std::chrono::duration<double> diff = end_time - start_time;
        double wall_time = diff.count();
        double cpu_time = ((double) (cpu_end - cpu_start)) / CLOCKS_PER_SEC;
        double tok_per_sec = generated_tokens / wall_time;
        double cpu_utilization = (cpu_time / wall_time) * 100.0;
        double current_ram = get_ram_usage_mb();

        std::cout << "\n\033[90m┌──────────────────────────────────────────────┐\033[0m\n";
        std::cout << "\033[90m│\033[0m \033[1;35m⚡ PERFORMANCE METRICS\033[0m                         \033[90m│\033[0m\n";
        std::cout << "\033[90m├──────────────────────────────────────────────┤\033[0m\n";
        std::cout << "\033[90m│\033[0m Tokens Generated : " << std::left << std::setw(25) << generated_tokens << " \033[90m│\033[0m\n";
        std::cout << "\033[90m│\033[0m Time Taken       : " << std::left << std::fixed << std::setprecision(3) << std::setw(21) << wall_time << " sec \033[90m│\033[0m\n";
        std::cout << "\033[90m│\033[0m Speed            : \033[1;32m" << std::left << std::fixed << std::setprecision(2) << std::setw(21) << tok_per_sec << " tok/s\033[0m \033[90m│\033[0m\n";
        std::cout << "\033[90m│\033[0m RAM Usage        : " << std::left << std::fixed << std::setprecision(2) << std::setw(21) << current_ram << " MB  \033[90m│\033[0m\n";
        std::cout << "\033[90m│\033[0m CPU Utilization  : " << std::left << std::fixed << std::setprecision(1) << std::setw(21) << cpu_utilization << " %   \033[90m│\033[0m\n";
        std::cout << "\033[90m└──────────────────────────────────────────────┘\033[0m\n";
    }
    
    std::cout << "Exiting. Goodbye!" << std::endl;
    return 0;
}