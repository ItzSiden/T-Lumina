#include <iostream>
#include "core/tokenizer.h"
int main() {
    Tokenizer tok("vocab_tinyllama.bin", "llama");
    auto ids = tok.encode("hello");
    std::cout << "Tokens: ";
    for (int id : ids) std::cout << id << " ";
    std::cout << "\n";
    for (int id : ids) std::cout << "[" << tok.decode(id) << "] ";
    std::cout << "\n";
    // vocab এর কিছু tokens দেখো
    for (int i = 0; i < 10; ++i)
        std::cout << i << ": [" << tok.id_to_token[i] << "]\n";
    return 0;