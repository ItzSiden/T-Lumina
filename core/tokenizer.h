#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <algorithm>

// -------------------------------------------------------
// SentencePiece-style BPE tokenizer
// Reads vocab.bin (same directory as the binary)
// Format: repeated  [ uint32 length | utf8 bytes ]
// -------------------------------------------------------
class Tokenizer {
public:
    static constexpr int BOS = 1;
    static constexpr int EOS = 2;

    // Keep old names for main.cpp compat
    int bos_token = BOS;
    int eos_token = EOS;

    std::vector<std::string>          id_to_token;
    std::unordered_map<std::string,int> token_to_id;

    explicit Tokenizer(const std::string& vocab_path = "vocab.bin") {
        std::ifstream f(vocab_path, std::ios::binary);
        if (!f) {
            std::cerr << "[Tokenizer] Warning: '" << vocab_path << "' not found.\n";
            return;
        }
        uint32_t len;
        while (f.read(reinterpret_cast<char*>(&len), 4)) {
            std::string s(len, '\0');
            if (len > 0) f.read(&s[0], len);
            int id = static_cast<int>(id_to_token.size());
            id_to_token.push_back(s);
            token_to_id.emplace(s, id);
        }
        std::cout << "[Tokenizer] Loaded " << id_to_token.size() << " tokens.\n";
    }

    // Encode: greedy longest-match BPE with Llama '▁' space convention
    std::vector<int> encode(const std::string& text) const {
        // Prepend leading space marker as Llama SentencePiece does
        static const std::string SPACE_SYM = "\xe2\x96\x81";
        std::string processed = SPACE_SYM;
        for (char c : text)
            processed += (c == ' ') ? SPACE_SYM : std::string(1, c);

        std::vector<int> ids;
        size_t pos = 0;
        while (pos < processed.size()) {
            size_t best_len = 0;
            int    best_id  = -1;
            size_t max_try  = std::min(processed.size() - pos, (size_t)32);
            for (size_t len = max_try; len > 0; --len) {
                auto it = token_to_id.find(processed.substr(pos, len));
                if (it != token_to_id.end()) {
                    best_len = len;
                    best_id  = it->second;
                    break;
                }
            }
            if (best_id >= 0) {
                ids.push_back(best_id);
                pos += best_len;
            } else {
                pos++;  // skip unknown byte
            }
        }
        return ids;
    }

    // Decode: convert '▁' back to space
    std::string decode(int id) const {
        if (id == EOS || id < 0 || id >= static_cast<int>(id_to_token.size()))
            return "";
        std::string word = id_to_token[id];
        static const std::string SPACE_SYM = "\xe2\x96\x81";
        size_t p;
        while ((p = word.find(SPACE_SYM)) != std::string::npos)
            word.replace(p, 3, " ");
        return word;
    }
};
