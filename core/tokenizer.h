#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <algorithm>

class Tokenizer {
public:
    int bos_token = 50256;
    int eos_token = 50256;
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;

    Tokenizer() {
        std::ifstream f("vocab.bin", std::ios::binary);
        if (!f) {
            std::cerr << "Warning: 'vocab.bin' not found! Run export_vocab.py first." << std::endl;
            return;
        }

        uint32_t len;
        int id = 0;
        while (f.read(reinterpret_cast<char*>(&len), 4)) {
            std::string s(len, ' ');
            if (len > 0) f.read(&s[0], len);
            id_to_token.push_back(s);
            
            // Map string to ID. Avoid overwriting earlier tokens (prefer lower IDs for common words)
            if (token_to_id.find(s) == token_to_id.end()) {
                token_to_id[s] = id;
            }
            id++;
        }
    }

    // Encoding: Greedy Longest-Prefix Match (Fast and effective for English)
    std::vector<int> encode(const std::string& text) {
        std::vector<int> ids;
        size_t pos = 0;
        
        while (pos < text.length()) {
            bool matched = false;
            // Max GPT-2 token length is rarely above 32 chars
            size_t max_len = std::min(text.length() - pos, (size_t)32); 
            
            for (size_t len = max_len; len > 0; --len) {
                std::string sub = text.substr(pos, len);
                auto it = token_to_id.find(sub);
                if (it != token_to_id.end()) {
                    ids.push_back(it->second);
                    pos += len;
                    matched = true;
                    break;
                }
            }
            // Fallback (should theoretically never happen as GPT2 has byte-level coverage)
            if (!matched) pos++; 
        }
        return ids;
    }
    
    // Decoding: Convert IDs back to words perfectly
    std::string decode(int id) {
        if (id == eos_token || id < 0 || id >= static_cast<int>(id_to_token.size())) {
            return "";
        }
        return id_to_token[id];
    }
};