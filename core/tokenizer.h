#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <algorithm>

class Tokenizer {
public:
    int bos_token = 1;
    int eos_token = 2; // TinyLlama uses 2 as EOS
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;

    Tokenizer() {
        std::ifstream f("vocab.bin", std::ios::binary);
        if (!f) {
            std::cerr << "Warning: 'vocab.bin' not found!" << std::endl;
            return;
        }

        uint32_t len;
        int id = 0;
        while (f.read(reinterpret_cast<char*>(&len), 4)) {
            std::string s(len, ' ');
            if (len > 0) f.read(&s[0], len);
            id_to_token.push_back(s);
            
            if (token_to_id.find(s) == token_to_id.end()) {
                token_to_id[s] = id;
            }
            id++;
        }
    }

    // ⚡ FIXED: Llama SentencePiece Space Conversion
    std::vector<int> encode(const std::string& text) {
        std::vector<int> ids;
        
        // Llama usually expects a starting space block
        std::string processed = "\xe2\x96\x81"; 
        for (char c : text) {
            if (c == ' ') processed += "\xe2\x96\x81";
            else processed += c;
        }

        size_t pos = 0;
        while (pos < processed.length()) {
            bool matched = false;
            size_t max_len = std::min(processed.length() - pos, (size_t)32); 
            
            for (size_t len = max_len; len > 0; --len) {
                std::string sub = processed.substr(pos, len);
                auto it = token_to_id.find(sub);
                if (it != token_to_id.end()) {
                    ids.push_back(it->second);
                    pos += len;
                    matched = true;
                    break;
                }
            }
            if (!matched) pos++; // Skip unknown chars safely
        }
        return ids;
    }
    
    std::string decode(int id) {
        if (id == eos_token || id < 0 || id >= static_cast<int>(id_to_token.size())) {
            return "";
        }
        std::string word = id_to_token[id];
        // Clean up Llama's block character back to standard space for console
        size_t block_pos;
        while ((block_pos = word.find("\xe2\x96\x81")) != std::string::npos) {
            word.replace(block_pos, 3, " ");
        }
        return word;
    }
};