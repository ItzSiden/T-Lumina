#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdint>

// ─────────────────────────────────────────────────────────
// Tokenizer — GPT-2 (BPE) এবং LLaMA (SentencePiece) দুটোই support
//
// vocab.bin format (same for both):
//   [uint32 len][utf8 bytes] × vocab_size
//
// tokenizer_type.txt:
//   "gpt2"   → GPT-2 / TinyLlama-v2 style (Ġ = space prefix)  
//   "llama"  → LLaMA / TinyLlama style (▁ = U+2581 space prefix)
// ─────────────────────────────────────────────────────────

enum class TokenizerType { GPT2, LLAMA };

class Tokenizer {
public:
    int bos_token = 1;
    int eos_token = 2;

    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
    TokenizerType type = TokenizerType::GPT2;

    // vocab_path: vocab.bin path
    // type_hint: "gpt2" বা "llama" — config থেকে আসে
    Tokenizer(const std::string& vocab_path = "vocab.bin",
              const std::string& type_hint  = "gpt2") {

        // Type detect
        if (type_hint == "llama" || type_hint == "tinyllama" ||
            type_hint == "mistral") {
            type      = TokenizerType::LLAMA;
            bos_token = 1;
            eos_token = 2;
        } else {
            // GPT-2
            type      = TokenizerType::GPT2;
            bos_token = 50256;
            eos_token = 50256;
        }

        // vocab.bin load
        std::ifstream f(vocab_path, std::ios::binary);
        if (!f) {
            std::cerr << "[Tokenizer] Warning: '" << vocab_path
                      << "' not found!\n";
            return;
        }

        uint32_t len;
        int id = 0;
        while (f.read(reinterpret_cast<char*>(&len), 4)) {
            std::string s(len, '\0');
            if (len > 0) f.read(&s[0], len);
            id_to_token.push_back(s);
            if (token_to_id.find(s) == token_to_id.end())
                token_to_id[s] = id;
            id++;
        }

        std::cout << "[Tokenizer] Loaded " << id_to_token.size()
                  << " tokens ("
                  << (type == TokenizerType::LLAMA ? "LLaMA/SP" : "GPT-2/BPE")
                  << ")\n";
    }

    // ── Encode ──────────────────────────────────────────
    std::vector<int> encode(const std::string& text) const {
        if (type == TokenizerType::LLAMA)
            return encode_llama(text);
        else
            return encode_gpt2(text);
    }

    // ── Decode ──────────────────────────────────────────
    std::string decode(int id) const {
        if (id < 0 || id >= static_cast<int>(id_to_token.size()))
            return "";
        if (id == eos_token) return "";

        const std::string& tok = id_to_token[id];

        if (type == TokenizerType::LLAMA)
            return decode_llama_token(tok);
        else
            return decode_gpt2_token(tok);
    }

private:
    // ────────────────────────────────────────────────────
    // GPT-2 encoding: Greedy longest-prefix match
    // Space → "Ġ" (U+0120) prefix
    // ────────────────────────────────────────────────────
    std::vector<int> encode_gpt2(const std::string& text) const {
        // GPT-2: spaces become Ġ prefix on next token
        std::string processed;
        for (size_t i = 0; i < text.size(); ++i) {
            if (text[i] == ' ' && i > 0)
                processed += "\xc4\xa0";  // Ġ (U+0120)
            else
                processed += text[i];
        }

        std::vector<int> ids;
        size_t pos = 0;
        while (pos < processed.size()) {
            bool matched = false;
            size_t max_l = std::min(processed.size() - pos, (size_t)32);
            for (size_t l = max_l; l > 0; --l) {
                auto it = token_to_id.find(processed.substr(pos, l));
                if (it != token_to_id.end()) {
                    ids.push_back(it->second);
                    pos += l;
                    matched = true;
                    break;
                }
            }
            if (!matched) pos++;
        }
        return ids;
    }

    // ────────────────────────────────────────────────────
    // LLaMA/SentencePiece encoding
    // Space → "▁" (U+2581) prefix
    // ────────────────────────────────────────────────────
    std::vector<int> encode_llama(const std::string& text) const {
        // SentencePiece: space → ▁ (U+2581 = \xe2\x96\x81)
        std::string processed = "\xe2\x96\x81";  // BOS space
        for (char c : text) {
            if (c == ' ')
                processed += "\xe2\x96\x81";
            else
                processed += c;
        }

        std::vector<int> ids;
        // BOS token add করো
        ids.push_back(bos_token);

        size_t pos = 0;
        while (pos < processed.size()) {
            bool matched = false;
            // UTF-8 aware max length (▁ = 3 bytes, so max 32 chars = ~96 bytes)
            size_t max_l = std::min(processed.size() - pos, (size_t)64);
            for (size_t l = max_l; l > 0; --l) {
                // UTF-8 boundary check — incomplete multibyte skip
                if (!is_valid_utf8_boundary(processed, pos, l)) continue;
                auto it = token_to_id.find(processed.substr(pos, l));
                if (it != token_to_id.end()) {
                    ids.push_back(it->second);
                    pos += l;
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                // Byte fallback: unknown char → skip
                // UTF-8 এ next char এ jump
                pos += utf8_char_len(processed[pos]);
            }
        }
        return ids;
    }

    // ────────────────────────────────────────────────────
    // GPT-2 decode: Ġ → space
    // ────────────────────────────────────────────────────
    std::string decode_gpt2_token(const std::string& tok) const {
        std::string result;
        size_t i = 0;
        while (i < tok.size()) {
            // Ġ = 0xC4 0xA0
            if (i + 1 < tok.size() &&
                (uint8_t)tok[i] == 0xC4 && (uint8_t)tok[i+1] == 0xA0) {
                result += ' ';
                i += 2;
            } else {
                result += tok[i++];
            }
        }
        return result;
    }

    // ────────────────────────────────────────────────────
    // LLaMA decode: ▁ → space
    // ────────────────────────────────────────────────────
    std::string decode_llama_token(const std::string& tok) const {
        std::string result;
        size_t i = 0;
        while (i < tok.size()) {
            // ▁ = 0xE2 0x96 0x81
            if (i + 2 < tok.size() &&
                (uint8_t)tok[i]   == 0xE2 &&
                (uint8_t)tok[i+1] == 0x96 &&
                (uint8_t)tok[i+2] == 0x81) {
                result += ' ';
                i += 3;
            } else {
                result += tok[i++];
            }
        }
        return result;
    }

    // ────────────────────────────────────────────────────
    // UTF-8 helpers
    // ────────────────────────────────────────────────────
    static size_t utf8_char_len(char c) {
        uint8_t u = static_cast<uint8_t>(c);
        if (u < 0x80) return 1;
        if (u < 0xE0) return 2;
        if (u < 0xF0) return 3;
        return 4;
    }

    // substring [pos, pos+len) টা valid UTF-8 boundary তে শেষ হচ্ছে কিনা
    static bool is_valid_utf8_boundary(const std::string& s,
                                        size_t pos, size_t len) {
        size_t end = pos + len;
        if (end > s.size()) return false;
        // end position টা continuation byte (10xxxxxx) হলে invalid
        if (end < s.size()) {
            uint8_t next = static_cast<uint8_t>(s[end]);
            if ((next & 0xC0) == 0x80) return false;
        }
        return true;
    }
};