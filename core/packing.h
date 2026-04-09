#pragma once
#include <cstdint>
#include <cstddef>

struct UnpackLUT {
    int8_t table[256][5];
    constexpr UnpackLUT() : table{} {
        for (int i = 0; i < 256; ++i) {
            int val = i;
            for (int j = 0; j < 5; ++j) {
                table[i][j] = (val % 3) - 1;
                val /= 3;
            }
        }
    }
};

constexpr UnpackLUT unpack_lut;

inline void unpack_5in8(const uint8_t* packed, int8_t* unpacked, size_t original_size) {
    size_t num_bytes = (original_size + 4) / 5;
    for (size_t i = 0; i < num_bytes; ++i) {
        uint8_t val = packed[i];
        const int8_t* lut_row = unpack_lut.table[val];
        
        if (i * 5 + 4 < original_size) {
            unpacked[i * 5 + 0] = lut_row[0];
            unpacked[i * 5 + 1] = lut_row[1];
            unpacked[i * 5 + 2] = lut_row[2];
            unpacked[i * 5 + 3] = lut_row[3];
            unpacked[i * 5 + 4] = lut_row[4];
        } else {
            for (int j = 0; j < 5; ++j) {
                if (i * 5 + j < original_size) {
                    unpacked[i * 5 + j] = lut_row[j];
                }
            }
        }
    }
}

inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
        exp++; mant &= ~0x400;
    } else if (exp == 0x1F) {
        exp = 0xFF;
    } else {
        exp += (127 - 15);
    }
    
    uint32_t res = (sign << 31) | (exp << 23) | (mant << 13);
    float f; 
    __builtin_memcpy(&f, &res, 4);
    return f;
}