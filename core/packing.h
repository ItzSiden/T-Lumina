#pragma once
#include <cstdint>
#include <cstddef>

// -------------------------------------------------------
// Base-3 5-in-8 Unpacking (LUT-accelerated)
// Each byte encodes 5 ternary weights {-1, 0, +1}.
// Python packs as: byte = sum(w[j]+1) * 3^j  for j in 0..4
// -------------------------------------------------------
struct UnpackLUT {
    int8_t table[256][5];
    constexpr UnpackLUT() : table{} {
        for (int i = 0; i < 256; ++i) {
            int v = i;
            for (int j = 0; j < 5; ++j) {
                table[i][j] = static_cast<int8_t>((v % 3) - 1);
                v /= 3;
            }
        }
    }
};

static constexpr UnpackLUT unpack_lut;

inline void unpack_5in8(const uint8_t* packed, int8_t* out, size_t original_size) {
    size_t num_bytes = (original_size + 4) / 5;
    for (size_t i = 0; i < num_bytes; ++i) {
        const int8_t* row = unpack_lut.table[packed[i]];
        size_t base = i * 5;
        size_t rem  = original_size - base;
        size_t take = rem < 5 ? rem : 5;
        for (size_t j = 0; j < take; ++j)
            out[base + j] = row[j];
    }
}

// -------------------------------------------------------
// FP16 → FP32  (IEEE 754 half-precision conversion)
// -------------------------------------------------------
inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;                        // ±0
        } else {
            // Subnormal: normalise
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000u | (mant << 13); // ±Inf or NaN
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float result;
    __builtin_memcpy(&result, &f, 4);
    return result;
}
