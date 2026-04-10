#pragma once
#include <vector>
#include <cstdint>
#include <string>

// -------------------------------------------------------
// Tensor: owns either float* (FP32) or int8_t* (ternary)
// -------------------------------------------------------
struct Tensor {
    float*   data     = nullptr;
    int8_t*  data_i8  = nullptr;
    size_t   size     = 0;
    std::vector<int> shape;

    ~Tensor() {
        delete[] data;
        delete[] data_i8;
    }

    Tensor() = default;
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& o) noexcept
        : data(o.data), data_i8(o.data_i8), size(o.size), shape(std::move(o.shape))
    { o.data = nullptr; o.data_i8 = nullptr; }

    Tensor& operator=(Tensor&& o) noexcept {
        if (this != &o) {
            delete[] data;   delete[] data_i8;
            data    = o.data;    o.data    = nullptr;
            data_i8 = o.data_i8; o.data_i8 = nullptr;
            size    = o.size;
            shape   = std::move(o.shape);
        }
        return *this;
    }
};

// -------------------------------------------------------
// TensorRaw: temporary holder while parsing the .bin file
//
// Type encoding (matches Python exporter):
//   type 1 → FP32  (Python saves param.to(float32).numpy())
//   type 2 → Packed ternary bytes  (uint8 packed)
//   type 4 → int32 scalar          (original_size metadata)
//   type 5 → float32 scalar        (alpha metadata)
// -------------------------------------------------------
struct TensorRaw {
    int  type      = 0;
    std::vector<uint8_t> data;
    int  int_val   = 0;
    float float_val = 0.0f;
};
