#pragma once
#include <vector>
#include <cstdint>
#include <string>

struct Tensor {
    float* data = nullptr;
    int8_t* data_i8 = nullptr;
    size_t size = 0;
    std::vector<int> shape;

    ~Tensor() {
        if (data) delete[] data;
        if (data_i8) delete[] data_i8;
    }

    Tensor() = default;
    
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    
    Tensor(Tensor&& other) noexcept {
        data = other.data;
        data_i8 = other.data_i8;
        size = other.size;
        shape = std::move(other.shape);
        other.data = nullptr;
        other.data_i8 = nullptr;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            if (data) delete[] data;
            if (data_i8) delete[] data_i8;
            data = other.data;
            data_i8 = other.data_i8;
            size = other.size;
            shape = std::move(other.shape);
            other.data = nullptr;
            other.data_i8 = nullptr;
        }
        return *this;
    }
};

struct TensorRaw {
    int type; 
    std::vector<uint8_t> data;
    std::vector<int> shape;
    int int_val = 0;
    float float_val = 0.0f; // Alpha এর জন্য যুক্ত করা হলো
};