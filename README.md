# 🌟 T-Lumina

**T-Lumina** is a high-performance, lightweight Large Language Model (LLM) architecture and inference engine written in C++. It is specifically engineered to bring modern AI capabilities to **legacy hardware** (e.g., Intel Core 2 Duo) by utilizing **Ternary Neural Networks (TNNs)**.

By representing weights as $\{-1, 0, 1\}$, T-Lumina replaces expensive floating-point multiplications with simple **Integer Addition and Subtraction**, enabling fast local inference on CPUs without modern AVX-512 or high-end GPUs.

---

## 🚀 Key Innovations

### 1. 5-in-8 Base-3 Weight Packing
Conventional ternary models often use 2-bit packing, which wastes bits. T-Lumina uses a custom **5-in-8 packing algorithm** that packs **5 ternary weights into a single 8-bit byte** using Base-3 encoding.
* **Efficiency:** Reduces memory footprint by **~80%** compared to FP16.
* **Performance:** Utilizes a pre-computed Look-Up Table (LUT) for O(1) unpacking.

### 2. Addition-Only FFN Kernels (SIMD Optimized)
The Feed-Forward Network (FFN) eliminates traditional matrix multiplications (MatMul).
* **AVX2/SSE Optimization:** Uses hardware-level SIMD instructions to perform parallel additions/subtractions.
* **Hardware Democratization:** Designed to run smoothly on systems with low RAM and less powerfull processors.

### 3. Progressive Training Pipeline
T-Lumina is trained using a progressive quantization approach:
* Starts at **8-bit (INT8)** to stabilize features.
* Decays to **4-bit**.
* Finalizes at **1.58-bit Ternary** to ensure minimum perplexity loss while maximizing speed.

---

## 🛠️ Technical Specifications

| Feature | Specification |
| :--- | :--- |
| **Architecture** | Transformer-based (Decoder only) |
| **Bit-width** | 1.58-bit (Ternary weights) |
| **Parameters** | 34.2M (Storyteller Baseline) |
| **Engine** | Pure C++ (Zero Dependencies) |
| **Optimizations** | AVX2, SSE, Custom 5-in-8 Packing |
| **Target Data** | TinyStories / General Knowledge |

---

## 💻 Installation & Compilation

Since T-Lumina has **zero external dependencies**, you only need a C++ compiler supporting C++17.

### Compile the Inference Engine:
```bash
g++ -O3 -mavx2 -std=c++17 main.cpp core/model.cpp -o tlumina_inference
````

### Run the Model:

```bash
./tlumina_inference
```

-----

## 📂 Project Structure

  * `/core`: Contains the C++ engine (Tensors, KV-Cache, SIMD Kernels).
  * `/scripts`: Python scripts for training (`train.py`) and 5-in-8 packing.
  * `tokenizer.h`: Custom BPE-based tokenizer interface.
  * `packing.h`: Implementation of the Base-3 unpacking logic.

-----

## 🛤️ Roadmap

  - [ ] Implement Ternary Attention (Q/K/V quantization).
  - [ ] Support for larger 1B+ Parameter models.
  - [ ] **T-Lumina Video:** Exploring Ternary Diffusion for low-cost video generation.
  - [ ] Mobile-optimized Android/iOS builds.

-----

Created by **Abdul Aleem**, an independent researcher and software developer from Bangladesh.

-----

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

```

---
