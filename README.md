# 🌟 T-Lumina: Ternary-First LLM Architecture

<div align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue.svg" alt="C++17">
  <img src="https://img.shields.io/badge/Optimization-AVX2%20SIMD-success" alt="AVX2">
  <img src="https://img.shields.io/badge/Dependencies-Zero-brightgreen" alt="Zero Dependencies">
  <img src="https://img.shields.io/badge/License-MIT-purple.svg" alt="License">
</div>

<br>

**T-Lumina** is a breakthrough, high-performance, lightweight Large Language Model (LLM) architecture and inference engine written entirely in pure C++. It is specifically engineered to democratize AI by bringing modern LLM capabilities to **legacy hardware** and edge devices utilizing **Ternary Neural Networks (TNNs)**.

By representing Feed-Forward Network (FFN) weights strictly as $\{-1, 0, 1\}$, T-Lumina replaces expensive floating-point matrix multiplications with highly optimized, branchless **Integer Addition and Subtraction**. 

---

## 🚀 Architectural Innovations

### 1. Multiplication-Free AVX2 Kernels
The core Feed-Forward Network (accounting for ~70% of LLM parameters) eliminates standard MatMul operations. T-Lumina utilizes custom **AVX2/SSE SIMD instructions** to perform parallel additions and subtractions via bitmasking, achieving blistering inference speeds on pure CPU.

### 2. Custom 5-in-8 Base-3 Weight Packing
Standard 2-bit quantization wastes memory limits. T-Lumina introduces a mathematically rigorous **Base-3 packing algorithm** that compresses **5 ternary weights into a single 8-bit byte (uint8)**. 
* **Compression:** Reduces FFN memory footprint by **~80%** compared to standard FP16.
* **O(1) Unpacking:** Uses a pre-computed C++ Look-Up Table (LUT) for instantaneous runtime unrolling into CPU cache.

### 3. Progressive Bit-Annealing Training
T-Lumina models are trained in PyTorch using a progressive quantization schedule:
1. **INT8 (0-15k steps):** Stabilizes feature extraction.
2. **4-bit (15k-30k steps):** Forces network to learn robust intermediate representations.
3. **1.58-bit Ternary (30k+ steps):** Finalizes at $\{-1, 0, 1\}$ + $\alpha$ scaling, minimizing perplexity loss.

---

## 📊 Live Benchmark (Storyteller-34M)

Running on a standard CPU environment (Single-Threaded):

| Metric | Result |
| :--- | :--- |
| **Parameters** | 34.2 Million |
| **Generation Speed** | **~106.47 tokens/sec** ⚡ |
| **RAM Usage** | **224.68 MB** |
| **Weight Quantization**| FFN: 1.58-bit (Ternary) \| Attn/Norm: FP32 |

---

## 💻 Installation & Compilation

T-Lumina has **zero external dependencies**. No PyTorch, No GGML, No ONNX. Just pure C++17.

### 1. Compile the Inference Engine
```bash
g++ -O3 -march=native -std=c++17 -Wall main.cpp core/model.cpp -o tlumina_inference
```

### 2. Run the Engine
```bash
./tlumina_inference
```
*(Simply type your prompt in the CLI and watch the real-time ternary stream!)*

---

## 📂 Repository Structure

* `/core/model.h & .cpp`: Core model architecture, LayerNorm, and FP32 Attention.
* `/core/ternary_ffn.h`: Multiplication-free AVX2 SIMD FFN implementation.
* `/core/packing.h`: Base-3 to 5-in-8 LUT decoding.
* `/scripts/train.py`: Supercharged PyTorch training script (AMP, Causal Mask fixed).
* `/scripts/export_direct.py`: Custom PyTorch -> Binary exporter with Alpha scaling.

---

## 🛤️ Roadmap

- [ ] **Multi-threading (OpenMP):** To scale speed for 1B+ parameter models.
- [ ] **Ternary Attention:** Full Q/K/V quantization for an end-to-end 100% ternary pipeline.
- [ ] **T-Lumina Video:** Exploring Ternary Diffusion models for ultra-low-cost local video generation.
- [ ] **Android/iOS JNI Ports:** Running T-Lumina locally on mid-range smartphones.

---

## 🏆 Author & Copyright

**Architecture invented and engineered by:**<br>
**© 2026 Abdul Aleem** <br>
*Student & AI Researcher | Dinajpur, Bangladesh*

---

## 📄 License
This project is licensed under the MIT License. Feel free to use, modify, and distribute, but please attribute the original author.
