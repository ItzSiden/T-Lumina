# export_vocab_tinyllama.py
from transformers import AutoTokenizer
import struct

print("Loading TinyLlama Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

print("Exporting tokens to vocab_tinyllama.bin...")
with open("vocab_tinyllama.bin", "wb") as f:
    for i in range(tokenizer.vocab_size):
        # Decode getting the exact string (handles spaces and special chars)
        text = tokenizer.decode([i])
        text_bytes = text.encode('utf-8')
        
        # Write length of string (4 bytes), then the string itself
        f.write(struct.pack("I", len(text_bytes)))
        f.write(text_bytes)

print("✅ vocab_tinyllama.bin Exported Successfully!")
print(f"Vocab Size: {tokenizer.vocab_size}")
