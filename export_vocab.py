# export_vocab.py
from transformers import AutoTokenizer
import struct

print("Loading GPT-2 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

print("Exporting tokens to vocab.bin...")
with open("vocab.bin", "wb") as f:
    for i in range(tokenizer.vocab_size):
        # Decode getting the exact string (handles spaces and special chars)
        text = tokenizer.decode([i])
        text_bytes = text.encode('utf-8')
        
        # Write length of string (4 bytes), then the string itself
        f.write(struct.pack("I", len(text_bytes)))
        f.write(text_bytes)

print("✅ vocab.bin Exported Successfully!")