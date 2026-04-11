from transformers import AutoTokenizer
import struct

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

with open("vocab.bin", "wb") as f:
    for i in range(tokenizer.vocab_size):
        text = tokenizer.decode([i])
        text_bytes = text.encode('utf-8')
        f.write(struct.pack("I", len(text_bytes)))
        f.write(text_bytes)

print("✅ TinyLlama Vocab exported! Download vocab.bin")