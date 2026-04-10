import torch
import struct
import math

MODEL_PATH = "/content/drive/MyDrive/tlumina_tinystories/tlumina_final.pt"

print(f"Loading T-Lumina Final Model from {MODEL_PATH}...")
state = torch.load(MODEL_PATH, map_location="cpu")

def pack_ternary_5in8(w_strict):
    original_size = w_strict.numel()
    w_t = w_strict.flatten().to(torch.int8)
    mapped_flat = w_t + 1
    remainder = original_size % 5
    if remainder != 0:
        mapped_padded = torch.nn.functional.pad(mapped_flat, (0, 5 - remainder), value=1)
    else:
        mapped_padded = mapped_flat

    mapped_matrix = mapped_padded.view(-1, 5).to(torch.int32)
    packed = (mapped_matrix[:, 0] +
              mapped_matrix[:, 1] * 3 +
              mapped_matrix[:, 2] * 9 +
              mapped_matrix[:, 3] * 27 +
              mapped_matrix[:, 4] * 81).to(torch.uint8)
    return packed, original_size

OUTPUT = "tlumina_model.bin"
ternary_keywords = ["ffn.gate.weight_fp", "ffn.up.weight_fp", "ffn.down.weight_fp"]

print(f"Exporting directly to {OUTPUT}...")

with open(OUTPUT, "wb") as f:
    for name, param in state.items():
        if not isinstance(param, torch.Tensor):
            continue

        is_ternary = any(k in name for k in ternary_keywords)

        if is_ternary:
            w_c = torch.clamp(param, -1.5, 1.5)
            w_c = w_c - w_c.mean()
            alpha = w_c.abs().mean().item() + 1e-6
            w_q = torch.sign(w_c) * alpha
            w_strict = torch.sign(w_q)

            packed, orig_size = pack_ternary_5in8(w_strict)

            name_p = (name + "_packed").encode()
            f.write(struct.pack("I", len(name_p)))
            f.write(name_p)
            f.write(struct.pack("I", 2))
            data = packed.numpy().tobytes()
            f.write(struct.pack("I", len(data)))
            f.write(data)

            name_s = (name + "_size").encode()
            f.write(struct.pack("I", len(name_s)))
            f.write(name_s)
            f.write(struct.pack("I", 4))
            f.write(struct.pack("I", 4))
            f.write(struct.pack("i", orig_size))

            name_a = (name + "_alpha").encode()
            f.write(struct.pack("I", len(name_a)))
            f.write(name_a)
            f.write(struct.pack("I", 5))
            f.write(struct.pack("I", 4))
            f.write(struct.pack("f", alpha))

            print(f"📦 Packed Ternary: {name} (Alpha: {alpha:.4f})")

        else:
            # ⚡ Reverted back to FP16 for memory saving!
            name_b = name.encode()
            f.write(struct.pack("I", len(name_b)))
            f.write(name_b)
            f.write(struct.pack("I", 1))
            fp16_tensor = param.to(torch.float16) # FP16 e export
            data = fp16_tensor.numpy().tobytes()
            f.write(struct.pack("I", len(data)))
            f.write(data)
            print(f"➡️ Saved FP32: {name}")

print("Export Complete! Download tlumina_model.bin")
