# ====================================================
# T-LUMINA - STORYTELLER EDITION (TINYSTORIES + TOP-P)
# ⚡ SUPERCHARGED SPEEDUP VERSION ⚡
# ====================================================

from google.colab import drive
drive.mount('/content/drive')

!pip install transformers datasets -q

import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import math, os, itertools, time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

SAVE_DIR = "/content/drive/MyDrive/tlumina_tinystories"
os.makedirs(SAVE_DIR, exist_ok=True)

# ====================================================
# Progressive Ternary Layer
# ====================================================
class AdaptiveTernaryLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=True):
        super().__init__()
        self.weight_fp = nn.Parameter(
            torch.empty(out_f, in_f).normal_(0, 0.02)
        )
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
        self.bit_width = 8  # starts at INT8

    def forward(self, x):
        if self.bit_width >= 8:
            w = self.weight_fp
        elif self.bit_width == 4:
            scale = self.weight_fp.abs().max() / 7 + 1e-8
            w_q   = torch.round(self.weight_fp / scale).clamp(-7, 7)
            w     = self.weight_fp + (w_q * scale - self.weight_fp).detach()
        else:
            w_c   = torch.clamp(self.weight_fp, -1.5, 1.5)
            w_c   = w_c - w_c.mean()
            alpha = w_c.abs().mean() + 1e-6
            w_q   = torch.sign(w_c) * alpha
            w     = w_c + (w_q - w_c).detach()

        return F.linear(x, w.to(x.dtype),
                        self.bias.to(x.dtype) if self.bias is not None else None)

    def set_bit_width(self, bits):
        self.bit_width = bits

# ====================================================
# T-Lumina Architecture
# ====================================================
class TLuminaFFN(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.gate = AdaptiveTernaryLinear(d_model, d_ffn, bias=False)
        self.up   = AdaptiveTernaryLinear(d_model, d_ffn, bias=False)
        self.down = AdaptiveTernaryLinear(d_ffn, d_model, bias=False)
        self.act  = nn.SiLU()

    def forward(self, x):
        return self.down(self.act(self.gate(x)) * self.up(x))

    def set_bit_width(self, bits):
        self.gate.set_bit_width(bits)
        self.up.set_bit_width(bits)
        self.down.set_bit_width(bits)

class TLuminaBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ffn):
        super().__init__()
        self.attn    = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn     = TLuminaFFN(d_model, d_ffn)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        # ⚡ FIX: Explicit bool mask without is_causal flag (Fixes PyTorch Error)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)

        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask, need_weights=False)

        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x

class TLumina(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8,
                 d_ffn=1024, n_layers=8, max_len=256):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.blocks  = nn.ModuleList([
            TLuminaBlock(d_model, n_heads, d_ffn)
            for _ in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.embed(x) + self.pos_emb[:, :x.size(1)]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def set_bit_width(self, bits):
        for block in self.blocks:
            block.ffn.set_bit_width(bits)
        print(f"  → All FFN layers set to {bits}-bit")

# ====================================================
# Dataset & DataLoader (⚡ Optimized)
# ====================================================
print("\nLoading TinyStories dataset...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

raw = load_dataset("roneneldan/TinyStories", split="train[:50000]")
val_raw = load_dataset("roneneldan/TinyStories", split="validation[:2000]")

def tokenize(ex):
    return tokenizer(ex["text"])

tokenized = raw.map(tokenize, batched=True, remove_columns=["text"])
val_tok = val_raw.map(tokenize, batched=True, remove_columns=["text"])

def group(examples):
    ids = list(itertools.chain(*examples["input_ids"]))
    total = (len(ids) // 256) * 256
    return {"input_ids": [ids[i:i+256] for i in range(0, total, 256)]}

data = tokenized.map(group, batched=True, remove_columns=tokenized.column_names)
val_data = val_tok.map(group, batched=True, remove_columns=val_tok.column_names)

# ⚡ SPEEDUP: DataLoader for parallel loading and Batching
BATCH_SIZE = 16

def collate_fn(batch):
    return torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)

train_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(f"Train: {len(data)} examples | Batch Size: {BATCH_SIZE} | GPU Util maximized 🚀")

# ====================================================
# Model Init
# ====================================================
vocab_size = tokenizer.vocab_size
model = TLumina(
    vocab_size = vocab_size,
    d_model    = 256,
    n_heads    = 8,
    d_ffn      = 1024,
    n_layers   = 8,
    max_len    = 256
).to(device)

params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"T-Lumina params: {params:.1f}M")

# ====================================================
# Evaluation & Robust Generation
# ====================================================
def eval_ppl(model, num_batches=20):
    model.eval()
    total = 0
    count = 0
    with torch.no_grad():
        for i, ids in enumerate(val_loader):
            if i >= num_batches: break
            ids = ids.to(device)
            # ⚡ WARNING FIXED: AMP in eval
            with torch.amp.autocast('cuda', enabled=(device=="cuda")):
                logits = model(ids)
                loss   = F.cross_entropy(
                    logits[:, :-1].reshape(-1, vocab_size),
                    ids[:, 1:].reshape(-1)
                )
            if not math.isnan(loss.item()):
                total += loss.item()
                count += 1
    return math.exp(total / max(1, count))

def quick_gen(prompt="Once upon a time, there was a little", temp=0.8, top_p=0.9):
    model.eval()
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        # ⚡ WARNING FIXED: AMP in Generation
        with torch.amp.autocast('cuda', enabled=(device=="cuda")):
            for _ in range(40):
                logits = model(ids)[:, -1, :] / temp

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                ids = torch.cat([ids, next_id], dim=1)

                if next_id.item() == tokenizer.eos_token_id:
                    break

    return tokenizer.decode(ids[0], skip_special_tokens=True)

# ====================================================
# Training - Progressive Bit Annealing (50k Steps)
# ====================================================
TOTAL_STEPS = 50000
optimizer   = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS)

# ⚡ SPEEDUP: Mixed Precision Scaler
scaler = torch.amp.GradScaler('cuda', enabled=(device=="cuda"))

def get_bit_width(step):
    if step < 15000:   return 8
    elif step < 30000: return 4
    else:              return 2

start     = 0
best_ppl  = float('inf')
ckpt_path = f"{SAVE_DIR}/ckpt.pt"

if os.path.exists(ckpt_path):
    print("Resuming...")
    c = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(c['model'])
    optimizer.load_state_dict(c['optimizer'])
    scheduler.load_state_dict(c['scheduler'])
    if 'scaler' in c: scaler.load_state_dict(c['scaler'])
    start    = c['step']
    best_ppl = c.get('best_ppl', float('inf'))
    print(f"Resumed from step {start}")

print(f"\n{'='*60}")
print(f"⚡ T-LUMINA SUPERCHARGED | {start} → {TOTAL_STEPS} steps")
print(f"Batch Size: {BATCH_SIZE} | AMP: Enabled | Causal Mask: Fixed")
print(f"{'='*60}")

current_bits = get_bit_width(start)
model.set_bit_width(current_bits)

t_start = time.time()
train_iter = iter(train_loader)

for step in range(start, TOTAL_STEPS):
    model.train()

    new_bits = get_bit_width(step)
    if new_bits != current_bits:
        current_bits = new_bits
        model.set_bit_width(current_bits)
        print(f"\n🔄 Step {step}: Switching to {current_bits}-bit!")

    # ⚡ SPEEDUP: Get batch from DataLoader
    try:
        ids = next(train_iter).to(device)
    except StopIteration:
        train_iter = iter(train_loader)
        ids = next(train_iter).to(device)

    optimizer.zero_grad()

    # ⚡ SPEEDUP: Automatic Mixed Precision (AMP)
    with torch.amp.autocast('cuda', enabled=(device=="cuda")):
        logits = model(ids)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, vocab_size),
            ids[:, 1:].reshape(-1),
            label_smoothing=0.1
        )

    # ⚡ SPEEDUP: Scaled Backpropagation
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    # ⚡ UPDATE: প্রতি ১০০ স্টেপে শুধু লগ প্রিন্ট করবে
    if (step + 1) % 100 == 0:
        elapsed  = time.time() - t_start
        per_step = elapsed / (step - start + 1)
        eta      = per_step * (TOTAL_STEPS - step - 1)
        print(f"Step {step+1:>6} | loss={loss.item():.4f} | {elapsed/60:.1f}m | ETA:{eta/60:.1f}m | bits={current_bits} ⚡")

    # ⚡ UPDATE: প্রতি ২৫০০ স্টেপে Autosave করবে
    if (step + 1) % 2500 == 0:
        torch.save({
            'step':      step + 1,
            'model':     model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler':    scaler.state_dict(),
            'loss':      loss.item(),
            'best_ppl':  best_ppl,
        }, ckpt_path)
        print(f"💾 Checkpoint Saved at Step {step+1}")

    # ⚡ SPEEDUP: Eval frequency reduced to every 5000 steps
    if (step + 1) % 5000 == 0:
        ppl = eval_ppl(model)
        if ppl < best_ppl:
            best_ppl = ppl
            torch.save(model.state_dict(), f"{SAVE_DIR}/best.pt")
            star = "⭐"
        else:
            star = ""
        gen = quick_gen()
        print(f"\n  PPL: {ppl:.2f} (best: {best_ppl:.2f}) {star}")
        print(f"  Gen: {gen}\n")

# ====================================================
# Final
# ====================================================
print("\nFinal evaluation...")
ppl = eval_ppl(model, num_batches=100)
print(f"\nFinal PPL: {ppl:.4f}")
print(f"Best PPL:  {best_ppl:.4f}")

model.cpu()
torch.save(model.state_dict(), f"{SAVE_DIR}/tlumina_final.pt")

import json
config = {
    "vocab_size": vocab_size,
    "d_model": 256,
    "n_heads": 8,
    "d_ffn": 1024,
    "n_layers": 8,
    "max_len": 256
}
with open(f"{SAVE_DIR}/config.json", "w") as f:
    json.dump(config, f)

print(f"\n✅ Saved to {SAVE_DIR}")
print("মডেল এখন গল্প বলার জন্য এবং C++ এ প্যাকিং এর জন্য রেডি! 🚀")
