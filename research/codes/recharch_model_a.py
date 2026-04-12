# ====================================================
# MODEL A: FP32 BASELINE (Standard FFN + Standard Attn)
# ====================================================

!pip install transformers datasets -q

import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import math, itertools, time

# === STRICT SEED FOR A/B TESTING ===
torch.manual_seed(1337)
torch.cuda.manual_seed_all(1337)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight

def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)

def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    xq_r = xq.float().reshape(xq.shape[:-1] + (-1, 2))
    xk_r = xk.float().reshape(xk.shape[:-1] + (-1, 2))
    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(2)
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(2)
    xq_out = torch.empty_like(xq_r)
    xk_out = torch.empty_like(xk_r)
    xq_out[..., 0] = xq_r[..., 0] * freqs_cos - xq_r[..., 1] * freqs_sin
    xq_out[..., 1] = xq_r[..., 0] * freqs_sin + xq_r[..., 1] * freqs_cos
    xk_out[..., 0] = xk_r[..., 0] * freqs_cos - xk_r[..., 1] * freqs_sin
    xk_out[..., 1] = xk_r[..., 0] * freqs_sin + xk_r[..., 1] * freqs_cos
    return xq_out.flatten(3).type_as(xq), xk_out.flatten(3).type_as(xk)

class TLuminaAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, freqs_cos, freqs_sin, mask):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos[:seqlen], freqs_sin[:seqlen])
        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None: scores = scores + mask[:seqlen, :seqlen]
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv).transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

# ⚡ BASELINE FP32 FFN ⚡
class BaselineFFN(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ffn, bias=False)
        self.up   = nn.Linear(d_model, d_ffn, bias=False)
        self.down = nn.Linear(d_ffn, d_model, bias=False)
        self.act  = nn.SiLU()
    def forward(self, x):
        return self.down(self.act(self.gate(x)) * self.up(x))

class TLuminaBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ffn):
        super().__init__()
        self.attn  = TLuminaAttention(d_model, n_heads)
        self.ffn   = BaselineFFN(d_model, d_ffn)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, freqs_cos, freqs_sin, mask):
        x = x + self.attn(self.norm1(x), freqs_cos, freqs_sin, mask)
        x = x + self.ffn(self.norm2(x))
        return x

class TLumina(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, d_ffn=1024, n_layers=8, max_len=256):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TLuminaBlock(d_model, n_heads, d_ffn) for _ in range(n_layers)])
        self.norm   = RMSNorm(d_model)
        self.head   = nn.Linear(d_model, vocab_size, bias=False)
        freqs_cos, freqs_sin = precompute_freqs_cis(d_model // n_heads, max_len * 2)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, x):
        bsz, seqlen = x.shape
        h = self.embed(x)
        mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)
        for block in self.blocks:
            h = block(h, self.freqs_cos, self.freqs_sin, mask)
        return self.head(self.norm(h))

# === WIKITEXT-2 DATASET ===
print("\nLoading WikiText-2 dataset...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
val_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

raw = raw.filter(lambda x: len(x["text"].strip()) > 10)
val_raw = val_raw.filter(lambda x: len(x["text"].strip()) > 10)

def tokenize(ex): return tokenizer(ex["text"])
tokenized = raw.map(tokenize, batched=True, remove_columns=["text"])
val_tok = val_raw.map(tokenize, batched=True, remove_columns=["text"])

def group(examples):
    ids = list(itertools.chain(*examples["input_ids"]))
    total = (len(ids) // 256) * 256
    return {"input_ids": [ids[i:i+256] for i in range(0, total, 256)]}

# FIX: remove_columns used here to drop attention_mask and avoid length mismatch error!
data = tokenized.map(group, batched=True, remove_columns=tokenized.column_names)
val_data = val_tok.map(group, batched=True, remove_columns=val_tok.column_names)

BATCH_SIZE = 16
def collate_fn(batch): return torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
train_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

vocab_size = tokenizer.vocab_size
model = TLumina(vocab_size=vocab_size).to(device)
print(f"FP32 Baseline params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

def eval_ppl(model, num_batches=40):
    model.eval()
    total, count = 0, 0
    with torch.no_grad():
        for i, ids in enumerate(val_loader):
            if i >= num_batches: break
            ids = ids.to(device)
            with torch.amp.autocast('cuda', enabled=(device=="cuda")):
                logits = model(ids)
                loss   = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), ids[:, 1:].reshape(-1))
            if not math.isnan(loss.item()):
                total += loss.item()
                count += 1
    return math.exp(total / max(1, count))

TOTAL_STEPS = 5000
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS)
scaler = torch.amp.GradScaler('cuda', enabled=(device=="cuda"))

print(f"\n{'='*60}")
print(f"🚀 MODEL A: FP32 BASELINE | {TOTAL_STEPS} steps | WikiText-2")
print(f"{'='*60}")

best_ppl = float('inf')
t_start = time.time()
train_iter = iter(train_loader)

for step in range(TOTAL_STEPS):
    model.train()
    try: ids = next(train_iter).to(device)
    except StopIteration:
        train_iter = iter(train_loader)
        ids = next(train_iter).to(device)

    optimizer.zero_grad()
    with torch.amp.autocast('cuda', enabled=(device=="cuda")):
        logits = model(ids)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), ids[:, 1:].reshape(-1), label_smoothing=0.1)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    if (step + 1) % 100 == 0:
        elapsed  = time.time() - t_start
        print(f"Step {step+1:>5} | loss={loss.item():.4f} | {elapsed/60:.1f}m")

    if (step + 1) % 1000 == 0 or (step + 1) == TOTAL_STEPS:
        ppl = eval_ppl(model)
        if ppl < best_ppl: best_ppl = ppl
        print(f"\n  Validation PPL: {ppl:.2f}\n")

print(f"\n✅ MODEL A Training Complete! Final PPL: {best_ppl:.2f}")