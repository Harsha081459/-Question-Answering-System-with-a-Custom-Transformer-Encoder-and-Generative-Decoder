import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, interleave_datasets
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer


@dataclass
class ModelConfig:
    vocab_size: int
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12


SIZE_PRESETS = {
    "base": dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072),
    "large": dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096),
}


class BertEmbeddings(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.position_embeddings = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.token_type_embeddings = nn.Embedding(cfg.type_vocab_size, cfg.hidden_size)
        self.layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids):
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)
        x = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(pos_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.hidden_size % cfg.num_attention_heads == 0
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.attn_dropout = nn.Dropout(cfg.attention_probs_dropout_prob)
        self.proj_dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(self, x, attention_mask):
        bsz, seq_len, hidden = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = attention_mask[:, None, None, :].to(dtype=scores.dtype)
        scores = scores.masked_fill(mask == 0, -1e4)
        probs = F.softmax(scores, dim=-1)
        probs = self.attn_dropout(probs)
        ctx = torch.matmul(probs, v).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        out = self.out_proj(ctx)
        out = self.proj_dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.attn = MultiHeadSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.ffn = FeedForward(cfg)

    def forward(self, x, attention_mask):
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class BertEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embeddings = BertEmbeddings(cfg)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        self.final_ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.embeddings(input_ids, token_type_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.final_ln(x)
        return x


class MLMHead(nn.Module):
    def __init__(self, cfg: ModelConfig, tied_embedding: nn.Embedding):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.decoder = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(cfg.vocab_size))
        self.decoder.bias = self.bias
        self.decoder.weight = tied_embedding.weight

    def forward(self, x):
        x = self.dense(x)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x


class BertForMLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.encoder = BertEncoder(cfg)
        self.mlm = MLMHead(cfg, self.encoder.embeddings.word_embeddings)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        x = self.encoder(input_ids, token_type_ids, attention_mask)
        logits = self.mlm(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return logits, loss


class StreamingTextDataset(IterableDataset):
    def __init__(self, tokenizer, max_len=128, seed=42):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.seed = seed
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        # Prefer parquet-backed datasets to avoid old script-based dataset failures.
        wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        web = load_dataset("allenai/c4", "en", split="train", streaming=True)
        self.stream = interleave_datasets([wiki, web], probabilities=[0.7, 0.3], seed=seed)
        self.stream = self.stream.shuffle(seed=seed, buffer_size=20000)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        token_buffer = []
        _ = random.Random(self.seed + worker_id)
        # Compatibility: some datasets versions expose "shard", others "to_iterable_dataset".
        if num_workers > 1 and hasattr(self.stream, "shard"):
            stream = self.stream.shard(num_shards=num_workers, index=worker_id)
        else:
            stream = self.stream
        for ex in stream:
            if num_workers > 1:
                # Manual worker partition fallback when .shard() is unavailable.
                ex_idx = getattr(self, "_worker_example_idx", 0)
                self._worker_example_idx = ex_idx + 1
                if (ex_idx % num_workers) != worker_id:
                    continue
            text = ex.get("text", "")
            if not text or len(text) < 20:
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
            if len(ids) == 0:
                continue
            token_buffer.extend(ids)
            while len(token_buffer) >= (self.max_len - 2):
                chunk = token_buffer[: self.max_len - 2]
                token_buffer = token_buffer[self.max_len - 2 :]
                input_ids = [self.cls_id] + chunk + [self.sep_id]
                attention_mask = [1] * len(input_ids)
                token_type_ids = [0] * len(input_ids)
                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                }


def collate_mlm(batch, tokenizer, mlm_prob=0.15):
    """Data collator that dynamically masks tokens for Masked Language Modeling (MLM)."""
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id
    vocab_size = tokenizer.vocab_size
    max_len = max(len(x["input_ids"]) for x in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    token_type_ids = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, x in enumerate(batch):
        length = len(x["input_ids"])
        input_ids[i, :length] = torch.tensor(x["input_ids"], dtype=torch.long)
        attention_mask[i, :length] = torch.tensor(x["attention_mask"], dtype=torch.long)
        token_type_ids[i, :length] = torch.tensor(x["token_type_ids"], dtype=torch.long)

    # Set all labels to -100 (ignored by cross-entropy) by default
    labels = torch.full_like(input_ids, -100)
    
    # Do not mask special tokens ([CLS], [SEP], [PAD])
    special_mask = (
        (input_ids == tokenizer.cls_token_id)
        | (input_ids == tokenizer.sep_token_id)
        | (input_ids == tokenizer.pad_token_id)
    )
    prob = torch.full(input_ids.shape, mlm_prob)
    prob.masked_fill_(special_mask, 0.0)
    masked = torch.bernoulli(prob).bool()
    
    # The label for masked tokens is the original token ID
    labels[masked] = input_ids[masked]

    # MLM Replacement Strategy:
    # 80% of the time, replace the masked token with [MASK]
    replace_prob = torch.rand(input_ids.shape)
    mask80 = masked & (replace_prob < 0.8)
    input_ids[mask80] = mask_id
    
    # 10% of the time, replace the masked token with a random word
    rand10 = masked & (replace_prob >= 0.8) & (replace_prob < 0.9)
    random_words = torch.randint(low=0, high=vocab_size, size=input_ids.shape, dtype=torch.long)
    input_ids[rand10] = random_words[rand10]
    
    # The remaining 10% of the time, keep the original token intact.

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": labels,
    }


def make_scheduler(optimizer, warmup_steps, total_steps):
    """Creates a learning rate scheduler with linear warmup and linear decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))
    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, step, out_dir, cfg, tokenizer):
    """Saves the model weights, optimizer state, configuration, and tokenizer."""
    ckpt_dir = os.path.join(out_dir, f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
        },
        os.path.join(ckpt_dir, "checkpoint.pt"),
    )
    with open(os.path.join(ckpt_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    tokenizer.save_pretrained(ckpt_dir)
    latest_file = os.path.join(out_dir, "latest_checkpoint.txt")
    with open(latest_file, "w", encoding="utf-8") as f:
        f.write(ckpt_dir)


def load_checkpoint(model, optimizer, scheduler, resume_path, device, model_only=False):
    """Loads a checkpoint, dynamically resizing position embeddings if the sequence length changed."""
    payload = torch.load(resume_path, map_location=device)
    model_state = payload["model"]
    model_pos = model.encoder.embeddings.position_embeddings.weight
    ckpt_pos = model_state.get("encoder.embeddings.position_embeddings.weight")
    if ckpt_pos is not None and ckpt_pos.shape != model_pos.shape:
        if ckpt_pos.shape[1] != model_pos.shape[1]:
            raise ValueError(
                f"Position embedding hidden size mismatch: ckpt={ckpt_pos.shape}, model={model_pos.shape}"
            )
        new_pos = model_pos.detach().clone()
        copy_len = min(ckpt_pos.shape[0], model_pos.shape[0])
        new_pos[:copy_len] = ckpt_pos[:copy_len]
        model_state["encoder.embeddings.position_embeddings.weight"] = new_pos
        print(
            f"Resized position embeddings from {ckpt_pos.shape[0]} to {model_pos.shape[0]} (copied {copy_len})."
        )

    model.load_state_dict(model_state, strict=True)
    if not model_only:
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
    return int(payload.get("step", 0))


def main():
    """Main entry point for BERT MLM pretraining."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")
    parser.add_argument("--size", type=str, choices=["base", "large"], default="base")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="checkpoints_pretrain")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--precision", type=str, choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--resume_latest", action="store_true")
    parser.add_argument("--resume_model_only", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda" and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.sep_token

    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, max_position_embeddings=args.seq_len)
    for k, v in SIZE_PRESETS[args.size].items():
        setattr(cfg, k, v)

    model = BertForMLM(cfg).to(device)

    dataset = StreamingTextDataset(tokenizer=tokenizer, max_len=args.seq_len, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=lambda b: collate_mlm(b, tokenizer),
    )
    data_iter = iter(loader)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-8)
    warmup_steps = int(args.max_steps * args.warmup_ratio)
    scheduler = make_scheduler(optimizer, warmup_steps, args.max_steps)
    # Compat: torch<2.3 may not expose torch.amp.GradScaler
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(enabled=(device == "cuda" and args.precision == "fp16"))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and args.precision == "fp16"))

    os.makedirs(args.out_dir, exist_ok=True)
    model.train()
    global_step = 0
    running_loss = 0.0
    last_log_time = time.time()
    tokens_since_log = 0

    resume_ckpt_file = ""
    if args.resume_checkpoint:
        resume_ckpt_file = args.resume_checkpoint
    elif args.resume_latest:
        latest_file = Path(args.out_dir) / "latest_checkpoint.txt"
        if latest_file.exists():
            ckpt_dir = latest_file.read_text(encoding="utf-8").strip()
            resume_ckpt_file = str(Path(ckpt_dir) / "checkpoint.pt")

    if resume_ckpt_file:
        global_step = load_checkpoint(
            model,
            optimizer,
            scheduler,
            resume_ckpt_file,
            device,
            model_only=args.resume_model_only,
        )
        if args.resume_model_only:
            global_step = 0
        print(f"Resumed from: {resume_ckpt_file} at step={global_step}")

    try:
        while global_step < args.max_steps:
            optimizer.zero_grad(set_to_none=True)
            for _ in range(args.grad_accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                token_type_ids = batch["token_type_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(device == "cuda")):
                    _, loss = model(input_ids, token_type_ids, attention_mask, labels)
                    loss = loss / args.grad_accum
                scaler.scale(loss).backward()
                running_loss += loss.item() * args.grad_accum
                tokens_since_log += int(attention_mask.sum().item())

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            if global_step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                ppl = math.exp(min(avg_loss, 20))
                lr = scheduler.get_last_lr()[0]
                now = time.time()
                dt = max(1e-6, now - last_log_time)
                toks_per_sec = tokens_since_log / dt
                print(f"step={global_step} loss={avg_loss:.4f} ppl={ppl:.2f} lr={lr:.6e} tok/s={toks_per_sec:.0f}")
                running_loss = 0.0
                tokens_since_log = 0
                last_log_time = now

            if global_step % args.save_every == 0:
                save_checkpoint(model, optimizer, scheduler, global_step, args.out_dir, cfg, tokenizer)
                print(f"Saved checkpoint at step {global_step}")
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, saving checkpoint...")
    finally:
        save_checkpoint(model, optimizer, scheduler, global_step, args.out_dir, cfg, tokenizer)
        print(f"Training stopped. Last saved step={global_step}.")


if __name__ == "__main__":
    main()
