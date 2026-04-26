import argparse
import json
import os
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from generative_data import GenQADataConfig, NO_ANSWER_TEXT, NO_ANSWER_SENTENCE, build_dataloaders
from generative_evaluation import evaluate_model
from standard_generative_decoder import DecoderConfig, GenerativeQAModel as StandardGenerativeQAModel
from main_hybrid_decoder import GenerativeQAModelHybrid
from mlm_pretraining import ModelConfig


def make_scheduler(optimizer, warmup_steps, total_steps):
    """Creates a learning rate scheduler with linear warmup and linear decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))

    return LambdaLR(optimizer, lr_lambda)


def load_encoder_from_pretrain(model, pretrain_ckpt_path: str):
    """Loads weights from a pretrained MLM checkpoint strictly into the encoder submodule."""
    payload = torch.load(pretrain_ckpt_path, map_location="cpu", weights_only=False)
    state = payload["model"]
    enc_state = {k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(enc_state, strict=True)
    return int(payload.get("step", 0))


def _remap_decoder_state_keys(state: dict, decoder_variant: str):
    """Translates state dict keys when switching between standard and hybrid decoder architectures."""
    if decoder_variant not in {"standard", "hybrid"}:
        raise ValueError("decoder_variant must be one of: standard, hybrid")

    remapped = {}
    num_changed = 0
    for k, v in state.items():
        nk = k
        # Map generic linear/attention names to the specific variant names.
        if decoder_variant == "hybrid":
            nk = nk.replace(".multihead_attn.", ".cross_attn.")
            nk = nk.replace(".linear1.", ".ffn_fc1.")
            nk = nk.replace(".linear2.", ".ffn_fc2.")
        else:
            nk = nk.replace(".cross_attn.", ".multihead_attn.")
            nk = nk.replace(".ffn_fc1.", ".linear1.")
            nk = nk.replace(".ffn_fc2.", ".linear2.")
        if nk != k:
            num_changed += 1
        remapped[nk] = v
    return remapped, num_changed


def load_model_from_gen_checkpoint(model, checkpoint_path: str, decoder_variant: str = "standard"):
    """Loads a full model checkpoint, adapting keys and tensor shapes where necessary."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_raw = payload["model"] if "model" in payload else payload
    state, changed = _remap_decoder_state_keys(state_raw, decoder_variant=decoder_variant)
    if changed:
        print(f"[init] Remapped {changed} decoder tensor keys for decoder_variant={decoder_variant}")
    current_state = model.state_dict()

    # Dynamic resizing: Allows loading a checkpoint trained with a different max target length.
    # It copies existing position embeddings and leaves new positions uninitialized (or cuts off excess).
    pos_key = "decoder_embeddings.position_embeddings.weight"
    if pos_key in state and pos_key in current_state:
        ckpt_pos = state[pos_key]
        cur_pos = current_state[pos_key]
        if ckpt_pos.shape != cur_pos.shape:
            resized = cur_pos.clone()
            copy_len = min(ckpt_pos.size(0), cur_pos.size(0))
            resized[:copy_len] = ckpt_pos[:copy_len]
            state[pos_key] = resized
            print(
                f"[init] Resized decoder position embeddings "
                f"{tuple(ckpt_pos.shape)} -> {tuple(cur_pos.shape)} "
                f"(copied first {copy_len} positions)"
            )

    model.load_state_dict(state, strict=True)
    return int(payload.get("step", 0)), float(payload.get("best_metric", -1e9))


def save_gen_checkpoint(path: str, model, optimizer, scheduler, scaler, step, epoch, best_metric, enc_cfg, dec_cfg):
    """Saves a training checkpoint including model weights, optimizer/scheduler state, and configurations."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": step,
            "epoch": epoch,
            "best_metric": best_metric,
            "encoder_config": enc_cfg.__dict__,
            "decoder_config": dec_cfg.to_dict(),
        },
        path,
    )


def load_gen_checkpoint(path: str, model, optimizer, scheduler, scaler, device):
    """Loads a training checkpoint to resume the training loop exactly where it left off."""
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    return int(payload.get("step", 0)), int(payload.get("epoch", 0)), float(payload.get("best_metric", -1e9))


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    device,
    grad_accum_steps,
    max_grad_norm,
    label_smoothing,
    global_step,
    amp_dtype,
    amp_enabled,
):
    """Runs a single pass over the training dataloader, updating model weights with gradient accumulation."""
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc="train", leave=False)
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(pbar, start=1):
        enc_ids = batch["encoder_input_ids"].to(device, non_blocking=True)
        enc_ttype = batch["encoder_token_type_ids"].to(device, non_blocking=True)
        enc_mask = batch["encoder_attention_mask"].to(device, non_blocking=True)
        dec_ids = batch["decoder_input_ids"].to(device, non_blocking=True)
        dec_mask = batch["decoder_attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            out = model(
                encoder_input_ids=enc_ids,
                encoder_token_type_ids=enc_ttype,
                encoder_attention_mask=enc_mask,
                decoder_input_ids=dec_ids,
                decoder_attention_mask=dec_mask,
                labels=labels,
                label_smoothing=label_smoothing,
            )
            loss = out["loss"] / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        running_loss += loss.item() * grad_accum_steps

        if i % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        pbar.set_postfix(loss=f"{running_loss / max(1, i):.4f}", step=global_step)

    return global_step, running_loss / max(1, len(train_loader))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer_path", default="checkpoints_pretrain_base_seq256/step_20000")
    p.add_argument("--pretrain_ckpt", default="checkpoints_pretrain_base_seq256/step_20000/checkpoint.pt")
    p.add_argument("--init_from_checkpoint", default="")
    p.add_argument("--output_dir", default="checkpoints_generative_qa")
    p.add_argument("--max_input_len", type=int, default=256)
    p.add_argument("--max_target_len", type=int, default=48)
    p.add_argument("--train_batch_size", type=int, default=6)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--encoder_lr", type=float, default=8e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--eval_every_epochs", type=int, default=1)
    p.add_argument("--save_every_epochs", type=int, default=1)
    p.add_argument("--resume_path", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze_warmup_epochs", type=int, default=2)
    p.add_argument("--unfreeze_top_layers", type=int, default=4)
    p.add_argument("--no_squad_v2", action="store_true")
    p.add_argument("--answerable_repeat", type=int, default=1)
    p.add_argument("--no_answer_repeat", type=int, default=1)
    p.add_argument("--target_style", choices=["span", "sentence"], default="span")
    p.add_argument("--no_answer_target_text", default=NO_ANSWER_TEXT)
    p.add_argument("--instruction_prefix", default="")
    p.add_argument("--decoder_variant", choices=["standard", "hybrid"], default="standard")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    args = p.parse_args()

    if args.answerable_repeat < 1:
        raise ValueError("--answerable_repeat must be >= 1")
    if args.no_answer_repeat < 1:
        raise ValueError("--no_answer_repeat must be >= 1")
    if args.target_style == "sentence" and args.no_answer_target_text == NO_ANSWER_TEXT:
        args.no_answer_target_text = NO_ANSWER_SENTENCE

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(
        f"Decoder variant: {args.decoder_variant} | "
        f"Target style: {args.target_style} | "
        f"No-answer target: {args.no_answer_target_text} | "
        f"Instruction prefix: {args.instruction_prefix or '(none)'}"
    )
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tok_cfg = GenQADataConfig(
        tokenizer_path=args.tokenizer_path,
        max_input_len=args.max_input_len,
        max_target_len=args.max_target_len,
        include_squad_v2=not args.no_squad_v2,
        answerable_repeat=args.answerable_repeat,
        no_answer_repeat=args.no_answer_repeat,
        target_style=args.target_style,
        no_answer_target_text=args.no_answer_target_text,
        instruction_prefix=args.instruction_prefix,
        seed=args.seed,
    )
    tokenizer, train_loader, val_loader = build_dataloaders(
        cfg=tok_cfg,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )

    enc_cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=args.max_input_len,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
    )
    cfg_path = Path(args.tokenizer_path) / "model_config.json"
    if cfg_path.exists():
        enc_cfg = ModelConfig(**json.loads(cfg_path.read_text(encoding="utf-8")))
        enc_cfg.max_position_embeddings = args.max_input_len

    dec_cfg = DecoderConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=512,
        num_layers=4,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=args.max_target_len + 8,
        dropout=0.1,
    )

    if args.decoder_variant == "hybrid":
        model = GenerativeQAModelHybrid(enc_cfg, dec_cfg).to(device)
    else:
        model = StandardGenerativeQAModel(enc_cfg, dec_cfg).to(device)

    _ = load_encoder_from_pretrain(model, args.pretrain_ckpt)
    if args.init_from_checkpoint:
        init_step, init_best = load_model_from_gen_checkpoint(
            model, args.init_from_checkpoint, decoder_variant=args.decoder_variant
        )
        print(f"Initialized from generative checkpoint: {args.init_from_checkpoint} (step={init_step}, best={init_best:.4f})")

    encoder_params = []
    decoder_params = []
    # Separate parameters into two groups to apply different learning rates.
    # The pretrained encoder generally needs a smaller LR to avoid catastrophic forgetting,
    # while the randomly initialized decoder needs a larger LR to learn effectively.
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(p)
        else:
            decoder_params.append(p)
    optimizer = AdamW(
        [
            {"params": decoder_params, "lr": args.lr},
            {"params": encoder_params, "lr": args.encoder_lr},
        ],
        weight_decay=args.weight_decay,
    )
    total_updates = max(1, (len(train_loader) // args.grad_accum) * args.epochs)
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = make_scheduler(optimizer, warmup_steps, total_updates)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and args.fp16))
    amp_enabled = device == "cuda" and (args.fp16 or args.bf16)
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    start_epoch = 0
    global_step = 0
    best_metric = -1e9
    latest_path = os.path.join(args.output_dir, "latest.pt")
    best_path = os.path.join(args.output_dir, "best.pt")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.resume_path:
        global_step, start_epoch, best_metric = load_gen_checkpoint(
            args.resume_path, model, optimizer, scheduler, scaler, device
        )
        print(f"Resumed from {args.resume_path} @step={global_step}, epoch={start_epoch}, best={best_metric:.4f}")

    history = []
    for epoch in range(start_epoch, args.epochs):
        # Gradual unfreezing schedule:
        # 1. Freeze the entire encoder early on to let the decoder align to the representations.
        # 2. Unfreeze the top layers of the encoder.
        # 3. Unfreeze the entire model for fine-tuning.
        if epoch < args.freeze_warmup_epochs:
            model.freeze_encoder()
        elif epoch == args.freeze_warmup_epochs:
            model.unfreeze_encoder_top_layers(args.unfreeze_top_layers)
        else:
            model.unfreeze_encoder_all()

        global_step, train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if (device == "cuda" and args.fp16) else None,
            device=device,
            grad_accum_steps=args.grad_accum,
            max_grad_norm=args.max_grad_norm,
            label_smoothing=args.label_smoothing,
            global_step=global_step,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )

        metrics = {}
        if (epoch + 1) % args.eval_every_epochs == 0:
            metrics = evaluate_model(
                model=model,
                tokenizer=tokenizer,
                val_loader=val_loader,
                device=device,
                beam_size=4,
                max_new_tokens=args.max_target_len,
                length_penalty=1.0,
                no_answer_text=args.no_answer_target_text,
            )
            score = metrics["f1"] + 0.25 * metrics["rougeL"]
            if score > best_metric:
                best_metric = score
                save_gen_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler if (device == "cuda" and args.fp16) else None,
                    global_step,
                    epoch + 1,
                    best_metric,
                    enc_cfg,
                    dec_cfg,
                )
                print(f"[best] epoch={epoch+1} score={score:.4f}")

        save_gen_checkpoint(
            latest_path,
            model,
            optimizer,
            scheduler,
            scaler if (device == "cuda" and args.fp16) else None,
            global_step,
            epoch + 1,
            best_metric,
            enc_cfg,
            dec_cfg,
        )

        row = {"epoch": epoch + 1, "train_loss": train_loss, **metrics}
        history.append(row)
        print(json.dumps(row, indent=2))

    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
