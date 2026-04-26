import argparse
import json
from pathlib import Path

import evaluate
import torch

from generative_data import NO_ANSWER_TEXT, GenQADataConfig, build_dataloaders, normalize_text
from standard_generative_decoder import DecoderConfig, GenerativeQAModel as StandardGenerativeQAModel
from main_hybrid_decoder import GenerativeQAModelHybrid
from mlm_pretraining import ModelConfig


def exact_match(pred: str, gold: str) -> float:
    """Computes the exact match score between prediction and ground truth."""
    return float(normalize_text(pred) == normalize_text(gold))


def f1_score(pred: str, gold: str) -> float:
    """Computes the token-level F1 score between prediction and ground truth."""
    p_toks = normalize_text(pred).split()
    g_toks = normalize_text(gold).split()
    common = {}
    for t in p_toks:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in g_toks:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    prec = overlap / max(1, len(p_toks))
    rec = overlap / max(1, len(g_toks))
    return 2 * prec * rec / max(1e-8, prec + rec)


def _decode_generated_ids(tokenizer, out_ids, bos: int, eos: int, pad: int) -> str:
    """Decodes a list of token IDs back into a text string, ignoring special tokens."""
    text_ids = []
    for tid in out_ids:
        if tid in {bos, pad}:
            continue
        if tid == eos:
            break
        text_ids.append(tid)
    return tokenizer.decode(text_ids, skip_special_tokens=True).strip()


def _build_target_ids(tokenizer, text: str, bos: int, eos: int, max_new_tokens: int, device: str) -> torch.Tensor:
    """Tokenizes and formats target text into a padded tensor sequence for probability evaluation."""
    ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max(1, max_new_tokens - 1),
    )["input_ids"]
    seq = [bos] + ids + [eos]
    return torch.tensor([seq], dtype=torch.long, device=device)


def _select_gate_threshold(
    is_noans_flags,
    score_diffs,
    threshold_points: int,
):
    """Scans across possible logprob thresholds to find the one that optimally balances unanswerable and answerable accuracy."""
    if not score_diffs:
        return 0.0, {"threshold_scan": [], "gate_balance": 0.0}

    lo = min(score_diffs)
    hi = max(score_diffs)
    if lo == hi:
        thresholds = [lo]
    else:
        # Create a linear grid of candidate thresholds between the min and max observed logprob differences.
        steps = max(3, threshold_points)
        thresholds = [lo + (hi - lo) * (i / (steps - 1)) for i in range(steps)]
    thresholds.append(0.0)
    thresholds = sorted(set(round(t, 6) for t in thresholds))

    best = {
        "threshold": 0.0,
        "gate_balance": -1.0,
        "no_answer_accuracy": 0.0,
        "answerable_accuracy": 0.0,
    }
    for th in thresholds:
        noans_total = 0
        noans_correct = 0
        ans_total = 0
        ans_correct = 0
        for is_noans, diff in zip(is_noans_flags, score_diffs):
            pred_noans = diff > th
            if is_noans:
                noans_total += 1
                if pred_noans:
                    noans_correct += 1
            else:
                ans_total += 1
                if not pred_noans:
                    ans_correct += 1
        noans_acc = 100.0 * noans_correct / max(1, noans_total)
        ans_acc = 100.0 * ans_correct / max(1, ans_total)
        gate_balance = 0.5 * (noans_acc + ans_acc)
        if gate_balance > best["gate_balance"]:
            best = {
                "threshold": th,
                "gate_balance": gate_balance,
                "no_answer_accuracy": noans_acc,
                "answerable_accuracy": ans_acc,
            }
    best["threshold_scan"] = [thresholds[0], thresholds[-1], len(thresholds)]
    return float(best["threshold"]), best


@torch.no_grad()
def evaluate_model(
    model,
    tokenizer,
    val_loader,
    device,
    beam_size=4,
    max_new_tokens=32,
    length_penalty=1.0,
    no_answer_text=NO_ANSWER_TEXT,
    enable_no_answer_gate=False,
    no_answer_threshold=0.0,
    tune_no_answer_threshold=False,
    threshold_points=101,
    max_eval_examples=0,
):
    """Runs generation and evaluation metrics (EM, F1, ROUGE, BLEU) over a dataset."""
    model.eval()
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    bos = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.pad_token_id
    eos = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.pad_token_id
    pad = tokenizer.pad_token_id

    gate_active = enable_no_answer_gate or tune_no_answer_threshold
    noans_target_ids = None
    if gate_active:
        noans_target_ids = _build_target_ids(
            tokenizer=tokenizer,
            text=no_answer_text,
            bos=bos,
            eos=eos,
            max_new_tokens=max_new_tokens,
            device=device,
        )

    raw_preds = []
    golds = []
    is_noans_flags = []
    score_diffs = []

    processed = 0
    stop_early = max_eval_examples is not None and int(max_eval_examples) > 0
    for batch in val_loader:
        bsz = batch["encoder_input_ids"].size(0)
        for i in range(bsz):
            if stop_early and processed >= int(max_eval_examples):
                break
            enc_ids = batch["encoder_input_ids"][i : i + 1].to(device)
            enc_ttype = batch["encoder_token_type_ids"][i : i + 1].to(device)
            enc_mask = batch["encoder_attention_mask"][i : i + 1].to(device)
            if gate_active:
                # Generate the best answer normally and get its sequence log probability.
                out, pred_logprob, _ = model.generate(
                    encoder_input_ids=enc_ids,
                    encoder_token_type_ids=enc_ttype,
                    encoder_attention_mask=enc_mask,
                    bos_token_id=bos,
                    eos_token_id=eos,
                    pad_token_id=pad,
                    max_new_tokens=max_new_tokens,
                    beam_size=beam_size,
                    length_penalty=length_penalty,
                    return_logprob=True,
                )
                out_ids = out[0].tolist()
                
                # Force the model to decode the "No answer" string and compute its sequence log probability.
                noans_logprob = model.sequence_logprob(
                    encoder_input_ids=enc_ids,
                    encoder_token_type_ids=enc_ttype,
                    encoder_attention_mask=enc_mask,
                    target_ids=noans_target_ids,
                    pad_token_id=pad,
                    normalize_by_length=True,
                )[0].item()
                
                # Store the difference to decide whether to override the prediction later.
                score_diffs.append(noans_logprob - pred_logprob)
            else:
                out_ids = model.generate(
                    encoder_input_ids=enc_ids,
                    encoder_token_type_ids=enc_ttype,
                    encoder_attention_mask=enc_mask,
                    bos_token_id=bos,
                    eos_token_id=eos,
                    pad_token_id=pad,
                    max_new_tokens=max_new_tokens,
                    beam_size=beam_size,
                    length_penalty=length_penalty,
                )[0].tolist()

            pred = _decode_generated_ids(tokenizer, out_ids, bos=bos, eos=eos, pad=pad)
            gold = batch["target_text"][i].strip()
            is_noans = normalize_text(gold) == normalize_text(no_answer_text)
            raw_preds.append(pred)
            golds.append(gold)
            is_noans_flags.append(is_noans)
            processed += 1
        if stop_early and processed >= int(max_eval_examples):
            break

    selected_threshold = float(no_answer_threshold)
    tuning_report = None
    if tune_no_answer_threshold:
        selected_threshold, tuning_report = _select_gate_threshold(
            is_noans_flags=is_noans_flags,
            score_diffs=score_diffs,
            threshold_points=threshold_points,
        )

    preds = []
    refs = []
    ems = []
    f1s = []
    noans_correct = 0
    noans_total = 0
    ans_correct = 0
    ans_total = 0
    for idx, (raw_pred, gold, is_noans) in enumerate(zip(raw_preds, golds, is_noans_flags)):
        if gate_active:
            # Override the generated text with the unanswerable text if the logprob difference exceeds the threshold.
            pred = no_answer_text if score_diffs[idx] > selected_threshold else raw_pred
        else:
            pred = raw_pred
        preds.append(pred)
        refs.append([gold])
        ems.append(exact_match(pred, gold))
        f1s.append(f1_score(pred, gold))

        pred_noans = normalize_text(pred) == normalize_text(no_answer_text)
        if is_noans:
            noans_total += 1
            if pred_noans:
                noans_correct += 1
        else:
            ans_total += 1
            if not pred_noans:
                ans_correct += 1

    rouge_res = rouge.compute(predictions=preds, references=[x[0] for x in refs])
    bleu_res = bleu.compute(predictions=preds, references=refs)
    metrics = {
        "exact_match": 100.0 * sum(ems) / max(1, len(ems)),
        "f1": 100.0 * sum(f1s) / max(1, len(f1s)),
        "rougeL": 100.0 * rouge_res["rougeL"],
        "bleu": 100.0 * bleu_res["bleu"],
        "no_answer_accuracy": 100.0 * noans_correct / max(1, noans_total),
        "answerable_accuracy": 100.0 * ans_correct / max(1, ans_total),
        "num_examples": len(ems),
        "gate_enabled": bool(gate_active),
        "no_answer_threshold": float(selected_threshold) if gate_active else None,
    }
    if tuning_report is not None:
        metrics["tuned_threshold_report"] = tuning_report
    return metrics


def main():
    """Main entry point for parsing arguments and running the generative model evaluation pipeline."""
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--max_input_len", type=int, default=256)
    p.add_argument("--max_target_len", type=int, default=48)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--target_style", choices=["span", "sentence"], default="span")
    p.add_argument("--no_answer_target_text", default=NO_ANSWER_TEXT)
    p.add_argument("--instruction_prefix", default="")
    p.add_argument("--decoder_variant", choices=["standard", "hybrid"], default="standard")
    p.add_argument("--beam_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--length_penalty", type=float, default=1.0)
    p.add_argument("--no_answer_text", default=NO_ANSWER_TEXT)
    p.add_argument("--enable_no_answer_gate", action="store_true")
    p.add_argument("--no_answer_threshold", type=float, default=0.0)
    p.add_argument("--tune_no_answer_threshold", action="store_true")
    p.add_argument("--threshold_points", type=int, default=101)
    p.add_argument("--max_eval_examples", type=int, default=0)
    p.add_argument("--out_json", default="generative_eval_metrics.json")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    enc_cfg = ModelConfig(**ckpt["encoder_config"])
    dec_cfg = DecoderConfig(**ckpt["decoder_config"])
    if args.decoder_variant == "hybrid":
        model = GenerativeQAModelHybrid(enc_cfg, dec_cfg)
    else:
        model = StandardGenerativeQAModel(enc_cfg, dec_cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    cfg = GenQADataConfig(
        tokenizer_path=args.tokenizer_path,
        max_input_len=args.max_input_len,
        max_target_len=args.max_target_len,
        include_squad_v2=True,
        target_style=args.target_style,
        no_answer_target_text=args.no_answer_target_text,
        instruction_prefix=args.instruction_prefix,
    )
    tokenizer, _, val_loader = build_dataloaders(
        cfg=cfg,
        train_batch_size=1,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )

    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        val_loader=val_loader,
        device=device,
        beam_size=args.beam_size,
        max_new_tokens=args.max_new_tokens,
        length_penalty=args.length_penalty,
        no_answer_text=args.no_answer_text,
        enable_no_answer_gate=args.enable_no_answer_gate,
        no_answer_threshold=args.no_answer_threshold,
        tune_no_answer_threshold=args.tune_no_answer_threshold,
        threshold_points=args.threshold_points,
        max_eval_examples=args.max_eval_examples,
    )
    Path(args.out_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {args.out_json}")


if __name__ == "__main__":
    main()
