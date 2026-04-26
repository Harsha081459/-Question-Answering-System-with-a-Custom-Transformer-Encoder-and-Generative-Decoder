import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from extractive_finetuning import BertForQuestionAnswering
from mlm_pretraining import ModelConfig


def parse_args():
    """Parses command-line arguments for the extractive inference script."""
    p = argparse.ArgumentParser(description="Run QA inference on custom question/context.")
    p.add_argument("--model_dir", required=True, help="Path to finetuned model folder")
    p.add_argument("--pretrain_config_dir", default="", help="Path with model_config.json if missing in model_dir")
    p.add_argument("--question", required=True, help="Question text")
    p.add_argument("--context", required=True, help="Context paragraph")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--doc_stride", type=int, default=64)
    p.add_argument("--n_best", type=int, default=20)
    p.add_argument("--max_answer_length", type=int, default=30)
    p.add_argument("--no_answer_threshold", type=float, default=None, help="Use for SQuAD v2 style no-answer")
    return p.parse_args()


def load_model_and_tokenizer(model_dir: Path, pretrain_config_dir: Path | None):
    """Loads the model configuration, tokenizer, and pretrained safetensors weights."""
    config_path = model_dir / "model_config.json"
    if not config_path.exists():
        # Fallback: If the fine-tuned model dir lacks the config, use the pretrain config dir.
        if pretrain_config_dir is None:
            raise FileNotFoundError(
                "model_config.json missing in model_dir. Pass --pretrain_config_dir containing model_config.json."
            )
        config_path = pretrain_config_dir / "model_config.json"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = ModelConfig(**json.load(f))

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.sep_token

    model = BertForQuestionAnswering(cfg)
    state_path = model_dir / "model.safetensors"
    if not state_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in {model_dir}")
    state = load_file(str(state_path))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, tokenizer, cfg


def main():
    """Runs the inference logic by tokenizing the input, computing logits, and extracting the best answer span."""
    args = parse_args()
    model_dir = Path(args.model_dir)
    pre_cfg_dir = Path(args.pretrain_config_dir) if args.pretrain_config_dir else None

    model, tokenizer, cfg = load_model_and_tokenizer(model_dir, pre_cfg_dir)

    max_length = min(args.max_length, cfg.max_position_embeddings)
    doc_stride = min(args.doc_stride, max(8, max_length // 4))

    enc = tokenizer(
        [args.question],
        [args.context],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    start_logits = out["start_logits"].cpu().numpy()
    end_logits = out["end_logits"].cpu().numpy()

    best_score = -1e30
    best_text = ""
    best_null_score = 1e30

    for i in range(input_ids.shape[0]):
        # A single question/context pair might be split into multiple chunks (features) if it exceeds max_length.
        offsets = enc["offset_mapping"][i].tolist()
        seq_ids = enc.sequence_ids(i)
        ids_i = input_ids[i].tolist()
        cls_idx = ids_i.index(tokenizer.cls_token_id)
        # Compute the null score (the score for predicting the CLS token, implying no answer)
        null_score = float(start_logits[i][cls_idx] + end_logits[i][cls_idx])
        if null_score < best_null_score:
            best_null_score = null_score

        # Get the top `n_best` candidate start and end token indices.
        s_idx = start_logits[i].argsort()[-1 : -args.n_best - 1 : -1].tolist()
        e_idx = end_logits[i].argsort()[-1 : -args.n_best - 1 : -1].tolist()
        for s in s_idx:
            for e in e_idx:
                if s >= len(offsets) or e >= len(offsets):
                    continue
                # Ensure the selected tokens actually belong to the context paragraph (sequence_id == 1).
                if seq_ids[s] != 1 or seq_ids[e] != 1:
                    continue
                # Reject impossible answers where the end index is before the start, or the span is too long.
                if e < s or (e - s + 1) > args.max_answer_length:
                    continue
                st, en = offsets[s][0], offsets[e][1]
                if st is None or en is None:
                    continue
                
                # Keep track of the highest scoring valid span across all chunks.
                score = float(start_logits[i][s] + end_logits[i][e])
                if score > best_score:
                    best_score = score
                    best_text = args.context[st:en]

    output = {
        "question": args.question,
        "answer": best_text,
        "span_score": best_score,
        "null_score": best_null_score,
        "score_diff_null_minus_span": best_null_score - best_score,
    }

    if args.no_answer_threshold is not None:
        output["no_answer_threshold"] = args.no_answer_threshold
        # SQuAD v2 style prediction: If the null score minus the best span score exceeds the threshold, predict no answer.
        output["predict_no_answer"] = (best_null_score - best_score) > args.no_answer_threshold
        if output["predict_no_answer"]:
            output["answer"] = ""

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

