import argparse
import json

import torch
from transformers import AutoTokenizer

from standard_generative_decoder import DecoderConfig, GenerativeQAModel as StandardGenerativeQAModel
from main_hybrid_decoder import GenerativeQAModelHybrid
from mlm_pretraining import ModelConfig


def parse_args():
    """Parses command-line arguments for generative QA inference."""
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", required=True, help="Path to best.pt or latest.pt")
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--context", required=True)
    p.add_argument("--max_input_len", type=int, default=256)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--beam_size", type=int, default=4)
    p.add_argument("--length_penalty", type=float, default=1.0)
    p.add_argument("--instruction_prefix", default="")
    p.add_argument("--decoder_variant", choices=["standard", "hybrid"], default="standard")
    p.add_argument("--enable_no_answer_gate", action="store_true")
    p.add_argument("--no_answer_text", default="The context does not contain the answer.")
    p.add_argument("--no_answer_threshold", type=float, default=0.0)
    return p.parse_args()


def decode_generated_ids(tokenizer, out_ids, bos: int, eos: int, pad: int) -> str:
    """Decodes a list of token IDs back into a text string, ignoring special tokens."""
    text_ids = []
    for t in out_ids:
        if t in {bos, pad}:
            continue
        if t == eos:
            break
        text_ids.append(t)
    return tokenizer.decode(text_ids, skip_special_tokens=True).strip()


def build_target_ids(tokenizer, text: str, bos: int, eos: int, max_new_tokens: int, device: str) -> torch.Tensor:
    """Tokenizes and formats target text into a padded tensor sequence for probability evaluation."""
    ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max(1, max_new_tokens - 1),
    )["input_ids"]
    seq = [bos] + ids + [eos]
    return torch.tensor([seq], dtype=torch.long, device=device)


def main():
    """Main entry point for running generative QA inference from the command line."""
    args = parse_args()
    payload = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    enc_cfg = ModelConfig(**payload["encoder_config"])
    dec_cfg = DecoderConfig(**payload["decoder_config"])
    if args.decoder_variant == "hybrid":
        model = GenerativeQAModelHybrid(enc_cfg, dec_cfg)
    else:
        model = StandardGenerativeQAModel(enc_cfg, dec_cfg)
    model.load_state_dict(payload["model"], strict=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.sep_token

    if args.instruction_prefix:
        inp = f"{args.instruction_prefix.strip()} question: {args.question} context: {args.context}"
    else:
        inp = f"question: {args.question} context: {args.context}"
    enc = tok(
        [inp],
        truncation=True,
        max_length=min(args.max_input_len, enc_cfg.max_position_embeddings),
        return_tensors="pt",
    )
    enc_ids = enc["input_ids"].to(device)
    enc_mask = enc["attention_mask"].to(device)
    enc_ttype = enc.get("token_type_ids", torch.zeros_like(enc_ids)).to(device)

    bos = tok.cls_token_id if tok.cls_token_id is not None else tok.pad_token_id
    eos = tok.sep_token_id if tok.sep_token_id is not None else tok.pad_token_id
    pad = tok.pad_token_id

    if args.enable_no_answer_gate:
        # Generate the best answer normally and get its sequence log probability.
        out, pred_logprob, _ = model.generate(
            encoder_input_ids=enc_ids,
            encoder_token_type_ids=enc_ttype,
            encoder_attention_mask=enc_mask,
            bos_token_id=bos,
            eos_token_id=eos,
            pad_token_id=pad,
            max_new_tokens=args.max_new_tokens,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            return_logprob=True,
        )
        out_ids = out[0].tolist()
    else:
        out_ids = model.generate(
            encoder_input_ids=enc_ids,
            encoder_token_type_ids=enc_ttype,
            encoder_attention_mask=enc_mask,
            bos_token_id=bos,
            eos_token_id=eos,
            pad_token_id=pad,
            max_new_tokens=args.max_new_tokens,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
        )[0].tolist()

    raw_answer = decode_generated_ids(tok, out_ids, bos=bos, eos=eos, pad=pad)

    output = {"question": args.question, "answer": raw_answer}
    if args.enable_no_answer_gate:
        noans_ids = build_target_ids(
            tokenizer=tok,
            text=args.no_answer_text,
            bos=bos,
            eos=eos,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        
        # Force the model to decode the "No answer" string and compute its sequence log probability.
        noans_logprob = model.sequence_logprob(
            encoder_input_ids=enc_ids,
            encoder_token_type_ids=enc_ttype,
            encoder_attention_mask=enc_mask,
            target_ids=noans_ids,
            pad_token_id=pad,
            normalize_by_length=True,
        )[0].item()
        
        # Compare the forced no-answer logprob against the generated answer logprob.
        # If the difference exceeds the threshold, override the prediction with the no-answer text.
        score_diff = noans_logprob - pred_logprob
        gated = score_diff > args.no_answer_threshold
        output = {
            "question": args.question,
            "answer": args.no_answer_text if gated else raw_answer,
            "raw_answer": raw_answer,
            "gate": {
                "enabled": True,
                "no_answer_text": args.no_answer_text,
                "no_answer_threshold": args.no_answer_threshold,
                "score_diff": score_diff,
                "pred_avg_logprob": pred_logprob,
                "no_answer_avg_logprob": noans_logprob,
                "selected_no_answer": gated,
            },
        }

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
