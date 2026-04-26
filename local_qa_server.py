import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import AutoTokenizer

from standard_generative_decoder import DecoderConfig, GenerativeQAModel as StandardGenerativeQAModel
from extractive_inference import load_model_and_tokenizer
from mlm_pretraining import ModelConfig

try:
    from main_hybrid_decoder import GenerativeQAModelHybrid

    HYBRID_DECODER_AVAILABLE = True
except Exception:
    GenerativeQAModelHybrid = None
    HYBRID_DECODER_AVAILABLE = False


DEFAULT_NO_ANSWER_TEXT = "The context does not contain the answer."

DEFAULT_GENERATIVE_CHECKPOINT_PATH = Path("checkpoints_generative_qa_hybrid_span_noans_upweight_20260426_111025/latest.pt")
DEFAULT_GENERATIVE_TOKENIZER_PATH = Path("checkpoints_generative_qa_hybrid_span_noans_upweight_20260426_111025")
DEFAULT_GENERATIVE_SECONDARY_CHECKPOINT_PATH = Path("checkpoints_generative_qa_stageE_tradeoff/best.pt")
DEFAULT_GENERATIVE_SECONDARY_TOKENIZER_PATH = Path("checkpoints_generative_qa_stageE_tradeoff")


def _find_checkpoint_in_directory(path: Path) -> Path | None:
    if not path.is_dir():
        return None
    for suffix in ("*.pt", "*.bin"):
        matches = sorted(path.glob(suffix))
        if matches:
            return matches[0]
    return None


def resolve_generative_paths(checkpoint_path: Path, tokenizer_path: Path) -> tuple[Path, Path]:
    initial_checkpoint_path = checkpoint_path
    initial_tokenizer_path = tokenizer_path

    if checkpoint_path.is_dir():
        found = _find_checkpoint_in_directory(checkpoint_path)
        if found is not None:
            checkpoint_path = found
            print(f"Resolved generative checkpoint directory {initial_checkpoint_path} to file {checkpoint_path}.")

    if checkpoint_path.exists() and tokenizer_path.exists():
        return checkpoint_path, tokenizer_path

    print(
        f"[warn] Requested generative checkpoint or tokenizer not available: {checkpoint_path} / {tokenizer_path}."
    )
    if DEFAULT_GENERATIVE_SECONDARY_CHECKPOINT_PATH.exists() and DEFAULT_GENERATIVE_SECONDARY_TOKENIZER_PATH.exists():
        print(
            f"[info] Falling back to secondary generative model at {DEFAULT_GENERATIVE_SECONDARY_CHECKPOINT_PATH} "
            f"with tokenizer {DEFAULT_GENERATIVE_SECONDARY_TOKENIZER_PATH}."
        )
        return DEFAULT_GENERATIVE_SECONDARY_CHECKPOINT_PATH, DEFAULT_GENERATIVE_SECONDARY_TOKENIZER_PATH

    return checkpoint_path, tokenizer_path


class ExtractiveService:
    def __init__(
        self,
        model_dir: Path,
        pretrain_config_dir: Path | None,
        max_length: int,
        doc_stride: int,
        n_best: int,
        max_answer_length: int,
    ):
        self.model, self.tokenizer, self.cfg = load_model_and_tokenizer(model_dir, pretrain_config_dir)
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.n_best = n_best
        self.max_answer_length = max_answer_length
        self.model_name = f"extractive:{model_dir.name}"

    @torch.no_grad()
    def answer(self, question: str, context: str, no_answer_threshold: float | None = None):
        max_length = min(self.max_length, self.cfg.max_position_embeddings)
        doc_stride = min(self.doc_stride, max(8, max_length // 4))

        enc = self.tokenizer(
            question,
            context,
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

        out = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        start_logits = out["start_logits"].cpu().numpy()
        end_logits = out["end_logits"].cpu().numpy()

        cls_token_id = self.tokenizer.cls_token_id
        if cls_token_id is None:
            cls_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        best_score = -1e30
        best_text = ""
        best_null_score = 1e30

        for i in range(input_ids.shape[0]):
            offsets = enc["offset_mapping"][i].tolist()
            seq_ids = enc.sequence_ids(i)
            ids_i = input_ids[i].tolist()
            cls_idx = ids_i.index(cls_token_id) if cls_token_id in ids_i else 0
            null_score = float(start_logits[i][cls_idx] + end_logits[i][cls_idx])
            if null_score < best_null_score:
                best_null_score = null_score

            s_idx = start_logits[i].argsort()[-1 : -self.n_best - 1 : -1].tolist()
            e_idx = end_logits[i].argsort()[-1 : -self.n_best - 1 : -1].tolist()
            for s in s_idx:
                for e in e_idx:
                    if s >= len(offsets) or e >= len(offsets):
                        continue
                    if seq_ids[s] != 1 or seq_ids[e] != 1:
                        continue
                    if e < s or (e - s + 1) > self.max_answer_length:
                        continue
                    st, en = offsets[s][0], offsets[e][1]
                    if st is None or en is None:
                        continue
                    score = float(start_logits[i][s] + end_logits[i][e])
                    if score > best_score:
                        best_score = score
                        best_text = context[st:en]

        result = {
            "answer": best_text,
            "span_score": best_score,
            "null_score": best_null_score,
            "score_diff_null_minus_span": best_null_score - best_score,
            "predict_no_answer": False,
        }

        if no_answer_threshold is not None:
            predict_no_answer = (best_null_score - best_score) > float(no_answer_threshold)
            result["predict_no_answer"] = bool(predict_no_answer)
            if predict_no_answer:
                result["answer"] = ""

        return result


class GenerativeService:
    def __init__(
        self,
        checkpoint_path: Path,
        tokenizer_path: Path,
        decoder_variant: str,
        max_input_len: int,
        max_new_tokens: int,
        beam_size: int,
        length_penalty: float,
        instruction_prefix: str,
        no_answer_text: str,
    ):
        payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        self.enc_cfg = ModelConfig(**payload["encoder_config"])
        self.dec_cfg = DecoderConfig(**payload["decoder_config"])
        self.model, self.loaded_variant = self._build_model(payload, decoder_variant)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.sep_token

        self.max_input_len = max_input_len
        self.max_new_tokens = max_new_tokens
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.instruction_prefix = instruction_prefix
        self.no_answer_text = no_answer_text
        self.model_name = f"generative:{checkpoint_path.parent.name}:{self.loaded_variant}"

    def _candidate_variants(self, decoder_variant: str):
        variant = decoder_variant.strip().lower()
        if variant == "standard":
            return ["standard"]
        if variant == "hybrid":
            if not HYBRID_DECODER_AVAILABLE:
                raise RuntimeError("Hybrid decoder was requested but the module is not available.")
            return ["hybrid"]
        if variant == "auto":
            return ["hybrid", "standard"] if HYBRID_DECODER_AVAILABLE else ["standard"]
        raise ValueError("decoder_variant must be one of: auto, standard, hybrid")

    def _instantiate_model(self, variant: str):
        if variant == "standard":
            return StandardGenerativeQAModel(self.enc_cfg, self.dec_cfg)
        if variant == "hybrid" and HYBRID_DECODER_AVAILABLE:
            return GenerativeQAModelHybrid(self.enc_cfg, self.dec_cfg)
        raise RuntimeError(f"Unsupported decoder variant: {variant}")

    def _build_model(self, payload, decoder_variant: str):
        errors = []

        for variant in self._candidate_variants(decoder_variant):
            try:
                model = self._instantiate_model(variant)
                model.load_state_dict(payload["model"], strict=True)
                return model, variant
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{variant}: {exc}")
        joined = " | ".join(errors)
        raise RuntimeError(f"Could not load generative checkpoint with requested variant(s): {joined}")

    @staticmethod
    def _decode_generated_ids(tokenizer, out_ids, bos: int, eos: int, pad: int) -> str:
        text_ids = []
        for tid in out_ids:
            if tid in {bos, pad}:
                continue
            if tid == eos:
                break
            text_ids.append(tid)
        return tokenizer.decode(text_ids, skip_special_tokens=True).strip()

    @staticmethod
    def _build_target_ids(tokenizer, text: str, bos: int, eos: int, max_new_tokens: int, device: str):
        ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max(1, max_new_tokens - 1),
        )["input_ids"]
        seq = [bos] + ids + [eos]
        return torch.tensor([seq], dtype=torch.long, device=device)

    @torch.no_grad()
    def answer(
        self,
        question: str,
        context: str,
        no_answer_threshold: float | None = None,
        instruction_prefix: str | None = None,
    ):
        prefix = self.instruction_prefix if instruction_prefix is None else instruction_prefix
        if prefix and prefix.strip():
            inp = f"{prefix.strip()} question: {question} context: {context}"
        else:
            inp = f"question: {question} context: {context}"

        enc = self.tokenizer(
            [inp],
            truncation=True,
            max_length=min(self.max_input_len, self.enc_cfg.max_position_embeddings),
            return_tensors="pt",
        )

        enc_ids = enc["input_ids"].to(self.device)
        enc_mask = enc["attention_mask"].to(self.device)
        enc_ttype = enc.get("token_type_ids", torch.zeros_like(enc_ids)).to(self.device)

        bos = self.tokenizer.cls_token_id if self.tokenizer.cls_token_id is not None else self.tokenizer.pad_token_id
        eos = self.tokenizer.sep_token_id if self.tokenizer.sep_token_id is not None else self.tokenizer.pad_token_id
        pad = self.tokenizer.pad_token_id

        gate_enabled = no_answer_threshold is not None
        if gate_enabled:
            out, pred_logprob, _ = self.model.generate(
                encoder_input_ids=enc_ids,
                encoder_token_type_ids=enc_ttype,
                encoder_attention_mask=enc_mask,
                bos_token_id=bos,
                eos_token_id=eos,
                pad_token_id=pad,
                max_new_tokens=self.max_new_tokens,
                beam_size=self.beam_size,
                length_penalty=self.length_penalty,
                return_logprob=True,
            )
            out_ids = out[0].tolist()
            raw_answer = self._decode_generated_ids(self.tokenizer, out_ids, bos=bos, eos=eos, pad=pad)

            noans_ids = self._build_target_ids(
                tokenizer=self.tokenizer,
                text=self.no_answer_text,
                bos=bos,
                eos=eos,
                max_new_tokens=self.max_new_tokens,
                device=self.device,
            )
            noans_logprob = self.model.sequence_logprob(
                encoder_input_ids=enc_ids,
                encoder_token_type_ids=enc_ttype,
                encoder_attention_mask=enc_mask,
                target_ids=noans_ids,
                pad_token_id=pad,
                normalize_by_length=True,
            )[0].item()

            score_diff = noans_logprob - pred_logprob
            selected_no_answer = score_diff > float(no_answer_threshold)
            return {
                "answer": self.no_answer_text if selected_no_answer else raw_answer,
                "raw_answer": raw_answer,
                "predict_no_answer": bool(selected_no_answer),
                "gate": {
                    "enabled": True,
                    "threshold": float(no_answer_threshold),
                    "score_diff": float(score_diff),
                    "pred_avg_logprob": float(pred_logprob),
                    "no_answer_avg_logprob": float(noans_logprob),
                },
            }

        out_ids = self.model.generate(
            encoder_input_ids=enc_ids,
            encoder_token_type_ids=enc_ttype,
            encoder_attention_mask=enc_mask,
            bos_token_id=bos,
            eos_token_id=eos,
            pad_token_id=pad,
            max_new_tokens=self.max_new_tokens,
            beam_size=self.beam_size,
            length_penalty=self.length_penalty,
        )[0].tolist()

        answer = self._decode_generated_ids(self.tokenizer, out_ids, bos=bos, eos=eos, pad=pad)
        return {
            "answer": answer,
            "raw_answer": answer,
            "predict_no_answer": answer.strip() == self.no_answer_text,
            "gate": {
                "enabled": False,
            },
        }


EXTRACTIVE_SERVICE: ExtractiveService | None = None
GENERATIVE_SERVICE: GenerativeService | None = None


class QAHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_POST(self):
        if self.path != "/api/answer":
            self.send_error(404, "Not Found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))

            mode = str(payload.get("mode", "extractive")).strip().lower()
            question = str(payload.get("question", "")).strip()
            context = str(payload.get("context", "")).strip()
            threshold = payload.get("no_answer_threshold", None)
            instruction_prefix = payload.get("instruction_prefix", None)

            if mode not in {"extractive", "generative"}:
                self._write_json(400, {"error": "mode must be either 'extractive' or 'generative'."})
                return

            if not question:
                self._write_json(400, {"error": "Question is required."})
                return
            if not context:
                self._write_json(400, {"error": "Context is required."})
                return

            if threshold is not None:
                try:
                    threshold = float(threshold)
                except (TypeError, ValueError):
                    self._write_json(400, {"error": "no_answer_threshold must be numeric."})
                    return

            if mode == "extractive":
                if EXTRACTIVE_SERVICE is None:
                    self._write_json(503, {"error": "Extractive model is not available on this server."})
                    return

                out = EXTRACTIVE_SERVICE.answer(question=question, context=context, no_answer_threshold=threshold)
                no_ans = bool(out.get("predict_no_answer", False))
                answer_text = out.get("answer", "").strip()
                if no_ans and not answer_text:
                    answer_text = DEFAULT_NO_ANSWER_TEXT

                self._write_json(
                    200,
                    {
                        "mode": "extractive",
                        "question": question,
                        "answer": answer_text,
                        "predict_no_answer": no_ans,
                        "model_name": EXTRACTIVE_SERVICE.model_name,
                        "meta": {
                            "span_score": out.get("span_score"),
                            "null_score": out.get("null_score"),
                            "score_diff_null_minus_span": out.get("score_diff_null_minus_span"),
                        },
                    },
                )
                return

            if GENERATIVE_SERVICE is None:
                self._write_json(503, {"error": "Generative model is not available on this server."})
                return

            if instruction_prefix is not None:
                instruction_prefix = str(instruction_prefix)

            out = GENERATIVE_SERVICE.answer(
                question=question,
                context=context,
                no_answer_threshold=threshold,
                instruction_prefix=instruction_prefix,
            )
            self._write_json(
                200,
                {
                    "mode": "generative",
                    "question": question,
                    "answer": out.get("answer", "").strip(),
                    "raw_answer": out.get("raw_answer", "").strip(),
                    "predict_no_answer": bool(out.get("predict_no_answer", False)),
                    "model_name": GENERATIVE_SERVICE.model_name,
                    "meta": {
                        "gate": out.get("gate", {}),
                    },
                },
            )
        except json.JSONDecodeError:
            self._write_json(400, {"error": "Invalid JSON body."})
        except Exception as exc:  # noqa: BLE001
            self._write_json(500, {"error": f"Inference failed: {exc}"})

    def _write_json(self, status: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args():
    p = argparse.ArgumentParser(description="Local QA server with extractive and generative modes.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)

    p.add_argument(
        "--extractive_model_dir",
        default="checkpoints_qa_squad_v2_lr5e-5_len256_e3",
        help="Directory containing extractive model.safetensors and tokenizer files.",
    )
    p.add_argument(
        "--extractive_pretrain_config_dir",
        default="checkpoints_pretrain_base_seq256/step_20000",
        help="Directory containing model_config.json if missing in extractive model dir.",
    )
    p.add_argument("--extractive_max_length", type=int, default=256)
    p.add_argument("--extractive_doc_stride", type=int, default=64)
    p.add_argument("--extractive_n_best", type=int, default=20)
    p.add_argument("--extractive_max_answer_length", type=int, default=30)

    p.add_argument(
        "--generative_checkpoint_path",
        default="checkpoints_generative_qa_hybrid_span_noans_upweight_20260426_111025/latest.pt",
        help="Path to generative checkpoint (.pt).",
    )
    p.add_argument(
        "--generative_tokenizer_path",
        default="checkpoints_generative_qa_hybrid_span_noans_upweight_20260426_111025",
        help="Tokenizer directory for generative model.",
    )
    p.add_argument(
        "--generative_decoder_variant",
        choices=["auto", "standard", "hybrid"],
        default="auto",
        help="Decoder variant for loading generative checkpoint. auto prefers hybrid first, then standard.",
    )
    p.add_argument("--generative_max_input_len", type=int, default=256)
    p.add_argument("--generative_max_new_tokens", type=int, default=48)
    p.add_argument("--generative_beam_size", type=int, default=5)
    p.add_argument("--generative_length_penalty", type=float, default=1.05)
    p.add_argument(
        "--generative_instruction_prefix",
        default="Answer in one concise sentence based only on the context.",
    )
    p.add_argument(
        "--generative_no_answer_text",
        default=DEFAULT_NO_ANSWER_TEXT,
    )
    return p.parse_args()


def main():
    global EXTRACTIVE_SERVICE, GENERATIVE_SERVICE
    args = parse_args()

    try:
        EXTRACTIVE_SERVICE = ExtractiveService(
            model_dir=Path(args.extractive_model_dir),
            pretrain_config_dir=Path(args.extractive_pretrain_config_dir) if args.extractive_pretrain_config_dir else None,
            max_length=args.extractive_max_length,
            doc_stride=args.extractive_doc_stride,
            n_best=args.extractive_n_best,
            max_answer_length=args.extractive_max_answer_length,
        )
        print(f"Loaded extractive model: {EXTRACTIVE_SERVICE.model_name}")
    except Exception as exc:  # noqa: BLE001
        EXTRACTIVE_SERVICE = None
        print(f"[warn] Could not load extractive model: {exc}")

    try:
        generative_checkpoint_path = Path(args.generative_checkpoint_path)
        generative_tokenizer_path = Path(args.generative_tokenizer_path)
        generative_checkpoint_path, generative_tokenizer_path = resolve_generative_paths(
            generative_checkpoint_path,
            generative_tokenizer_path,
        )

        GENERATIVE_SERVICE = GenerativeService(
            checkpoint_path=generative_checkpoint_path,
            tokenizer_path=generative_tokenizer_path,
            decoder_variant=args.generative_decoder_variant,
            max_input_len=args.generative_max_input_len,
            max_new_tokens=args.generative_max_new_tokens,
            beam_size=args.generative_beam_size,
            length_penalty=args.generative_length_penalty,
            instruction_prefix=args.generative_instruction_prefix,
            no_answer_text=args.generative_no_answer_text,
        )
        print(f"Loaded generative model: {GENERATIVE_SERVICE.model_name}")
    except Exception as exc:  # noqa: BLE001
        GENERATIVE_SERVICE = None
        print(f"[warn] Could not load generative model: {exc}")

    if EXTRACTIVE_SERVICE is None and GENERATIVE_SERVICE is None:
        raise RuntimeError("Neither extractive nor generative model could be loaded.")

    server = ThreadingHTTPServer((args.host, args.port), QAHandler)
    print(f"Local QA server running at http://{args.host}:{args.port}")
    print("POST /api/answer with JSON: {\"mode\":\"extractive|generative\", \"context\":..., \"question\":..., \"no_answer_threshold\": optional}")
    server.serve_forever()


if __name__ == "__main__":
    main()
