import argparse
import collections
import json
import os
from pathlib import Path

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from mlm_pretraining import BertEncoder, ModelConfig


class BertForQuestionAnswering(nn.Module):
    """Model for extractive question answering using a BERT encoder."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.encoder = BertEncoder(cfg)
        self.qa_outputs = nn.Linear(cfg.hidden_size, 2)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        start_positions=None,
        end_positions=None,
    ):
        """Computes start and end logits, and loss if labels are provided."""
        hidden = self.encoder(input_ids, token_type_ids, attention_mask)
        logits = self.qa_outputs(hidden)
        start_logits, end_logits = logits[..., 0], logits[..., 1]

        loss = None
        if start_positions is not None and end_positions is not None:
            start_loss = nn.functional.cross_entropy(start_logits, start_positions)
            end_loss = nn.functional.cross_entropy(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

        if loss is None:
            return {"start_logits": start_logits, "end_logits": end_logits}
        return {"loss": loss, "start_logits": start_logits, "end_logits": end_logits}


def parse_args():
    """Parses command-line arguments for training and evaluation."""
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--dataset", type=str, default="squad", choices=["squad", "squad_v2"])
    p.add_argument("--output_dir", type=str, default="checkpoints_qa_squad")
    p.add_argument("--max_length", type=int, default=384)
    p.add_argument("--doc_stride", type=int, default=128)
    p.add_argument("--per_device_batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--num_train_epochs", type=float, default=2.0)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()


def load_pretrained_encoder(model: BertForQuestionAnswering, checkpoint_dir: str):
    """Loads a pretrained BERT encoder from a checkpoint directory."""
    ckpt_path = Path(checkpoint_dir) / "checkpoint.pt"
    payload = torch.load(str(ckpt_path), map_location="cpu")
    state = payload["model"]
    # Strip the "encoder." prefix from the state dict keys so they can be 
    # loaded directly into the self.encoder submodule.
    enc_state = {k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(enc_state, strict=True)
    return int(payload.get("step", 0))


def main():
    """Main entry point for training and evaluating the extractive QA model."""
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    with open(ckpt_dir / "model_config.json", "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg = ModelConfig(**cfg_dict)
    if args.max_length > cfg.max_position_embeddings:
        raise ValueError(
            f"--max_length ({args.max_length}) exceeds pretrained max_position_embeddings "
            f"({cfg.max_position_embeddings}). Use --max_length {cfg.max_position_embeddings}."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.sep_token

    model = BertForQuestionAnswering(cfg)
    restored_step = load_pretrained_encoder(model, str(ckpt_dir))
    print(f"Loaded encoder weights from step={restored_step}")

    dataset = load_dataset(args.dataset)
    train_examples = dataset["train"]
    eval_examples = dataset["validation"]

    max_length = args.max_length
    # Keep stride safely below tokenizer's effective max len to avoid tokenizers panic.
    doc_stride = min(args.doc_stride, max(8, max_length // 4))

    def prepare_train_features(examples):
        """Tokenizes and formats training examples into features with start and end positions."""
        tokenized = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        start_positions = []
        end_positions = []
        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = tokenized.sequence_ids(i)
            sample_idx = sample_mapping[i]
            answers = examples["answers"][sample_idx]

            if len(answers["answer_start"]) == 0:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Find the first and last token indices that belong to the context (sequence_id == 1).
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # If the answer is completely outside the current feature's context span, label it with the CLS token.
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                # Otherwise, shift the token_start_index and token_end_index inwards to precisely match the answer boundaries.
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_positions.append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_positions.append(token_end_index + 1)

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        return tokenized

    def prepare_validation_features(examples):
        """Tokenizes and formats validation examples into features with offset mappings for evaluation."""
        tokenized = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        tokenized["example_id"] = []

        for i in range(len(tokenized["input_ids"])):
            sequence_ids = tokenized.sequence_ids(i)
            sample_idx = sample_mapping[i]
            tokenized["example_id"].append(examples["id"][sample_idx])
            # Set the offset_mapping to None for tokens not part of the context (e.g. question tokens) 
            # to prevent the model from predicting answers outside the context.
            tokenized["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(tokenized["offset_mapping"][i])
            ]
        return tokenized

    train_features = train_examples.map(
        prepare_train_features,
        batched=True,
        remove_columns=train_examples.column_names,
    )
    eval_features = eval_examples.map(
        prepare_validation_features,
        batched=True,
        remove_columns=eval_examples.column_names,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=100,
        evaluation_strategy="no",
        save_strategy="epoch",
        save_total_limit=2,
        dataloader_num_workers=args.num_workers,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_features,
        eval_dataset=eval_features.remove_columns(["example_id", "offset_mapping"]),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    eval_features_for_model = eval_features.remove_columns(["example_id", "offset_mapping"])
    predictions, _, _ = trainer.predict(eval_features_for_model)
    start_logits, end_logits = predictions

    example_id_to_index = {k: i for i, k in enumerate(eval_examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, f in enumerate(eval_features):
        features_per_example[example_id_to_index[f["example_id"]]].append(i)

    n_best = 20
    max_answer_length = 30
    final_predictions = {}

    for example_index, example in enumerate(eval_examples):
        feature_indices = features_per_example[example_index]
        min_null_score = None
        valid_answers = []

        context = example["context"]
        for feature_index in feature_indices:
            s_logit = start_logits[feature_index]
            e_logit = end_logits[feature_index]
            offsets = eval_features[feature_index]["offset_mapping"]
            cls_index = eval_features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = s_logit[cls_index] + e_logit[cls_index]
            if min_null_score is None or min_null_score > feature_null_score:
                min_null_score = feature_null_score

            # Consider the top `n_best` start and end logits.
            start_indexes = np.argsort(s_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(e_logit)[-1 : -n_best - 1 : -1].tolist()
            for s in start_indexes:
                for e in end_indexes:
                    # Ignore out-of-bounds indices and non-context tokens.
                    if s >= len(offsets) or e >= len(offsets):
                        continue
                    if offsets[s] is None or offsets[e] is None:
                        continue
                    # Ignore impossible answers where the end is before the start or the length exceeds the limit.
                    if e < s or (e - s + 1) > max_answer_length:
                        continue
                    
                    # Map the valid token spans back to the original text via character offsets.
                    start_char = offsets[s][0]
                    end_char = offsets[e][1]
                    valid_answers.append(
                        {"score": s_logit[s] + e_logit[e], "text": context[start_char:end_char]}
                    )

        if valid_answers:
            best = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best = {"text": ""}
        final_predictions[example["id"]] = best["text"]

    metric = evaluate.load("squad_v2" if args.dataset == "squad_v2" else "squad")
    formatted_predictions = []
    for k, v in final_predictions.items():
        pred = {"id": k, "prediction_text": v}
        if args.dataset == "squad_v2":
            pred["no_answer_probability"] = 0.0
        formatted_predictions.append(pred)
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]
    results = metric.compute(predictions=formatted_predictions, references=references)

    em_key = "exact_match" if "exact_match" in results else "exact"
    f1_key = "f1"
    if em_key in results and f1_key in results:
        print(f"Validation EM: {results[em_key]:.2f}")
        print(f"Validation F1: {results[f1_key]:.2f}")
    else:
        print(f"Validation metrics: {results}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(Path(args.output_dir) / "squad_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved model + metrics to: {args.output_dir}")


if __name__ == "__main__":
    main()
