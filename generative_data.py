import re
import string
from dataclasses import dataclass

import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


NO_ANSWER_TEXT = "No answer in context."
NO_ANSWER_SENTENCE = "The context does not contain the answer."


def normalize_text(s: str) -> str:
    """Normalizes text by removing articles, punctuation, and extra whitespace, and lowercasing."""
    def remove_articles(text):
        return " ".join([w for w in text.split() if w not in {"a", "an", "the"}])

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


@dataclass
class GenQADataConfig:
    tokenizer_path: str
    max_input_len: int = 256
    max_target_len: int = 48
    include_squad_v2: bool = True
    answerable_repeat: int = 1
    no_answer_repeat: int = 1
    target_style: str = "span"  # "span" or "sentence"
    no_answer_target_text: str = NO_ANSWER_TEXT
    instruction_prefix: str = ""
    seed: int = 42


def build_tokenizer(tokenizer_path: str):
    """Loads a fast tokenizer and ensures the pad token is set."""
    tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.sep_token
    return tok


def _has_answer(example) -> bool:
    """Helper to check if a SQuAD example contains an answer."""
    return len(example["answers"]["text"]) > 0


def _is_no_answer(example) -> bool:
    """Helper to check if a SQuAD example is unanswerable."""
    return len(example["answers"]["text"]) == 0


def load_train_val(
    include_squad_v2: bool = True,
    answerable_repeat: int = 1,
    no_answer_repeat: int = 1,
):
    """Loads SQuAD datasets, optionally including v2, and allows upsampling specific example types."""
    ds1 = load_dataset("squad")
    train = [ds1["train"]]
    val = [ds1["validation"]]
    if include_squad_v2:
        ds2 = load_dataset("squad_v2")
        train.append(ds2["train"])
        val.append(ds2["validation"])
    train_ds = concatenate_datasets(train)
    val_ds = concatenate_datasets(val)
    if include_squad_v2 and answerable_repeat > 1:
        # Upsample answerable examples to balance the dataset.
        answerable_ds = train_ds.filter(_has_answer)
        train_ds = concatenate_datasets([train_ds] + [answerable_ds] * (answerable_repeat - 1))
    if include_squad_v2 and no_answer_repeat > 1:
        # Upsample unanswerable examples to balance the dataset.
        no_answer_ds = train_ds.filter(_is_no_answer)
        train_ds = concatenate_datasets([train_ds] + [no_answer_ds] * (no_answer_repeat - 1))
    return train_ds, val_ds


def _slice_sentence_around_index(context: str, char_index: int) -> str:
    """Extracts the full sentence containing the character at the specified index."""
    if not context:
        return ""
    if char_index < 0:
        char_index = 0
    if char_index >= len(context):
        char_index = len(context) - 1

    # Find the nearest sentence-ending punctuation to the left of the char_index.
    left_candidates = [
        context.rfind(".", 0, char_index),
        context.rfind("?", 0, char_index),
        context.rfind("!", 0, char_index),
    ]
    left = max(left_candidates)
    start = 0 if left == -1 else left + 1

    # Find the nearest sentence-ending punctuation to the right of the char_index.
    right_positions = []
    for ch in (".", "?", "!"):
        pos = context.find(ch, char_index)
        if pos != -1:
            right_positions.append(pos + 1)
    end = min(right_positions) if right_positions else len(context)
    return context[start:end].strip()


def _find_answer_sentence(context: str, answers: dict) -> str:
    """Finds the complete sentence in the context that contains the answer."""
    if not context:
        return ""
    answer_texts = answers.get("text", [])
    if not answer_texts:
        return ""

    answer_text = answer_texts[0].strip()
    answer_starts = answers.get("answer_start", [])
    answer_start = answer_starts[0] if answer_starts else -1
    
    # If the dataset didn't provide a valid answer_start, try to find it via substring search.
    if answer_start is None or answer_start < 0:
        answer_start = context.lower().find(answer_text.lower())

    if answer_start is not None and answer_start >= 0:
        # Extract the sentence around the found start index.
        sent = _slice_sentence_around_index(context, int(answer_start))
        if sent:
            return sent

    # Fallback logic: split the context into sentences via regex if the index search failed.
    for sent in re.split(r"(?<=[.!?])\s+", context):
        s = sent.strip()
        if s and answer_text.lower() in s.lower():
            return s

    return answer_text


def add_targets(example, target_style: str = "span", no_answer_target_text: str = NO_ANSWER_TEXT):
    """Generates the target text for generation based on whether we want exact spans or full sentences."""
    answers = example["answers"]
    if len(answers["text"]) == 0:
        target = no_answer_target_text.strip()
    else:
        if target_style == "sentence":
            target = _find_answer_sentence(example.get("context", ""), answers)
        else:
            target = answers["text"][0].strip()
    return {"target_text": target}


def preprocess_dataset(
    dataset,
    tokenizer,
    max_input_len: int,
    max_target_len: int,
    target_style: str = "span",
    no_answer_target_text: str = NO_ANSWER_TEXT,
    instruction_prefix: str = "",
):
    """Tokenizes and formats the inputs and targets for generative QA."""
    if target_style not in {"span", "sentence"}:
        raise ValueError("target_style must be one of: span, sentence")

    dataset = dataset.map(
        add_targets,
        fn_kwargs={
            "target_style": target_style,
            "no_answer_target_text": no_answer_target_text,
        },
    )

    def _tok(examples):
        if instruction_prefix:
            inputs = [
                f"{instruction_prefix.strip()} question: {q} context: {c}"
                for q, c in zip(examples["question"], examples["context"])
            ]
        else:
            inputs = [f"question: {q} context: {c}" for q, c in zip(examples["question"], examples["context"])]
        model_inputs = tokenizer(
            inputs,
            truncation=True,
            max_length=max_input_len,
            padding=False,
        )
        targets = tokenizer(
            examples["target_text"],
            truncation=True,
            max_length=max_target_len,
            padding=False,
        )
        model_inputs["labels_ids"] = targets["input_ids"]
        model_inputs["target_text"] = examples["target_text"]
        return model_inputs

    keep_cols = ["id", "question", "context", "answers"]
    tokenized = dataset.map(_tok, batched=True, remove_columns=[c for c in dataset.column_names if c not in keep_cols])
    return tokenized


def _pad_2d(seqs, pad_val: int):
    """Pads a list of sequences to a 2D tensor of uniform shape."""
    max_len = max(len(x) for x in seqs)
    out = torch.full((len(seqs), max_len), pad_val, dtype=torch.long)
    for i, x in enumerate(seqs):
        out[i, : len(x)] = torch.tensor(x, dtype=torch.long)
    return out


def collate_generative(batch, tokenizer):
    """Collates a batch of examples into padded tensors suitable for a sequence-to-sequence model."""
    pad = tokenizer.pad_token_id
    bos = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    eos = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    if bos is None:
        bos = pad
    if eos is None:
        eos = pad

    enc_ids = _pad_2d([x["input_ids"] for x in batch], pad)
    enc_attn = _pad_2d([x["attention_mask"] for x in batch], 0)
    enc_ttype = torch.zeros_like(enc_ids)

    tgt = []
    for x in batch:
        ids = x["labels_ids"]
        seq = [bos] + ids + [eos]
        tgt.append(seq)
    tgt_full = _pad_2d(tgt, pad)
    
    # For teacher forcing: decoder input is target shifted right, labels are the unshifted target.
    dec_in = tgt_full[:, :-1]
    labels = tgt_full[:, 1:].clone()
    dec_attn = (dec_in != pad).long()
    
    # Mask out padding tokens in the labels with -100 so they are ignored in cross-entropy loss.
    labels[labels == pad] = -100

    return {
        "encoder_input_ids": enc_ids,
        "encoder_attention_mask": enc_attn,
        "encoder_token_type_ids": enc_ttype,
        "decoder_input_ids": dec_in,
        "decoder_attention_mask": dec_attn,
        "labels": labels,
        "target_text": [x["target_text"] for x in batch],
        "id": [x["id"] for x in batch],
        "question": [x["question"] for x in batch],
        "context": [x["context"] for x in batch],
        "answers": [x["answers"] for x in batch],
    }


def build_dataloaders(
    cfg: GenQADataConfig,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int = 2,
):
    """Creates the training and validation dataloaders for the generative QA task."""
    tokenizer = build_tokenizer(cfg.tokenizer_path)
    train_ds, val_ds = load_train_val(
        include_squad_v2=cfg.include_squad_v2,
        answerable_repeat=cfg.answerable_repeat,
        no_answer_repeat=cfg.no_answer_repeat,
    )
    train_tok = preprocess_dataset(
        train_ds,
        tokenizer,
        cfg.max_input_len,
        cfg.max_target_len,
        target_style=cfg.target_style,
        no_answer_target_text=cfg.no_answer_target_text,
        instruction_prefix=cfg.instruction_prefix,
    )
    val_tok = preprocess_dataset(
        val_ds,
        tokenizer,
        cfg.max_input_len,
        cfg.max_target_len,
        target_style=cfg.target_style,
        no_answer_target_text=cfg.no_answer_target_text,
        instruction_prefix=cfg.instruction_prefix,
    )

    train_loader = DataLoader(
        train_tok,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate_generative(b, tokenizer),
    )
    val_loader = DataLoader(
        val_tok,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate_generative(b, tokenizer),
    )
    return tokenizer, train_loader, val_loader
