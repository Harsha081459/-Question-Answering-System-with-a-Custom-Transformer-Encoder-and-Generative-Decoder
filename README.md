# 🧠 Generative Question Answering — From Scratch

A custom **Transformer-based Question Answering system** built entirely from scratch in PyTorch. The project spans the full pipeline: from **pretraining a BERT encoder** on raw text, to building a **custom Transformer decoder**, to **fine-tuning an end-to-end encoder–decoder model** on SQuAD and SQuAD v2.

> **No pretrained model weights from HuggingFace are used.**  
> The encoder, decoder, data pipeline, beam search, and evaluation — everything is written from the ground up.

---

## ✨ Highlights

- **Custom BERT Encoder** — 12-layer, 768-dim Transformer pretrained with Masked Language Modeling (MLM).
- **Custom Hybrid Decoder** — 4-layer cross-attention Transformer decoder with pre-norm and tied embeddings.
- **3-Stage Training Pipeline** — MLM pretraining → Decoder training with gradual encoder unfreezing → End-to-end fine-tuning.
- **Extractive + Generative QA** — Supports both span-extraction and free-form sentence generation.
- **No-Answer Gating** — Logprob-based threshold mechanism to detect unanswerable questions (SQuAD v2).
- **Interactive Demo** — Local web server with a clean HTML/JS frontend for live QA.

---

## 📁 Repository Structure

- `config.yaml` — Full pretraining configuration.
- `mlm_pretraining.py` — Stage 1: BERT encoder pretraining with MLM.
- `main_hybrid_decoder.py` — Custom hybrid decoder architecture.
- `generative_data.py` — SQuAD v1/v2 data pipeline for generative QA.
- `generative_finetuning.py` — Stage 2–3: Decoder + end-to-end fine-tuning.
- `generative_evaluation.py` — Evaluation (EM, F1, ROUGE-L, BLEU).
- `generative_inference.py` — CLI inference for generative model.
- `extractive_finetuning.py` & `extractive_inference.py` — Extractive QA scripts.
- `local_qa_server.py` & `qa_demo_local.html` — Interactive QA web UI.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Interactive Demo
```bash
python local_qa_server.py --port 8001
# Then open qa_demo_local.html in your browser
```

### 3. Run Inference (CLI)
```bash
python generative_inference.py \
  --checkpoint_path checkpoints_generative_qa_stageB/best.pt \
  --tokenizer_path checkpoints_generative_qa_stageB \
  --decoder_variant hybrid \
  --question "When was Hyderabad founded?" \
  --context "Hyderabad was founded in 1591 by Muhammad Quli Qutb Shah."
```

*(See `README_MLM_PRETRAINING.md` and `README_GENERATIVE_DECODER.md` for full training and evaluation commands).*

---

## 🛠️ Tech Stack & Architecture Notes

- **Tech Stack:** PyTorch (Core), HuggingFace Transformers (Tokenizer only), HuggingFace Datasets.
- **Architecture:** 12-layer encoder (768-dim), 4-layer hybrid decoder. Tied embeddings for parameter efficiency.
- **Training:** Gradual unfreezing schedule, differential learning rates, FP16/BF16 mixed precision.
- **Gating Mechanism:** Compares beam search output logprob against a forced "no-answer" sequence logprob to detect unanswerable questions without a separate classification head.
