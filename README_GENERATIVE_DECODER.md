# Generative QA (Phase 3) - From Scratch Decoder + Reused Encoder

This phase adds a trainable Transformer decoder on top of your pretrained encoder checkpoint.

## Files
- `main_hybrid_decoder.py` - main custom hybrid decoder
- `standard_generative_decoder.py` - earlier Transformer decoder baseline
- `generative_data.py` - SQuAD/SQuAD-v2 data pipeline
- `generative_finetuning.py` - staged training (freeze -> partial unfreeze -> full)
- `generative_evaluation.py` - EM/F1/ROUGE-L/BLEU/no-answer accuracy
- `generative_inference.py` - inference CLI
- `run_generative_finetuning_nohup.sh` - nohup launcher

## Install
```bash
pip install -r requirements.txt
```

## Train
```bash
chmod +x run_generative_finetuning_nohup.sh
./run_generative_finetuning_nohup.sh
```

## High-impact training sequence (recommended)

### Stage A: Curriculum on SQuAD v1 only (answerable behavior first)
```bash
python generative_finetuning.py \
  --tokenizer_path checkpoints_pretrain_base_seq256/step_20000 \
  --pretrain_ckpt checkpoints_pretrain_base_seq256/step_20000/checkpoint.pt \
  --output_dir checkpoints_generative_qa_stageA_v1 \
  --decoder_variant hybrid \
  --no_squad_v2 \
  --epochs 3 \
  --train_batch_size 16 \
  --eval_batch_size 8 \
  --grad_accum 1 \
  --lr 3e-4 \
  --encoder_lr 8e-5 \
  --freeze_warmup_epochs 1 \
  --unfreeze_top_layers 4 \
  --fp16
```

### Stage B: Continue on SQuAD v1+v2 with answerable rebalancing
```bash
python generative_finetuning.py \
  --tokenizer_path checkpoints_pretrain_base_seq256/step_20000 \
  --pretrain_ckpt checkpoints_pretrain_base_seq256/step_20000/checkpoint.pt \
  --init_from_checkpoint checkpoints_generative_qa_stageA_v1/best.pt \
  --output_dir checkpoints_generative_qa_stageB_v1v2 \
  --decoder_variant hybrid \
  --answerable_repeat 2 \
  --epochs 4 \
  --train_batch_size 16 \
  --eval_batch_size 8 \
  --grad_accum 1 \
  --lr 2.5e-4 \
  --encoder_lr 5e-5 \
  --freeze_warmup_epochs 0 \
  --unfreeze_top_layers 4 \
  --fp16
```

Monitor:
```bash
tail -f logs/generative_train_*.log
nvidia-smi
```

## Evaluate
```bash
python generative_evaluation.py \
  --checkpoint_path checkpoints_generative_qa/best.pt \
  --tokenizer_path checkpoints_generative_qa \
  --decoder_variant hybrid \
  --max_input_len 256 \
  --max_target_len 48 \
  --eval_batch_size 8 \
  --num_workers 4 \
  --beam_size 4 \
  --max_new_tokens 32 \
  --length_penalty 1.0 \
  --out_json checkpoints_generative_qa/generative_eval_metrics.json
```

## Gated no-answer calibration (recommended for SQuAD v2 style behavior)
Tune a no-answer threshold on validation and save metrics:
```bash
python generative_evaluation.py \
  --checkpoint_path checkpoints_generative_qa_stageE_tradeoff/best.pt \
  --tokenizer_path checkpoints_generative_qa_stageE_tradeoff \
  --decoder_variant hybrid \
  --target_style sentence \
  --no_answer_text "The context does not contain the answer." \
  --instruction_prefix "Answer in one concise sentence based only on the context." \
  --beam_size 5 \
  --max_new_tokens 48 \
  --length_penalty 1.05 \
  --tune_no_answer_threshold \
  --threshold_points 101 \
  --out_json checkpoints_generative_qa_stageE_tradeoff/generative_eval_metrics_gated.json
```

Use the tuned threshold at inference time:
```bash
python generative_inference.py \
  --checkpoint_path checkpoints_generative_qa_stageE_tradeoff/best.pt \
  --tokenizer_path checkpoints_generative_qa_stageE_tradeoff \
  --decoder_variant hybrid \
  --question "Who won the Nobel Peace Prize in 2023?" \
  --context "Mars is the fourth planet from the Sun and is often called the Red Planet." \
  --instruction_prefix "Answer in one concise sentence based only on the context." \
  --enable_no_answer_gate \
  --no_answer_text "The context does not contain the answer." \
  --no_answer_threshold 0.0
```

## Inference
```bash
python generative_inference.py \
  --checkpoint_path checkpoints_generative_qa/best.pt \
  --tokenizer_path checkpoints_generative_qa \
  --decoder_variant hybrid \
  --question "When was Hyderabad founded?" \
  --context "Hyderabad was founded in 1591 by Muhammad Quli Qutb Shah. It is the capital of Telangana." \
  --max_input_len 256 \
  --max_new_tokens 32 \
  --beam_size 4 \
  --length_penalty 1.0
```

```bash
python generative_inference.py \
  --checkpoint_path checkpoints_generative_qa/best.pt \
  --tokenizer_path checkpoints_generative_qa \
  --decoder_variant hybrid \
  --question "What is the population of Hyderabad?" \
  --context "Hyderabad was founded in 1591 by Muhammad Quli Qutb Shah. It is the capital of Telangana." \
  --max_input_len 256 \
  --max_new_tokens 32 \
  --beam_size 4 \
  --length_penalty 1.0
```

## Sentence-target continuation training (decoder learns full sentence answers)
```bash
python generative_finetuning.py \
  --tokenizer_path checkpoints_pretrain_base_seq256/step_20000 \
  --pretrain_ckpt checkpoints_pretrain_base_seq256/step_20000/checkpoint.pt \
  --init_from_checkpoint checkpoints_generative_qa_stageB_v1v2_run1/best.pt \
  --output_dir checkpoints_generative_qa_stageC_sentence \
  --decoder_variant hybrid \
  --target_style sentence \
  --no_answer_target_text "The context does not contain the answer." \
  --instruction_prefix "Answer in one concise sentence based only on the context." \
  --answerable_repeat 2 \
  --no_answer_repeat 1 \
  --max_target_len 96 \
  --epochs 2 \
  --train_batch_size 16 \
  --eval_batch_size 8 \
  --grad_accum 1 \
  --lr 1.5e-4 \
  --encoder_lr 3e-5 \
  --freeze_warmup_epochs 0 \
  --unfreeze_top_layers 4 \
  --fp16
```

## Stage D robustness tuning (recommended after Stage C)
```bash
python generative_finetuning.py \
  --tokenizer_path checkpoints_pretrain_base_seq256/step_20000 \
  --pretrain_ckpt checkpoints_pretrain_base_seq256/step_20000/checkpoint.pt \
  --init_from_checkpoint checkpoints_generative_qa_stageC_sentence_run2/best.pt \
  --output_dir checkpoints_generative_qa_stageD_balanced \
  --decoder_variant hybrid \
  --target_style sentence \
  --no_answer_target_text "The context does not contain the answer." \
  --instruction_prefix "Answer in one concise sentence based only on the context." \
  --answerable_repeat 1 \
  --no_answer_repeat 3 \
  --max_target_len 96 \
  --epochs 1 \
  --train_batch_size 24 \
  --eval_batch_size 12 \
  --grad_accum 1 \
  --lr 6e-5 \
  --encoder_lr 1e-5 \
  --freeze_warmup_epochs 0 \
  --unfreeze_top_layers 6 \
  --fp16
```

## Runtime expectation (RTX 4060 Ti)
- 6 epochs with defaults: typically ~6 to 14 hours (depends on dataloader/network/cache)
