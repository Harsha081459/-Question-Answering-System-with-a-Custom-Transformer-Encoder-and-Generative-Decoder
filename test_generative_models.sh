#!/bin/bash
cd "/home/sem6/main file"
Q1="Who created Python?"
C1="Python was created by Guido van Rossum and first released in 1991. It is a dynamically typed programming language."

Q2="What is the capital of Mars?"
C2="Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System."

models=(
  "checkpoints_generative_qa_stageA_v1"
  "checkpoints_generative_qa_stageB_v1v2_run1"
  "checkpoints_generative_qa_stageC_sentence_run2"
  "checkpoints_generative_qa_stageD_balanced_run2"
)

for m in "${models[@]}"; do
  echo "======================================"
  echo "MODEL: $m"
  
  if [[ "$m" == *"stageC"* ]] || [[ "$m" == *"stageD"* ]]; then
    prefix="--instruction_prefix \"Answer in one concise sentence based only on the context.\""
  else
    prefix=""
  fi
  
  echo "Question 1 (Answerable): $Q1"
  eval "python generative_inference.py --checkpoint_path \"$m/best.pt\" --tokenizer_path \"$m\" --decoder_variant hybrid --question \"$Q1\" --context \"$C1\" $prefix"
  
  echo ""
  echo "Question 2 (Unanswerable): $Q2"
  eval "python generative_inference.py --checkpoint_path \"$m/best.pt\" --tokenizer_path \"$m\" --decoder_variant hybrid --question \"$Q2\" --context \"$C2\" $prefix"
  echo ""
done
