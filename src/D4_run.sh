#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ../env

# Summarization - baseline
echo "Generating summaries with [baseline]"
python3 baseline.py

# Evaluation
echo "Outputting evaluation results"
python3 run_rouge_eval.py \
    --summary_path ../outputs/D4/ \
    --model_path ~/dropbox/22-23/575x/Data/models/devtest/ \
    --rouge_methods rouge1,rouge2 \
    --eval_output_path ../results/D4/ \
    --eval_output_filename rouge_scores.out

