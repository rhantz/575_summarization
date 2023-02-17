#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ../env

# Preprocess document
echo "Preprocessing documents"
python3 proc_docset.py

# Summarization - tf-idf
echo "Generating summaries with [tf-idf]"
python3 tf_idf.py 0.3 10

# Summarizarion - ILP
echo "Generating summaries with [ILP]"
python3 ilp.py devtest

# Summarization - baseline
echo "Generating summaries with [baseline]"
python3 baseline.py

# Evaluation
echo "Outputting evaluation results"
python3 run_rouge_eval.py \
    --summary_path ../outputs/D3/ \
    --model_path ~/dropbox/22-23/575x/Data/models/devtest/ \
    --rouge_methods rouge1,rouge2 \
    --eval_output_path ../results/D3/ \
    --eval_output_filename rouge_scores.out

