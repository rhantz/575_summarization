#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ../env

# Preprocess document
echo "Preprocessing documents"
python3 proc_docset.py

# Summarization - tf-idf
echo "Generating summaries with [tf-idf]: position ordering"
python3 tf_idf.py \
    --input_dir devtest \
    --cosine_sim 0.25 \
    --word_co 8 \
    --gram_type unigram \
    --info_order yes \
    --output_dir ../outputs/D4

# Summarizarion - ILP
echo "Generating summaries with [ILP]: without info ordering"
python3 ilp.py \
    --input_dir devtest \
    --concept_type skipgrams \
    --skipgram_degree 1 \
    --remove_punctuation \
    --min_sent_length 5  \
    --output_dir ../outputs/D4/ilp_cos

# Information ordering for [ILP] summaries
echo "Reordering summaries generated with [ILP]: cosine similarity ordering"
python3 io_cos_jac.py c ../outputs/D4/ilp_cos ../outputs/D4

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

