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
    --output_dir ../outputs/D5/before_CR/devtest

# Summarizarion - ILP
echo "Generating summaries with [ILP]: without info ordering"
python3 ilp.py \
    --input_dir devtest \
    --concept_type skipgrams \
    --skipgram_degree 1 \
    --remove_punctuation \
    --min_sent_length 5  \
    --output_dir ../outputs/D5/ilp_cos/devtest

# Information ordering for [ILP] summaries
echo "Reordering summaries generated with [ILP]: cosine similarity ordering"
python3 io_cos_jac.py c ../outputs/D5/ilp_cos/devtest ../outputs/D5/before_CR/devtest

# Summarization - baseline
# echo "Generating summaries with [baseline]"
# python3 baseline.py ../outputs/devtest ../outputs/D5/before_CR/devtest

# Content Realization
echo "Performing content realization"
python3 name_resolution.py \
    --preCR_dir ../outputs/D5/before_CR/devtest \
    --source_dir ../outputs/devtest \
    --postCR_dir ../outputs/D5_devtest \

# Evaluation
echo "Outputting evaluation results"
python3 run_rouge_eval.py \
    --summary_path ../outputs/D5_devtest/ \
    --model_path ~/dropbox/22-23/575x/Data/models/devtest/ \
    --rouge_methods rouge1,rouge2 \
    --eval_output_path ../results/ \
    --eval_output_filename D5_devtest_rouge_scores.out

