#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ../env

# Preprocess document
echo "Preprocessing documents"
python3 proc_docset.py