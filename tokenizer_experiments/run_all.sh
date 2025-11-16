#!/bin/bash
# Copyright 2025 Pathway Technology, Inc.
# 
# Master script to run complete tokenizer comparison experiment

set -e  # Exit on error

echo "======================================================================"
echo "Tokenizer Comparison Experiment - Complete Pipeline"
echo "======================================================================"
echo ""

# Step 1: Data Preparation
echo "STEP 1: Preparing tokenized datasets"
echo "----------------------------------------------------------------------"
for tokenizer in bielik whitespace sentencepiece; do
    echo ""
    echo "Preparing data for: $tokenizer"
    python prepare_data.py --tokenizer_type $tokenizer
done

# Step 2: Training
echo ""
echo "======================================================================"
echo "STEP 2: Training models"
echo "----------------------------------------------------------------------"
for tokenizer in bielik whitespace sentencepiece; do
    echo ""
    echo "Training model with: $tokenizer"
    python train.py --tokenizer_type $tokenizer
done

# Step 3: Evaluation
echo ""
echo "======================================================================"
echo "STEP 3: Evaluating models"
echo "----------------------------------------------------------------------"
for tokenizer in bielik whitespace sentencepiece; do
    echo ""
    echo "Evaluating: $tokenizer"
    python evaluate.py --tokenizer_type $tokenizer
done

# Step 4: Qualitative Analysis
echo ""
echo "======================================================================"
echo "STEP 4: Qualitative analysis"
echo "----------------------------------------------------------------------"
python qualitative_analysis.py

# Done
echo ""
echo "======================================================================"
echo "EXPERIMENT COMPLETE!"
echo "======================================================================"
echo ""
echo "Results available in:"
echo "  - results/training_metrics_*.json"
echo "  - results/evaluation_*.json"
echo "  - results/qualitative_analysis.json"
echo "  - results/qualitative_report.txt"
echo ""
