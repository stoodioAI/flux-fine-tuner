#!/bin/bash
set -euo pipefail

# Configuration
DATASETS_DIR="$HOME/datasets"
OUTDIR_REL="output"

echo "=== Flux Fine Tuner Training Script ==="
echo "Current directory: $(pwd)"

# Check if we're in the right directory
if [[ ! -f "train.py" ]]; then
    echo "Error: train.py not found. Please run this script from the flux-fine-tuner directory."
    exit 1
fi

# Find the first zip file in datasets directory
DATASET_ZIP=""
if [[ -d "$DATASETS_DIR" ]]; then
    for zipfile in "$DATASETS_DIR"/*.zip; do
        if [[ -f "$zipfile" ]]; then
            DATASET_ZIP="$zipfile"
            break
        fi
    done
fi

# Check if we found a dataset
if [[ -z "$DATASET_ZIP" ]]; then
    echo "✗ No dataset found in $DATASETS_DIR"
    echo "Please place a .zip file in the datasets directory"
    exit 1
fi

DATASET_NAME=$(basename "$DATASET_ZIP")
echo "✓ Using dataset: $DATASET_NAME ($(du -h "$DATASET_ZIP" | cut -f1))"

# Create output directory
mkdir -p "$OUTDIR_REL"
echo "✓ Output directory ready: $OUTDIR_REL"

# Check for required environment variables
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "✗ ERROR: HF_TOKEN environment variable is required"
    echo "Please set it with: export HF_TOKEN=your_token_here"
    echo "You can get a token from: https://huggingface.co/settings/tokens"
    exit 1
fi
echo "✓ HF_TOKEN is set"

echo "Starting training with cog..."
echo "Model: black-forest-labs/FLUX.1-Krea-dev"
echo "================================"

# run training using cog train with the train function
cog train -e HF_TOKEN="${HF_TOKEN:-}" \
  -i input_images="@$DATASET_ZIP" \
  -i trigger_word="helena" \
  -i autocaption=true \
  -i steps=1500 \
  -i lora_rank=32 \
  -i optimizer="adamw8bit" \
  -i batch_size=1 \
  -i resolution="512,768,1024,2048" \
  -i learning_rate=4e-4 \
  -i caption_dropout_rate=0.05 \
  -i cache_latents_to_disk=false

echo "================================"
echo "Training completed!"
echo "Check the output directory for your trained model."
