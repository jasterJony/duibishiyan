#!/bin/bash
# Batch training script for 6 models on coco8 dataset with 2 epochs
# Models: Faster R-CNN, Cascade R-CNN, Mask R-CNN, HTC, SCNet, Grid R-CNN

set -e  # Exit on error

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Create work_dirs if not exists
mkdir -p work_dirs

# Training function
train_model() {
    local model_name=$1
    local config_path=$2
    local work_dir="work_dirs/${model_name}_coco8_2e"
    
    echo "=============================================="
    echo "Training: $model_name"
    echo "Config: $config_path"
    echo "Work Dir: $work_dir"
    echo "=============================================="
    
    python tools/train.py "$config_path" --work-dir "$work_dir"
    
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] $model_name training completed!"
        return 0
    else
        echo "[FAILED] $model_name training failed!"
        return 1
    fi
}

# Main execution
echo "=========================================="
echo "Batch Training Script for MMDetection"
echo "Dataset: coco8"
echo "Epochs: 2"
echo "Models: Faster R-CNN, Cascade R-CNN, Mask R-CNN, HTC, SCNet, Grid R-CNN"
echo "=========================================="
echo ""

# Check if coco8 dataset exists
if [ ! -d "data/coco8" ]; then
    echo "[WARNING] data/coco8 directory not found!"
    echo "Please ensure your coco8 dataset is placed at: $PROJECT_ROOT/data/coco8/"
    echo "Expected structure:"
    echo "  data/coco8/"
    echo "  ├── annotations/"
    echo "  │   ├── instances_train2017.json"
    echo "  │   └── instances_val2017.json"
    echo "  ├── train2017/"
    echo "  └── val2017/"
    echo ""
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Record start time
START_TIME=$(date +%s)

# Define models and configs as simple arrays
MODEL_NAMES="faster_rcnn cascade_rcnn mask_rcnn htc scnet grid_rcnn"
CONFIG_faster_rcnn="configs/faster_rcnn/faster-rcnn_r50_fpn_2e_coco8.py"
CONFIG_cascade_rcnn="configs/cascade_rcnn/cascade-rcnn_r50_fpn_2e_coco8.py"
CONFIG_mask_rcnn="configs/mask_rcnn/mask-rcnn_r50_fpn_2e_coco8.py"
CONFIG_htc="configs/htc/htc_r50_fpn_2e_coco8.py"
CONFIG_scnet="configs/scnet/scnet_r50_fpn_2e_coco8.py"
CONFIG_grid_rcnn="configs/grid_rcnn/grid-rcnn_r50_fpn_gn-head_2e_coco8.py"

# Train all models
FAILED_MODELS=""
SUCCESS_COUNT=0
FAIL_COUNT=0

for model_name in $MODEL_NAMES; do
    # Get config path using indirect variable reference
    config_var="CONFIG_${model_name}"
    config_path=$(eval echo \$$config_var)
    
    if train_model "$model_name" "$config_path"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_MODELS="$FAILED_MODELS $model_name"
    fi
    echo ""
done

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Summary
echo "=========================================="
echo "Batch Training Summary"
echo "=========================================="
echo "Total time: $((DURATION / 60)) minutes $((DURATION % 60)) seconds"
echo "Successful: $SUCCESS_COUNT / 6"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo "All 6 models trained successfully!"
else
    echo "Failed models:$FAILED_MODELS"
fi

echo ""
echo "Trained models saved in work_dirs/"
ls -la work_dirs/ 2>/dev/null | grep coco8 || echo "No models found yet"
