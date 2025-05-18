#!/usr/bin/env bash
model_map=(
    "phi-4"
    "gemma-3-4b"
    "llama-3.1-8b"
    "llama-3.2-3b"
    "qwen3-14b"
)


# loop over all models in model_map
for model in "${model_map[@]}"; do
    echo "--------------------------------------------"
    echo "Timing $model with NONE"
    ./profile-wrapper.sh 2 python training.py --model $model --profiling none
    echo "--------------------------------------------"

    echo "NSYS"
    ./profile-wrapper.sh 2 nsys profile --trace=cuda --sample=none --cpuctxsw=none python training.py --model $model --profiling none

    echo "--------------------------------------------"

    echo "PROTON"
    ./profile-wrapper.sh 2 proton training.py --model $model --profiling proton

    echo "--------------------------------------------"

    echo "TORCH"
    ./profile-wrapper.sh 2 python training.py --profile_torch --model $model --profiling torch
    echo "--------------------------------------------"
done
