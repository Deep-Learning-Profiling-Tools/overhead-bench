#!/usr/bin/env bash
model_map=(
    "phi-4"
    "gemma-3-4b"
    "llama-3.1-8b"
    "llama-3.2-3b"
    "qwen3-14b"
    "mistral-7b"
    # "llama-4-scout"
)


# loop over all models in model_map
for model in "${model_map[@]}"; do
    echo "--------------------------------------------"
    echo "Timing $model with NONE"
    ./profile-wrapper.sh 5 python training.py --model $model --profiling none
    sleep 5

    echo "NSYS"
    ./profile-wrapper.sh 5 nsys profile --trace=cuda --sample=none --cpuctxsw=none -o $model.nsys python training.py --model $model --profiling none
    sleep 5
    echo "--------------------------------------------"

    echo "PROTON"
    ./profile-wrapper.sh 5 proton training.py --model $model --profiling proton
    sleep 5
    echo "--------------------------------------------"

    echo "TORCH"
    ./profile-wrapper.sh 5 python training.py --profile_torch --model $model --profiling torch
    sleep 5
    echo "--------------------------------------------"
done
