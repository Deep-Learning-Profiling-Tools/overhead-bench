#!/bin/bash

# Runs each profiler N times, times each, and prints the results

# Number of trials
NUM_TRIALS=5

# Create a timestamped results file
TIMESTAMP=$(date +%s)
RESULTS_FILE="results_${TIMESTAMP}.csv"
echo "run,script,baseline,proton,torch,nsys" > "$RESULTS_FILE"
# also include nn-*.py scripts in tk directory
for SCRIPT in [0-9]*-*.py tk/[0-9]*-*.py; do

    BASELINE_TIMES=()
    PROTON_TIMES=()
    TORCH_TIMES=()
    NSYS_TIMES=()

    echo -e "\n===== Benchmarking $SCRIPT ====="

    cd "$(dirname "$0")"
    # run once to warm up
    python "$SCRIPT" 2>&1 >/dev/null
    
    echo "No Profiling for baseline"
    for ((i=1; i<=NUM_TRIALS; i++)); do
        t=$(/usr/bin/time -f "%e" python "$SCRIPT" 2>&1 >/dev/null)
        sleep 3
        BASELINE_TIMES+=("$t")
        echo "Run $i: $t s"
    done

    echo -e "\nProfiling with --profiler proton..."
    for ((i=1; i<=NUM_TRIALS; i++)); do
        t=$(/usr/bin/time -f "%e" python "$SCRIPT" --profiler proton 2>&1 >/dev/null)
        sleep 3
        PROTON_TIMES+=("$t")
        echo "Run $i: $t s"
    done

    echo -e "\nProfiling with --profiler torch..."
    for ((i=1; i<=NUM_TRIALS; i++)); do
        t=$(/usr/bin/time -f "%e" python "$SCRIPT" --profiler torch 2>&1 >/dev/null)
        sleep 3
        TORCH_TIMES+=("$t")
        echo "Run $i: $t s"
    done

    echo -e "\nProfiling with nsys..."
    for ((i=1; i<=NUM_TRIALS; i++)); do
        t=$(/usr/bin/time -f "%e" bash -c "nsys profile --trace=cuda --sample=none --cpuctxsw=none python \"$SCRIPT\" > /dev/null" 2>&1)
        sleep 3
        NSYS_TIMES+=("$t")
        echo "Run $i: $t s"
    done
    # delete nsys-rep files
    rm -f *.nsys-rep

    # Print results

    echo -e "\n==== Results (seconds) ===="
    echo "baseline: [${BASELINE_TIMES[*]}]"
    echo "proton:  [${PROTON_TIMES[*]}]"
    echo "torch:   [${TORCH_TIMES[*]}]"
    echo "nsys:    [${NSYS_TIMES[*]}]"
    Append results to the global CSV file
    for ((i=0; i<NUM_TRIALS; i++)); do
        echo "$((i+1)),$(basename "$SCRIPT"),${BASELINE_TIMES[$i]},${PROTON_TIMES[$i]},${TORCH_TIMES[$i]},${NSYS_TIMES[$i]}" >> "$RESULTS_FILE"
    done
    echo -e "\n==== Results in CSV format ===="
    echo "run,script, baseline,proton,torch,nsys"
    for ((i=0; i<NUM_TRIALS; i++)); do
        echo "$((i+1)),$(basename "$SCRIPT"),${BASELINE_TIMES[$i]},${PROTON_TIMES[$i]},${TORCH_TIMES[$i]},${NSYS_TIMES[$i]}"
    done

done