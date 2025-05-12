#!/bin/bash

# Script to benchmark profiler overheads for 01-vector-add.py
# Runs each profiler 5 times, times each, and prints the results


# Find all Python scripts in current directory, run if the first two letters are not "01", "02", "03", "04", "05", "06"
for SCRIPT in *.py; do
    BASELINE_TIMES=()
    PROTON_TIMES=()
    TORCH_TIMES=()
    NSYS_TIMES=()


    echo -e "\n===== Benchmarking $SCRIPT ====="

    cd "$(dirname "$0")"
    # run once to warm up
    python "$SCRIPT" 2>&1 >/dev/null
    
    echo "No Profiling for baseline"
    for i in {1..5}; do
        t=$(/usr/bin/time -f "%e" python "$SCRIPT" 2>&1 >/dev/null)
        sleep 3
        BASELINE_TIMES+=("$t")
        echo "Run $i: $t s"
    done

    echo -e "\nProfiling with --profiler proton..."
    for i in {1..5}; do
        t=$(/usr/bin/time -f "%e" python "$SCRIPT" --profiler proton 2>&1 >/dev/null)
        sleep 3
        PROTON_TIMES+=("$t")
        echo "Run $i: $t s"
    done

    echo -e "\nProfiling with --profiler torch..."
    for i in {1..5}; do
        t=$(/usr/bin/time -f "%e" python "$SCRIPT" --profiler torch 2>&1 >/dev/null)
        sleep 3
        TORCH_TIMES+=("$t")
        echo "Run $i: $t s"
    done

    echo -e "\nProfiling with nsys..."
    for i in {1..5}; do
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
    # Print CSV format and save to file
    # Save results to CSV file
    echo "run,script,baseline,proton,torch,nsys" > "${SCRIPT%.*}_results.csv"
    for i in {0..4}; do
        echo "$((i+1)),${SCRIPT},${BASELINE_TIMES[$i]},${PROTON_TIMES[$i]},${TORCH_TIMES[$i]},${NSYS_TIMES[$i]}" >> "${SCRIPT%.*}_results.csv"
    done
    echo -e "\n==== Results in CSV format ===="
    echo "run,script, baseline,proton,torch,nsys"
    for i in {0..4}; do
        echo "$((i+1)),${SCRIPT},${BASELINE_TIMES[$i]},${PROTON_TIMES[$i]},${TORCH_TIMES[$i]},${NSYS_TIMES[$i]}"
    done
done