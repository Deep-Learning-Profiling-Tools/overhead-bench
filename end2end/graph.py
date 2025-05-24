import re
import csv
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np

def parse_experiments(log_path):
    # Patterns for experiment headers
    first_hdr = re.compile(r"Timing\s+(?P<model>\S+)\s")
    prof_hdr = re.compile(r"^(?P<profiler>NSYS|PROTON|TORCH)\s*$")
    run_pattern = re.compile(r"Run\s*#?(?P<number>\d+)[: of]*.*?(?P<duration>\d+\.\d+)\s*seconds")

    experiments = []
    current = None
    current_model = None

    with open(log_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()

            # Detect first header: Timing <model> NONE
            m1 = first_hdr.search(line)
            if m1:
                current_model = m1.group('model')
                current = {'model': current_model, 'profiler_type': 'NONE', 'runs': []}
                experiments.append(current)
                continue

            # Detect subsequent headers: NSYS, PROTON, TORCH
            m2 = prof_hdr.search(line)
            if m2 and current_model:
                profiler = m2.group('profiler')
                current = {'model': current_model, 'profiler_type': profiler, 'runs': []}
                experiments.append(current)
                continue

            # Parse run durations
            if current:
                run = run_pattern.search(line)
                if run:
                    current['runs'].append({'run_number': int(run.group('number')), 'duration_s': float(run.group('duration'))})
    return experiments


def write_csv(experiments, out_path):
    with open(out_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['model','profiler_type','run_number','duration_s'])
        writer.writeheader()
        for exp in experiments:
            for run in exp['runs']:
                writer.writerow({'model': exp['model'],'profiler_type': exp['profiler_type'],'run_number': run['run_number'],'duration_s': run['duration_s']})


def plot_experiments(experiments):
    # Group by model and profiler
    models = sorted({exp['model'] for exp in experiments})
    # Calculate baseline (NONE) averages per model
    base_avgs = {}
    for model in models:
        runs = next((exp['runs'] for exp in experiments if exp['model']==model and exp['profiler_type']=='NONE'), [])
        base_avgs[model] = (sum(r['duration_s'] for r in runs)/len(runs)) if runs else 0.0

    # Define desired profiler order, excluding NONE
    all_profs = {exp['profiler_type'] for exp in experiments if exp['profiler_type']!='NONE'}
    ordered = ['PROTON', 'NSYS', 'TORCH']
    profilers = [p for p in ordered if p in all_profs]

    # Calculate percentage difference from NONE
    pct_diff = {prof: [] for prof in profilers}
    for model in models:
        base = base_avgs.get(model, 0.0)
        for prof in profilers:
            runs = next((exp['runs'] for exp in experiments if exp['model']==model and exp['profiler_type']==prof), [])
            avg = (sum(r['duration_s'] for r in runs)/len(runs)) if runs else 0.0
            percent = ((avg - base) / base) * 100 if base > 0 else 0.0
            pct_diff[prof].append(percent)

    # Plot grouped bar chart
    x = np.arange(len(models))
    width = 0.8 / len(profilers)
    fig, ax = plt.subplots()
    for i, prof in enumerate(profilers):
        ax.bar(x + i * width, pct_diff[prof], width, label=prof)

    ax.set_xticks(x + width * (len(profilers) - 1) / 2)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Percentage Increase in Training Time (%)')
    ax.set_title('Percentage Overhead of Profilers for Unsloth Model Training')
    ax.legend()
    plt.tight_layout()
    plt.savefig('unsloth_profiling.png')
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and optionally plot experiment timings.')
    parser.add_argument('log_file', help='Path to the log file')
    parser.add_argument('out_csv', help='Output CSV file path')
    parser.add_argument('--plot', action='store_true', help='Show grouped bar chart with percent difference')
    args = parser.parse_args()

    experiments = parse_experiments(args.log_file)
    write_csv(experiments, args.out_csv)
    total_runs = sum(len(exp['runs']) for exp in experiments)
    print(f"Extracted {total_runs} runs across {len(experiments)} experiments to {args.out_csv}")

    if args.plot:
        plot_experiments(experiments)
