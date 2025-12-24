import re
import csv
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np

def parse_sizes(listing_path):
    """
    Parse an 'ls -l' style file listing, extracting model, profiler type, and size.
    .hatchet    -> PROTON
    unsloth_trace_*.json -> TORCH
    .nsys.nsys-rep -> NSYS
    """
    patterns = [
        ('PROTON', re.compile(r'^(?P<model>[\w\.\-]+)\.hatchet$')),
        ('TORCH',  re.compile(r'^unsloth_trace_(?P<model>[\w\.\-]+)\.json$')),
        ('NSYS',   re.compile(r'^(?P<model>[\w\.\-]+)\.nsys\.nsys-rep$'))
    ]
    records = []

    with open(listing_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 9:
                continue
            size = int(parts[4])
            filename = parts[8]
            for profiler, pat in patterns:
                m = pat.match(filename)
                if m:
                    model = m.group('model')
                    records.append({'model': model, 'profiler_type': profiler, 'size_bytes': size})
                    break
    return records


def write_csv(records, out_path):
    """Write parsed size records to CSV."""
    with open(out_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['model', 'profiler_type', 'size_bytes'])
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def plot_sizes(records):
    """Plot grouped bar chart of percent size difference vs PROTON baseline."""
    # Unique models
    models = sorted({r['model'] for r in records})
    # Baseline sizes (PROTON)
    baseline = {m: next((r['size_bytes'] for r in records if r['model']==m and r['profiler_type']=='PROTON'), 0)
                for m in models}
    # Ordered profilers excluding baseline
    ordered = ['PROTON', 'NSYS', 'TORCH']
    profilers = [p for p in ordered if p!='PROTON' and any(r['profiler_type']==p for r in records)]

    # Compute percent differences
    pct = {p: [] for p in profilers}
    for m in models:
        base = baseline[m]
        for p in profilers:
            size = next((r['size_bytes'] for r in records if r['model']==m and r['profiler_type']==p), 0)
            pct[p].append(((size - base) / base) if base > 0 else 0)

    # Plot
    x = np.arange(len(models))
    width = 0.8 / len(profilers)
    fig, ax = plt.subplots()
    for i, p in enumerate(profilers):
        ax.bar(x + i*width, pct[p], width, label=p)
    ax.set_xticks(x + width*(len(profilers)-1)/2)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Relative Profile Size Difference in Comparison to Proton')
    ax.set_title('Profile File Size Comparison by Model')
    ax.legend()
    plt.tight_layout()
    plt.savefig("memsizes.png")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and compare profile file sizes.')
    parser.add_argument('listing_file', help='Path to ls -l listing file')
    parser.add_argument('out_csv', help='Output CSV path')
    parser.add_argument('--plot', action='store_true', help='Show grouped percent-difference bar chart')
    args = parser.parse_args()

    records = parse_sizes(args.listing_file)
    write_csv(records, args.out_csv)
    print(f"Extracted {len(records)} profile size records to {args.out_csv}")
    if args.plot:
        plot_sizes(records)

