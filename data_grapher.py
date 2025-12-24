import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Load your data
df = pd.read_csv('all_results.csv')

# 2) Compute mean of the four methods per script
metrics = ['baseline', 'proton', 'torch', 'nsys']
grouped = (
    df
    .groupby('script')[metrics]
    .mean()
    .reset_index()
)
print(grouped)

# 3) Compute % difference from baseline
#    (value - baseline) / baseline * 100
for m in metrics:
    # print(grouped[m])
    grouped[m] = (grouped[m] - grouped['baseline']) / grouped['baseline'] * 100
    print(m)

# 4) Prepare for plotting
x_labels  = grouped['script']
data      = grouped[metrics]
n_groups  = len(x_labels)
n_metrics = len(metrics)
bar_width = 0.8 / n_metrics
x         = np.arange(n_groups)

# 5) Create the grouped bar chart
fig, ax = plt.subplots(figsize=(10, 5))

for i, m in enumerate(metrics):
    ax.bar(
        x + i * bar_width,
        data[m],
        width=bar_width,
        label=m
    )

# 6) Formatting
ax.set_xlabel('Script')
ax.set_ylabel('Percent Difference from Baseline (%)')
ax.set_title('Average Performance Difference from Baseline by Script')
ax.set_xticks(x + bar_width*(n_metrics-1)/2)
ax.set_xticklabels(x_labels, rotation=30, ha='right')
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')  # reference line at 0%
ax.legend(title='Implementation')

plt.tight_layout()
plt.savefig('overhead_bench.png', dpi=300)
plt.show()
