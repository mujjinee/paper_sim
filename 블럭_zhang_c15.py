import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = [
    ["1/20", 35.68, 14.19, 34.89, 13.91],
    ["1/10", 35.71, 13.85, 35.14, 13.42],
    ["1/5", 35.62, 13.75, 36.28, 12.71],
    ["1/2", 36.82, 13.48, 41.09, 11.88],
    ["1/1", 38.04, 13.32, 44.95, 11.44],
    ["2/1", 42.83, 12.83, 46.11, 11.38],
    ["5/1", 49.51, 12.12, 48.27, 11.38],
    ["10/1", 59.01, 11.47, 49.21, 11.36],
    ["20/1", 66.07, 11.07, 49.61, 11.36],
    ["1/0", 68.85, 11.17, 50.07, 11.36]
]

columns = ["Label", "nRMSE", "Gap", "Paper_nRMSE", "Paper_Gap"]
df = pd.DataFrame(data, columns=columns)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

fig, ax1 = plt.subplots(figsize=(12, 6))

x = np.arange(len(df['Label']))

# Left Y-axis: nRMSE
color_nrmse_exp = '#1f77b4'
color_nrmse_paper = '#aec7e8'

line1 = ax1.plot(x, df['nRMSE'], marker='o', color=color_nrmse_exp, linewidth=2.5, label='Experiment nRMSE (Left)')
line2 = ax1.plot(x, df['Paper_nRMSE'], marker='s', color=color_nrmse_paper, linestyle='--', linewidth=2, label='Paper nRMSE (Left)')

ax1.set_xlabel('Label (W1 / W2)', fontsize=12, fontweight='bold')
ax1.set_ylabel('nRMSE (%)', fontsize=12, color='#1f77b4', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='#1f77b4')
ax1.set_xticks(x)
ax1.set_xticklabels(df['Label'], rotation=45, fontsize=10)
ax1.set_ylim(30, 75)
ax1.grid(True, linestyle=':', alpha=0.6)

# Right Y-axis: Gap
ax2 = ax1.twinx()
color_gap_exp = '#d62728'
color_gap_paper = '#ff9896'

line3 = ax2.plot(x, df['Gap'], marker='^', color=color_gap_exp, linewidth=2.5, label='Experiment Gap (Right)')
line4 = ax2.plot(x, df['Paper_Gap'], marker='D', color=color_gap_paper, linestyle='--', linewidth=2, label='Paper Gap (Right)')

ax2.set_ylabel('Gap (%)', fontsize=12, color='#d62728', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#d62728')
ax2.set_ylim(9, 16)
ax2.grid(False)

# Combine legends
lines = line1 + line2 + line3 + line4
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

plt.title('c = 1.5: nRMSE (Left Y) & Gap (Right Y) Dual Axis Plot', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('dual_y_axis_c1_5.png', dpi=300)
plt.close()