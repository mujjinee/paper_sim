import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = [
    ["AR", None, None, 36.11, -0.75, 36.11, 14.94, 34.76, 15.04, 1.35, -0.10],
    ["1/20", 1, 20, 36.29, -1.09, 35.69, 14.54, 34.89, 13.91, 0.80, 0.63],
    ["1/10", 1, 10, None, None, 35.68, 14.35, 35.14, 13.42, 0.54, 0.93],
    ["1/5", 1, 5, None, None, 35.29, 14.09, 36.28, 12.71, -0.99, 1.38],
    ["1/2", 1, 2, None, None, 36.98, 14.18, 41.09, 11.88, -4.11, 2.30],
    ["1/1", 1, 1, None, None, 37.46, 13.70, 44.95, 11.44, -7.49, 2.26],
    ["2/1", 2, 1, None, None, 39.92, 12.95, 46.11, 11.38, -6.19, 1.57],
    ["5/1", 5, 1, None, None, 45.10, 11.50, 48.27, 11.38, -3.17, 0.12],
    ["10/1", 10, 1, None, None, 54.75, 10.94, 49.21, 11.36, 5.54, -0.42],
    ["20/1", 20, 1, None, None, 63.15, 10.33, 49.61, 11.36, 13.54, -1.03],
    ["1/0", 1, 0, None, None, 67.84, 10.69, 50.07, 11.36, 17.77, -0.67]
]

columns = ["Label", "W1", "W2", "Old_nRMSE", "Old_Gap", "Mod_nRMSE", "Mod_Gap", "Paper_nRMSE", "Paper_Gap", "Delta_nRMSE", "Delta_Gap"]
df = pd.DataFrame(data, columns=columns)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

fig, ax1 = plt.subplots(figsize=(12, 6))

x = np.arange(len(df['Label']))

# Left Y-axis: nRMSE
color_nrmse_mod = '#1f77b4'
color_nrmse_paper = '#aec7e8'

line1 = ax1.plot(x, df['Mod_nRMSE'], marker='o', color=color_nrmse_mod, linewidth=2.5, label='Modified nRMSE (Left)')
line2 = ax1.plot(x, df['Paper_nRMSE'], marker='s', color=color_nrmse_paper, linestyle='--', linewidth=2, label='Paper nRMSE (Left)')

ax1.set_xlabel('Label', fontsize=12, fontweight='bold')
ax1.set_ylabel('nRMSE (%)', fontsize=12, color='#1f77b4', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='#1f77b4')
ax1.set_xticks(x)
ax1.set_xticklabels(df['Label'], rotation=45, fontsize=10)
ax1.set_ylim(30, 72)
ax1.grid(True, linestyle=':', alpha=0.6)

# Right Y-axis: Gap
ax2 = ax1.twinx()
color_gap_mod = '#d62728'
color_gap_paper = '#ff9896'

line3 = ax2.plot(x, df['Mod_Gap'], marker='^', color=color_gap_mod, linewidth=2.5, label='Modified Gap (Right)')
line4 = ax2.plot(x, df['Paper_Gap'], marker='D', color=color_gap_paper, linestyle='--', linewidth=2, label='Paper Gap (Right)')

ax2.set_ylabel('Gap (%)', fontsize=12, color='#d62728', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#d62728')
ax2.set_ylim(8, 18)
ax2.grid(False) # avoid overlapping grid lines

# Combine legends
lines = line1 + line2 + line3 + line4
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

plt.title('Modified vs Paper: nRMSE (Left Y) & Gap (Right Y) Dual Axis Plot', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('dual_y_axis_plot.png', dpi=300)
plt.close()