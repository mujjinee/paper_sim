import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

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

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

x = np.arange(len(df['Label']))
labels = df['Label']

# Plot 1: nRMSE Comparison
axes[0].plot(x, df['Mod_nRMSE'], marker='o', color='#1f77b4', linewidth=2.5, label='Modified nRMSE')
axes[0].plot(x, df['Paper_nRMSE'], marker='s', color='#ff7f0e', linestyle='--', linewidth=2.5, label='Paper nRMSE')
# Old nRMSE points if available
axes[0].scatter([0, 1], df['Old_nRMSE'].iloc[:2], color='#2ca02c', zorder=5, s=60, label='Original nRMSE (AR, 1/20)')

axes[0].set_title('nRMSE Comparison (%)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('nRMSE (%)', fontsize=11)
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, rotation=45)
axes[0].legend(frameon=True)
axes[0].grid(True, linestyle=':', alpha=0.6)

# Plot 2: Gap Comparison
axes[1].plot(x, df['Mod_Gap'], marker='o', color='#1f77b4', linewidth=2.5, label='Modified Gap')
axes[1].plot(x, df['Paper_Gap'], marker='s', color='#ff7f0e', linestyle='--', linewidth=2.5, label='Paper Gap')
axes[1].scatter([0, 1], df['Old_Gap'].iloc[:2], color='#d62728', zorder=5, s=60, label='Original Gap (AR, 1/20)')

axes[1].set_title('Gap Comparison (%)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Gap (%)', fontsize=11)
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, rotation=45)
axes[1].legend(frameon=True)
axes[1].grid(True, linestyle=':', alpha=0.6)

# Plot 3: Delta (%p)
axes[2].bar(x - 0.2, df['Delta_nRMSE'], width=0.4, label='Δ nRMSE (%p)', color='#9467bd', alpha=0.85)
axes[2].bar(x + 0.2, df['Delta_Gap'], width=0.4, label='Δ Gap (%p)', color='#8c564b', alpha=0.85)
axes[2].axhline(0, color='gray', linestyle='--', alpha=0.7)
axes[2].set_title('Delta vs Paper (%p)', fontsize=13, fontweight='bold')
axes[2].set_ylabel('Percentage Points (%p)', fontsize=11)
axes[2].set_xticks(x)
axes[2].set_xticklabels(labels, rotation=45)
axes[2].legend(frameon=True)
axes[2].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('modified_comparison1.png', dpi=300)
plt.close()