
'''

6.1 방법 4 (SPO+ loss) 실행 결과

'''


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data parsing
data = [
    ["AR", None, None, 36.11, 25.41, 34.76, 15.04, 1.35, 10.37],
    ["1/20", 1, 20, 60.35, 19.41, 34.89, 13.91, 25.46, 5.50],
    ["1/10", 1, 10, 55.64, 20.79, 35.14, 13.42, 20.50, 7.37],
    ["1/5", 1, 5, 54.42, 21.23, 36.28, 12.71, 18.14, 8.52],
    ["1/2", 1, 2, 54.40, 21.37, 41.09, 11.88, 13.31, 9.49],
    ["1/1", 1, 1, 54.59, 21.31, 44.95, 11.44, 9.64, 9.87],
    ["2/1", 2, 1, 53.64, 21.52, 46.11, 11.38, 7.53, 10.14],
    ["5/1", 5, 1, 53.79, 21.49, 48.27, 11.38, 5.52, 10.11],
    ["10/1", 10, 1, 58.86, 20.17, 49.21, 11.36, 9.65, 8.81],
    ["20/1", 20, 1, 58.86, 20.17, 49.61, 11.36, 9.25, 8.81]
]

columns = ["Label", "W1", "W2", "nRMSE", "Gap", "Paper_nRMSE", "Paper_Gap", "Delta_nRMSE", "Delta_Gap"]
df = pd.DataFrame(data, columns=columns)

print(df)

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

x = df['Label']

# Plot 1: nRMSE Comparison
axes[0].plot(x, df['nRMSE'], marker='o', color='#1f77b4', linewidth=2, label='Current nRMSE')
axes[0].plot(x, df['Paper_nRMSE'], marker='s', color='#ff7f0e', linestyle='--', linewidth=2, label='Paper nRMSE')
axes[0].set_title('nRMSE Comparison (%)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('nRMSE (%)', fontsize=11)
axes[0].legend(frameon=True)
axes[0].grid(True, linestyle=':', alpha=0.6)

# Plot 2: Gap Comparison
axes[1].plot(x, df['Gap'], marker='o', color='#2ca02c', linewidth=2, label='Current Gap')
axes[1].plot(x, df['Paper_Gap'], marker='s', color='#d62728', linestyle='--', linewidth=2, label='Paper Gap')
axes[1].set_title('Gap Comparison (%)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Gap (%)', fontsize=11)
axes[1].legend(frameon=True)
axes[1].grid(True, linestyle=':', alpha=0.6)

# Plot 3: Delta Comparison
axes[2].bar(np.arange(len(x)) - 0.2, df['Delta_nRMSE'], width=0.4, label='Δ nRMSE (%p)', color='#9467bd', alpha=0.85)
axes[2].bar(np.arange(len(x)) + 0.2, df['Delta_Gap'], width=0.4, label='Δ Gap (%p)', color='#8c564b', alpha=0.85)
axes[2].set_xticks(np.arange(len(x)))
axes[2].set_xticklabels(x)
axes[2].set_title('Delta (Difference) (%p)', fontsize=13, fontweight='bold')
axes[2].set_ylabel('Percentage Points (%p)', fontsize=11)
axes[2].legend(frameon=True)
axes[2].grid(True, linestyle=':', alpha=0.6)

for ax in axes:
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('comparison_plots.png', dpi=300)
plt.show()