import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Paper reference values
penalty_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
paper_ar_nrmse = [34.76] * 11
paper_ar_gap = [9, 11, 12, 13, 14, 15.04, 16, 18, 19, 20, 22]
paper_prop_nrmse = [46, 34, 35, 36, 40, 44.45, 50, 55, 61, 64, 67]
paper_prop_gap = [8, 10, 11, 11, 11, 11.44, 11, 11.5, 11.5, 11.5, 11.5]

# Implemented (Proposed/AR) values
impl_data = [
    [0, 36.11, 6.42, 57.92, 6.58],
    [10, 36.11, 8.13, 36.96, 7.76],
    [20, 36.11, 9.83, 37.35, 9.66],
    [30, 36.11, 11.53, 37.47, 11.13],
    [40, 36.11, 13.24, 37.85, 12.22],
    [50, 36.11, 14.94, 37.46, 13.70],
    [60, 36.11, 16.64, 37.07, 15.09],
    [70, 36.11, 18.35, None, None],
    [80, 36.11, 20.05, None, None],
    [90, 36.11, 21.76, None, None],
    [100, 36.11, 23.46, None, None]
]

impl_df = pd.DataFrame(impl_data, columns=["Penalty", "Impl_AR_nRMSE", "Impl_AR_Gap", "Impl_Prop_nRMSE", "Impl_Prop_Gap"])

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Colors
c_ar_paper = '#1f77b4'
c_ar_impl = '#aec7e8'
c_prop_paper = '#d62728'
c_prop_impl = '#ff9896'

# Figure 5(a): nRMSE vs Penalty
axes[0].plot(penalty_levels, paper_ar_nrmse, marker='o', color=c_ar_paper, linestyle='--', linewidth=2, label='Paper AR')
axes[0].plot(impl_df['Penalty'], impl_df['Impl_AR_nRMSE'], marker='o', color=c_ar_paper, linewidth=2, label='Impl AR')
axes[0].plot(penalty_levels, paper_prop_nrmse, marker='s', color=c_prop_paper, linestyle='--', linewidth=2, label='Paper Proposed')
axes[0].plot(impl_df['Penalty'], impl_df['Impl_Prop_nRMSE'], marker='s', color=c_prop_paper, linewidth=2, label='Impl Proposed')

axes[0].set_title('(a) nRMSE vs. Penalty', fontsize=14, fontweight='bold', pad=12)
axes[0].set_xlabel('Penalty (%)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('nRMSE (%)', fontsize=12, fontweight='bold')
axes[0].set_xticks(penalty_levels)
axes[0].set_xticklabels([f'{p}%' for p in penalty_levels])
axes[0].legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
axes[0].grid(True, linestyle=':', alpha=0.6)

# Figure 5(b): Gap vs Penalty
axes[1].plot(penalty_levels, paper_ar_gap, marker='o', color=c_ar_paper, linestyle='--', linewidth=2, label='Paper AR')
axes[1].plot(impl_df['Penalty'], impl_df['Impl_AR_Gap'], marker='o', color=c_ar_paper, linewidth=2, label='Impl AR')
axes[1].plot(penalty_levels, paper_prop_gap, marker='s', color=c_prop_paper, linestyle='--', linewidth=2, label='Paper Proposed')
axes[1].plot(impl_df['Penalty'], impl_df['Impl_Prop_Gap'], marker='s', color=c_prop_paper, linewidth=2, label='Impl Proposed')

axes[1].set_title('(b) Gap vs. Penalty', fontsize=14, fontweight='bold', pad=12)
axes[1].set_xlabel('Penalty (%)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Gap (%)', fontsize=12, fontweight='bold')
axes[1].set_xticks(penalty_levels)
axes[1].set_xticklabels([f'{p}%' for p in penalty_levels])
axes[1].legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.suptitle('Figure 5: Paper vs. Implemented Performance Comparison Across Penalty Levels', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig5_paper_vs_impl.png', dpi=300, bbox_inches='tight')
plt.close()