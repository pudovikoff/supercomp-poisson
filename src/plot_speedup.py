#!/usr/bin/env python3
"""
Plot MPI speedup and execution time performance from Table 2.
Usage: python3 src/plot_speedup.py
"""

import matplotlib.pyplot as plt
import numpy as np

processes = [1, 2, 4, 8, 16, 1, 4, 8, 16, 32]
grid_sizes = ['400×600', '400×600', '400×600', '400×600', '400×600',
              '800×1200', '800×1200', '800×1200', '800×1200', '800×1200']

# OpenMP
algo = 'OpenMP'
x_arg = 'OpenMP threads'
times = [10.59, 5.03, 2.54, 2.08, 1.07, 79.06, 20.03, 16.56, 5.13, 4.26]
speedup = [1.0, 2.11, 4.17, 5.09, 9.9, 1.0, 3.95, 4.78, 15.41, 18.56]



# # MPI
# algo = 'MPI'
# x_arg = 'MPI proccesses'
# times = [7.82, 3.89, 2.01, 1.09, 0.75, 46.49, 11.49, 5.82, 3.20, 4.22]
# speedup = [1.0, 2.01, 3.89, 7.17, 10.43, 1.0, 4.05, 7.99, 14.53, 11.02]

SAVE_PATH = f'results/speedup_graph_{algo}.png'

# Split data by grid size
grid_400x600_p = processes[:5]
grid_400x600_t = times[:5]
grid_400x600_s = speedup[:5]

grid_800x1200_p = processes[5:]
grid_800x1200_t = times[5:]
grid_800x1200_s = speedup[5:]

# Create figure with 1x2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('MPI Performance Analysis: Speedup and Execution Time', fontsize=16, fontweight='bold')

# Plot 1: Speedup for both grids (LEFT)
ax = axes[0]
# Plot ideal speedup (linear)
ideal_p_400 = np.array(grid_400x600_p)
ideal_s_400 = ideal_p_400
ax.plot(ideal_p_400, ideal_s_400, '--', linewidth=2, color='gray', alpha=0.5, label='Ideal Speedup')

# Actual speedup for both grids
ax.plot(grid_400x600_p, grid_400x600_s, 'o-', linewidth=2.5, markersize=9, color='#2E86AB', label='400×600')
ax.plot(grid_800x1200_p, grid_800x1200_s, 's-', linewidth=2.5, markersize=9, color='#A23B72', label='800×1200')

ax.set_xlabel(f'Number of {x_arg}', fontsize=12, fontweight='bold')
ax.set_ylabel('Speedup (×)', fontsize=12, fontweight='bold')
ax.set_title('Speedup: Both Grid Sizes', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xticks([1, 2, 4, 8, 16, 32])
ax.legend(fontsize=11, loc='upper left')
ax.set_xlim(0.5, 33)

# Add value labels for 400×600
for p, s in zip(grid_400x600_p, grid_400x600_s):
    ax.text(p, s + 0.4, f'{s:.2f}×', ha='center', fontsize=9, fontweight='bold', color='#2E86AB')

# Add value labels for 800×1200
for p, s in zip(grid_800x1200_p, grid_800x1200_s):
    ax.text(p, s - 0.7, f'{s:.2f}×', ha='center', fontsize=9, fontweight='bold', color='#A23B72')

# Plot 2: Execution Time for both grids (RIGHT)
ax = axes[1]
ax.plot(grid_400x600_p, grid_400x600_t, 'o-', linewidth=2.5, markersize=9, color='#2E86AB', label='400×600')
ax.plot(grid_800x1200_p, grid_800x1200_t, 's-', linewidth=2.5, markersize=9, color='#A23B72', label='800×1200')

ax.set_xlabel(f'Number of {x_arg}', fontsize=12, fontweight='bold')
ax.set_ylabel('Execution Time (s)', fontsize=12, fontweight='bold')
ax.set_title('Execution Time: Both Grid Sizes', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xticks([1, 2, 4, 8, 16, 32])
ax.legend(fontsize=11, loc='upper right')
ax.set_xlim(0.5, 33)

# Add value labels for 400×600
for p, t in zip(grid_400x600_p, grid_400x600_t):
    ax.text(p, t + 0.2, f'{t:.2f}s', ha='center', fontsize=9, fontweight='bold', color='#2E86AB')

# Add value labels for 800×1200
for p, t in zip(grid_800x1200_p, grid_800x1200_t):
    ax.text(p, t + 1.0, f'{t:.2f}s', ha='center', fontsize=9, fontweight='bold', color='#A23B72')

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
print(f"✓ Plot saved to {SAVE_PATH}")

# Print summary statistics
print(f"\n=== {algo} Performance Summary ===")
print("\n400×600 Grid:")
print(f"  Sequential (1 proc):     {grid_400x600_t[0]:.2f}s")
print(f"  Best speedup (16 procs): {grid_400x600_s[-1]:.2f}x")
print(f"  Efficiency at 16 procs:  {(grid_400x600_s[-1] / 16) * 100:.1f}%")

print("\n800×1200 Grid:")
print(f"  Sequential (1 proc):     {grid_800x1200_t[0]:.2f}s")
print(f"  Best speedup (16 procs): {grid_800x1200_s[3]:.2f}x")
print(f"  Efficiency at 16 procs:  {(grid_800x1200_s[3] / 16) * 100:.1f}%")