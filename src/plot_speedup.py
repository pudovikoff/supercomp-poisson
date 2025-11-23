#!/usr/bin/env python3
"""
Plot speedup and execution time performance for OpenMP, MPI, and MPI+OpenMP.
Usage: python3 src/plot_speedup.py [--mode {openmp|mpi|hybrid}]
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse

# ============================================================================
# DATA CONFIGURATION SECTION
# Edit the appropriate section below with your measured data
# ============================================================================

# OpenMP data
OPENMP_DATA = {
    'processes': [1, 2, 4, 8, 16, 1, 4, 8, 16, 32],
    'times': [10.59, 5.03, 2.54, 2.08, 1.07, 79.06, 20.03, 16.56, 5.13, 4.26],
    'speedup': [1.0, 2.11, 4.17, 5.09, 9.9, 1.0, 3.95, 4.78, 15.41, 18.56],
    'algo': 'OpenMP',
    'x_label': 'OpenMP threads',
    'title_prefix': 'OpenMP'
}

# MPI data
MPI_DATA = {
    'processes': [1, 2, 4, 8, 16, 1, 4, 8, 16, 32],
    'times': [7.82, 3.89, 2.01, 1.09, 0.75, 46.49, 11.49, 5.82, 3.20, 4.22],
    'speedup': [1.0, 2.01, 3.89, 7.17, 10.43, 1.0, 4.05, 7.99, 14.53, 11.02],
    'algo': 'MPI',
    'x_label': 'MPI processes',
    'title_prefix': 'MPI'
}

# MPI+OpenMP hybrid data
# Format: each configuration is (mpi_procs, omp_threads)
# For x-axis labeling, we use "mpi_procs×omp_threads" format
HYBRID_DATA = {
    # Configuration: [(MPI, OMP), ...]
    'configs_400x600': [(2, 1), (2, 2), (2, 4), (2, 8)],
    'configs_800x1200': [(4, 1), (4, 2), (4, 4), (4, 8)],
    # Measured times
    'times': [3.96, 2.01, 1.03, 0.94, 11.62, 5.88, 2.99, 2.73],
    # Speedup relative to first config in each group
    'speedup': [1.0, 1.97, 3.83, 4.21, 1.0, 1.98, 3.89, 4.26],
    'algo': 'MPI+OpenMP',
    'x_label': 'Configuration (MPI×OMP)',
    'title_prefix': 'MPI+OpenMP Hybrid'
}

def plot_standard(data, save_path):
    """Plot for OpenMP and MPI (standard format with process counts)."""
    processes = data['processes']
    times = data['times']
    speedup = data['speedup']
    x_label = data['x_label']
    title_prefix = data['title_prefix']
    
    # Split data by grid size
    grid_400x600_p = processes[:5]
    grid_400x600_t = times[:5]
    grid_400x600_s = speedup[:5]
    
    grid_800x1200_p = processes[5:]
    grid_800x1200_t = times[5:]
    grid_800x1200_s = speedup[5:]

    # Create figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{title_prefix} Performance Analysis: Speedup and Execution Time', fontsize=16, fontweight='bold')

    # Plot 1: Speedup for both grids (LEFT)
    ax = axes[0]
    # Plot ideal speedup (linear)
    ideal_p_400 = np.array(grid_400x600_p)
    ideal_s_400 = ideal_p_400
    ax.plot(ideal_p_400, ideal_s_400, '--', linewidth=2, color='gray', alpha=0.5, label='Ideal Speedup')
    
    # Actual speedup for both grids
    ax.plot(grid_400x600_p, grid_400x600_s, 'o-', linewidth=2.5, markersize=9, color='#2E86AB', label='400×600')
    ax.plot(grid_800x1200_p, grid_800x1200_s, 's-', linewidth=2.5, markersize=9, color='#A23B72', label='800×1200')
    
    ax.set_xlabel(f'Number of {x_label}', fontsize=12, fontweight='bold')
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
    
    ax.set_xlabel(f'Number of {x_label}', fontsize=12, fontweight='bold')
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {save_path}")
    
    # Print summary statistics
    print(f"\n=== {data['algo']} Performance Summary ===")
    print("\n400×600 Grid:")
    print(f"  Base time (1 proc):      {grid_400x600_t[0]:.2f}s")
    print(f"  Best speedup (16 procs): {grid_400x600_s[-1]:.2f}x")
    print(f"  Efficiency at 16 procs:  {(grid_400x600_s[-1] / 16) * 100:.1f}%")
    
    print("\n800×1200 Grid:")
    print(f"  Base time (1 proc):      {grid_800x1200_t[0]:.2f}s")
    print(f"  Best speedup (16 procs): {grid_800x1200_s[3]:.2f}x")
    print(f"  Efficiency at 16 procs:  {(grid_800x1200_s[3] / 16) * 100:.1f}%")


def plot_hybrid(data, save_path):
    """Plot for MPI+OpenMP hybrid (special format with configuration labels)."""
    configs_400 = data['configs_400x600']
    configs_800 = data['configs_800x1200']
    times = data['times']
    speedup = data['speedup']
    title_prefix = data['title_prefix']
    
    # Split data by grid size
    n_400 = len(configs_400)
    grid_400x600_t = times[:n_400]
    grid_400x600_s = speedup[:n_400]
    
    grid_800x1200_t = times[n_400:]
    grid_800x1200_s = speedup[n_400:]
    
    # Create x-axis labels
    labels_400 = [f"{m}×{o}" for m, o in configs_400]
    labels_800 = [f"{m}×{o}" for m, o in configs_800]
    
    # X-axis positions
    x_400 = np.arange(len(configs_400))
    x_800 = np.arange(len(configs_800))
    
    # Create figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{title_prefix} Performance Analysis: Speedup and Execution Time', fontsize=16, fontweight='bold')
    
    # Plot 1: Speedup for both grids (LEFT)
    ax = axes[0]
    
    # Plot data
    ax.plot(x_400, grid_400x600_s, 'o-', linewidth=2.5, markersize=9, color='#2E86AB', label='400×600 (2 MPI)')
    ax.plot(x_800, grid_800x1200_s, 's-', linewidth=2.5, markersize=9, color='#A23B72', label='800×1200 (4 MPI)')
    
    ax.set_xlabel('Configuration (MPI×OpenMP)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (×)', fontsize=12, fontweight='bold')
    ax.set_title('Speedup vs Thread Configuration', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_400)
    ax.set_xticklabels(labels_400)
    ax.legend(fontsize=11, loc='upper left')
    
    # Add value labels for 400×600
    for x, s in zip(x_400, grid_400x600_s):
        ax.text(x, s + 0.15, f'{s:.2f}×', ha='center', fontsize=9, fontweight='bold', color='#2E86AB')
    
    # Add value labels for 800×1200
    for x, s in zip(x_800, grid_800x1200_s):
        ax.text(x, s - 0.25, f'{s:.2f}×', ha='center', fontsize=9, fontweight='bold', color='#A23B72')
    
    # Plot 2: Execution Time for both grids (RIGHT)
    ax = axes[1]
    ax.plot(x_400, grid_400x600_t, 'o-', linewidth=2.5, markersize=9, color='#2E86AB', label='400×600 (2 MPI)')
    ax.plot(x_800, grid_800x1200_t, 's-', linewidth=2.5, markersize=9, color='#A23B72', label='800×1200 (4 MPI)')
    
    ax.set_xlabel('Configuration (MPI×OpenMP)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (s)', fontsize=12, fontweight='bold')
    ax.set_title('Execution Time vs Thread Configuration', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_800)
    ax.set_xticklabels(labels_800)
    ax.legend(fontsize=11, loc='upper right')
    
    # Add value labels for 400×600
    for x, t in zip(x_400, grid_400x600_t):
        ax.text(x, t + 0.15, f'{t:.2f}s', ha='center', fontsize=9, fontweight='bold', color='#2E86AB')
    
    # Add value labels for 800×1200
    for x, t in zip(x_800, grid_800x1200_t):
        ax.text(x, t + 0.4, f'{t:.2f}s', ha='center', fontsize=9, fontweight='bold', color='#A23B72')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {save_path}")
    
    # Print summary statistics
    print(f"\n=== {data['algo']} Performance Summary ===")
    print("\n400×600 Grid (2 MPI processes):")
    print(f"  Base config (2×1):       {grid_400x600_t[0]:.2f}s")
    print(f"  Best config (2×8):       {grid_400x600_t[-1]:.2f}s, speedup {grid_400x600_s[-1]:.2f}x")
    
    print("\n800×1200 Grid (4 MPI processes):")
    print(f"  Base config (4×1):       {grid_800x1200_t[0]:.2f}s")
    print(f"  Best config (4×8):       {grid_800x1200_t[-1]:.2f}s, speedup {grid_800x1200_s[-1]:.2f}x")


def main():
    parser = argparse.ArgumentParser(description='Plot performance graphs for Poisson solver')
    parser.add_argument('--mode', choices=['openmp', 'mpi', 'hybrid'], default='mpi',
                        help='Which version to plot (default: mpi)')
    args = parser.parse_args()
    
    # Select data based on mode
    if args.mode == 'openmp':
        data = OPENMP_DATA
        save_path = 'results/speedup_graph_OpenMP.png'
        plot_standard(data, save_path)
    elif args.mode == 'mpi':
        data = MPI_DATA
        save_path = 'results/speedup_graph_MPI.png'
        plot_standard(data, save_path)
    elif args.mode == 'hybrid':
        data = HYBRID_DATA
        save_path = 'results/speedup_graph_MPI_OpenMP.png'
        plot_hybrid(data, save_path)


if __name__ == '__main__':
    main()
