#!/usr/bin/env python3
"""
Plot speedup and efficiency for all implementations (OpenMP, MPI, MPI+OpenMP, GPU versions).
Baseline: Sequential version = 1800 s
Usage: python3 src/cuda_res.py
"""

import matplotlib.pyplot as plt
import numpy as np

# Sequential baseline
SEQUENTIAL_TIME = 970.629623  # seconds

# Data for all implementations
DATA = {
    'implementations': [
        'Sequential',
        'OpenMP\n16×8',
        'MPI\n20 procs',
        'MPI+OpenMP\n10×16',
        '1 GPU',
        '2 GPU'
    ],
    'times': [
        970.629623,      # Sequential
        70.43,       # OpenMP 16×8
        45.816858,   # MPI 20
        65.774599,   # MPI 10×16
        9.18,   # 1 GPU
        6.27    # 2 GPU
    ],
    'colors': [
        '#808080',   # Gray for sequential
        '#2E86AB',   # Blue (OpenMP)
        '#A23B72',   # Purple (MPI)
        '#F18F01',   # Orange (Hybrid)
        '#C73E1D',   # Red-brown (1 GPU)
        '#6A994E'    # Green (2 GPU)
    ]
}

def calculate_speedup_efficiency(times, baseline):
    """Calculate speedup and efficiency relative to baseline."""
    speedup = [baseline / t for t in times]
    # Efficiency: speedup / number_of_resources
    # We estimate resource count from speedup (for non-GPU we use process count)
    resources = [1, 16*8, 20, 10*16, 1, 2]
    efficiency = [s / r * 100 for s, r in zip(speedup, resources)]
    return speedup, efficiency

def plot_speedup_efficiency():
    """Create plots for speedup and efficiency."""
    times = DATA['times']
    implementations = DATA['implementations']
    colors = DATA['colors']
    
    # Calculate metrics
    speedup, efficiency = calculate_speedup_efficiency(times, SEQUENTIAL_TIME)
    
    # X-axis positions
    x = np.arange(len(implementations))
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('CUDA and Parallel Performance Analysis: Speedup and Efficiency, 2000×3200 Grid',
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Speedup (no lines connecting)
    ax = axes[0]
    ax.scatter(x, speedup, s=200, c=colors, edgecolors='black', linewidth=1.5, zorder=3)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Implementation Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (×)', fontsize=12, fontweight='bold')
    ax.set_title('Speedup vs Sequential', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(implementations, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(speedup) * 1.15)
    
    # Add value labels for speedup
    for i, s in enumerate(speedup):
        ax.text(i, s + 1.5, f'{s:.2f}×', ha='center', fontsize=10, fontweight='bold')
    
    # Plot 2: Execution Time (log scale)
    ax = axes[1]
    ax.scatter(x, times, s=200, c=colors, edgecolors='black', linewidth=1.5, zorder=3)
    ax.set_yscale('log')
    
    ax.set_xlabel('Implementation Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (s, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Execution Time Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(implementations, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', which='both')
    ax.set_ylim(5, 3000)
    
    # Add value labels for execution time
    for i, t in enumerate(times):
        ax.text(i, t * 1.3, f'{t:.2f}s', ha='center', fontsize=10, fontweight='bold')
    
    plt.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.12, wspace=0.3)
    plt.savefig('results/cuda_performance.png', dpi=300)
    print(f"✓ Plot saved to results/cuda_performance.png")
    
    # Print detailed summary
    print(f"\n{'='*70}")
    print(f"CUDA and Parallel Performance Analysis Summary")
    print(f"{'='*70}")
    print(f"Grid: 2000×3200")
    print(f"Baseline (Sequential): {SEQUENTIAL_TIME:.1f}s")
    print(f"{'='*70}\n")
    
    print(f"{'Implementation':<20} {'Time (s)':<15} {'Speedup':<12} {'Efficiency':<12}")
    print(f"{'-'*70}")
    
    for i, impl in enumerate(implementations):
        impl_name = impl.replace('\n', ' ')
        print(f"{impl_name:<20} {times[i]:<15.3f} {speedup[i]:<12.2f}x {efficiency[i]:<12.1f}%")
    
    print(f"\n{'='*70}")
    print(f"Key Observations:")
    print(f"{'='*70}")
    print(f"Best performer: 2 GPU - {speedup[5]:.2f}x speedup")
    print(f"2 GPU vs 1 GPU: {speedup[4]/speedup[5]:.2f}x improvement")
    print(f"2 GPU vs MPI 20: {speedup[2]/speedup[5]:.2f}x improvement")
    print(f"GPU efficiency limited by CPU-GPU transfers (visible in timing breakdowns)")

if __name__ == '__main__':
    plot_speedup_efficiency()
