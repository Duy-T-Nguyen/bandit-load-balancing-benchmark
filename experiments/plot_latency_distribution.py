import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.stats import gamma
import matplotlib.cm as cm

# Apply SciencePlots style
plt.style.use(['science', 'no-latex'])

# Custom rcParams for premium, clean styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['grid.color'] = '#e2e8f0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.6
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9


def plot_arms_distribution():
    print("Generating Latency Distribution Plot for the 10 Arms...")
    K = 10
    k = 5
    SEED = 42
    
    rng = np.random.default_rng(SEED)
    mean_latencies = rng.uniform(10.0, 100.0, size=K)
    
    # Sort for representation in the plot (so the legend and colors match speed)
    sorted_indices = np.argsort(mean_latencies)
    sorted_means = mean_latencies[sorted_indices]
    
    # Define range for X (latency in ms)
    x = np.linspace(0.1, 220.0, 1000)
    
    plt.figure(figsize=(10, 6))
    
    # Using 'coolwarm' or a nice teal-to-red colormap
    # Blue is fast (low latency), red is slow (high latency)
    cmap = plt.colormaps['coolwarm']
    colors = [cmap(val) for val in np.linspace(0.0, 1.0, K)]
    
    for i, orig_idx in enumerate(sorted_indices):
        mean_lat = sorted_means[i]
        theta = mean_lat / k
        pdf = gamma.pdf(x, a=k, scale=theta)
        
        # Label with arm number and its mean latency
        label = f'Arm {orig_idx} (Mean = {mean_lat:.2f} ms)'
        plt.plot(x, pdf, label=label, color=colors[i], linewidth=2.0)
        # Optional: shade under the curve
        plt.fill_between(x, pdf, color=colors[i], alpha=0.08)
        
    plt.xlabel('Response latency L (ms)', fontsize=12)
    plt.ylabel('Probability density (PDF)', fontsize=12)
    plt.title('Response-latency distribution (Gamma, k = 5) across the 10 arms', fontsize=14, fontweight='bold', pad=15)
    plt.xlim(0, 220)
    plt.ylim(0, 0.055)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Custom legend with nice styling
    plt.legend(title='10 servers (sorted fastest first)', title_fontsize=11, loc='upper right', fontsize=9, framealpha=0.9)
    
    # Draw vertical line for L_min = 1.0 ms
    plt.axvline(x=1.0, color='gray', linestyle=':', linewidth=1.5, label='L_min = 1.0 ms')
    
    plt.tight_layout()
    os.makedirs('report/Images', exist_ok=True)
    filename = 'figures/fig_arms_pdf.pdf'
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved PDF distribution plot to: {filename}")

if __name__ == '__main__':
    plot_arms_distribution()
