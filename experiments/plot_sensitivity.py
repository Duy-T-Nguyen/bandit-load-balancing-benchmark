"""
plot_sensitivity.py
===================
Generates cumulative regret plots for the extended sensitivity scenarios:
  1. multi_abrupt
  2. periodic_drift
Reads results from experiment/results.pkl.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

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



def generate_sensitivity_plots():
    print("Generating Sensitivity Analysis Plots...")
    results_path = 'results/results.pkl'
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found. Please run run_sensitivity_experiments.py first.")
        return
        
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
        
    os.makedirs('report/Images', exist_ok=True)
    
    scenarios = ['multi_abrupt', 'periodic_drift']
    methods = ['RoundRobin', 'LeastConnections', 'EpsilonGreedy', 'UCB', 'ThompsonSampling', 'SW-UCB', 'D-UCB']
    
    colors = {
        'RoundRobin': '#9ea8b5',
        'LeastConnections': '#7d8a99',
        'EpsilonGreedy': '#e09f3e',
        'UCB': '#3f8efc',
        'ThompsonSampling': '#2ec4b6',
        'SW-UCB': '#ff477e',
        'D-UCB': '#7209b7'
    }
    
    titles = {
        'multi_abrupt': 'Cumulative dynamic regret — multi-abrupt scenario',
        'periodic_drift': 'Cumulative dynamic regret — periodic-drift scenario'
    }
    
    for scenario in scenarios:
        plt.figure(figsize=(10, 6))
        
        for method in methods:
            if method not in results[scenario]:
                print(f"Warning: {method} not found in results for {scenario}")
                continue
            data = results[scenario][method]
            mean = data['regret_mean']
            std = data['regret_std']
            t = np.arange(len(mean))
            
            # SEM (Standard Error of the Mean) over N=150
            sem = std / np.sqrt(150.0)
            
            plt.plot(t, mean, label=method, color=colors[method], linewidth=2.0)
            plt.fill_between(t, mean - sem, mean + sem, color=colors[method], alpha=0.15)
            
        plt.xlabel('Decision round (t)', fontsize=12)
        plt.ylabel('Cumulative regret', fontsize=12)
        plt.title(titles[scenario], fontsize=14, pad=15)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        if scenario == 'multi_abrupt':
            # Add vertical lines at the 3 breakpoints
            for bp in [2500, 5000, 7500]:
                plt.axvline(x=bp, color='red', linestyle=':', linewidth=1.2)
            plt.axvline(x=2500, color='red', linestyle=':', linewidth=1.2, label='Breakpoints ($t_b$)')
            
        elif scenario == 'periodic_drift':
            # Add vertical lines at the peaks/valleys of the sine wave (every 1000 rounds)
            for bp in [1000, 3000, 5000, 7000, 9000]:
                plt.axvline(x=bp, color='gray', linestyle=':', linewidth=1.0, alpha=0.7)
            
        plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
        plt.tight_layout()
        
        filename = f'figures/fig_sensitivity_{scenario}_regret.pdf'
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved: {filename}")

if __name__ == '__main__':
    generate_sensitivity_plots()
