"""
plot_drift_sweep.py — Figure for the drift-frequency sweep (Amendment A5).
Reads experiment/results_drift_sweep.pkl, emits fig4_8_drift_sweep.pdf:
final dynamic regret vs. stable-phase length P (log x-axis), per-instance SEM bands.
"""
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'no-latex'])
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.6

METHODS = ['UCB', 'ThompsonSampling', 'SW-UCB-Default', 'D-UCB-Default', 'D-UCB']
LABELS = {
    'UCB': 'UCB (≡ SW-UCB tuned)',
    'ThompsonSampling': r'TS (tuned, $\sigma_0{=}0{.}25$)',
    'SW-UCB-Default': r'SW-UCB ($\tau{=}200$)',
    'D-UCB-Default': r'D-UCB ($\gamma{=}0{.}99$)',
    'D-UCB': r'D-UCB ($\gamma{=}1{.}0$)',
}
COLORS = {
    'UCB': '#3f8efc',
    'ThompsonSampling': '#2ec4b6',
    'SW-UCB-Default': '#ff477e',
    'D-UCB-Default': '#7209b7',
    'D-UCB': '#b5179e',
}


def main():
    path = 'results/results_drift_sweep.pkl'
    if not os.path.exists(path):
        print(f"Error: {path} not found. Run run_drift_frequency_sweep.py first.")
        return
    with open(path, 'rb') as f:
        R = pickle.load(f)
    periods = sorted(R.keys())

    plt.figure(figsize=(10, 6))
    for m in METHODS:
        means, sems = [], []
        for P in periods:
            per_inst = np.array(R[P][m]['per_instance']['final_regret'])
            means.append(per_inst.mean())
            sems.append(per_inst.std(ddof=1) / np.sqrt(len(per_inst)))
        means, sems = np.array(means), np.array(sems)
        ls = '--' if m.endswith('Default') else '-'
        plt.plot(periods, means, marker='o', label=LABELS[m], color=COLORS[m],
                 linewidth=2.0, linestyle=ls)
        plt.fill_between(periods, means - sems, means + sems, color=COLORS[m], alpha=0.15)

    plt.xscale('log')
    plt.xticks(periods, [str(p) for p in periods])
    plt.xlabel('Stable-phase length $P$ (rounds, log scale) — drift denser to the left', fontsize=12)
    plt.ylabel('Final cumulative dynamic regret', fontsize=12)
    plt.title('Drift-frequency sweep (dense\\_abrupt): no crossover appears\n'
              'for the forgetting mechanisms across the whole range', fontsize=13, pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    os.makedirs('report/Images', exist_ok=True)
    plt.savefig('figures/fig4_8_drift_sweep.pdf', dpi=300)
    plt.close()
    print("Saved: figures/fig4_8_drift_sweep.pdf")


if __name__ == '__main__':
    main()
