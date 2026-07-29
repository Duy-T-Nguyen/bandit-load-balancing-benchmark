"""
plot.py — generates all Chapter-4 figures/tables from results.pkl (Amendment 1).

Figure conventions after Amendment 1:
  * Line plots show the 7-method comparison where SW-UCB / D-UCB appear as their
    DEFAULT configurations (tau=200, gamma=0.99): the tuned configurations hit the
    UCB anchor (tau=T, gamma=1.0) and coincide with the UCB curve, so plotting
    them would just overdraw UCB (stated in the captions).
  * EpsilonGreedy / ThompsonSampling appear as their TUNED configurations
    (c=2, sigma_0=0.25); their Default variants are covered in the tuned-vs-default
    table.
  * fig4_5_optrate (new): rolling ground-truth optimal-arm rate around t_b —
    the primary visual for the Amendment-A1 adaptation metrics.
"""
import os
import json
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'no-latex'])
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

SCENARIOS = ['stationary', 'gradual_drift', 'abrupt_drift']

# Methods shown in line plots (see module docstring for the rationale)
PLOT_METHODS = ['RoundRobin', 'LeastConnections', 'EpsilonGreedy', 'UCB',
                'ThompsonSampling', 'SW-UCB-Default', 'D-UCB-Default']

LABELS = {
    'RoundRobin': 'Round Robin',
    'LeastConnections': 'Least Connections',
    'EpsilonGreedy': r'$\epsilon$-greedy (tuned, $c{=}2$)',
    'EpsilonGreedy-Default': r'$\epsilon$-greedy (default, $c{=}0{.}1$)',
    'UCB': 'UCB',
    'ThompsonSampling': r'TS (tuned, $\sigma_0{=}0{.}25$)',
    'ThompsonSampling-Default': r'TS (default, $\sigma_0{=}1$)',
    'SW-UCB': r'SW-UCB (tuned, $\tau{=}10^4$)',
    'SW-UCB-Default': r'SW-UCB ($\tau{=}200$)',
    'D-UCB': r'D-UCB (tuned, $\gamma{=}1$)',
    'D-UCB-Default': r'D-UCB ($\gamma{=}0{.}99$)',
}

COLORS = {
    'RoundRobin': '#9ea8b5',
    'LeastConnections': '#7d8a99',
    'EpsilonGreedy': '#e09f3e',
    'UCB': '#3f8efc',
    'ThompsonSampling': '#2ec4b6',
    'SW-UCB': '#ff477e',
    'SW-UCB-Default': '#ff477e',
    'D-UCB': '#7209b7',
    'D-UCB-Default': '#7209b7',
}


def generate_plots_and_tables():
    print("Generating Plots and Tables...")
    results_path = 'results/results.pkl'
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found. Please run run_experiments.py first.")
        return
    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    os.makedirs('report/Images', exist_ok=True)

    # --- Figures 4.1-4.3: cumulative regret curves ---
    for scenario in SCENARIOS:
        plt.figure(figsize=(10, 6))
        for method in PLOT_METHODS:
            data = results[scenario][method]
            mean = data['regret_mean']
            sem = data['regret_std'] / np.sqrt(150.0)
            t = np.arange(len(mean))
            plt.plot(t, mean, label=LABELS[method], color=COLORS[method], linewidth=2.0)
            plt.fill_between(t, mean - sem, mean + sem, color=COLORS[method], alpha=0.15)
        plt.xlabel('Decision round (t)', fontsize=12)
        plt.ylabel('Cumulative regret', fontsize=12)
        plt.title(f'Cumulative dynamic regret — {scenario.replace("_", " ").title()} scenario',
                  fontsize=14, pad=15)
        plt.grid(True, linestyle='--', alpha=0.5)
        if scenario == 'abrupt_drift':
            plt.axvline(x=5000, color='red', linestyle='--', linewidth=1.5,
                        label='Breakpoint ($t_b$)')
        plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
        plt.tight_layout()
        filename = f'figures/fig4_{SCENARIOS.index(scenario)+1}_{scenario}_regret.pdf'
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved: {filename}")

    # --- Figure 4.4: smoothed instantaneous regret around the breakpoint ---
    plt.figure(figsize=(10, 6))
    t_b = 5000
    for method in ['UCB', 'ThompsonSampling', 'SW-UCB-Default', 'D-UCB-Default']:
        inst_mean = results['abrupt_drift'][method]['inst_regret_mean']
        w = 50
        smooth = np.convolve(inst_mean, np.ones(w) / w, mode='same')
        rng_plot = np.arange(t_b - 200, t_b + 1500)
        ls = '--' if method.endswith('Default') else '-'
        plt.plot(rng_plot, smooth[rng_plot], label=LABELS[method],
                 color=COLORS[method], linewidth=2.0, linestyle=ls)
    plt.axvline(x=t_b, color='red', linestyle='--', linewidth=1.5, label='Breakpoint ($t_b$)')
    plt.xlabel('Decision round (t)', fontsize=12)
    plt.ylabel('Instantaneous regret (smoothed, w=50)', fontsize=12)
    plt.title('Instantaneous regret around the breakpoint ($t_b = 5000$)', fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('figures/fig4_4_abrupt_zoom.pdf', dpi=300)
    plt.close()
    print("Saved: figures/fig4_4_abrupt_zoom.pdf")

    # --- Figure 4.6: exploration rate bar chart (metric renamed per A2) ---
    plt.figure(figsize=(10, 6))
    bar_methods = ['EpsilonGreedy', 'UCB', 'ThompsonSampling', 'SW-UCB-Default', 'D-UCB-Default']
    rates = [results['stationary'][m]['exploration_rate_mean'] for m in bar_methods]
    sems = [results['stationary'][m]['exploration_rate_std'] / np.sqrt(150.0) for m in bar_methods]
    plt.bar([LABELS[m] for m in bar_methods], rates, yerr=sems,
            color=[COLORS[m] for m in bar_methods], capsize=5, alpha=0.85,
            edgecolor='black', width=0.5)
    plt.ylabel('Empirical exploration rate', fontsize=12)
    plt.title('Empirical exploration rate, stationary environment', fontsize=14, pad=15)
    plt.xticks(fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('figures/fig4_6_non_greedy_rate.pdf', dpi=300)
    plt.close()
    print("Saved: figures/fig4_6_non_greedy_rate.pdf")

    # --- Table 4.4: main regret table (all 11 methods, grouped) ---
    order = ['RoundRobin', 'LeastConnections',
             'EpsilonGreedy', 'EpsilonGreedy-Default', 'UCB',
             'ThompsonSampling', 'ThompsonSampling-Default',
             'SW-UCB', 'SW-UCB-Default', 'D-UCB', 'D-UCB-Default']
    tex_names = {
        'RoundRobin': 'Round Robin', 'LeastConnections': 'Least Connections',
        'EpsilonGreedy': r'$\epsilon$-greedy (tuned, $c=2$)',
        'EpsilonGreedy-Default': r'$\epsilon$-greedy (default, $c=0{.}1$)',
        'UCB': 'UCB',
        'ThompsonSampling': r'TS (tuned, $\sigma_0=0{.}25$)',
        'ThompsonSampling-Default': r'TS (default, $\sigma_0=1$)',
        'SW-UCB': r'SW-UCB (tuned, $\tau=10000$)$^{\dagger}$',
        'SW-UCB-Default': r'SW-UCB (default, $\tau=200$)',
        'D-UCB': r'D-UCB (tuned, $\gamma=1{.}0$)$^{\dagger}$',
        'D-UCB-Default': r'D-UCB (default, $\gamma=0{.}99$)',
    }
    s = r"""\begin{table}[htbp]
\centering
\small
\caption{Final cumulative dynamic regret ($T = 10.000$), mean $\pm$ std over $N=150$ trajectories (5 instances $\times$ 30 runs). $^{\dagger}$The optimal tuned values for SW-UCB and D-UCB land exactly on the UCB anchor ($\tau = T$, $\gamma = 1$), so both configurations degenerate to UCB.}
\label{tab:regret_comparison_final}
\begin{tabular}{lccc}
\toprule
\textbf{Algorithm} & \textbf{Stationary} & \textbf{Gradual Drift} & \textbf{Abrupt Drift} \\ \midrule
"""
    for i, method in enumerate(order):
        row = tex_names[method]
        for sc in SCENARIOS:
            row += f" & {results[sc][method]['regret_mean'][-1]:.2f} $\\pm$ {results[sc][method]['regret_std'][-1]:.2f}"
        s += row + " \\\\\n"
        if method in ('LeastConnections', 'UCB', 'ThompsonSampling-Default'):
            s += "\\addlinespace\n"
    s += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    with open('results/table_4_4.txt', 'w', encoding='utf-8') as f:
        f.write(s)
    print("Saved: experiment/table_4_4.txt")

    # --- Table 4.5: NEW adaptation metrics (Amendment A1) ---
    s = r"""\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{Post-breakpoint adaptation ($t_b = 5000$, abrupt-drift scenario) under the ground-truth-anchored metric: share of rounds on the new optimal arm $\text{opt}(W)$, and time to majority $t_{50}$ (median [IQR]; runs that never reach it are censored at the maximum). Round Robin is the control: $\text{opt}(W) \approx 1/K = 0{.}10$.}
\label{tab:adaptation_speed}
\begin{tabular}{lccccc}
\toprule
\textbf{Algorithm} & \textbf{opt(100)} & \textbf{opt(500)} & \textbf{opt(1000)} & \textbf{$t_{50}$ median [IQR]} & \textbf{\% censored} \\ \midrule
"""
    for method in ['UCB', 'ThompsonSampling', 'EpsilonGreedy',
                   'SW-UCB-Default', 'D-UCB-Default', 'D-UCB',
                   'LeastConnections', 'RoundRobin']:
        d = results['abrupt_drift'][method]
        if 'opt_rate_100_mean' not in d:
            continue
        s += (f"{tex_names[method]} & {d['opt_rate_100_mean']:.3f} & {d['opt_rate_500_mean']:.3f} & "
              f"{d['opt_rate_1000_mean']:.3f} & {d['t50_median']:.0f} [{d['t50_q25']:.0f}; {d['t50_q75']:.0f}] & "
              f"{d['t50_censored_pct']:.1f}\\% \\\\\n")
    s += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    with open('results/table_4_5.txt', 'w', encoding='utf-8') as f:
        f.write(s)
    print("Saved: experiment/table_4_5.txt")

    # --- Table 4.6: tuned vs default for the FOUR tuned algorithms ---
    s = r"""\begin{table}[htbp]
\centering
\small
\caption{Final cumulative dynamic regret, default configuration versus tuned configuration (Amendment A4: 4 algorithms, an equal budget of 8 grid points each).}
\label{tab:tuning_comparison}
\begin{tabular}{llccc}
\toprule
\textbf{Algorithm} & \textbf{Configuration} & \textbf{Stationary} & \textbf{Gradual Drift} & \textbf{Abrupt Drift} \\ \midrule
"""
    pairs = [('EpsilonGreedy', r'$\epsilon$-greedy', '$c=0{.}1$', '$c=2$'),
             ('ThompsonSampling', 'TS', r'$\sigma_0=1$', r'$\sigma_0=0{.}25$'),
             ('SW-UCB', 'SW-UCB', r'$\tau=200$', r'$\tau=10000$'),
             ('D-UCB', 'D-UCB', r'$\gamma=0{.}99$', r'$\gamma=1{.}0$')]
    for i, (key, name, dcfg, tcfg) in enumerate(pairs):
        for cfg_label, mkey in [(f'Default ({dcfg})', f'{key}-Default'), (f'Tuned ({tcfg})', key)]:
            row = f"{name} & {cfg_label}"
            for sc in SCENARIOS:
                row += f" & {results[sc][mkey]['regret_mean'][-1]:.2f} $\\pm$ {results[sc][mkey]['regret_std'][-1]:.2f}"
            s += row + " \\\\\n"
        if i < len(pairs) - 1:
            s += "\\midrule\n"
    s += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    with open('results/table_4_6.txt', 'w', encoding='utf-8') as f:
        f.write(s)
    print("Saved: experiment/table_4_6.txt")

    # --- Table 4.7: tuning grid results (from best_params.json) ---
    with open('results/best_params.json') as f:
        bp = json.load(f)
    grid = bp['grid_results']
    algo_meta = [('EpsilonGreedy', r'$\epsilon$-greedy — $c$'),
                 ('ThompsonSampling', r'TS — $\sigma_0$'),
                 ('SW-UCB', r'SW-UCB — $\tau$'),
                 ('D-UCB', r'D-UCB — $\gamma$')]
    s = r"""\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{Grid search results (Amendment A4): mean regret over the 3 main scenarios $\times$ 5 tuning instances $\times$ 30 runs. Bold marks the optimum. The final grid points of SW-UCB ($\tau = T$) and D-UCB ($\gamma = 1$) are the anchors that degenerate to UCB.}
\label{tab:hyperparam_sensitivity}
\begin{tabular}{cc|cc|cc|cc}
\toprule
\multicolumn{2}{c|}{\textbf{$\epsilon$-greedy ($c$)}} & \multicolumn{2}{c|}{\textbf{TS ($\sigma_0$)}} & \multicolumn{2}{c|}{\textbf{SW-UCB ($\tau$)}} & \multicolumn{2}{c}{\textbf{D-UCB ($\gamma$)}} \\
Value & Regret & Value & Regret & Value & Regret & Value & Regret \\ \midrule
"""
    cols = []
    for key, _ in algo_meta:
        vals = list(grid[key].items())
        best_v = min(vals, key=lambda kv: kv[1]['objective_mean_regret'])[0]
        col = []
        for v, r in vals:
            cell_v, cell_r = v, f"{r['objective_mean_regret']:.1f}"
            if v == best_v:
                cell_v, cell_r = rf"\textbf{{{v}}}", rf"\textbf{{{cell_r}}}"
            col.append((cell_v, cell_r))
        cols.append(col)
    for i in range(8):
        s += " & ".join(f"{cols[j][i][0]} & {cols[j][i][1]}" for j in range(4)) + " \\\\\n"
    s += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    with open('results/table_4_7.txt', 'w', encoding='utf-8') as f:
        f.write(s)
    print("Saved: experiment/table_4_7.txt")

    # --- Appendix table: legacy tau_adapt (kept for transparency, A1) ---
    s = r"""\begin{table}[htbp]
\centering
\small
\caption{[Appendix] The OLD adaptation metric $\tau_{\text{adapt}}$ (self-referential threshold $\eta = 1{.}1\rho_{\min}$), kept only for transparency. It was replaced because it rewards policies with high baseline regret: Round Robin ``adapts'' in $\approx 2$ rounds while learning nothing.}
\label{tab:legacy_tau_adapt}
\begin{tabular}{lc}
\toprule
\textbf{Algorithm} & \textbf{old $\tau_{\text{adapt}}$ (mean $\pm$ std)} \\ \midrule
"""
    for method in order:
        d = results['abrupt_drift'][method]
        if 'tau_adapt_mean' in d:
            s += f"{tex_names[method]} & {d['tau_adapt_mean']:.1f} $\\pm$ {d['tau_adapt_std']:.1f} \\\\\n"
    s += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    with open('results/table_appendix_legacy_tau.txt', 'w', encoding='utf-8') as f:
        f.write(s)
    print("Saved: experiment/table_appendix_legacy_tau.txt")


# ── Figure 4.5: rolling optimal-arm rate around t_b (recomputed trajectories) ──
def _optrate_worker(args):
    import sys
    from src.env import LatencyEnvironment
    from src.policies import (EpsilonGreedy, UCB, ThompsonSampling,
                              SlidingWindowUCB, DiscountedUCB)
    from src.simulator import run_simulation
    method, instance_seed, run_seed, params = args
    env = LatencyEnvironment('abrupt_drift', K=10, T=10000, L_min=1.0,
                             instance_seed=instance_seed)
    if method == 'UCB':
        policy = UCB(K=10)
    elif method == 'ThompsonSampling':
        policy = ThompsonSampling(K=10, sigma_0=params['ThompsonSampling']['sigma_0'])
    elif method == 'EpsilonGreedy':
        policy = EpsilonGreedy(K=10, c=params['EpsilonGreedy']['c'], d=1.0)
    elif method == 'SW-UCB-Default':
        policy = SlidingWindowUCB(K=10, tau=200, T=10000)
    elif method == 'D-UCB-Default':
        policy = DiscountedUCB(K=10, gamma=0.99)
    sim = run_simulation(env, policy, 10000, run_seed)
    return (sim['arms'] == env.optimal_arms).astype(float)


def plot_opt_rate_curve():
    print("Generating Figure 4.5 (rolling optimal-arm rate) — recomputing trajectories...")
    with open('results/best_params.json') as f:
        params = json.load(f)
    methods = ['UCB', 'ThompsonSampling', 'EpsilonGreedy', 'SW-UCB-Default', 'D-UCB-Default']
    SEED, RUNS, N_EVAL = 42, 30, 5
    tasks, idx = [], {}
    for m in methods:
        idx[m] = []
        for inst in range(N_EVAL):
            for run in range(RUNS):
                tasks.append((m, SEED + 5 + inst, SEED + 500 + inst * 100 + run, params))
                idx[m].append(len(tasks) - 1)
    with ProcessPoolExecutor(max_workers=4) as ex:
        flat = list(ex.map(_optrate_worker, tasks, chunksize=8))

    t_b, w = 5000, 100
    kernel = np.ones(w) / w
    plt.figure(figsize=(10, 6))
    for m in methods:
        correct = np.mean(np.array([flat[i] for i in idx[m]]), axis=0)   # (T,)
        rolling = np.convolve(correct, kernel, mode='same')
        rng_plot = np.arange(t_b - 500, t_b + 2500)
        ls = '--' if m.endswith('Default') else '-'
        plt.plot(rng_plot, rolling[rng_plot], label=LABELS[m],
                 color=COLORS[m], linewidth=2.0, linestyle=ls)
    plt.axvline(x=t_b, color='red', linestyle='--', linewidth=1.5, label='Breakpoint ($t_b$)')
    plt.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.2, label='Majority threshold (50\\%)')
    plt.axhline(y=0.1, color='gray', linestyle='-.', linewidth=1.0, alpha=0.7,
                label='Random baseline ($1/K$)')
    plt.ylim(-0.02, 1.0)
    plt.xlabel('Decision round (t)', fontsize=12)
    plt.ylabel('Optimal-arm selection rate (rolling, w=100)', fontsize=12)
    plt.title('Ground-truth optimal-arm selection rate around the breakpoint', fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='center right', fontsize=8)
    plt.tight_layout()
    plt.savefig('figures/fig4_5_optrate.pdf', dpi=300)
    plt.close()
    print("Saved: figures/fig4_5_optrate.pdf")


def plot_environment_dynamics():
    from src.env import LatencyEnvironment
    print("Generating Environment Dynamics Illustration...")
    K, T, L_MIN, SEED = 10, 10000, 1.0, 42
    arms_to_plot = [0, 4, 9]
    colors = {0: '#3f8efc', 4: '#2ec4b6', 9: '#ff477e'}
    titles = {
        'stationary': 'Stationary scenario',
        'gradual_drift': 'Gradual-drift scenario',
        'abrupt_drift': 'Abrupt-drift scenario'
    }
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for i, scenario in enumerate(SCENARIOS):
        env = LatencyEnvironment(scenario=scenario, K=K, T=T, L_min=L_MIN, instance_seed=SEED)
        t = np.arange(T)
        for arm in arms_to_plot:
            axes[i].plot(t, env.k * env.theta[arm, :], label=f'Arm {arm}',
                         color=colors[arm], linewidth=2.0)
        axes[i].set_ylabel('Expected latency (ms)', fontsize=11)
        axes[i].set_title(titles[scenario], fontsize=12, fontweight='bold')
        axes[i].grid(linestyle='--', alpha=0.5)
        if i == 0:
            axes[i].legend(loc='upper right')
    axes[-1].set_xlabel('Decision round (t)', fontsize=11)
    plt.tight_layout()
    os.makedirs('report/Images', exist_ok=True)
    plt.savefig('figures/fig4_0_env_dynamics.pdf', dpi=300)
    plt.close()
    print("Saved: figures/fig4_0_env_dynamics.pdf")


if __name__ == '__main__':
    plot_environment_dynamics()
    generate_plots_and_tables()
    plot_opt_rate_curve()
