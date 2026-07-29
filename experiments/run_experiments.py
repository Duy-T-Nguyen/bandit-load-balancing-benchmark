"""
run_experiments.py — Main evaluation (rewritten per Amendment 1).

Protocol changes vs. the original (see evaluate-instruction.md, Amendment 1):
  * A1: adaptation measured by opt_rate(W) and t_50 (ground-truth-anchored,
    shared absolute threshold). Legacy tau_adapt still computed for the
    appendix comparison.
  * A2: both exploration_rate (non-greedy vs. own empirical best; legacy
    metric renamed) and non_optimal_rate (vs. ground-truth a*(t)) reported.
  * A3: reward normalization is now online/self-calibrated inside the
    simulator (no oracle constants).
  * A4: four tuned algorithms (EpsilonGreedy c, ThompsonSampling sigma_0,
    SW-UCB tau, D-UCB gamma) each evaluated in Tuned and Default variants.
  * A6: per-instance aggregates stored to enable paired statistics (n=5
    independent instances; the 150 trajectories are NOT independent).
Seed families unchanged: instance_seed = SEED+5+inst, run_seed = SEED+500+inst*100+run.
"""
import os
import json
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.env import LatencyEnvironment
from src.policies import (RoundRobin, LeastConnections, EpsilonGreedy, UCB,
                          ThompsonSampling, SlidingWindowUCB, DiscountedUCB)
from src.simulator import run_simulation
from src.metrics import (compute_regret, compute_cumulative_reward,
                         compute_average_latency, compute_adaptation_speed,
                         compute_opt_rate, compute_time_to_majority,
                         compute_non_optimal_rate)

K = 10
T = 10_000
L_MIN = 1.0
RUNS = 30
SEED = 42
N_EVAL_INSTANCES = 5
NUM_WORKERS = 4

OPT_RATE_WINDOWS = (100, 500, 1000)   # Amendment A1
T50_W = 100
T50_LEVEL = 0.5

# Theoretical/legacy defaults for the Default variants (Amendment A4)
DEFAULTS = {
    'EpsilonGreedy': {'c': 0.1, 'd': 1.0},
    'ThompsonSampling': {'sigma_0': 1.0},
    'SW-UCB': {'tau': 200},
    'D-UCB': {'gamma': 0.99},
}

METHODS = [
    'RoundRobin', 'LeastConnections',
    'EpsilonGreedy', 'EpsilonGreedy-Default',
    'UCB',
    'ThompsonSampling', 'ThompsonSampling-Default',
    'SW-UCB', 'SW-UCB-Default',
    'D-UCB', 'D-UCB-Default',
]

SCENARIOS = ['stationary', 'gradual_drift', 'abrupt_drift']


def build_policy(method, best_params):
    if method == 'RoundRobin':
        return RoundRobin(K=K)
    if method == 'LeastConnections':
        return LeastConnections(K=K)
    if method == 'EpsilonGreedy':
        p = best_params['EpsilonGreedy']
        return EpsilonGreedy(K=K, c=p['c'], d=p.get('d', 1.0))
    if method == 'EpsilonGreedy-Default':
        return EpsilonGreedy(K=K, **DEFAULTS['EpsilonGreedy'])
    if method == 'UCB':
        return UCB(K=K)
    if method == 'ThompsonSampling':
        return ThompsonSampling(K=K, sigma_0=best_params['ThompsonSampling']['sigma_0'])
    if method == 'ThompsonSampling-Default':
        return ThompsonSampling(K=K, **DEFAULTS['ThompsonSampling'])
    if method == 'SW-UCB':
        return SlidingWindowUCB(K=K, tau=best_params['SW-UCB']['tau'], T=T)
    if method == 'SW-UCB-Default':
        return SlidingWindowUCB(K=K, tau=DEFAULTS['SW-UCB']['tau'], T=T)
    if method == 'D-UCB':
        return DiscountedUCB(K=K, gamma=best_params['D-UCB']['gamma'])
    if method == 'D-UCB-Default':
        return DiscountedUCB(K=K, **DEFAULTS['D-UCB'])
    raise ValueError(f"Unknown method: {method}")


def run_single_eval_task(args):
    scenario, instance_seed, run_seed, method, best_params = args
    env = LatencyEnvironment(scenario=scenario, K=K, T=T, L_min=L_MIN,
                             instance_seed=instance_seed)
    policy = build_policy(method, best_params)
    sim = run_simulation(env, policy, T, run_seed)

    regret_curve = compute_regret(env.optimal_rewards, sim['rewards'])
    cum_reward_curve = compute_cumulative_reward(sim['rewards'])
    avg_lat_curve = compute_average_latency(sim['latencies'])
    exploration_rate = float(np.mean(sim['non_greedy_choices']))          # A2 (legacy, renamed)
    non_optimal_rate = compute_non_optimal_rate(sim['arms'], env.optimal_arms)  # A2 (new)
    inst_regret = env.optimal_rewards - sim['rewards']

    tau_adapt_legacy = None
    opt_rate = None
    t50 = None
    if scenario == 'abrupt_drift':
        t_b = T // 2
        tau_adapt_legacy = compute_adaptation_speed(inst_regret, t_b)      # appendix only
        opt_rate = compute_opt_rate(sim['arms'], env.optimal_arms, t_b,
                                    windows=OPT_RATE_WINDOWS)              # A1
        t50 = compute_time_to_majority(sim['arms'], env.optimal_arms, t_b,
                                       w=T50_W, level=T50_LEVEL)           # A1

    return (regret_curve, cum_reward_curve, avg_lat_curve, exploration_rate,
            non_optimal_rate, inst_regret, tau_adapt_legacy, opt_rate, t50)


def aggregate(results_list, indices):
    """Aggregate one (scenario, method) cell. Task order is instance-major
    (inst 0: runs 0..29, inst 1: runs 30..59, ...) — used for per-instance stats."""
    regrets, cum_rewards, avg_lats = [], [], []
    expl_rates, non_opt_rates, inst_regrets = [], [], []
    tau_legacy, opt_rates, t50_deltas, t50_censored = [], [], [], []

    for idx in indices:
        (rc, cr, al, er, nor, ir, tal, orate, t50) = results_list[idx]
        regrets.append(rc); cum_rewards.append(cr); avg_lats.append(al)
        expl_rates.append(er); non_opt_rates.append(nor); inst_regrets.append(ir)
        if tal is not None:
            tau_legacy.append(tal)
        if orate is not None:
            opt_rates.append(orate)
        if t50 is not None:
            t50_deltas.append(t50[0]); t50_censored.append(t50[1])

    regrets = np.array(regrets)
    cum_rewards = np.array(cum_rewards)
    avg_lats = np.array(avg_lats)
    inst_regrets = np.array(inst_regrets)
    expl_rates = np.array(expl_rates)
    non_opt_rates = np.array(non_opt_rates)

    n = len(indices)
    n_inst = n // RUNS
    final_regret = regrets[:, -1]

    out = {
        # legacy keys (kept so existing plot scripts keep working)
        'regret_mean': np.mean(regrets, axis=0),
        'regret_std': np.std(regrets, axis=0),
        'cum_reward_mean': np.mean(cum_rewards, axis=0),
        'avg_lat_mean': np.mean(avg_lats, axis=0),
        'non_greedy_rate_mean': float(np.mean(expl_rates)),
        'non_greedy_rate_std': float(np.std(expl_rates)),
        'inst_regret_mean': np.mean(inst_regrets, axis=0),
        # A2
        'exploration_rate_mean': float(np.mean(expl_rates)),
        'exploration_rate_std': float(np.std(expl_rates)),
        'non_optimal_rate_mean': float(np.mean(non_opt_rates)),
        'non_optimal_rate_std': float(np.std(non_opt_rates)),
        # A6: per-instance aggregates (independent units, n=5)
        'per_instance': {
            'final_regret': [float(np.mean(final_regret[i * RUNS:(i + 1) * RUNS]))
                             for i in range(n_inst)],
            'non_optimal_rate': [float(np.mean(non_opt_rates[i * RUNS:(i + 1) * RUNS]))
                                 for i in range(n_inst)],
            'exploration_rate': [float(np.mean(expl_rates[i * RUNS:(i + 1) * RUNS]))
                                 for i in range(n_inst)],
        },
    }

    if tau_legacy:
        out['tau_adapt_mean'] = float(np.mean(tau_legacy))   # legacy, appendix only
        out['tau_adapt_std'] = float(np.std(tau_legacy))
    if opt_rates:
        for W in OPT_RATE_WINDOWS:
            vals = np.array([o[W] for o in opt_rates])
            out[f'opt_rate_{W}_mean'] = float(np.mean(vals))
            out[f'opt_rate_{W}_std'] = float(np.std(vals))
            out['per_instance'][f'opt_rate_{W}'] = [
                float(np.mean(vals[i * RUNS:(i + 1) * RUNS])) for i in range(n_inst)]
    if t50_deltas:
        deltas = np.array(t50_deltas, dtype=float)
        censored = np.array(t50_censored, dtype=bool)
        out['t50_median'] = float(np.median(deltas))
        out['t50_q25'] = float(np.percentile(deltas, 25))
        out['t50_q75'] = float(np.percentile(deltas, 75))
        out['t50_censored_pct'] = float(np.mean(censored) * 100.0)
        out['t50_values'] = deltas.tolist()
        out['t50_censored_flags'] = censored.tolist()

    return out


def run_evaluation():
    print("Final Evaluation Protocol (Amendment 1: new metrics, online "
          "normalization, 4 tuned algorithms + Default variants)")

    best_params_path = 'results/best_params.json'
    if os.path.exists(best_params_path):
        with open(best_params_path) as f:
            best_params = json.load(f)
        print("Loaded tuned parameters: " + ", ".join(
            f"{a}={ {k: v for k, v in best_params[a].items()} }"
            for a in ['EpsilonGreedy', 'ThompsonSampling', 'SW-UCB', 'D-UCB']
            if a in best_params))
    else:
        best_params = dict(DEFAULTS)
        print(f"Warning: {best_params_path} not found. Using defaults for all.")

    tasks = []
    task_indices = {}
    for scenario in SCENARIOS:
        task_indices[scenario] = {}
        for method in METHODS:
            task_indices[scenario][method] = []
            for inst in range(N_EVAL_INSTANCES):
                for run in range(RUNS):
                    run_seed = SEED + 500 + inst * 100 + run
                    tasks.append((scenario, SEED + 5 + inst, run_seed, method, best_params))
                    task_indices[scenario][method].append(len(tasks) - 1)

    print(f"Total evaluation tasks: {len(tasks)} on {NUM_WORKERS} workers...")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        all_results = list(executor.map(run_single_eval_task, tasks, chunksize=8))

    results = {}
    for scenario in SCENARIOS:
        results[scenario] = {}
        for method in METHODS:
            results[scenario][method] = aggregate(all_results, task_indices[scenario][method])

    with open('results/results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("\nEvaluation complete. Saved experiment/results.pkl")

    # Human-readable summary
    summary = {}
    for scenario in SCENARIOS:
        summary[scenario] = {}
        for method in METHODS:
            d = results[scenario][method]
            s = {
                'final_cumulative_regret_mean': float(d['regret_mean'][-1]),
                'final_cumulative_regret_std': float(d['regret_std'][-1]),
                'final_cumulative_reward_mean': float(d['cum_reward_mean'][-1]),
                'average_latency_ms_mean': float(d['avg_lat_mean'][-1]),
                'exploration_rate_mean': d['exploration_rate_mean'],
                'exploration_rate_std': d['exploration_rate_std'],
                'non_optimal_rate_mean': d['non_optimal_rate_mean'],
                'non_optimal_rate_std': d['non_optimal_rate_std'],
                'per_instance_final_regret': d['per_instance']['final_regret'],
            }
            if 'tau_adapt_mean' in d:
                s['legacy_tau_adapt_mean'] = d['tau_adapt_mean']
                s['legacy_tau_adapt_std'] = d['tau_adapt_std']
            for W in OPT_RATE_WINDOWS:
                if f'opt_rate_{W}_mean' in d:
                    s[f'opt_rate_{W}_mean'] = d[f'opt_rate_{W}_mean']
                    s[f'opt_rate_{W}_std'] = d[f'opt_rate_{W}_std']
            if 't50_median' in d:
                s['t50_median'] = d['t50_median']
                s['t50_iqr'] = [d['t50_q25'], d['t50_q75']]
                s['t50_censored_pct'] = d['t50_censored_pct']
            summary[scenario][method] = s

    with open('results/results_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
    print("Saved experiment/results_summary.json")

    # ── Amendment A1 mandatory sanity check ─────────────────────────────────
    rr = results['abrupt_drift']['RoundRobin']
    orate = rr.get('opt_rate_1000_mean')
    cens = rr.get('t50_censored_pct')
    print("\nSanity check (Amendment A1): RoundRobin on abrupt_drift")
    print(f"  opt_rate@1000 = {orate:.4f} (expected ~ 1/K = {1.0 / K:.2f})")
    print(f"  t50 censored  = {cens:.1f}% (expected ~ 100%)")
    ok = orate is not None and abs(orate - 1.0 / K) < 0.05 and cens is not None and cens > 95.0
    print("  => PASS" if ok else "  => FAIL — DO NOT publish these numbers; investigate!")


if __name__ == '__main__':
    run_evaluation()
