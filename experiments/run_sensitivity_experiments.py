"""
run_sensitivity_experiments.py
================================
Extended sensitivity analysis for 2 additional scenarios:
  1. multi_abrupt   — 3 breakpoints at t ∈ {2500, 5000, 7500}
  2. periodic_drift — sinusoidal drift with period = 2000 rounds

Mathematical rigor guarantees
------------------------------
* Same N_total = 150 trajectories (5 eval instances × 30 MC runs) as the
  main evaluation, ensuring identical statistical power.
* Same seed family as run_experiments.py:
    instance_seed  = SEED + 5 + inst  (inst = 0..4)
    run_seed       = SEED + 500 + inst*100 + run
  This is disjoint from the TUNING seed family (SEED + inst).
* Same tuned hyperparameters loaded from best_params.json.
* tau_adapt is computed at EVERY breakpoint for multi_abrupt, then
  summarised as mean across breakpoints per trajectory.
"""

import os
import json
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# ── local imports ─────────────────────────────────────────────────────────────
from src.env import LatencyEnvironment
from src.policies import (
    RoundRobin, LeastConnections, EpsilonGreedy,
    UCB, ThompsonSampling, SlidingWindowUCB, DiscountedUCB,
)
from src.simulator import run_simulation
from src.metrics import (
    compute_regret, compute_cumulative_reward,
    compute_average_latency, compute_adaptation_speed,
    compute_opt_rate, compute_time_to_majority, compute_non_optimal_rate,
)

# ── constants (must match run_experiments.py) ─────────────────────────────────
K            = 10
T            = 10_000
L_MIN        = 1.0
RUNS         = 30
SEED         = 42
N_EVAL       = 5          # evaluation instances
NUM_WORKERS  = 4

# Breakpoints for multi_abrupt (must match env.py implementation)
MULTI_ABRUPT_BREAKPOINTS = [2500, 5000, 7500]

# Periodic drift: half-period is the natural "breakpoint" distance
PERIODIC_PERIOD = 2000
# Peaks (local maxima → local minimum crossings) where optimal arm may change
# Conservatively: check every half-period
PERIODIC_BREAKPOINTS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]

SCENARIOS = ['multi_abrupt', 'periodic_drift']

METHODS = [
    'RoundRobin', 'LeastConnections', 'EpsilonGreedy',
    'UCB', 'ThompsonSampling',
    'SW-UCB', 'D-UCB',
]


# ── worker ────────────────────────────────────────────────────────────────────
def run_single_task(args):
    scenario, instance_seed, run_seed, method, best_params = args

    env = LatencyEnvironment(
        scenario=scenario, K=K, T=T, L_min=L_MIN,
        instance_seed=instance_seed,
    )

    # Build policy (tuned params per Amendment A4)
    if method == 'RoundRobin':
        policy = RoundRobin(K=K)
    elif method == 'LeastConnections':
        policy = LeastConnections(K=K)
    elif method == 'EpsilonGreedy':
        p = best_params.get('EpsilonGreedy', {'c': 0.1, 'd': 1.0})
        policy = EpsilonGreedy(K=K, c=p['c'], d=p.get('d', 1.0))
    elif method == 'UCB':
        policy = UCB(K=K)
    elif method == 'ThompsonSampling':
        p = best_params.get('ThompsonSampling', {'sigma_0': 1.0})
        policy = ThompsonSampling(K=K, sigma_0=p['sigma_0'])
    elif method == 'SW-UCB':
        policy = SlidingWindowUCB(K=K, tau=best_params['SW-UCB']['tau'], T=T)
    elif method == 'D-UCB':
        policy = DiscountedUCB(K=K, gamma=best_params['D-UCB']['gamma'])
    else:
        raise ValueError(f"Unknown method: {method}")

    sim          = run_simulation(env, policy, T, run_seed)
    inst_regret  = env.optimal_rewards - sim['rewards']   # shape (T,)

    regret_curve = compute_regret(env.optimal_rewards, sim['rewards'])
    cum_reward   = compute_cumulative_reward(sim['rewards'])
    avg_lat      = compute_average_latency(sim['latencies'])
    non_greedy   = float(np.mean(sim['non_greedy_choices']))

    non_optimal = compute_non_optimal_rate(sim['arms'], env.optimal_arms)  # A2

    # ── Adaptation metrics ─────────────────────────────────────────────────
    # Legacy tau_adapt (appendix only) + Amendment A1 metrics per breakpoint,
    # averaged per trajectory. For multi_abrupt each phase lasts 2500 rounds,
    # so opt_rate windows (<=1000) fit and t_50 search is capped at phase end.
    tau_adapt_values = []
    opt_rate_values = []          # list of dicts {W: rate}
    t50_values, t50_censored = [], []

    if scenario == 'multi_abrupt':
        breakpoints = MULTI_ABRUPT_BREAKPOINTS
        phase_len = 2500
    elif scenario == 'periodic_drift':
        # Only measure at half-periods where a rank reversal actually occurs
        # in this instance's environment profile.
        opt_arms = env.optimal_arms  # shape (T,)
        breakpoints = []
        for t_b in PERIODIC_BREAKPOINTS:
            if t_b > 0 and t_b + 100 < T:
                # Check if optimal arm changes around this point
                arm_before = opt_arms[max(0, t_b - 10)]
                arm_after  = opt_arms[min(T - 1, t_b + 10)]
                if arm_before != arm_after:
                    breakpoints.append(t_b)
        phase_len = PERIODIC_PERIOD // 2
    else:
        breakpoints = []
        phase_len = T

    for t_b in breakpoints:
        if t_b + 100 < T:   # need at least w=100 rounds after breakpoint
            tau = compute_adaptation_speed(inst_regret, t_b, w=100, eta_factor=1.1)
            tau_adapt_values.append(tau)
            # Amendment A1 metrics, capped at the end of the current phase
            t_end = min(t_b + phase_len, T)
            windows = tuple(w for w in (100, 500, 1000) if t_b + w <= t_end)
            if windows:
                opt_rate_values.append(
                    compute_opt_rate(sim['arms'], env.optimal_arms, t_b, windows=windows))
            d, cens = compute_time_to_majority(
                sim['arms'], env.optimal_arms, t_b, w=100, level=0.5, t_end=t_end)
            t50_values.append(d)
            t50_censored.append(cens)

    mean_tau_adapt = float(np.mean(tau_adapt_values)) if tau_adapt_values else None
    # Average the A1 metrics across breakpoints for this trajectory
    mean_opt_rate = None
    if opt_rate_values:
        keys = set.intersection(*(set(o.keys()) for o in opt_rate_values))
        mean_opt_rate = {W: float(np.mean([o[W] for o in opt_rate_values])) for W in keys}
    mean_t50 = float(np.mean(t50_values)) if t50_values else None
    frac_censored = float(np.mean(t50_censored)) if t50_censored else None

    return (
        regret_curve,
        cum_reward,
        avg_lat,
        non_greedy,
        inst_regret,
        mean_tau_adapt,
        len(tau_adapt_values),   # number of breakpoints actually measured
        non_optimal,
        mean_opt_rate,
        mean_t50,
        frac_censored,
    )


# ── main evaluation ───────────────────────────────────────────────────────────
def run_sensitivity_evaluation():
    print("=" * 60)
    print("Sensitivity Analysis — Extended Scenario Evaluation")
    print(f"  Scenarios   : {SCENARIOS}")
    print(f"  N_total     : {N_EVAL} instances × {RUNS} runs = {N_EVAL*RUNS}")
    print(f"  Algorithms  : {METHODS}")
    print("=" * 60)

    # Load tuned hyperparameters (must exist from prior tuning)
    best_params_path = 'results/best_params.json'
    if os.path.exists(best_params_path):
        with open(best_params_path) as f:
            best_params = json.load(f)
        print(f"Loaded tuned params: SW-UCB τ={best_params['SW-UCB']['tau']}, "
              f"D-UCB γ={best_params['D-UCB']['gamma']}")
    else:
        best_params = {'SW-UCB': {'tau': 200}, 'D-UCB': {'gamma': 0.99}}
        print("WARNING: best_params.json not found — using theoretical defaults.")

    # Build task list
    tasks       = []
    task_index  = {}   # (scenario, method) -> list of task positions

    for scenario in SCENARIOS:
        task_index[scenario] = {}
        for method in METHODS:
            task_index[scenario][method] = []
            for inst in range(N_EVAL):
                for run in range(RUNS):
                    instance_seed = SEED + 5 + inst          # eval seed family
                    run_seed      = SEED + 500 + inst*100 + run
                    tasks.append((scenario, instance_seed, run_seed, method, best_params))
                    task_index[scenario][method].append(len(tasks) - 1)

    print(f"Total tasks: {len(tasks)} — dispatching to {NUM_WORKERS} workers...")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        all_results = list(executor.map(run_single_task, tasks))

    print("All tasks complete. Aggregating results...")

    # Aggregate
    results = {}
    for scenario in SCENARIOS:
        results[scenario] = {}
        for method in METHODS:
            indices = task_index[scenario][method]

            regrets, cum_rewards, avg_lats = [], [], []
            non_greedy_rates, inst_regrets, tau_adapts = [], [], []
            n_breakpoints_list = []
            non_optimals, opt_rates, t50s, t50_cens = [], [], [], []

            for idx in indices:
                (rc, cr, al, ng, ir, tau, n_bp, nor, orate, t50, cens) = all_results[idx]
                regrets.append(rc)
                cum_rewards.append(cr)
                avg_lats.append(al)
                non_greedy_rates.append(ng)
                inst_regrets.append(ir)
                if tau is not None:
                    tau_adapts.append(tau)
                n_breakpoints_list.append(n_bp)
                non_optimals.append(nor)
                if orate is not None:
                    opt_rates.append(orate)
                if t50 is not None:
                    t50s.append(t50)
                    t50_cens.append(cens)

            regrets      = np.array(regrets)       # (N, T)
            cum_rewards  = np.array(cum_rewards)
            avg_lats     = np.array(avg_lats)
            inst_regrets = np.array(inst_regrets)

            results[scenario][method] = {
                'regret_mean':         np.mean(regrets, axis=0),
                'regret_std':          np.std(regrets, axis=0),
                'cum_reward_mean':     np.mean(cum_rewards, axis=0),
                'avg_lat_mean':        np.mean(avg_lats, axis=0),
                'non_greedy_rate_mean': float(np.mean(non_greedy_rates)),
                'non_greedy_rate_std':  float(np.std(non_greedy_rates)),
                'inst_regret_mean':    np.mean(inst_regrets, axis=0),
                'tau_adapt_mean':      float(np.mean(tau_adapts))  if tau_adapts else None,
                'tau_adapt_std':       float(np.std(tau_adapts))   if tau_adapts else None,
                'n_breakpoints_measured': float(np.mean(n_breakpoints_list)),
                # Amendment A2
                'non_optimal_rate_mean': float(np.mean(non_optimals)),
                'non_optimal_rate_std':  float(np.std(non_optimals)),
                # Amendment A6: per-instance final regret (chunks of RUNS, instance-major order)
                'per_instance': {
                    'final_regret': [float(np.mean(np.array(regrets)[i*RUNS:(i+1)*RUNS, -1]))
                                     for i in range(len(indices) // RUNS)],
                },
            }
            # Amendment A1 metrics (breakpoint-averaged per trajectory)
            if opt_rates:
                keys = set.intersection(*(set(o.keys()) for o in opt_rates))
                for W in sorted(keys):
                    vals = np.array([o[W] for o in opt_rates])
                    results[scenario][method][f'opt_rate_{W}_mean'] = float(np.mean(vals))
                    results[scenario][method][f'opt_rate_{W}_std'] = float(np.std(vals))
            if t50s:
                t50s_arr = np.array(t50s)
                results[scenario][method]['t50_median'] = float(np.median(t50s_arr))
                results[scenario][method]['t50_q25'] = float(np.percentile(t50s_arr, 25))
                results[scenario][method]['t50_q75'] = float(np.percentile(t50s_arr, 75))
                results[scenario][method]['t50_censored_pct'] = float(np.mean(t50_cens) * 100.0)

    # Merge into existing results.pkl
    pkl_path = 'results/results.pkl'
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            existing = pickle.load(f)
        existing.update(results)
        merged = existing
    else:
        merged = results

    with open(pkl_path, 'wb') as f:
        pickle.dump(merged, f)
    print(f"Saved merged results to {pkl_path}")

    # Print a quick summary table
    print("\n" + "=" * 60)
    print("SUMMARY — Final cumulative regret (Mean ± STD)")
    print("=" * 60)
    print(f"{'Algorithm':<20} {'multi_abrupt':>20} {'periodic_drift':>20}")
    print("-" * 60)
    for method in METHODS:
        row = f"{method:<20}"
        for sc in SCENARIOS:
            d = results[sc][method]
            row += f"  {d['regret_mean'][-1]:>7.2f} ± {d['regret_std'][-1]:>6.2f}"
        print(row)

    print("\nTau_adapt (mean ± std across breakpoints and trajectories):")
    print(f"{'Algorithm':<20} {'multi_abrupt':>25} {'periodic_drift':>25}")
    print("-" * 70)
    for method in METHODS:
        row = f"{method:<20}"
        for sc in SCENARIOS:
            d = results[sc][method]
            if d['tau_adapt_mean'] is not None:
                row += f"  {d['tau_adapt_mean']:>8.1f} ± {d['tau_adapt_std']:>7.1f}"
            else:
                row += f"  {'N/A':>18}"
        print(row)

    return results


if __name__ == '__main__':
    run_sensitivity_evaluation()
