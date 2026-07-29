"""
run_drift_frequency_sweep.py — Amendment A5: drift-frequency sweep.

Neutral research question (pre-registered in evaluate-instruction.md Amendment A5):
  "How short must the stable phase be before forgetting mechanisms (SW-UCB,
   D-UCB) pay off against plain UCB?"

Design:
  * Scenario `dense_abrupt(P)`: breakpoints every P rounds, each phase draws a
    fresh independent set of mean latencies (random redraw, no forced swap).
  * Sweep axis: P in {250, 500, 1000, 2500, 5000}.
  * Methods: UCB + Tuned and Default variants of TS / SW-UCB / D-UCB.
  * Same evaluation seed family as run_experiments.py:
      instance_seed = SEED + 5 + inst,  run_seed = SEED + 500 + inst*100 + run
  * Metrics per cell: final dynamic regret and whole-horizon non_optimal_rate
    (equivalently 1 - overall correct-arm rate — well-defined for every P),
    plus per-instance aggregates for paired statistics (Amendment A6).
  * Commitment: report the crossover point if the curves cross; report its
    absence otherwise. No post-hoc selection of P values.

Output: experiment/results_drift_sweep.pkl and experiment/results_drift_sweep.json
"""
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.env import LatencyEnvironment
from src.policies import UCB, ThompsonSampling, SlidingWindowUCB, DiscountedUCB
from src.simulator import run_simulation
from src.metrics import compute_regret, compute_non_optimal_rate

K = 10
T = 10_000
L_MIN = 1.0
RUNS = 30
SEED = 42
N_EVAL = 5
NUM_WORKERS = 4

PERIODS = [250, 500, 1000, 2500, 5000]

DEFAULTS = {
    'ThompsonSampling': {'sigma_0': 1.0},
    'SW-UCB': {'tau': 200},
    'D-UCB': {'gamma': 0.99},
}

METHODS = [
    'UCB',
    'ThompsonSampling', 'ThompsonSampling-Default',
    'SW-UCB', 'SW-UCB-Default',
    'D-UCB', 'D-UCB-Default',
]


def build_policy(method, best_params):
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


def run_single_task(args):
    period, instance_seed, run_seed, method, best_params = args
    env = LatencyEnvironment(scenario='dense_abrupt', K=K, T=T, L_min=L_MIN,
                             instance_seed=instance_seed, period=period)
    policy = build_policy(method, best_params)
    sim = run_simulation(env, policy, T, run_seed)
    final_regret = float(compute_regret(env.optimal_rewards, sim['rewards'])[-1])
    non_opt = compute_non_optimal_rate(sim['arms'], env.optimal_arms)
    return final_regret, non_opt


def run_sweep():
    print("Drift-frequency sweep (Amendment A5)")
    print(f"  P values : {PERIODS}")
    print(f"  Methods  : {METHODS}")

    with open('results/best_params.json') as f:
        best_params = json.load(f)

    tasks, index = [], {}
    for P in PERIODS:
        index[P] = {}
        for method in METHODS:
            index[P][method] = []
            for inst in range(N_EVAL):
                for run in range(RUNS):
                    run_seed = SEED + 500 + inst * 100 + run
                    tasks.append((P, SEED + 5 + inst, run_seed, method, best_params))
                    index[P][method].append(len(tasks) - 1)

    print(f"Total tasks: {len(tasks)} on {NUM_WORKERS} workers...")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        flat = list(ex.map(run_single_task, tasks, chunksize=8))

    results = {}
    for P in PERIODS:
        results[P] = {}
        for method in METHODS:
            idxs = index[P][method]
            regs = np.array([flat[i][0] for i in idxs])
            nors = np.array([flat[i][1] for i in idxs])
            n_inst = len(idxs) // RUNS
            results[P][method] = {
                'final_regret_mean': float(np.mean(regs)),
                'final_regret_std': float(np.std(regs)),
                'non_optimal_rate_mean': float(np.mean(nors)),
                'non_optimal_rate_std': float(np.std(nors)),
                'per_instance': {
                    'final_regret': [float(np.mean(regs[i * RUNS:(i + 1) * RUNS]))
                                     for i in range(n_inst)],
                    'non_optimal_rate': [float(np.mean(nors[i * RUNS:(i + 1) * RUNS]))
                                         for i in range(n_inst)],
                },
            }

    with open('results/results_drift_sweep.pkl', 'wb') as f:
        pickle.dump(results, f)
    with open('results/results_drift_sweep.json', 'w', encoding='utf-8') as f:
        json.dump({str(P): results[P] for P in PERIODS}, f, indent=4)
    print("Saved experiment/results_drift_sweep.pkl / .json")

    # Console summary + crossover detection vs UCB
    print(f"\n{'P (phase len)':>14} | " + " | ".join(f"{m:>24}" for m in METHODS))
    for P in PERIODS:
        print(f"{P:>14} | " + " | ".join(
            f"{results[P][m]['final_regret_mean']:>10.1f} ±{results[P][m]['final_regret_std']:>6.1f}      "
            for m in METHODS))
    print("\nCrossover check (per P: does the method beat UCB on mean regret?):")
    for method in METHODS:
        if method == 'UCB':
            continue
        wins = [P for P in PERIODS
                if results[P][method]['final_regret_mean'] < results[P]['UCB']['final_regret_mean']]
        print(f"  {method:26s} beats UCB at P in {wins if wins else 'NONE'}")


if __name__ == '__main__':
    run_sweep()
