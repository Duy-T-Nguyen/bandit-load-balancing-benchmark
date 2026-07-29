"""
tune.py — Hyperparameter tuning protocol (rewritten per Amendment A4).

Changes vs. the original protocol (see evaluate-instruction.md, Amendment 1):
  * Every algorithm with a sensitive free parameter is tuned with the SAME
    budget of 8 grid points: SW-UCB (tau), D-UCB (gamma), EpsilonGreedy (c),
    ThompsonSampling (sigma_0). UCB1 is parameter-free and is not tuned.
  * Grids include ANCHOR points that reduce the non-stationary variants to
    plain UCB behaviour (tau = T, gamma = 1.0), so a boundary optimum at the
    anchor is itself an interpretable result ("optimal = forget nothing"),
    not an out-of-grid extrapolation.
  * Tuning objective: mean final dynamic regret across the THREE main
    scenarios (stationary, gradual_drift, abrupt_drift) on the tuning
    instances — because the chosen value is used unchanged in all scenarios.
    Unweighted average (declared: gradual, having the largest regret scale,
    naturally carries the most weight).
  * Seed family unchanged and disjoint from evaluation:
    instance_seed = SEED + inst, run_seed = SEED + inst*100 + run.
Full grid results are stored in best_params.json under 'grid_results' for the
report's sensitivity table.
"""
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.env import LatencyEnvironment
from src.policies import EpsilonGreedy, ThompsonSampling, SlidingWindowUCB, DiscountedUCB
from src.simulator import run_simulation
from src.metrics import compute_regret

K = 10
T = 10_000
L_MIN = 1.0
RUNS = 30
SEED = 42
N_TUNING_INSTANCES = 5
NUM_WORKERS = 4

SCENARIOS = ['stationary', 'gradual_drift', 'abrupt_drift']

# Equal budget: exactly 8 grid points per tuned algorithm (Amendment A4).
GRIDS = {
    'SW-UCB': {
        'param': 'tau',
        'values': [50, 100, 200, 500, 1000, 2000, 5000, 10000],  # 10000 = T -> UCB1 anchor
        'cls': SlidingWindowUCB,
        'kwargs': lambda v: {'tau': int(v), 'T': T},
    },
    'D-UCB': {
        'param': 'gamma',
        'values': [0.90, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999, 1.0],  # 1.0 -> UCB anchor
        'cls': DiscountedUCB,
        'kwargs': lambda v: {'gamma': float(v)},
    },
    'EpsilonGreedy': {
        'param': 'c',
        'values': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],  # contains legacy 0.1 and the c>5 theorem regime
        'cls': EpsilonGreedy,
        'kwargs': lambda v: {'c': float(v), 'd': 1.0},
    },
    'ThompsonSampling': {
        'param': 'sigma_0',
        'values': [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
        'cls': ThompsonSampling,
        'kwargs': lambda v: {'sigma_0': float(v)},
    },
}


def run_single_task(args):
    """Worker: one simulation, returns final dynamic regret."""
    scenario, instance_seed, run_seed, cls, kwargs = args
    env = LatencyEnvironment(scenario=scenario, K=K, T=T, L_min=L_MIN,
                             instance_seed=instance_seed)
    policy = cls(K=K, **kwargs)
    history = run_simulation(env, policy, T, run_seed)
    return compute_regret(env.optimal_rewards, history['rewards'])[-1]


def tune_parameters():
    print("Hyperparameter tuning (Amendment A4): 4 algorithms x 8 points, "
          "objective = mean regret over 3 scenarios")

    best_params = {}
    grid_results = {}

    for algo, spec in GRIDS.items():
        values = spec['values']
        print(f"\n=== Tuning {algo} ({spec['param']}) ===")

        tasks = []
        for v in values:
            for scenario in SCENARIOS:
                for inst in range(N_TUNING_INSTANCES):
                    for run in range(RUNS):
                        run_seed = SEED + inst * 100 + run     # tuning seed family
                        tasks.append((scenario, SEED + inst, run_seed,
                                      spec['cls'], spec['kwargs'](v)))

        print(f"  {len(tasks)} simulations across {NUM_WORKERS} workers...")
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
            flat = list(ex.map(run_single_task, tasks, chunksize=8))

        # Aggregate: per point, mean over scenarios (each scenario averaged first)
        per_run = len(SCENARIOS) * N_TUNING_INSTANCES * RUNS
        grid_results[algo] = {}
        best_v, best_obj = None, float('inf')
        idx = 0
        for v in values:
            block = np.array(flat[idx:idx + per_run]).reshape(
                len(SCENARIOS), N_TUNING_INSTANCES * RUNS)
            idx += per_run
            per_scenario = block.mean(axis=1)
            objective = float(per_scenario.mean())
            grid_results[algo][str(v)] = {
                'objective_mean_regret': objective,
                'per_scenario': {sc: float(m) for sc, m in zip(SCENARIOS, per_scenario)},
            }
            marker = ''
            if objective < best_obj:
                best_obj, best_v = objective, v
                marker = '  <- best so far'
            print(f"  {spec['param']} = {v:>8} | mean regret (3-scenario avg) = "
                  f"{objective:9.2f} | " +
                  " ".join(f"{sc[:4]}={m:7.2f}" for sc, m in zip(SCENARIOS, per_scenario)) +
                  marker)

        best_params[algo] = {spec['param']: best_v}
        print(f"  BEST {algo}: {spec['param']} = {best_v} (objective {best_obj:.2f})")

    # Keep d explicit for EpsilonGreedy (fixed, declared)
    best_params['EpsilonGreedy']['d'] = 1.0
    best_params['grid_results'] = grid_results

    with open('results/best_params.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    print("\nSaved best parameters (+ full grid results) to experiment/best_params.json")

    # Boundary check (Amendment A4): warn if an optimum sits on a non-anchor edge
    for algo, spec in GRIDS.items():
        v = best_params[algo][spec['param']]
        vals = spec['values']
        if v == vals[0]:
            print(f"WARNING: {algo} optimum at LOWER grid edge ({v}) — "
                  f"a declared one-shot refinement may be needed.")
        elif v == vals[-1] and algo in ('EpsilonGreedy', 'ThompsonSampling'):
            print(f"WARNING: {algo} optimum at UPPER grid edge ({v}) — "
                  f"a declared one-shot refinement may be needed.")
        elif v == vals[-1]:
            print(f"NOTE: {algo} optimum at the UCB anchor ({v}) — interpretable "
                  f"result: 'forget nothing' is optimal at this drift density.")


if __name__ == '__main__':
    tune_parameters()
