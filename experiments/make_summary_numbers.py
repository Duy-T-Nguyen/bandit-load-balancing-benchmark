"""
make_summary_numbers.py — Amendment A6 + A8: every summary number quoted in the
report must come from this script, never typed by hand. Also runs the paired
statistical tests (unit of independence = instance, n=5) and the H1 two-part
check, and prints a digest for the author to read BEFORE any report text is
drafted.

Run from repo root: .venv/bin/python experiment/make_summary_numbers.py
"""
import json
import pickle

import numpy as np
from scipy import stats

SCENARIOS = ['stationary', 'gradual_drift', 'abrupt_drift']
TUNED = ['EpsilonGreedy', 'ThompsonSampling', 'SW-UCB', 'D-UCB']
MAB = ['EpsilonGreedy', 'UCB', 'ThompsonSampling', 'SW-UCB', 'D-UCB']
BASELINES = ['RoundRobin', 'LeastConnections']


def paired_test(a, b):
    """Paired comparison on per-instance means (n=5). Returns dict."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = a - b
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    t_stat, t_p = stats.ttest_rel(a, b)
    # bootstrap 95% CI of mean difference
    rng = np.random.default_rng(0)
    boots = [np.mean(diff[rng.integers(0, len(diff), len(diff))]) for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {'mean_diff': float(np.mean(diff)), 'wilcoxon_p': float(w_p),
            'ttest_p': float(t_p), 'boot_ci': (float(lo), float(hi)),
            'consistent_sign': bool(np.all(diff < 0) or np.all(diff > 0))}


def fmt_test(name_a, name_b, res):
    ci = res['boot_ci']
    sign = "CONSISTENT WIN on 5/5 instances" if res['consistent_sign'] else "NOT consistent across instances"
    return (f"    {name_a} vs {name_b}: mean regret difference = {res['mean_diff']:+.1f}, "
            f"Wilcoxon p={res['wilcoxon_p']:.3f}, t-test p={res['ttest_p']:.3f}, "
            f"CI95=[{ci[0]:.1f}; {ci[1]:.1f}] — {sign}")


def main():
    with open('results/results.pkl', 'rb') as f:
        R = pickle.load(f)
    S = json.load(open('results/results_summary.json'))

    def pi(scenario, method):
        return R[scenario][method]['per_instance']['final_regret']

    def m(scenario, method):
        return S[scenario][method]['final_cumulative_regret_mean']

    print("=" * 78)
    print("DIGEST - every summary figure reported in the paper (Amendment A8)")
    print("=" * 78)

    # ── 0. Mandatory sanity check (A1) ──────────────────────────────────────
    rr = S['abrupt_drift']['RoundRobin']
    print("\n[0] SANITY CHECK: new metric on Round Robin (abrupt):")
    print(f"    opt_rate@1000 = {rr.get('opt_rate_1000_mean', float('nan')):.4f} "
          f"(expected ~0.10) | t50 censored = {rr.get('t50_censored_pct', float('nan')):.1f}% "
          f"(expected ~100%)")

    # ── 1. H1 - test BOTH clauses (A8) ─────────────────────────────────────
    print("\n[1] H1 (stationary): 'UCB = TS' AND 'both > eps-greedy'")
    print(fmt_test('TS', 'UCB', paired_test(pi('stationary', 'ThompsonSampling'),
                                            pi('stationary', 'UCB'))))
    print(fmt_test('UCB', 'εG', paired_test(pi('stationary', 'UCB'),
                                            pi('stationary', 'EpsilonGreedy'))))
    print(fmt_test('TS', 'εG', paired_test(pi('stationary', 'ThompsonSampling'),
                                           pi('stationary', 'EpsilonGreedy'))))

    # ── 2. H2 (gradual): D-UCB vs UCB ───────────────────────────────────────
    print("\n[2] H2 (gradual_drift): D-UCB > UCB?")
    print(fmt_test('D-UCB', 'UCB', paired_test(pi('gradual_drift', 'D-UCB'),
                                               pi('gradual_drift', 'UCB'))))

    # ── 3. H3 (abrupt): adaptation speed under the NEW metric ──────────────────
    print("\n[3] H3 (abrupt): adaptation under the new metric (opt_rate/t50)")
    hdr = f"    {'Method':22s} {'opt@100':>8} {'opt@500':>8} {'opt@1000':>9} {'t50 med [IQR]':>20} {'censored':>9}"
    print(hdr)
    for method in ['UCB', 'ThompsonSampling', 'SW-UCB', 'SW-UCB-Default',
                   'D-UCB', 'D-UCB-Default', 'EpsilonGreedy', 'RoundRobin', 'LeastConnections']:
        s = S['abrupt_drift'][method]
        if 'opt_rate_100_mean' not in s:
            continue
        iqr = s.get('t50_iqr', [float('nan')] * 2)
        print(f"    {method:22s} {s['opt_rate_100_mean']:>8.3f} {s['opt_rate_500_mean']:>8.3f} "
              f"{s['opt_rate_1000_mean']:>9.3f} {s['t50_median']:>8.0f} "
              f"[{iqr[0]:.0f};{iqr[1]:.0f}]".ljust(14) + f" {s['t50_censored_pct']:>8.1f}%")
    print("    (Legacy tau_adapt kept in the appendix - see results_summary.json)")

    # ── 4. H4: MAB vs baselines, paired ─────────────────────────────────────
    print("\n[4] H4: MAB vs best baseline (per scenario, paired n=5)")
    for sc in SCENARIOS:
        best_bl = min(BASELINES, key=lambda b: m(sc, b))
        print(f"  -- {sc} (best baseline: {best_bl} = {m(sc, best_bl):.1f})")
        for method in MAB:
            res = paired_test(pi(sc, method), pi(sc, best_bl))
            ratio = m(sc, best_bl) / m(sc, method)
            print(fmt_test(method, best_bl, res) + f" | ratio {ratio:.1f}x")

    # ── 5. Tuned vs Default - step 5 of the original protocol ───────────────────────
    print("\n[5] Tuned vs Default (the 'reduced by X%-Y%' sentence comes from here, NOT typed by hand):")
    reductions = []
    for algo in TUNED:
        for sc in SCENARIOS:
            a, b = m(sc, algo), m(sc, f'{algo}-Default') if f'{algo}-Default' in S[sc] else (None, None)
            if f'{algo}-Default' not in S[sc]:
                continue
            b = m(sc, f'{algo}-Default')
            red = (b - a) / b * 100
            reductions.append(red)
            print(f"    {algo:18s} {sc:14s}: {b:8.2f} -> {a:8.2f}  ({red:+5.1f}%)")
    if reductions:
        print(f"    => Correct sentence for the paper: 'tuning changes regret by "
              f"{min(reductions):+.0f}% to {max(reductions):+.0f}%'")

    # ── 6. Jensen (A7) ──────────────────────────────────────────────────────
    print("\n[6] Jensen verification (A7): theoretical gap = 1/(k-1) = 25.0% (k=5), constant across arms")
    rng = np.random.default_rng(1)
    for theta in [4.0, 10.0, 19.0]:
        L = np.maximum(rng.gamma(5, theta, 1_000_000), 1.0)
        gap = (np.mean(1 / L) - 1 / np.mean(L)) / (1 / np.mean(L)) * 100
        print(f"    Monte Carlo theta={theta:5.1f}: gap = {gap:5.2f}%")

    print("\nDONE. Read this digest and write the H1-H4 conclusions "
          "BEFORE looking at the report draft.")


if __name__ == '__main__':
    main()
