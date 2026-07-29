"""
make_appendix_tables.py - generates the three appendix tables (B.1-B.3) from raw data,
following rule A8: every number in the paper is script-generated, never typed by hand.

Run from the repository root:  python -m experiments.make_appendix_tables

Ghi ra:
  experiment/table_appendix_B1_per_instance.txt   - regret per evaluation instance
                                                    (raw evidence for the "consistent on 5/5" criterion)
  experiment/table_appendix_B2_grid_scenario.txt  - SW-UCB/D-UCB grid search split by scenario
                                                    (evidence that the trend is monotone even within abrupt alone)
  experiment/table_appendix_B3_sweep.txt          - full data from the dense_abrupt(P) sweep

Numbers are printed ready to paste straight into LaTeX, avoiding the
cell-by-cell transcription that caused an error in an earlier draft (see lessons/, pitfall 10).
"""
import json

SUMMARY = 'results/results_summary.json'
PARAMS = 'results/best_params.json'
SWEEP = 'results/results_drift_sweep.json'

SCENARIOS = ['stationary', 'gradual_drift', 'abrupt_drift']
SCENARIO_VN = {'stationary': 'Stationary', 'gradual_drift': 'Gradual Drift',
               'abrupt_drift': 'Abrupt Drift'}

METHOD_LABELS = {  # display order = declaration order
    'RoundRobin': 'Round Robin',
    'LeastConnections': 'Least Connections',
    'EpsilonGreedy': r'$\epsilon$-greedy (tuned, $c=2$)',
    'EpsilonGreedy-Default': r'$\epsilon$-greedy (default, $c=0{.}1$)',
    'UCB': 'UCB',
    'ThompsonSampling': r'TS (tuned, $\sigma_0=0{.}25$)',
    'ThompsonSampling-Default': r'TS (default, $\sigma_0=1$)',
    'SW-UCB': r'SW-UCB (tuned, $\tau=10000$)',
    'SW-UCB-Default': r'SW-UCB (default, $\tau=200$)',
    'D-UCB': r'D-UCB (tuned, $\gamma=1{.}0$)',
    'D-UCB-Default': r'D-UCB (default, $\gamma=0{.}99$)',
}

SWEEP_LABELS = {
    'UCB': r'UCB ($\equiv$ SW-UCB tuned)',
    'ThompsonSampling': r'TS (tuned, $\sigma_0=0{.}25$)',
    'ThompsonSampling-Default': r'TS (default, $\sigma_0=1$)',
    'SW-UCB-Default': r'SW-UCB (default, $\tau=200$)',
    'D-UCB': r'D-UCB (tuned, $\gamma=1{.}0$)',
    'D-UCB-Default': r'D-UCB (default, $\gamma=0{.}99$)',
}


def fmt(x, nd=1):
    """Format a number for direct LaTeX inclusion."""
    return f'{x:.{nd}f}'


def make_b1(summary):
    lines = [
        r'\begin{table}[htbp]', r'\centering', r'\small',
        r'\setlength{\tabcolsep}{5pt}',
        r'\caption{Final dynamic regret per \emph{individual} evaluation instance (I1--I5; each cell is'
        r' the mean of 30 runs on that instance). These are the 5 independent units ($n=5$) used for every'
        r' paired test; the reader can verify the'
        " ``consistent on 5/5 instances'' criterion by comparing the sign between any two rows"
        r' column by column. The Mean column averages the 5 instances.}',
        r'\label{app:tab_per_instance}',
        r'\begin{tabular}{lrrrrrr}', r'\toprule',
        r'\textbf{Method} & \textbf{I1} & \textbf{I2} & \textbf{I3} & \textbf{I4} & \textbf{I5} & \textbf{Mean} \\',
    ]
    for scn in SCENARIOS:
        lines.append(r'\midrule')
        lines.append(r'\multicolumn{7}{l}{\textbf{' + SCENARIO_VN[scn] + r'}} \\[1pt]')
        for m, label in METHOD_LABELS.items():
            vals = summary[scn][m]['per_instance_final_regret']
            cells = ' & '.join(fmt(v) for v in vals)
            mean = sum(vals) / len(vals)
            lines.append(f'{label} & {cells} & {fmt(mean)} ' + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def make_b2(params):
    gr = params['grid_results']
    sw = sorted(gr['SW-UCB'].items(), key=lambda kv: float(kv[0]))
    du = sorted(gr['D-UCB'].items(), key=lambda kv: float(kv[0]))
    lines = [
        r'\begin{table}[htbp]', r'\centering', r'\small',
        r'\setlength{\tabcolsep}{3.5pt}',
        r'\caption{SW-UCB and D-UCB grid search split by scenario (mean regret'
        r' over the tuning instances). The monotone trend toward the anchor point holds within \emph{each}'
        r' individual scenario --- including abrupt drift --- so it is not an artefact of'
        r' averaging. Bold: the optimum (the UCB anchor point).}',
        r'\label{app:tab_grid_scenario}',
        r'\begin{tabular}{r rrrr @{\hspace{14pt}} r rrrr}', r'\toprule',
        r'\multicolumn{5}{c}{\textbf{SW-UCB ($\tau$)}} & \multicolumn{5}{c}{\textbf{D-UCB ($\gamma$)}} \\',
        r'\cmidrule(r{14pt}){1-5} \cmidrule{6-10}',
        r'$\tau$ & Stat. & Grad. & Abr. & TB & $\gamma$ & Stat. & Grad. & Abr. & TB \\',
        r'\midrule',
    ]
    for (tau, sv), (gam, dv) in zip(sw, du):
        sp, dp = sv['per_scenario'], dv['per_scenario']
        tau_disp = f'{int(float(tau))}'
        gam_disp = fmt(float(gam), 4).rstrip('0').rstrip(',') if float(gam) != 1.0 else '1,0'
        row = (f'{tau_disp} & {fmt(sp["stationary"])} & {fmt(sp["gradual_drift"])} & '
               f'{fmt(sp["abrupt_drift"])} & {fmt(sv["objective_mean_regret"])} & '
               f'{gam_disp} & {fmt(dp["stationary"])} & {fmt(dp["gradual_drift"])} & '
               f'{fmt(dp["abrupt_drift"])} & {fmt(dv["objective_mean_regret"])}')
        if float(tau) == 10000:  # row containing both anchor points
            row = ' & '.join(r'\textbf{' + c.strip() + '}' for c in row.split('&'))
        lines.append(row + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def make_b3(sweep):
    periods = sorted(sweep.keys(), key=int)
    lines = [
        r'\begin{table}[htbp]', r'\centering', r'\small',
        r'\setlength{\tabcolsep}{3pt}',
        r'\caption{Full data from the drift-frequency sweep'
        r': final dynamic regret (mean $\pm$ standard'
        r' deviation, $N=150$ trajectories) against stable-phase length $P$. The $P=5000$ column coincides with the'
        r' main abrupt scenario by design (same environment family and seeds).}',
        r'\label{app:tab_sweep}',
        r'\begin{tabular}{l' + 'c' * len(periods) + '}', r'\toprule',
        r'\textbf{Method} & ' +
        ' & '.join(r'$P{=}' + p + '$' for p in periods) + r' \\',
        r'\midrule',
    ]
    for m, label in SWEEP_LABELS.items():
        cells = []
        for p in periods:
            v = sweep[p][m]
            cells.append(f'{fmt(v["final_regret_mean"])} $\\pm$ {fmt(v["final_regret_std"])}')
        lines.append(f'{label} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def main():
    with open(SUMMARY) as f:
        summary = json.load(f)
    with open(PARAMS) as f:
        params = json.load(f)
    with open(SWEEP) as f:
        sweep = json.load(f)

    outputs = {
        'results/table_appendix_B1_per_instance.txt': make_b1(summary),
        'results/table_appendix_B2_grid_scenario.txt': make_b2(params),
        'results/table_appendix_B3_sweep.txt': make_b3(sweep),
    }
    for path, content in outputs.items():
        with open(path, 'w') as f:
            f.write(content + '\n')
        print(f'Saved: {path}')


if __name__ == '__main__':
    main()
