"""
make_appendix_tables.py — sinh tự động 3 bảng Phụ lục (B.1–B.3) từ dữ liệu gốc,
theo nguyên tắc A8 (mọi con số trong báo cáo sinh bằng script, không gõ tay).

Chạy từ gốc repo:  .venv/bin/python experiment/make_appendix_tables.py

Ghi ra:
  experiment/table_appendix_B1_per_instance.txt   — regret theo từng evaluation instance
                                                    (bằng chứng thô cho tiêu chí "nhất quán 5/5")
  experiment/table_appendix_B2_grid_scenario.txt  — grid search SW-UCB/D-UCB tách theo kịch bản
                                                    (bằng chứng cho claim "đơn điệu kể cả riêng abrupt")
  experiment/table_appendix_B3_sweep.txt          — số liệu đầy đủ của thí nghiệm quét dense_abrupt(P)

Số được in theo định dạng thập phân DẤU PHẨY để dán nguyên khối vào LaTeX — tránh
bước chép tay từng ô vốn đã gây lỗi ở bảng 4.7 (xem lession/12-soat-xet-vong-2.md, mục A3).
"""
import json

SUMMARY = 'results/results_summary.json'
PARAMS = 'results/best_params.json'
SWEEP = 'results/results_drift_sweep.json'

SCENARIOS = ['stationary', 'gradual_drift', 'abrupt_drift']
SCENARIO_VN = {'stationary': 'Stationary', 'gradual_drift': 'Gradual Drift',
               'abrupt_drift': 'Abrupt Drift'}

METHOD_LABELS = {  # thứ tự hiển thị = thứ tự khai báo
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
    """Định dạng số kiểu Việt Nam (dấu phẩy thập phân)."""
    return f'{x:.{nd}f}'.replace('.', ',')


def make_b1(summary):
    lines = [
        r'\begin{table}[htbp]', r'\centering', r'\small',
        r'\setlength{\tabcolsep}{5pt}',
        r'\caption{Hối tiếc động cuối kỳ theo \emph{từng} evaluation instance (I1--I5; mỗi ô là'
        r' trung bình của 30 run trên instance đó). Đây là 5 đơn vị độc lập ($n=5$) dùng cho mọi'
        r' kiểm định theo cặp tại Mục~\ref{subsec:monte_carlo_estimation}; người đọc có thể tự'
        " kiểm tra tiêu chí ``nhất quán 5/5 instance'' bằng cách so dấu hiệu giữa hai dòng bất kỳ"
        r' theo từng cột. Cột TB là trung bình 5 instance.}',
        r'\label{app:tab_per_instance}',
        r'\begin{tabular}{lrrrrrr}', r'\toprule',
        r'\textbf{Phương pháp} & \textbf{I1} & \textbf{I2} & \textbf{I3} & \textbf{I4} & \textbf{I5} & \textbf{TB} \\',
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
        r'\caption{Grid search của SW-UCB và D-UCB tách theo từng kịch bản (regret trung bình'
        r' trên tuning instances). Xu hướng đơn điệu về phía điểm neo giữ nguyên ở \emph{từng}'
        r' kịch bản riêng lẻ --- kể cả abrupt drift --- chứ không phải hệ quả của phép lấy trung'
        r' bình (Mục~\ref{sec:sensitivity_analysis}). In đậm: điểm tối ưu (điểm neo UCB).}',
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
        if float(tau) == 10000:  # hàng chứa cả hai điểm neo
            row = ' & '.join(r'\textbf{' + c.strip() + '}' for c in row.split('&'))
        lines.append(row + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def make_b3(sweep):
    periods = sorted(sweep.keys(), key=int)
    lines = [
        r'\begin{table}[htbp]', r'\centering', r'\small',
        r'\setlength{\tabcolsep}{3pt}',
        r'\caption{Số liệu đầy đủ của thí nghiệm quét tần suất drift (Mục'
        r'~\ref{subsec:drift_frequency_sweep}): hối tiếc động cuối kỳ (trung bình $\pm$ độ lệch'
        r' chuẩn, $N=150$ quỹ đạo) theo độ dài pha ổn định $P$. Cột $P=5000$ trùng kịch bản'
        r' abrupt chính theo thiết kế (cùng họ sinh môi trường và cùng seed).}',
        r'\label{app:tab_sweep}',
        r'\begin{tabular}{l' + 'c' * len(periods) + '}', r'\toprule',
        r'\textbf{Phương pháp} & ' +
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
