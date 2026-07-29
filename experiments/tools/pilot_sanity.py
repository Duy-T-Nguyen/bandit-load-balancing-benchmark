"""
pilot_sanity.py — 5 kiểm tra bắt buộc TRƯỚC khi chạy full pipeline (~30s).

Chạy từ gốc repo:  .venv/bin/python experiment/tools/pilot_sanity.py

Mục đích: bắt lỗi thiết kế/cài đặt bằng vài mô phỏng rẻ tiền trước khi đốt
hàng giờ compute. Đây chính là bộ kiểm đã chặn được các lỗi trong audit
2026-07-17 (xem lession/BAO-CAO-AUDIT.md, mục IV):
  [1] dense_abrupt thật sự đổi arm tối ưu giữa các pha
  [2] "Nhóm đối chứng vô lý": Round Robin phải ra opt_rate ≈ 1/K và t50 censored
      — nếu RR có điểm tốt, metric hỏng chứ không phải RR giỏi
  [3] Chuẩn hóa online không làm UCB nổ (khám phá bão hòa)
  [4] Điểm neo: SW-UCB(tau=T) phải tái tạo UCB chính xác; D-UCB(gamma=1) chạy được
  [5] Các cấu hình vùng lưới mới (c lớn, sigma_0 nhỏ) chạy ổn định
"""
import sys
import time
import pathlib

# Cho phép chạy từ gốc repo: đưa experiment/ vào sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from src.env import LatencyEnvironment
from src.policies import (RoundRobin, UCB, SlidingWindowUCB, DiscountedUCB,
                          EpsilonGreedy, ThompsonSampling)
from src.simulator import run_simulation
from src.metrics import (compute_regret, compute_opt_rate,
                         compute_time_to_majority, compute_non_optimal_rate)

K, T, SEED = 10, 10000, 42
ok = True

# [1] dense_abrupt builds + optimal arm changes between phases
env = LatencyEnvironment('dense_abrupt', K=K, T=T, instance_seed=SEED, period=500)
n_changes = int(np.sum(np.diff(env.optimal_arms) != 0))
print(f"[1] dense_abrupt(P=500): số lần đổi arm tối ưu = {n_changes} (kỳ vọng >=10 với 19 breakpoints)")
ok &= n_changes >= 10

# [2] RR sanity trên abrupt: opt_rate ~ 1/K, t50 censored
env_ab = LatencyEnvironment('abrupt_drift', K=K, T=T, instance_seed=SEED + 5)
sim_rr = run_simulation(env_ab, RoundRobin(K=K), T, SEED + 500)
orate = compute_opt_rate(sim_rr['arms'], env_ab.optimal_arms, T // 2)
t50 = compute_time_to_majority(sim_rr['arms'], env_ab.optimal_arms, T // 2)
print(f"[2] RR: opt_rate={orate}, t50={t50} (kỳ vọng ~0.1 mọi cửa sổ và censored=True)")
ok &= abs(orate[1000] - 1.0 / K) < 0.05 and t50[1] is True

# [3] Online norm: UCB trên stationary vẫn hội tụ
env_st = LatencyEnvironment('stationary', K=K, T=T, instance_seed=SEED + 5)
t0 = time.perf_counter()
sim_ucb = run_simulation(env_st, UCB(K=K), T, SEED + 500)
dt = time.perf_counter() - t0
reg_ucb = compute_regret(env_st.optimal_rewards, sim_ucb['rewards'])[-1]
nor_ucb = compute_non_optimal_rate(sim_ucb['arms'], env_st.optimal_arms)
print(f"[3] UCB stationary (online norm): regret={reg_ucb:.1f}, non_opt={nor_ucb:.2f}, {dt:.2f}s/sim")
ok &= reg_ucb < 300

# [4] Điểm neo: SW(tau=T) ≡ UCB trên cùng env/seed; D-UCB(gamma=1) chạy OK
sim_sw = run_simulation(env_st, SlidingWindowUCB(K=K, tau=T, T=T), T, SEED + 500)
reg_sw = compute_regret(env_st.optimal_rewards, sim_sw['rewards'])[-1]
sim_d1 = run_simulation(env_st, DiscountedUCB(K=K, gamma=1.0), T, SEED + 500)
reg_d1 = compute_regret(env_st.optimal_rewards, sim_d1['rewards'])[-1]
same = bool(np.array_equal(sim_sw['arms'], sim_ucb['arms']))
print(f"[4] Neo: SW(tau=T) regret={reg_sw:.1f} vs UCB={reg_ucb:.1f} | trùng từng quyết định: {same} | D-UCB(g=1) regret={reg_d1:.1f}")
ok &= same

# [5] Các cấu hình vùng lưới mới chạy ổn
reg_e = compute_regret(env_st.optimal_rewards,
                       run_simulation(env_st, EpsilonGreedy(K=K, c=5.0, d=1.0), T, SEED + 500)['rewards'])[-1]
reg_t = compute_regret(env_st.optimal_rewards,
                       run_simulation(env_st, ThompsonSampling(K=K, sigma_0=0.25), T, SEED + 500)['rewards'])[-1]
print(f"[5] eps-greedy(c=5) regret={reg_e:.1f}, TS(sigma_0=0.25) regret={reg_t:.1f}")

print("\n=> PILOT " + ("PASS — an toàn để chạy full pipeline" if ok else "FAIL — DỪNG LẠI, tìm bug trước"))
sys.exit(0 if ok else 1)
