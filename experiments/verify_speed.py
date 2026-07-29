import time
import numpy as np
from src.env import LatencyEnvironment
from src.policies import SlidingWindowUCB
from src.simulator import run_simulation

def verify():
    K = 10
    T = 10000
    L_MIN = 1.0
    SEED = 42
    
    print("Verifying environment creation speed...")
    t0 = time.perf_counter()
    env = LatencyEnvironment(scenario='abrupt_drift', K=K, T=T, L_min=L_MIN, instance_seed=SEED)
    t1 = time.perf_counter()
    env_time = (t1 - t0) * 1000
    print(f"Environment creation time: {env_time:.2f} ms")
    
    print("\nVerifying 10 consecutive simulations of Sliding Window UCB (the most compute-intensive policy)...")
    sim_times = []
    for run in range(10):
        policy = SlidingWindowUCB(K=K, tau=1000, T=T)
        t_start = time.perf_counter()
        sim = run_simulation(env, policy, T, SEED + run)
        t_end = time.perf_counter()
        sim_times.append((t_end - t_start) * 1000)
        
    print(f"Simulation execution times (ms): {['{:.2f}'.format(x) for x in sim_times]}")
    print(f"Average simulation time: {np.mean(sim_times):.2f} ms")

if __name__ == '__main__':
    verify()
