import numpy as np

# ── Online reward-normalization procedure (Amendment A3) ──────────────────────
# Fixed procedure parameters, shared by ALL learning policies. These are declared
# process constants (pre-registered in evaluate-instruction.md Amendment A3), not
# per-policy tunables, and use ONLY the policy's own observed rewards — no
# ground-truth/oracle information (the legacy constants r_min=0.005, r_max=0.150
# were derived from environment percentiles and have been removed).
N_WARMUP = 100        # rounds before switching from min/max to percentile bounds
PCTL_LO, PCTL_HI = 1.0, 99.0
UPDATE_EVERY = 500    # recompute percentile bounds every this many rounds


def run_simulation(env, policy, T, seed):
    """
    Simulates a policy in a given environment for T steps with a specific seed.
    """
    rng = np.random.default_rng(seed)

    arms = np.zeros(T, dtype=int)
    rewards = np.zeros(T)
    latencies = np.zeros(T)
    non_greedy_choices = np.zeros(T, dtype=int)

    # Running statistics for tracking empirical mean of rewards
    arm_counts = np.zeros(env.K)
    arm_rewards = np.zeros(env.K)
    empirical_means = np.zeros(env.K)

    # Online normalization bounds (from the policy's own observation history)
    r_lo, r_hi = 0.0, 0.0

    for t in range(T):
        # 1. Policy selects arm
        arm = policy.select_arm(t, rng)

        # 2. Environment samples latency and reward
        latency, reward = env.sample_latency(arm, t, rng)

        # 3. Track whether this choice was exploration
        # (i.e. chose an arm that is not the current empirical best-estimate arm)
        # Using pure Python loop for speed on small K=10 array
        best_empirical_arm = 0
        max_val = empirical_means[0]
        for idx in range(1, env.K):
            if empirical_means[idx] > max_val:
                max_val = empirical_means[idx]
                best_empirical_arm = idx

        if arm_counts[best_empirical_arm] > 0 and arm != best_empirical_arm:
            non_greedy_choices[t] = 1

        # 4. Update running statistics
        arm_counts[arm] += 1
        arm_rewards[arm] += reward
        empirical_means[arm] = arm_rewards[arm] / arm_counts[arm]

        # 5. Record raw observation, then update the online normalizer
        arms[t] = arm
        rewards[t] = reward
        latencies[t] = latency

        if t < N_WARMUP:
            # Warm-up: running min/max of observations so far
            r_lo = rewards[:t + 1].min()
            r_hi = rewards[:t + 1].max()
        elif t == N_WARMUP or t % UPDATE_EVERY == 0:
            r_lo, r_hi = np.percentile(rewards[:t + 1], [PCTL_LO, PCTL_HI])

        # 6. Feed the calibrated observation to the policy
        if r_hi - r_lo < 1e-12:
            reward_norm = 0.5  # cannot discriminate yet
        else:
            reward_norm = float(np.clip((reward - r_lo) / (r_hi - r_lo), 0.0, 1.0))
        policy.update(arm, reward_norm, latency, t)

    return {
        'arms': arms,
        'rewards': rewards,
        'latencies': latencies,
        'non_greedy_choices': non_greedy_choices
    }
