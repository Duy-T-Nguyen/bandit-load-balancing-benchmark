import numpy as np

def compute_regret(optimal_rewards, rewards):
    """
    Computes the dynamic regret over time.
    """
    return np.cumsum(optimal_rewards - rewards)

def compute_cumulative_reward(rewards):
    """
    Computes cumulative reward over time.
    """
    return np.cumsum(rewards)

def compute_average_latency(latencies):
    """
    Computes the running average latency.
    """
    return np.cumsum(latencies) / (np.arange(len(latencies)) + 1)

def compute_opt_rate(arms, optimal_arms, t_b, windows=(100, 500, 1000)):
    """
    Amendment A1: fraction of rounds selecting the ground-truth optimal arm a*(t)
    within the first W rounds after breakpoint t_b. Threshold-free, comparable
    across all policies. Sanity anchor: RoundRobin must yield ~1/K.
    """
    correct = (np.asarray(arms) == np.asarray(optimal_arms)).astype(float)
    T = len(correct)
    return {int(W): float(np.mean(correct[t_b:min(t_b + W, T)])) for W in windows}


def compute_time_to_majority(arms, optimal_arms, t_b, w=100, level=0.5, t_end=None):
    """
    Amendment A1: smallest delta such that the rolling rate (window w) of choosing
    the ground-truth optimal arm reaches `level`, measured from t_b. The threshold
    is ABSOLUTE and shared by all policies (unlike the legacy self-referential
    eta = 1.1 * rho_min). Returns (delta, censored); censored=True when the level
    is never reached within [t_b, t_end).
    """
    correct = (np.asarray(arms) == np.asarray(optimal_arms)).astype(float)
    T = len(correct) if t_end is None else min(int(t_end), len(correct))
    max_delta = T - t_b - w
    if max_delta <= 0:
        return 0, True
    cs = np.concatenate(([0.0], np.cumsum(correct)))
    for delta in range(max_delta):
        s = t_b + delta
        if (cs[s + w] - cs[s]) / w >= level:
            return delta, False
    return max_delta, True


def compute_non_optimal_rate(arms, optimal_arms):
    """
    Amendment A2: fraction of ALL rounds where the chosen arm differs from the
    ground-truth optimal arm a*(t). Distinct from the exploration rate
    (non-greedy vs. the agent's own empirical best), which is tracked in the
    simulator. Only this metric may support "wrong choice causes regret" claims.
    """
    return float(np.mean(np.asarray(arms) != np.asarray(optimal_arms)))


def compute_adaptation_speed(avg_instantaneous_regret, t_b, w=100, eta_factor=1.1):
    """
    LEGACY METRIC — kept for the appendix comparison only (Amendment A1).
    Flawed by design: eta = eta_factor * rho_min is self-referential (rho_min comes
    from the measured policy itself), so weaker policies get easier thresholds
    (empirically: RoundRobin "adapts" in ~2 rounds). Do NOT use for conclusions;
    use compute_opt_rate / compute_time_to_majority instead.
    tau_adapt = min { Delta >= 0 : (1/w) * sum_{s=t_b+Delta}^{t_b+Delta+w-1} rho(s) <= eta }
    where eta = eta_factor * rho_min, and rho_min is the mean instantaneous regret over the last 500 rounds.
    """
    T = len(avg_instantaneous_regret)
    # Estimate rho_min using the last 500 steps (stable phase)
    rho_min = np.mean(avg_instantaneous_regret[-500:])
    eta = eta_factor * rho_min
    
    # We search for Delta starting from 0
    max_delta = T - t_b - w
    for delta in range(max_delta):
        start = t_b + delta
        end = start + w
        window_mean = np.mean(avg_instantaneous_regret[start:end])
        if window_mean <= eta:
            return delta
            
    # Return max_delta if it never recovers to threshold
    return max_delta
