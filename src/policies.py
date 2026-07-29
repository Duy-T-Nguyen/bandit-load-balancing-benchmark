import numpy as np

class Policy:
    def __init__(self, K):
        self.K = K
        
    def select_arm(self, t, rng):
        raise NotImplementedError
        
    def update(self, arm, reward, latency, t):
        pass

class RoundRobin(Policy):
    def __init__(self, K):
        super().__init__(K)
        self.current = 0
        
    def select_arm(self, t, rng):
        arm = self.current
        self.current = (self.current + 1) % self.K
        return arm

class LeastConnections(Policy):
    def __init__(self, K, delta=10.0):
        super().__init__(K)
        self.delta = delta
        # active_requests[i] stores the completion time step (float) of requests sent to arm i
        self.active_requests = [[] for _ in range(K)]
        self.t = 0
        
    def select_arm(self, t, rng):
        self.t = t
        counts = np.zeros(self.K)
        for i in range(self.K):
            # Clean up completed requests whose completion step is <= t
            self.active_requests[i] = [c for c in self.active_requests[i] if c > t]
            counts[i] = len(self.active_requests[i])
        
        # Tie-break randomly using rng
        min_conn = np.min(counts)
        candidates = np.where(counts == min_conn)[0]
        return rng.choice(candidates)
        
    def update(self, arm, reward, latency, t):
        # Completion step is current step t + (latency in ms) / (delta ms per step)
        completion_time = t + latency / self.delta
        self.active_requests[arm].append(completion_time)

class EpsilonGreedy(Policy):
    def __init__(self, K, c=0.1, d=1.0):
        super().__init__(K)
        self.c = c
        self.d = d
        self.counts = np.zeros(K)
        self.values = np.zeros(K)
        
    def select_arm(self, t, rng):
        if t == 0:
            epsilon = 1.0
        else:
            epsilon = min(1.0, (self.c * self.K) / ((self.d ** 2) * t))
            
        if rng.random() < epsilon:
            return rng.integers(0, self.K)
            
        max_val = np.max(self.values)
        candidates = np.where(self.values == max_val)[0]
        return rng.choice(candidates)
        
    def update(self, arm, reward, latency, t):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] = ((n - 1) * self.values[arm] + reward) / n

class UCB(Policy):
    def __init__(self, K):
        super().__init__(K)
        self.counts = np.zeros(K)
        self.values = np.zeros(K)
        
    def select_arm(self, t, rng):
        # Play each arm once first
        unplayed = np.where(self.counts == 0)[0]
        if len(unplayed) > 0:
            return rng.choice(unplayed)
            
        # Standard UCB1 index: mean + sqrt(2 * ln(t) / n)
        ucb_values = self.values + np.sqrt(2.0 * np.log(t) / self.counts)
        max_ucb = np.max(ucb_values)
        candidates = np.where(ucb_values == max_ucb)[0]
        return rng.choice(candidates)
        
    def update(self, arm, reward, latency, t):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] = ((n - 1) * self.values[arm] + reward) / n

class ThompsonSampling(Policy):
    def __init__(self, K, sigma_0=1.0):
        super().__init__(K)
        self.sigma_0 = sigma_0
        self.counts = np.zeros(K)
        self.means = np.ones(K) * 0.5  # Prior mean initialized in the middle of [0, 1]
        
    def select_arm(self, t, rng):
        # Sample from Gaussian posterior for each arm
        # Posterior standard deviation is sigma_0 / sqrt(counts + 1)
        stds = self.sigma_0 / np.sqrt(self.counts + 1.0)
        samples = rng.normal(self.means, stds)
        max_sample = np.max(samples)
        candidates = np.where(samples == max_sample)[0]
        return rng.choice(candidates)
        
    def update(self, arm, reward, latency, t):
        self.counts[arm] += 1
        self.means[arm] += (reward - self.means[arm]) / self.counts[arm]

class SlidingWindowUCB(Policy):
    def __init__(self, K, tau=200, T=10000):
        super().__init__(K)
        self.tau = tau
        self.history_arms = np.zeros(T, dtype=int)
        self.history_rewards = np.zeros(T)
        self.step = 0
        
    def select_arm(self, t, rng):
        window_size = min(t, self.tau)
        if window_size == 0:
            return rng.integers(0, self.K)
            
        start_idx = t - window_size
        active_arms = self.history_arms[start_idx:t]
        active_rewards = self.history_rewards[start_idx:t]
        
        # Calculate counts and rewards in the window
        counts = np.bincount(active_arms, minlength=self.K)
        values = np.bincount(active_arms, weights=active_rewards, minlength=self.K)
        
        # If any arm was not played in the window, play it
        unplayed = np.where(counts == 0)[0]
        if len(unplayed) > 0:
            return rng.choice(unplayed)
            
        # Calculate SW-UCB indices
        means = values / counts
        ucb_values = means + np.sqrt(2.0 * np.log(window_size) / counts)
        max_ucb = max(ucb_values)
        candidates = [i for i in range(self.K) if ucb_values[i] == max_ucb]
        return rng.choice(candidates)
        
    def update(self, arm, reward, latency, t):
        self.history_arms[self.step] = arm
        self.history_rewards[self.step] = reward
        self.step += 1

class DiscountedUCB(Policy):
    def __init__(self, K, gamma=0.99):
        super().__init__(K)
        self.gamma = gamma
        self.n_discounted = np.zeros(K)
        self.r_discounted = np.zeros(K)
        
    def select_arm(self, t, rng):
        # If any arm has a discounted count near 0, select it first
        unplayed = np.where(self.n_discounted < 0.1)[0]
        if len(unplayed) > 0:
            return rng.choice(unplayed)
            
        # D-UCB index calculation (Garivier & Moulines 2011)
        n_total = np.sum(self.n_discounted)
        means = self.r_discounted / self.n_discounted
        ucb_values = means + 2.0 * np.sqrt(np.log(n_total) / self.n_discounted)
        max_ucb = np.max(ucb_values)
        candidates = np.where(ucb_values == max_ucb)[0]
        return rng.choice(candidates)
        
    def update(self, arm, reward, latency, t):
        # Update discounted counts and rewards recursively
        self.n_discounted *= self.gamma
        self.r_discounted *= self.gamma
        self.n_discounted[arm] += 1.0
        self.r_discounted[arm] += reward
