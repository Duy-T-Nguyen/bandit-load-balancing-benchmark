import numpy as np
from scipy.stats import gamma

class LatencyEnvironment:
    """
    Simulation environment for K resources over time horizon T.
    Generates latencies from Gamma distribution with shape parameter k.
    Supports stationary, gradual drift, abrupt drift, periodic drift, and multi abrupt scenarios.
    """
    def __init__(self, scenario, K=10, T=10000, L_min=1.0, k=5, instance_seed=42, period=None):
        self.scenario = scenario
        self.K = K
        self.T = T
        self.L_min = L_min
        self.k = k
        self.instance_seed = instance_seed
        self.period = period  # only used by 'dense_abrupt' (Amendment A5)
        
        # Initialize parameters for the scenario
        self.theta = np.zeros((self.K, self.T))
        self._generate_profiles()
        
        # Calculate ground-truth expected rewards for each arm and time step
        self.expected_rewards = np.zeros((self.K, self.T))
        self._calculate_expected_rewards()
        
        # Optimal reward and arm at each step
        self.optimal_rewards = np.max(self.expected_rewards, axis=0)
        self.optimal_arms = np.argmax(self.expected_rewards, axis=0)
        
    def _generate_profiles(self):
        rng = np.random.default_rng(self.instance_seed)
        
        # k is the shape parameter. Mean latency E[L] = k * theta => theta = E[L]/k
        # We target expected latencies in the range [10, 100] ms.
        if self.scenario == 'stationary':
            # Mean latencies are constant over time
            mean_latencies = rng.uniform(10.0, 100.0, size=self.K)
            for i in range(self.K):
                self.theta[i, :] = mean_latencies[i] / self.k
                
        elif self.scenario == 'gradual_drift':
            # Mean latencies drift linearly: E[L_i(t)] = E[L_{i,0}] + alpha_i * t
            mean_latencies_start = rng.uniform(20.0, 90.0, size=self.K)
            # Drift rate chosen so that total drift is bounded, maintaining positive latency
            drift_rates = rng.uniform(-5.0 / 1000.0, 5.0 / 1000.0, size=self.K)
            t = np.arange(self.T)
            for i in range(self.K):
                mean_lat_t = mean_latencies_start[i] + drift_rates[i] * t
                # Keep mean latency strictly bounded in a physical range [5.0, 150.0] ms
                mean_lat_t = np.clip(mean_lat_t, 5.0, 150.0)
                self.theta[i, :] = mean_lat_t / self.k
                
        elif self.scenario == 'abrupt_drift':
            # Piecewise constant mean latencies with a breakpoint at T/2
            t_b = self.T // 2
            mean_latencies_phase1 = rng.uniform(15.0, 95.0, size=self.K)
            # Shuffle or assign new random values for phase 2 to simulate independent load shifts
            mean_latencies_phase2 = rng.uniform(15.0, 95.0, size=self.K)
            
            for i in range(self.K):
                self.theta[i, :t_b] = mean_latencies_phase1[i] / self.k
                self.theta[i, t_b:] = mean_latencies_phase2[i] / self.k
                
        elif self.scenario == 'periodic_drift':
            # Mean latencies fluctuate sinusoidally representing load cycles
            mean_latencies_base = rng.uniform(30.0, 70.0, size=self.K)
            amplitudes = rng.uniform(10.0, 25.0, size=self.K)
            period = 2000.0
            t = np.arange(self.T)
            for i in range(self.K):
                mean_lat_t = mean_latencies_base[i] + amplitudes[i] * np.sin(2.0 * np.pi * t / period)
                mean_lat_t = np.clip(mean_lat_t, 5.0, 150.0)
                self.theta[i, :] = mean_lat_t / self.k
                
        elif self.scenario == 'multi_abrupt':
            # Multiple abrupt shifts at 2500, 5000, 7500 representing multiple network load steps
            mean_latencies_p1 = rng.uniform(15.0, 95.0, size=self.K)
            mean_latencies_p2 = rng.uniform(15.0, 95.0, size=self.K)
            mean_latencies_p3 = rng.uniform(15.0, 95.0, size=self.K)
            mean_latencies_p4 = rng.uniform(15.0, 95.0, size=self.K)
            
            for i in range(self.K):
                self.theta[i, :2500] = mean_latencies_p1[i] / self.k
                self.theta[i, 2500:5000] = mean_latencies_p2[i] / self.k
                self.theta[i, 5000:7500] = mean_latencies_p3[i] / self.k
                self.theta[i, 7500:] = mean_latencies_p4[i] / self.k
                
        elif self.scenario == 'dense_abrupt':
            # Amendment A5: breakpoints at regular intervals of `period` rounds.
            # Each phase draws a fresh, independent set of mean latencies from the
            # same family as abrupt_drift (random redraw — no forced swap), so the
            # optimal arm changes with probability ~ 1 - 1/K at each breakpoint.
            if not self.period or int(self.period) <= 0:
                raise ValueError("dense_abrupt requires a positive 'period'")
            P = int(self.period)
            n_phases = int(np.ceil(self.T / P))
            for ph in range(n_phases):
                means = rng.uniform(15.0, 95.0, size=self.K)
                start, end = ph * P, min((ph + 1) * P, self.T)
                self.theta[:, start:end] = (means / self.k)[:, None]

        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")
            
    def _calculate_expected_rewards(self):
        """
        Vectorized computation of analytical expected reward E[1/L_i(t)] under truncation:
        L_i(t) = max(L_raw, L_min)
        E[1/L_i(t)] = (1/L_min) * P(L_raw <= L_min) + E[1/L_raw * I(L_raw > L_min)]
        """
        # P(L_raw <= L_min) using self.theta (K, T)
        p_trunc = gamma.cdf(self.L_min, self.k, scale=self.theta)
        
        # E[1/L_raw * I(L_raw > L_min)] using self.theta (K, T)
        p_excess = 1.0 - gamma.cdf(self.L_min, self.k - 1, scale=self.theta)
        e_inv = (1.0 / (self.theta * (self.k - 1))) * p_excess
        
        self.expected_rewards = (1.0 / self.L_min) * p_trunc + e_inv

    def sample_latency(self, arm, t, rng):
        """
        Samples the realized latency and reward for the chosen arm at time step t.
        """
        scale = self.theta[arm, t]
        raw_latency = rng.gamma(self.k, scale)
        latency = max(raw_latency, self.L_min)
        reward = 1.0 / latency
        return latency, reward
