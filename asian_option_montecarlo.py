import numpy as np
import tqdm 
from scipy.stats import norm


class AsianOption:
    def __init__(self, S0, K, T, r, sigma):
        self.S0 = S0  # Initial stock price
        self.K = K    # Strike price
        self.T = T    # Time to maturity in years
        self.r = r    # Risk-free rate
        self.sigma = sigma  # Volatility
    
    def simulate_price_paths(self, N, M, S0=None):
        if S0 is None:
            S0 = self.S0
        
        dt = self.T / N  # time step
        drift = (self.r - 0.5 * self.sigma**2) * dt
        shock = self.sigma * np.sqrt(dt)
        
        S = np.zeros((N + 1, M))
        S[0] = S0
        
        iterator = tqdm.tqdm(range(1, N + 1))

        for t in iterator:
            z = np.random.standard_normal(M)
            S[t] = S[t - 1] * np.exp(drift + shock * z)
        
        return S


class FixedStrikeAsianOption(AsianOption):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class FixedStrikeAsianOption(AsianOption):
    def price(self, N, M, control_variate=True, S0=None):
        S = self.simulate_price_paths(N, M, S0)
        A = np.mean(S, axis=0) 
        
        payoff = np.maximum(A - self.K, 0)
        
        if control_variate:
            G = np.exp(np.mean(np.log(S), axis=0))
            geometric_option = np.exp(-self.r * self.T) * np.maximum(G - self.K, 0)
            
            adjusted_sigma = self.sigma / np.sqrt(3)  # adjusting the sigma as per geometric Asian option
            adjusted_r = 0.5 * (self.r + adjusted_sigma**2)  # adjusted rate for geometric Asian option
            
            d1 = (np.log(S0 / self.K) + (adjusted_r + 0.5 * adjusted_sigma**2) * self.T) / (adjusted_sigma * np.sqrt(self.T))
            d2 = d1 - adjusted_sigma * np.sqrt(self.T)
            
            geometric_option_true_value = np.exp(-self.r * self.T) * (S0 * np.exp(adjusted_r * self.T) * norm.cdf(d1) - self.K * norm.cdf(d2))
            
            payoff += geometric_option_true_value - geometric_option
            
        return np.exp(-self.r * self.T) * np.mean(payoff)


class FloatingStrikeAsianOption(AsianOption):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def price(self, N, M, control_variate=True, S0=None):
        S = self.simulate_price_paths(N, M, S0)
        A = np.mean(S[:-1], axis=0)  # Arithmetic mean of the prices excluding the last one
        
        payoff = np.maximum(A - S[-1], 0)  # Standard payoff for floating strike Asian option
        
        if control_variate:
            G = np.exp(np.mean(np.log(S[:-1]), axis=0))  # Geometric mean of the prices excluding the last one
            geometric_option = np.exp(-self.r * self.T) * np.maximum(G - S[-1], 0)  # Geometric Asian option payoff
            
            # Implement the true value of geometric Asian option here, similar to the FixedStrikeAsianOption.
            adjusted_sigma = self.sigma / np.sqrt(3)
            adjusted_r = 0.5 * (self.r + adjusted_sigma**2)
            
            d1 = (np.log(S0 / self.K) + (adjusted_r + 0.5 * adjusted_sigma**2) * self.T) / (adjusted_sigma * np.sqrt(self.T))
            d2 = d1 - adjusted_sigma * np.sqrt(self.T)
            
            geometric_option_true_value = np.exp(-self.r * self.T) * (S0 * np.exp(adjusted_r * self.T) * norm.cdf(d1) - self.K * norm.cdf(d2))
            
            payoff += geometric_option_true_value - geometric_option  # Adjusting the payoff with control variate
            
        return np.exp(-self.r * self.T) * np.mean(payoff)