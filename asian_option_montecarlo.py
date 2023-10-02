import numpy as np

class AsianOption:
    def __init__(self, S0, K, T, r, sigma):
        self.S0 = S0  # Initial stock price
        self.K = K    # Strike price
        self.T = T    # Time to maturity in years
        self.r = r    # Risk-free rate
        self.sigma = sigma  # Volatility
    
    def simulate_price_paths(self, N, M):
        dt = self.T / N  # time step
        drift = (self.r - 0.5 * self.sigma**2) * dt
        shock = self.sigma * np.sqrt(dt)
        
        S = np.zeros((N + 1, M))
        S[0] = self.S0
        
        # Simulate stock price paths
        for t in range(1, N + 1):
            z = np.random.standard_normal(M)
            S[t] = S[t - 1] * np.exp(drift + shock * z)
        return S


class FixedStrikeAsianOption(AsianOption):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def price(self, N, M, control_variate=True):
        S = self.simulate_price_paths(N, M)
        A = np.mean(S[1:], axis=0)
        payoff = np.maximum(A - self.K, 0)
        
        if control_variate:
            G = np.exp(np.mean(np.log(S[1:]), axis=0))  # Geometric average
            geometric_option = np.exp(-self.r * self.T) * np.maximum(G - self.K, 0)
            geometric_option_true = np.exp(-self.r * self.T) * max(np.exp(np.mean(np.log(self.S0)) + (self.r - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * np.random.standard_normal()) - self.K, 0)
            payoff = payoff - (geometric_option - geometric_option_true)
            
        return np.exp(-self.r * self.T) * np.mean(payoff)


class FloatingStrikeAsianOption(AsianOption):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def price(self, N, M):
        S = self.simulate_price_paths(N, M)
        A = np.mean(S[:-1], axis=0)
        payoff = np.maximum(A - S[-1], 0)
        return np.exp(-self.r * self.T) * np.mean(payoff)


# Example
S0 = 100
K = 100
T = 1
r = 0.05
sigma = 0.2
N = 252
M = 10000

fixed_strike_asian_option = FixedStrikeAsianOption(S0, K, T, r, sigma)
print("Fixed Strike Asian Option Price with Control Variate: ", fixed_strike_asian_option.price(N, M))

floating_strike_asian_option = FloatingStrikeAsianOption(S0, K, T, r, sigma)
print("Floating Strike Asian Option Price: ", floating_strike_asian_option.price(N, M))
