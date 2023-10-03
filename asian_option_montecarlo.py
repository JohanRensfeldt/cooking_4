import numpy as np
import tqdm
import matplotlib.pyplot as plt

from scipy.stats import norm

epsilon = 0.0001
gamma = 1

class AsianOption:
    def __init__(self, S0, K, T, r, sigma, gamma):
        self.S0 = S0  # Initial stock price
        self.K = K  # Strike price
        self.T = T  # Time to maturity in years
        self.r = r  # Risk-free rate
        self.sigma = sigma  # Volatility
        self.gamma = gamma  # Gamma value for the model
    
    def simulate_price_paths(self, N, M, S0=None, gamma=None):
        if S0 is None:
            S0 = self.S0
        if gamma is None:
            gamma = self.gamma
            
        dt = self.T / N  # time step
        S = np.zeros((N + 1, M))
        S[0] = S0
        
        iterator = tqdm.tqdm(range(1, N + 1))
        
        for t in iterator:
            dw = np.random.standard_normal(M)   
            S[t] = S[t - 1] * np.exp((self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * dw)

       
        return S

class FixedStrikeAsianOption(AsianOption):
    def price(self, N, M, control_variate=True, S0=None, gamma=None):
        S = self.simulate_price_paths(N, M, S0, gamma)
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
    def price(self, N, M, control_variate=True, S0=None, gamma=None):
        S = self.simulate_price_paths(N, M, S0, gamma)
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
    
class Analysis:
    def __init__(self, FixedStrikeAsianOption, FloatingStrikeAsianOption):
        self.FixedStrikeAsianOption = FixedStrikeAsianOption
        self.FloatingStrikeAsianOption = FloatingStrikeAsianOption

    def print_price_fixed_strike(self, N, M, S0=None, gamma=None):
        price = self.FixedStrikeAsianOption.price(N, M, S0=S0, gamma=gamma)
        print("Fixed Strike Asian Option Price: ", price)
    
    def print_price_floating_strike(self, N, M, S0=None, gamma=None):
        price = self.FloatingStrikeAsianOption.price(N, M, S0=S0, gamma=gamma)
        print("Floating Strike Asian Option Price: ", price)

    def price_vs_s0(self, N, M, OptionType):
            
        if OptionType == "Fixed":
            S0_range = np.arange(10, 150, 5)
            iterator = tqdm.tqdm(S0_range)
            prices = np.array([self.FixedStrikeAsianOption.price(N, M, S0=S0) for S0 in iterator])
            print(prices)
            plt.plot(S0_range, prices)
            plt.xlabel('Initial Stock Price (S0)')
            plt.ylabel('Option Price')
            plt.title('Fixed Strike Asian Option Price vs Initial Stock Price')
            plt.show()
        elif OptionType == "Floating":
            S0_range = np.arange(10, 150, 5)
            iterator = tqdm.tqdm(S0_range)
            prices = np.array([self.FloatingStrikeAsianOption.price(N, M, S0=S0) for S0 in iterator])
            print(prices)
            plt.plot(S0_range, prices)
            plt.xlabel('Initial Stock Price (S0)')
            plt.ylabel('Option Price')
            plt.title('Fixed Strike Asian Option Price vs Initial Stock Price')
            plt.show()
        else:
            raise ValueError("OptionType must be either Fixed or Floating")
        
    def delta_fixed(self, N, M, S0, epsilon=0.1, gamma=gamma):
        delta = np.zeros(5)
        for i in range(len(delta)):
            price = self.FixedStrikeAsianOption.price(N, M, S0=S0, gamma=gamma)
            price_epsilon = self.FixedStrikeAsianOption.price(N, M, S0=S0 + epsilon, gamma=gamma)
            delta[i] = (price_epsilon - price) / epsilon
        return np.mean(delta)
    
    def delta_floating(self, N, M, S0, epsilon=0.1, gamma=gamma):
        delta = np.zeros(5)
        for i in range(len(delta)):
            price = self.FloatingStrikeAsianOption.price(N, M, S0=S0, gamma=gamma)
            price_epsilon = self.FloatingStrikeAsianOption.price(N, M, S0=S0 + epsilon, gamma=gamma)
            delta[i] = (price_epsilon - price) / epsilon
        return np.mean(delta)

    def asian_option_turnbull_wakeman(self, S0, K, T, r, sigma, N):
        adjusted_sigma = sigma * np.sqrt((2 * N + 1) * (N + 1) / (6 * (N ** 2)))
        adjusted_r = 0.5 * adjusted_sigma ** 2 + (r * (N + 1)) / (2 * N)  
        
        d1 = (np.log(S0 / K) + (adjusted_r + 0.5 * adjusted_sigma ** 2) * T) / (adjusted_sigma * np.sqrt(T))
        d2 = d1 - adjusted_sigma * np.sqrt(T)
        
        return np.exp(-r * T) * (S0 * np.exp(adjusted_r * T) * norm.cdf(d1) - K * norm.cdf(d2))
