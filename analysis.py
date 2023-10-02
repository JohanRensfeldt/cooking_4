from asian_option_montecarlo import FixedStrikeAsianOption, FloatingStrikeAsianOption, AsianOption
import numpy as np
import matplotlib.pyplot as plt
import tqdm

S0 = 100
K = 50
T = 1
r = 0.05
sigma = 0.2
N = 1000
M = 100000

class Analysis:
    def __init__(self, FixedStrikeAsianOption, FloatingStrikeAsianOption):
        self.FixedStrikeAsianOption = FixedStrikeAsianOption
        self.FloatingStrikeAsianOption = FloatingStrikeAsianOption

    def print_price_fixed_strike(self, N, M, S0=None):
        price = self.FixedStrikeAsianOption.price(N, M, S0=S0)
        print("Fixed Strike Asian Option Price: ", price)
    
    def print_price_floating_strike(self, N, M, S0=None):
        price = self.FloatingStrikeAsianOption.price(N, M, S0=S0)
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

if __name__ == "__main__":
    fixed_strike_asian_option = FixedStrikeAsianOption(S0, K, T, r, sigma)  # create instances before passing
    floating_strike_asian_option = FloatingStrikeAsianOption(S0, K, T, r, sigma)  # create instances before passing
    
    analysis = Analysis(fixed_strike_asian_option, floating_strike_asian_option)
    #analysis.print_price_fixed_strike(N, M, S0=S0)
    #analysis.print_price_floating_strike(N, M, S0=S0)
    analysis.price_vs_s0(N, M, OptionType="Floating")
