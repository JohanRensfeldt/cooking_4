from asian_option_montecarlo import FixedStrikeAsianOption, FloatingStrikeAsianOption, Analysis
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy.stats import norm

S0 = 100
K = 110
T = 1
r = 0.05
sigma = 0.2
gamma = 1
N = 1000
M = 10000

if __name__ == "__main__":
    fixed_strike_asian_option = FixedStrikeAsianOption(S0, K, T, r, sigma, gamma)  # create instances before passing
    floating_strike_asian_option = FloatingStrikeAsianOption(S0, K, T, r, sigma, gamma)  # create instances before passing
    analysis = Analysis(fixed_strike_asian_option, floating_strike_asian_option)

    analytic = analysis.asian_option_turnbull_wakeman(S0, K, T, r, sigma, N)

    print("Analytic Asian Option Price: ", analytic)

    delta_fixed = analysis.delta_fixed(N, M, S0, gamma=gamma)

    print("Delta Fixed Strike Asian Option: ", delta_fixed)

    delta_floating = analysis.delta_floating(N, M, S0, gamma=gamma)

    print("Delta Floating Strike Asian Option: ", delta_floating)

    analysis.print_price_fixed_strike(N, M, S0=S0, gamma=gamma)
    analysis.print_price_floating_strike(N, M, S0=S0, gamma=gamma)
    analysis.price_vs_s0(N, M, OptionType="Floating")


