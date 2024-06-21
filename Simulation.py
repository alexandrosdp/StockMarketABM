import math
import numpy as np
from Fundamentalist import Fundamentalist
from Chartist import Chartist
import matplotlib.pyplot as plt


class Market:
    def __init__(self, fundamentalist, chartist, mu, prices, beta, alpha_w, alpha_O, alpha_p):
        self.fundamentalist = fundamentalist
        self.chartist = chartist
        self.mu = mu
        self.prices = prices  # Initial price
        self.beta = beta
        self.alpha_w = alpha_w
        self.alpha_O = alpha_O
        self.alpha_p = alpha_p
        self.A = [0,0]
        self.Nf = [0.5,0.5]
        self.Nc = [0.5,0.5]

    def update_market_fractions(self, t):
        self.Nf.append(1 / (1 + np.exp(-self.beta * self.A[t-1])))
        self.Nc.append(1 - self.Nf[t])

    def calculate_A(self, t):
        self.A.append(self.alpha_w * (self.fundamentalist.W[t] - self.chartist.W[t]) + self.alpha_O + self.alpha_p * (self.fundamentalist.pstar - self.prices[t])**2)

    def calculate_demands(self, t):
        self.fundamentalist.calculate_demand(self.prices, t)
        self.chartist.calculate_demand(self.prices, t)

    def update_price(self, t):
        new_price = self.prices[t] + self.mu * ( self.Nf[t] * self.fundamentalist.D[t] + self.Nc[t] * self.chartist.D[t])
        self.prices.append(new_price)

def run_simulation(initial_price, time_steps):
    fundamentalist = Fundamentalist(node_number=1,eta=0.991, alpha_w=2668, alpha_O=2.1, alpha_p=0, phi=1.00, sigma_f=0.681, pstar=0)
    chartist = Chartist(node_number=2, eta=0.991, chi=1.20, sigma_c=1.724)
    prices = [initial_price, initial_price, initial_price]  # Ensure enough initial prices for the first calculations
    market = Market(fundamentalist, chartist, mu=0.01, prices=prices, beta=1, alpha_w=2668, alpha_O=2.1, alpha_p=0)

    for t in range(2, time_steps):
        # Update portfolio performance
        market.fundamentalist.update_performance(market.prices, t)
        market.chartist.update_performance(market.prices, t)  

        #summarize performance over time
        market.fundamentalist.update_wealth(t)
        market.chartist.update_wealth(t)

        # type fractions
        market.update_market_fractions(t)

        # The A[t] dynamic is set up to handle several models
        market.calculate_A(t)

        # Calculate demands
        market.fundamentalist.calculate_demand(market.prices,t)
        market.chartist.calculate_demand(market.prices,t)
 
        # Update price
        market.update_price(t)

    return market.prices

if __name__ == '__main__':
    initial_price = 0
    T = 7000
    pstar = 0

    prices = run_simulation(initial_price, T)

    rr = np.array(prices[1:T+1]) - np.array(prices[0:T])

    plt.figure()
    plt.plot(np.exp(prices))
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Discrete Choice Approach: Wealth')
    plt.show()

    # plot returns
    fig_r, ax_r = plt.subplots()
    ax_r.plot(range(T), rr)
    plt.xlabel('Time')
    plt.ylabel('Returns')
    ax_r.set_title('Discrete Choice Approach:Wealth')
    plt.show()

