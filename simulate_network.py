import math
import numpy as np
from Fundamentalist import Fundamentalist
from Chartist import Chartist
import matplotlib.pyplot as plt
from Network import Network


class Market:
    def __init__(self, network, mu, prices, beta, alpha_w, alpha_O, alpha_p):
        self.network = network
        self.mu = mu
        self.prices = prices  # Initial price
        self.beta = beta
        self.alpha_w = alpha_w
        self.alpha_O = alpha_O
        self.alpha_p = alpha_p
        self.A = [0,0]
        self.average_demand = 0  # Total demand in the market
        
    def calculate_A(self, t):
        self.A.append(self.alpha_w * (self.fundamentalist.Wf[t] - self.chartist.Wc[t]) + self.alpha_O + self.alpha_p * (self.fundamentalist.pstar - self.prices[t])**2)

    def update_strategies(self, t):
        # loop over all agents and update their strategies
        for agent in self.network.trader_dictionary.values():
            agent_node_number = agent.node_number
            agent_profit_list = agent.profits
            neighbors = self.network.get_neighbors(agent.node_number)
            profits = [neighbor.profit for neighbor in neighbors]
            if agent.profit < np.max(profits):
                self.network.trader_dictionary[agent_node_number] = neighbors[np.argmax(profits)]
                self.network.trader_dictionary[agent_node_number].profit_list = agent_profit_list
                self.network.trader_dictionary[agent_node_number].node_number = agent_node_number

    def calculate_demands(self, t):
        # loop over all agents and update their demands
        keys = self.network.trader_dictionary.keys()
        demands = []

        for agent in self.network.trader_dictionary.values():
            demands.append(agent.calculate_demand(self.prices, t))
        self.average_demand =  np.sum(demands) / len(demands)

    def update_price(self, t):
        new_price = self.prices[t] + self.mu * self.average_demand
        self.prices.append(new_price)

def run_simulation(initial_price, time_steps):
    network = Network(network_type='barabasi', number_of_traders = 100, percent_fund=0.5, percent_chartist=0.5,new_node_edges=3)
    network.create_network()
    prices = [initial_price, initial_price, initial_price]  # Ensure enough initial prices for the first calculations
    market = Market(network, mu=0.01, prices=prices, beta=1, alpha_w=2668, alpha_O=2.1, alpha_p=0)

    for t in range(2, time_steps):
        
        # update strategies for all agents
        # market.network.update_strategies(t)
        # Calculate the demands of all agents
        market.calculate_demands(t)

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
    plt.plot(prices)
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

