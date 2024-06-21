import math
import numpy as np
from Fundamentalist import Fundamentalist
from Chartist import Chartist
import matplotlib.pyplot as plt
from Network import Network
import statsmodels.api as sm
from scipy.stats import kurtosis
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf
from utils import progress_bar, clear_progress_bar




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
            agent_W = agent.W
            agent_G = agent.G
            agent_D = agent.D
            agent_lookback_period = agent.lookback_period
            agent_max_risk = agent.max_risk

            neighbor_node_numbers = self.network.get_neighbors(agent.node_number)
            neighbors = [self.network.trader_dictionary[neighbor] for neighbor in neighbor_node_numbers]
            
            perfomances = [self.calculate_average_performance(neighbor, agent_lookback_period) for neighbor in neighbors]
            if self.calculate_average_performance(agent, agent_lookback_period) < np.max(perfomances):
                self.network.trader_dictionary[agent_node_number] = neighbors[np.argmax(perfomances)]  
                self.network.trader_dictionary[agent_node_number].node_number = agent_node_number
                self.network.trader_dictionary[agent_node_number].W = agent_W 
                self.network.trader_dictionary[agent_node_number].G = agent_G
                self.network.trader_dictionary[agent_node_number].D = agent_D
                self.network.trader_dictionary[agent_node_number].lookback_period = agent_lookback_period
                self.network.trader_dictionary[agent_node_number].max_risk = agent_max_risk


    def calculate_average_performance(self, agent, agent_lookback_period):
        if len(self.prices) < agent_lookback_period:
            return np.mean(agent.G)
        else:
            return np.mean(agent.G[-agent_lookback_period:])

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
    network = Network(network_type= 'barabasi', number_of_traders = 30, percent_fund=0.5, percent_chartist=0.5,new_node_edges=3, connection_probability=0.1)
    network.create_network()
    prices = [initial_price, initial_price, initial_price]  # Ensure enough initial prices for the first calculations
    market = Market(network, mu=0.01, prices=prices, beta=1, alpha_w=2668, alpha_O=2.1, alpha_p=0)

    for t in range(2, time_steps):

        for agent in network.trader_dictionary.values():
            agent.update_performance(market.prices, t)
            agent.update_wealth(t)
        
        # update strategies for all agents
        market.update_strategies(t)
        # Calculate the demands of all agents
        market.calculate_demands(t)

        # Update price
        market.update_price(t)

        progress_bar(t / time_steps) 

    clear_progress_bar()

    return market

if __name__ == '__main__':
    initial_price = 0
    T = 10000
    pstar = 0

    market = run_simulation(initial_price, T)

    prices = market.prices

    # store the profits of the agents
    profit_dict = {}
    for agent in market.network.trader_dictionary.values():
        profit_dict[agent.node_number] = agent.G

    wealth_dict = {}
    for agent in market.network.trader_dictionary.values():
        wealth_dict[agent.node_number] = agent.W
    
    demand_dict = {}
    for agent in market.network.trader_dictionary.values():
        demand_dict[agent.node_number] = agent.D

    # plot the wealth of the agents
    plt.figure()
    for agent in market.network.trader_dictionary.values():
        plt.plot(agent.W)
    plt.xlabel('Time')
    plt.ylabel('Wealth')
    plt.title('Discrete Choice Approach: Wealth')
    plt.show()

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

    plt.hist(rr.flatten(),bins=80, density=True, alpha=0.8, color='b', edgecolor='black', linewidth=1.2)
    # Fit a normal distribution to the data
    mu, std = norm.fit(rr.flatten())
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)

    # Plot normal distribution curve
    plt.plot(x, p, 'k', linewidth=2)
    plt.show()

