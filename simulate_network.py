import math
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import kurtosis, norm
from statsmodels.graphics.tsaplots import plot_acf

from Fundamentalist import Fundamentalist
from Chartist import Chartist
from Network import Network
from utils import progress_bar, clear_progress_bar

class Market:
    """
    A class representing the market environment.
    
    Attributes:
    ----------
    network : Network
        The network of traders.
    mu : float
        The market's sensitivity to average demand.
    prices : list
        A list of market prices over time.
    beta : float
        A parameter influencing trader behavior.
    alpha_w, alpha_O, alpha_p : float
        Parameters influencing the market dynamics.
    A : list
        A list to store the calculated A values.
    average_demand : float
        The average demand in the market.
    """
    def __init__(self, network, mu, prices, beta, alpha_w, alpha_O, alpha_p):
        self.network = network
        self.mu = mu
        self.prices = prices  # Initial price
        self.beta = beta
        self.alpha_w = alpha_w
        self.alpha_O = alpha_O
        self.alpha_p = alpha_p
        self.A = [0, 0]
        self.average_demand = 0  # Total demand in the market
        
    def calculate_A(self, t):
        """
        Calculate the A value for time t.
        
        Parameters:
        ----------
        t : int
            The current time step.
        """
        self.A.append(self.alpha_w * (self.fundamentalist.Wf[t] - self.chartist.Wc[t]) + 
                      self.alpha_O + 
                      self.alpha_p * (self.fundamentalist.pstar - self.prices[t])**2)

    def update_strategies(self, t):
        """
        Update strategies for all agents based on their performance.
        
        Parameters:
        ----------
        t : int
            The current time step.
        """
        for agent in self.network.trader_dictionary.values():
            agent_node_number = agent.node_number
            agent_W = agent.W
            agent_G = agent.G
            agent_D = agent.D
            agent_lookback_period = agent.lookback_period
            agent_max_risk = agent.max_risk

            neighbor_node_numbers = self.network.get_neighbors(agent.node_number)
            neighbors = [self.network.trader_dictionary[neighbor] for neighbor in neighbor_node_numbers]
            
            performances = [self.calculate_average_performance(neighbor, agent_lookback_period) for neighbor in neighbors]
            if self.calculate_average_performance(agent, agent_lookback_period) < np.max(performances):
                self.network.trader_dictionary[agent_node_number] = neighbors[np.argmax(performances)]
                self.network.trader_dictionary[agent_node_number].node_number = agent_node_number
                self.network.trader_dictionary[agent_node_number].W = agent_W 
                self.network.trader_dictionary[agent_node_number].G = agent_G
                self.network.trader_dictionary[agent_node_number].D = agent_D
                self.network.trader_dictionary[agent_node_number].lookback_period = agent_lookback_period
                self.network.trader_dictionary[agent_node_number].max_risk = agent_max_risk

    def calculate_average_performance(self, agent, agent_lookback_period):
        """
        Calculate the average performance of an agent.
        
        Parameters:
        ----------
        agent : Agent
            The agent whose performance is to be calculated.
        agent_lookback_period : int
            The period over which the performance is calculated.
        
        Returns:
        -------
        float
            The average performance of the agent.
        """
        if len(self.prices) < agent_lookback_period:
            return np.mean(agent.G)
        else:
            return np.mean(agent.G[-agent_lookback_period:])

    def calculate_demands(self, t):
        """
        Calculate the demands of all agents at time t.
        
        Parameters:
        ----------
        t : int
            The current time step.
        """
        keys = self.network.trader_dictionary.keys()
        demands = []

        for agent in self.network.trader_dictionary.values():
            demands.append(agent.calculate_demand(self.prices, t))
        self.average_demand = np.sum(demands) / len(demands)

    def update_price(self, t):
        """
        Update the market price based on the average demand.
        
        Parameters:
        ----------
        t : int
            The current time step.
        """
        new_price = self.prices[t] + self.mu * self.average_demand
        self.prices.append(new_price)

def run_simulation(initial_price, time_steps):
    """
    Run the market simulation.
    
    Parameters:
    ----------
    initial_price : float
        The initial price of the market.
    time_steps : int
        The number of time steps to simulate.
    
    Returns:
    -------
    Market
        The market object after the simulation.
    """
    network = Network(network_type='barabasi', number_of_traders=150, percent_fund=0.5, percent_chartist=0.5, new_node_edges=5, connection_probability=0.5)
    network.create_network()
    prices = [initial_price, initial_price, initial_price]  # Ensure enough initial prices for the first calculations
    market = Market(network, mu=0.03, prices=prices, beta=1, alpha_w=2668, alpha_O=2.1, alpha_p=0)

    for t in range(2, time_steps):
        for agent in network.trader_dictionary.values():
            agent.update_performance(market.prices, t)
            agent.update_wealth(t)
        
        # Update strategies for all agents
        market.update_strategies(t)
        # Calculate the demands of all agents
        market.calculate_demands(t)
        # Update price
        market.update_price(t)

        progress_bar(t / time_steps) 

    clear_progress_bar()
    
    network.display_network()

    return market
