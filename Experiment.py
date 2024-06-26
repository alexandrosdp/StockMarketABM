
"""
A Class to conduct experiments with the simulation
"""

from Network import Network
from simulate_network import Market
from utils import progress_bar, clear_progress_bar
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import pandas as pd


class Experiment():

    def __init__(self, initial_price, time_steps, network_type='barabasi', number_of_traders=150, percent_fund=0.5, percent_chartist=0.5, percent_rational=0.50, percent_risky=0.50, high_lookback=5, low_lookback=1, high_risk=0.50, low_risk=0.10, new_node_edges=5, connection_probability=0.5, mu=0.01, beta=1, alpha_w=2668, alpha_O=2.1, alpha_p=0):

        self.initial_price = initial_price
        self.time_steps = time_steps
        self.network_type = network_type
        self.number_of_traders = number_of_traders
        self.percent_fund = percent_fund
        self.percent_chartist = percent_chartist
        self.percent_rational = percent_rational
        self.percent_risky = percent_risky
        self.high_lookback = high_lookback
        self.low_lookback = low_lookback
        self.high_risk = high_risk
        self.low_risk = low_risk
        self.new_node_edges = new_node_edges
        self.connection_probability = connection_probability
        self.mu = mu
        self.beta = beta
        self.alpha_w = alpha_w
        self.alpha_O = alpha_O
        self.alpha_p = alpha_p

    def run_simulation(self):
        network = Network(network_type=self.network_type, number_of_traders=self.number_of_traders, percent_fund=self.percent_fund, percent_chartist=self.percent_chartist, percent_rational=self.percent_rational, percent_risky=self.percent_risky,
                          high_lookback=self.high_lookback, low_lookback=self.low_lookback, high_risk=self.high_risk, low_risk=self.low_risk, new_node_edges=self.new_node_edges, connection_probability=self.connection_probability)
        network.create_network()
        # Ensure enough initial prices for the first calculations
        prices = [self.initial_price, self.initial_price, self.initial_price]
        market = Market(network, mu=self.mu, prices=prices, beta=self.beta,
                        alpha_w=self.alpha_w, alpha_O=self.alpha_O, alpha_p=self.alpha_p)

        for t in range(2, self.time_steps):

            for agent in network.trader_dictionary.values():
                agent.update_performance(market.prices, t)
                agent.update_wealth(t)

            # update strategies for all agents
            market.update_strategies(t)
            # Calculate the demands of all agents
            market.calculate_demands(t)

            # Update price
            market.update_price(t)

            progress_bar(t / self.time_steps)

        clear_progress_bar()

        # network.display_network()
        return market

    def analyze_autocorrelation_of_returns(self, prices):    
        returns = np.diff(prices)

        # plot（ACF）
        plot_acf(returns, lags=40)
        plt.title('Autocorrelation Function (ACF) of Returns')
        plt.xlabel('Lags')
        plt.ylabel('Autocorrelation')
        plt.show()

        # Ljung-Box test
        ljung_box_result = acorr_ljungbox(returns, lags=[20], return_df=True)
        print("Ljung-Box Test Results:")
        print(ljung_box_result)


    def analyze_volatility_clustering(self):
        prices = self.run_simulation().prices

        # Calculate log returns.
        returns = np.log(prices[1:]) - np.log(prices[:-1])
        squared_returns = returns ** 2

        # Create a figure to house both subplots.
        plt.figure(figsize=(12, 6))

        # for squared returns
        plt.subplot(1, 2, 1)
        plt.plot(squared_returns, label='Squared Returns')
        plt.title('Squared Returns Over Time')
        plt.xlabel('Time')
        plt.ylabel('Squared Returns')
        plt.legend()

        # for the ACF of squared returns
        plt.subplot(1, 2, 2)
        plot_acf(squared_returns, lags=20, alpha=0.05)
        plt.title('ACF of Squared Returns')
        plt.xlabel('Lags')
        plt.ylabel('Autocorrelation')

        plt.tight_layout()
        plt.show()

    def crash_experiment(self):

    
        market = self.run_simulation()

        "TODO: Add logic to check for flash crash"

        crash_indices = []

        period = 30
        
        drop_magnitude = 0

        biggest_drop = None

        # get the index of highest negative returns
        indices = np.argsort(np.diff(market.prices))[0:30]

        drop_lengths = []

        for index in indices:

            start_index = index

            while (np.diff(market.prices)[start_index - 1] < 0) and start_index > 0:

                start_index -= 1

            end_index = start_index

            while (np.diff(market.prices)[end_index + 1] < 0) and end_index < len(np.diff(market.prices)) - 3:
                
                end_index += 1
            
            drop_lengths.append((start_index, end_index))


            # calculate drop magnitude
            if  np.sum(np.diff(market.prices)[start_index:end_index+1]) < drop_magnitude:
                drop_magnitude = np.sum(np.diff(market.prices)[start_index:end_index+1])
                biggest_drop = (start_index, end_index)

        if np.max(market.prices[end_index + 1: np.maximum(end_index+1+period, len(market.prices)-1)]) - market.prices[end_index+1] > np.sum(np.diff(market.prices)[start_index:end_index+1]) * 0.5 and np.sum(np.diff(market.prices)[start_index:end_index+1]) < - 0.03:
                print("Flash Crash Detected at time: ", index)
                crash_indices.append((start_index, end_index))

        if drop_magnitude <= -0.07:
            print("CRASH DETECTED")
            return 1
        else:
            return 0


        # print("Biggest Drop: ", drop_magnitude)
        # plt.figure()
        # plt.plot(np.exp(market.prices))
        # plt.scatter(biggest_drop[0], np.exp(market.prices)[biggest_drop[0]], color='red', label='Flash Crash start')
        # plt.scatter(biggest_drop[1]+1, np.exp(market.prices)[biggest_drop[1]+1], color='red', label='Flash Crash start')
        # plt.xlabel('Time')
        # plt.ylabel('Price')
        # plt.title('Discrete Choice Approach: Flash Crash Detection')
        # plt.show()

        # rr = np.array(market.prices[1:self.time_steps+1]) - np.array(market.prices[0:self.time_steps])
        # plt.figure()
        # plt.plot(range(self.time_steps), rr)
        # plt.xlabel('Time')
        # plt.ylabel('Returns')
        # plt.title('Discrete Choice Approach:Wealth')
        # plt.show()


    def multiple_runs_crash(self,n_runs):

        """Running the crash experiment multiple times"""


        crash_count = 0

        for i in range(n_runs):

            crash = self.crash_experiment() #Crash returns either 0  or 1
            crash_count += crash

        return crash_count


    def fat_tail_experiment(self):

        market = self.run_simulation()

        "TODO: Add logic to check for flat tails"





if __name__ == "__main__":


    params = {'initial_price': 0,
        'time_steps': 2000,
        'network_type': 'barabasi',
        'number_of_traders': 150,
        'percent_fund': 0.50,
        'percent_chartist': 0.50,
        'percent_rational': 0,
        'percent_risky': 0.50,
        'high_lookback': 10,
        'low_lookback': 1,
        'high_risk': 0.50,
        'low_risk': 0.10,
        'new_node_edges': 5,
        'connection_probability': 0.50,
        'mu': 0.01,
        'beta': 1,
        'alpha_w': 2668,
        'alpha_O': 2.1,
        'alpha_p': 0}


    experiment = Experiment(
        initial_price=params['initial_price'],
        time_steps=params['time_steps'],
        network_type=params['network_type'],
        number_of_traders=params['number_of_traders'],
        percent_fund=params['percent_fund'],
        percent_chartist=params['percent_chartist'],
        percent_rational=params['percent_rational'],
        percent_risky=params['percent_risky'],
        high_lookback=params['high_lookback'],
        low_lookback=params['low_lookback'],
        high_risk=params['high_risk'],
        low_risk=params['low_risk'],
        new_node_edges=params['new_node_edges'],
        connection_probability=params['connection_probability'],
        mu=params['mu'],
        beta=params['beta'],
        alpha_w=params['alpha_w'],
        alpha_O=params['alpha_O'],
        alpha_p=params['alpha_p']
    )
    
    crash_results_rationality = pd.read_csv('crash_results_rationality.csv')
    
    results_df = pd.DataFrame(params, index= [0])

    crash_count = experiment.multiple_runs_crash(1)

    results_df.loc[0,'No. of crashes'] = crash_count

    crash_results_rationality = pd.concat([crash_results_rationality, results_df], ignore_index=True)

    crash_results_rationality.to_csv('crash_results_rationality.csv')

    print(crash_results_rationality)

    
     
    # experiment.multiple_runs_crash(n_runs=30)

    # for params in experiment_params:
    #     num_crashes = run_experiment(params)
    #     params['num_crashes'] = num_crashes
    #     results_df = results_df.append(params, ignore_index=True)


    #  experiment.analyze_autocorrelation_of_returns(market.prices)
    # Analyze autocorrelation of prices
    # experiment.analyze_autocorrelation(market.prices)
    # do we do initial price from 0? here l set it as 1
    # Ensure initial_price is more than 0 to avoid log(0) issues.
    # experiment = Experiment(initial_price=1, time_steps=100)
    # experiment.analyze_volatility_clustering()
