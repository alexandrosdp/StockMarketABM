
"""
A Class to conduct experiments with the simulation
"""
import streamlit as st
from Network import Network
from simulate_network import Market
from utils import progress_bar, clear_progress_bar
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import kurtosis
from scipy.stats import norm
from tqdm import tqdm


class Experiment():

    def __init__(self, initial_price, time_steps, network_type='small_world', number_of_traders=150, percent_fund=0.5, percent_chartist=0.5, percent_rational=0.50, percent_risky=0.50, high_lookback=5, low_lookback=1, high_risk=0.50, low_risk=0.10, new_node_edges=5, connection_probability=0.5, mu=0.01, beta=1, alpha_w=2668, alpha_O=2.1, alpha_p=0):

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

            # progress_bar(t / self.time_steps)

        clear_progress_bar()

        # network.display_network()
        return market

    def analyze_autocorrelation_of_returns(self, prices):
        returns = np.diff(prices)
        # Ljung-Box test
        ljung_box_result = acorr_ljungbox(returns, lags=[20], return_df=True)
        
        plt.figure()
        plot_acf(returns, lags=40)
        plt.title('Autocorrelation Function (ACF) of Returns')
        plt.xlabel('Lags')
        plt.ylabel('Autocorrelation')
        st.pyplot(plt.gcf())  # Display the figure in Streamlit
        #plt.show()
        
        return ljung_box_result

    def analyze_volatility_clustering(self, prices):

        # Calculate log returns
        prices = np.exp(np.array(prices))
        # Convert the numpy array to a pandas Series
        # Calculate returns
        returns = prices[1:] - prices[:-1]
        # ARCH Test For Volatility Clustering
        arch_test = sm.stats.diagnostic.het_arch(returns.flatten())
        if arch_test[1] < 0.05:
            return 1, arch_test[1]
        else:
            return 0, arch_test[1]
        
        # return squared_returns

    def crash_experiment(self):

        market = self.run_simulation()

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

            counter = 1

            if end_index == len(np.diff(market.prices)) - 1:

                end_index = start_index
            else:

                while (np.diff(market.prices)[end_index + 1] < 0) and end_index < len(np.diff(market.prices)) - 3:

                    end_index += 1

            drop_lengths.append((start_index, end_index))

            # calculate drop magnitude
            if np.sum(np.diff(market.prices)[start_index:end_index+1]) < drop_magnitude:
                drop_magnitude = np.sum(np.diff(market.prices)[
                                        start_index:end_index+1])
                biggest_drop = (start_index, end_index)

        if np.max(market.prices[end_index + 1: np.maximum(end_index+1+period, len(market.prices)-1)]) - market.prices[end_index+1] > np.sum(np.diff(market.prices)[start_index:end_index+1]) * 0.5 and np.sum(np.diff(market.prices)[start_index:end_index+1]) < - 0.03:
            print("Flash Crash Detected at time: ", index)
            crash_indices.append((start_index, end_index))

        drop_magnitude_list = []

        if drop_magnitude <= -0.07:
            return 1, drop_magnitude
        else:
            return 0, drop_magnitude

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

    def multiple_runs_crash(self, n_runs):
        """Running the crash experiment multiple times"""

        crash_count = 0
        drop_magintude_list = []

        for i in range(n_runs):

            crash, drop_magintude = self.crash_experiment()  # Crash returns either 0  or 1
            crash_count += crash
            drop_magintude_list.append(drop_magintude)

        return crash_count, drop_magintude_list

    def fat_tail_experiment(self, T, prices,  plot=False):
        rr = np.array(prices[1:T+1]) - np.array(prices[0:T])
        if plot:
            plt.figure()
            sm.qqplot(rr.flatten(), line='s')  # 's' line fit standardizes the data to have the same scale
            plt.title('QQ Plot')
            plt.xlabel('Returns')
            plt.ylabel('Frequency')
            st.pyplot(plt.gcf())  # Display the figure in Streamlit
            plt.figure()
            plt.hist(rr.flatten(), bins = 50, density=True, alpha=0.8, color='b', edgecolor='black', linewidth=1.2)
            # Fit a normal distribution to the data
            mu, std = norm.fit(rr.flatten())
            xmin, xmax = plt.xlim()
            # Plot normal distribution curve
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2)
            st.pyplot(plt.gcf())  # Display the figure in Streamlit
            #plt.show()
        # Calculate kurtosis (K value)
        kurtosis_value = kurtosis(rr.flatten())
        return kurtosis_value


if __name__ == "__main__":
    ks_mean = []
    ks_ci_lower = []
    ks_ci_upper = []
    index = np.arange(0.01, 0.1, 0.01)
    for mu1 in index:
        ks = []
        for _ in tqdm(range(30)):
            experiment = Experiment(initial_price=0,
                                time_steps=500,
                                network_type="barabasi",
                                number_of_traders=150,
                                percent_fund=0.50,
                                percent_chartist=0.50,
                                percent_rational=0.50,
                                percent_risky=0.050,
                                high_lookback=10,
                                low_lookback=1,
                                high_risk=0.50,
                                low_risk=0.10,
                                new_node_edges=5,
                                connection_probability=0.50,
                                mu=mu1,
                                beta=1,
                                alpha_w=2668,
                                alpha_O=2.1,
                                alpha_p=0
                                )
            market = experiment.run_simulation()
            ks.append(experiment.fat_tail_experiment(500, market.prices))
        ks = np.array(ks)
        ks_mean.append(np.mean(ks))
        ks_ci_lower.append(np.percentile(ks, 2.5))
        ks_ci_upper.append(np.percentile(ks, 97.5))
    plt.figure()
    plt.plot(index, ks_mean, label='Mean')
    plt.plot(index, ks_ci_lower, label='Lower CI')
    plt.plot(index, ks_ci_upper, label='Upper CI')
    plt.fill_between(index, ks_ci_lower, ks_ci_upper, alpha=0.2, label='95% CI')
    plt.xlabel('Mu')
    plt.ylabel('Kurtosis')
    plt.title('Kurtosis vs Mu')
    plt.legend()
    plt.savefig('kurtosis_vs_mu.png')
    plt.show()
            
            
    # experiment.crash_experiment()
    """
    #Experiment to plot Distribution of Kurtosis
    ks = []
    for i in tqdm(range(500)):
        market = experiment.run_simulation()
        ks.append(experiment.fat_tail_experiment(500))
    plt.hist(ks,bins = 50, density=True, alpha=0.8, color='b', edgecolor='black', linewidth=1.2)
    plt.title('Kurtosis Distribution')
    plt.xlabel('Kurtosis')
    plt.ylabel('Frequency')
    plt.savefig('kurtosis_distribution.svg')
    plt.show()
     """

    
    # # Experiment to plot Distribution of Volatility Clustering
    # volatilities = []
    # for i in tqdm(range(500)):
    #     volatilities.append(experiment.fat_tail_experiment(500))

    # plt.hist(volatilities, bins=50, density=True, alpha=0.8,
    #          color='g', edgecolor='black', linewidth=1.2)
    # plt.title('Volatility Clustering Distribution')
    # plt.xlabel('Average Squared Returns')
    # plt.ylabel('Frequency')
    # plt.savefig('volatility_clustering_distribution.svg')
    # plt.show()

    # Experiment to plot Distribution of Flash Crashes
    # Run the crash experiment 500 times and collect the results
    crash_counts = []
    drop_magnitude_list = []

    """ for _ in tqdm(range(500)):
        crash_count, drop_magnitude = experiment.multiple_runs_crash(1)
        crash_counts.append(crash_count)
        drop_magnitude_list.extend(drop_magnitude)
        print("Crash Count: ", crash_counts.count(1)) """

    # Plot the distribution of crash counts
    """ plt.figure(figsize=(12, 6))
    plt.hist(crash_counts, bins=50, density=True, alpha=0.8,
             color='b', edgecolor='black', linewidth=1.2)
    plt.title('Distribution of Crash Counts')
    plt.xlabel('Crash Count')
    plt.ylabel('Frequency')
    plt.show() """

    # Plot the distribution of drop magnitudes
    """ plt.figure(figsize=(12, 6))
    plt.hist(drop_magnitude_list, bins=50, density=True, alpha=0.8,
             color='b', edgecolor='black', linewidth=1.2)
    plt.title('Distribution of Drop Magnitudes')
    plt.xlabel('Drop Magnitude')
    plt.ylabel('Frequency')
    plt.show() """

    # market = experiment.run_simulation()  # Ensure proper recepte market
    # experiment.analyze_autocorrelation_of_returns(market.prices)
    # Analyze autocorrelation of prices
    # experiment.analyze_autocorrelation(market.prices)
    # do we do initial price from 0? here l set it as 1
    # Ensure initial_price is more than 0 to avoid log(0) issues.
    # experiment = Experiment(initial_price=1, time_steps=100)
    # experiment.analyze_volatility_clustering()
