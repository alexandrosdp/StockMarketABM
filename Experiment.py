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
    """
    A class to conduct experiments with the simulation.
    
    Attributes:
        initial_price (float): Initial price of the asset.
        time_steps (int): Number of time steps to run the simulation.
        network_type (str): Type of network ('small_world', 'random', 'scale_free').
        number_of_traders (int): Number of traders in the network.
        percent_fund (float): Percentage of fundamental traders.
        percent_chartist (float): Percentage of chartist traders.
        percent_rational (float): Percentage of rational traders.
        percent_risky (float): Percentage of risky assets.
        high_lookback (int): High lookback period for traders.
        low_lookback (int): Low lookback period for traders.
        high_risk (float): High risk value for traders.
        low_risk (float): Low risk value for traders.
        new_node_edges (int): Number of new node edges.
        connection_probability (float): Connection probability for random networks.
        mu (float): Drift term for price evolution.
        beta (float): Impact factor of demand on price.
        alpha_w (float): Weight parameter.
        alpha_O (float): Offset parameter.
        alpha_p (float): Noise parameter.
    """

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
        """
        Runs the market simulation.

        Returns:
            Market: The market object containing the simulation results.
        """
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

            # Update strategies for all agents
            market.update_strategies(t)
            # Calculate the demands of all agents
            market.calculate_demands(t)
            # Update price
            market.update_price(t)

        clear_progress_bar()
        return market

    def analyze_autocorrelation_of_returns(self, prices):
        """
        Analyzes the autocorrelation of returns.

        Args:
            prices (list): List of prices from the simulation.

        Returns:
            DataFrame: Ljung-Box test results.
        """
        returns = np.diff(prices)
        # Ljung-Box test
        ljung_box_result = acorr_ljungbox(returns, lags=[20], return_df=True)
        
        plt.figure()
        plot_acf(returns, lags=40)
        plt.title('Autocorrelation Function (ACF) of Returns')
        plt.xlabel('Lags')
        plt.ylabel('Autocorrelation')
        st.pyplot(plt.gcf())  # Display the figure in Streamlit
        
        return ljung_box_result

    def analyze_volatility_clustering(self, prices, plot=False):
        """
        Analyzes volatility clustering.

        Args:
            prices (list): List of prices from the simulation.
            plot (bool): Whether to plot the results.

        Returns:
            tuple: Indicator of volatility clustering (1 or 0) and p-value of the ARCH test.
        """
        prices = np.exp(np.array(prices))
        returns = prices[1:] - prices[:-1]
        # ARCH Test For Volatility Clustering
        arch_test = sm.stats.diagnostic.het_arch(returns.flatten())
        if plot:
            plt.figure()
            plt.plot(returns**2)
            plt.title('Autocorrelation of Returns')
            plt.xlabel('Lags')
            plt.ylabel('Autocorrelation')
            st.pyplot(plt.gcf())
        if arch_test[1] < 0.05:
            return 1, arch_test[1]
        else:
            return 0, arch_test[1]

    def crash_experiment(self):
        """
        Conducts the crash experiment to detect flash crashes.

        Returns:
            tuple: Indicator of a crash (1 or 0) and the magnitude of the drop.
        """
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
                drop_magnitude = np.sum(np.diff(market.prices)[start_index:end_index+1])
                biggest_drop = (start_index, end_index)

        if np.max(market.prices[end_index + 1: np.maximum(end_index+1+period, len(market.prices)-1)]) - market.prices[end_index+1] > np.sum(np.diff(market.prices)[start_index:end_index+1]) * 0.5 and np.sum(np.diff(market.prices)[start_index:end_index+1]) < -0.03:
            print("Flash Crash Detected at time: ", index)
            crash_indices.append((start_index, end_index))

        drop_magnitude_list = []

        if drop_magnitude <= -0.07:
            return 1, drop_magnitude
        else:
            return 0, drop_magnitude

    def multiple_runs_crash(self, n_runs):
        """
        Runs the crash experiment multiple times.

        Args:
            n_runs (int): Number of runs.

        Returns:
            tuple: Total number of crashes and list of drop magnitudes.
        """
        crash_count = 0
        drop_magintude_list = []

        for i in range(n_runs):
            crash, drop_magintude = self.crash_experiment()  # Crash returns either 0 or 1
            crash_count += crash
            drop_magintude_list.append(drop_magintude)

        return crash_count, drop_magintude_list

    def fat_tail_experiment(self, T, prices, plot=False):
        """
        Conducts the fat tail experiment to analyze the distribution of returns.

        Args:
            T (int): Number of time steps.
            prices (list): List of prices from the simulation.
            plot (bool): Whether to plot the results.

        Returns:
            float: Kurtosis value of the returns.
        """
        rr = np.array(prices[1:T+1]) - np.array(prices[0:T])
        if plot:
            plt.figure()
            sm.qqplot(rr.flatten(), line='s')  # 's' line fit standardizes the data to have the same scale
            plt.title('QQ Plot')
            plt.xlabel('Returns')
            plt.ylabel('Frequency')
            st.pyplot(plt.gcf())  # Display the figure in Streamlit
            
            plt.figure()
            plt.hist(rr.flatten(), bins=50, density=True, alpha=0.8, color='b', edgecolor='black', linewidth=1.2)
            # Fit a normal distribution to the data
            mu, std = norm.fit(rr.flatten())
            xmin, xmax = plt.xlim()
            # Plot normal distribution curve
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2)
            st.pyplot(plt.gcf())  # Display the figure in Streamlit

        # Calculate kurtosis (K value)
        kurtosis_value = kurtosis(rr.flatten())
        return kurtosis_value
