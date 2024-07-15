import numpy as np
import matplotlib.pyplot as plt
import math

class Fundamentalist:
    """
    A class to represent a Fundamentalist trader.

    Attributes:
    ----------
    node_number : int
        The identifier for the trader node in the network.
    eta : float
        The smoothing parameter for updating wealth.
    alpha_w : float
        Parameter for wealth updating.
    alpha_O : float
        Parameter for demand calculation.
    alpha_p : float
        Parameter for demand calculation.
    phi : float
        The sensitivity of the trader to price changes relative to fundamental value.
    sigma_f : float
        The standard deviation of the noise term in the demand calculation.
    pstar : float
        The perceived fundamental value of the asset.
    lookback_period : int
        The period over which the trader looks back to calculate demand.
    max_risk : float
        The maximum risk level tolerated by the trader.
    W : list
        The list of wealth values over time.
    G : list
        The list of performance values over time.
    D : list
        The list of demand values over time.
    """

    def __init__(self, node_number, eta, alpha_w, alpha_O, alpha_p, phi, sigma_f, pstar, lookback_period, max_risk):
        self.type = 'Fundamentalist'
        self.node_number = node_number
        self.eta = eta
        self.alpha_w = alpha_w
        self.alpha_O = alpha_O
        self.alpha_p = alpha_p
        self.phi = phi
        self.sigma_f = sigma_f
        self.pstar = pstar
        self.lookback_period = lookback_period
        self.max_risk = max_risk
        self.W = [0, 0]
        self.G = [0, 0]
        self.D = [0, 0]

    def update_performance(self, prices, t):
        """
        Update the performance of the trader.

        Parameters:
        ----------
        prices : list
            List of prices over time.
        t : int
            The current time step.
        """
        self.G.append((np.exp(prices[t]) - np.exp(prices[t-1])) * self.D[t-2])

    def update_wealth(self, t):
        """
        Update the wealth of the trader.

        Parameters:
        ----------
        t : int
            The current time step.
        """
        self.W.append(self.eta * self.W[t-1] + (1 - self.eta) * self.G[t])

    def calculate_demand(self, P, t):
        """
        Calculate the demand of the trader.

        Parameters:
        ----------
        P : list
            List of prices over time.
        t : int
            The current time step.

        Returns:
        -------
        float
            The demand of the trader at time t.
        """
        # Use the last 90 prices or all available prices if less than 90
        P_v = P[-90:] if len(P) > 90 else P
        
        # Calculate annualized volatility
        vol = np.std(np.diff(P_v)) * np.sqrt(252)
        
        # If volatility is within the risk tolerance, update demand based on the difference from the fundamental value and random noise
        if vol <= self.max_risk:
            self.D.append(self.phi * (self.pstar - P[t]) + self.sigma_f * np.random.randn(1).item())
        else:
            self.D.append(0)
        
        return self.D[t]
