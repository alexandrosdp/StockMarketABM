import numpy as np
import matplotlib.pyplot as plt
import math

class Fundamentalist:
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
        self.W = [0,0]
        self.G = [0,0]
        self.D = [0,0]

    def update_performance(self, prices, t):
        self.G.append((np.exp(prices[t]) - np.exp(prices[t-1])) * self.D[t-2])

    def update_wealth(self, t):
        self.W.append(self.eta * self.W[t-1] + (1 - self.eta) * self.G[t])

    def calculate_demand(self, P, t):
        vol = np.std(np.diff(P)) * np.sqrt(252)
        if vol <= self.max_risk:
            self.D.append(self.phi * (self.pstar - P[t]) + self.sigma_f * np.random.randn(1).item())
        else:
            self.D.append(0)
        return self.D[t]
