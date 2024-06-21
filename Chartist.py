import numpy as np
import matplotlib.pyplot as plt
import math

class Chartist:
    def __init__(self, node_number,eta, chi, sigma_c):
        self.type = 'Chartist'
        self.node_number = node_number
        self.eta = eta
        self.chi = chi
        self.sigma_c = sigma_c
        self.W = [0,0]
        self.G = [0,0]
        self.D = [0,0]

    def update_performance(self, prices, t):
        self.G.append((np.exp(prices[t]) - np.exp(prices[t-1])) * self.D[t-2])

    def update_wealth(self, t):
        self.W.append(self.eta * self.W[t-1] + (1 - self.eta) * self.G[t])

    def calculate_demand(self, P, t):
        self.D.append(self.chi * (P[t] - P[t-1]) + self.sigma_c * np.random.randn(1).item())
        return self.D[t]

 