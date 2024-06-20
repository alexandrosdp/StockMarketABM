import numpy as np
import matplotlib.pyplot as plt
import math

class Chartist:
    def __init__(self,node_number,eta, chi, sigma_c):
        self.type = 'Chartist'
        self.node_number = node_number
        self.eta = eta
        self.chi = chi
        self.sigma_c = sigma_c
        self.Wc = [0,0]
        self.Gc = [0,0]
        self.Dc = [0,0]
        self.profit_list = []

    def update_performance(self, prices, t):
        self.Gc.append((np.exp(prices[t]) - np.exp(prices[t-1])) * self.Dc[t-2])

    def update_wealth(self, t):
        self.Wc.append(self.eta * self.Wc[t-1] + (1 - self.eta) * self.Gc[t])

    def calculate_demand(self, P, t):
        self.Dc.append(self.chi * (P[t] - P[t-1]) + self.sigma_c * np.random.randn(1).item())
        return self.Dc[t]
    def change_strategy(self, new_strategy, t):
        if new_strategy == 'fundamentalist':
            self.type = 'fundamentalist'


 