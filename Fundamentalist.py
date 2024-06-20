import numpy as np
import matplotlib.pyplot as plt
import math

class Fundamentalist:
    def __init__(self, node_number,eta, alpha_w, alpha_O, alpha_p, phi, sigma_f, pstar):
        self.type = 'undamentalist'
        self.node_number = node_number
        self.eta = eta
        self.alpha_w = alpha_w
        self.alpha_O = alpha_O
        self.alpha_p = alpha_p
        self.phi = phi
        self.sigma_f = sigma_f
        self.pstar = pstar
        self.Wf = [0,0]
        self.Gf = [0,0]
        self.Df = [0,0]

    def update_performance(self, prices, t):
        self.Gf.append((np.exp(prices[t]) - np.exp(prices[t-1])) * self.Df[t-2])

    def update_wealth(self, t):
        self.Wf.append(self.eta * self.Wf[t-1] + (1 - self.eta) * self.Gf[t])

    def calculate_demand(self, P, t):
        self.Df.append(self.phi * (self.pstar - P[t]) + self.sigma_f * np.random.randn(1).item())