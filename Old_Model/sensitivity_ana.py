
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from SALib.sample import sobol
from SALib.analyze import sobol as sobol_analyze
from scipy.stats import kurtosis
import time
from Experiment import *

def model(params):
    
    number_of_traders = int(params[0])
    if number_of_traders % 2 != 0:
        number_of_traders += 1  # Increment to make even if odd

    exp = Experiment(
        initial_price=0,
        time_steps= 1000,
        network_type='barabasi',
        number_of_traders=number_of_traders,
        percent_fund = 0.5,
        percent_chartist=0.5,
        percent_rational=params[1],
        percent_risky=params[2],
        high_lookback=int(params[3]),
        low_lookback=1,
        high_risk=params[4],
        low_risk=0.01,
        new_node_edges=int(params[5]),
        connection_probability=0.5,
        mu=0.01,
        beta=1,
        alpha_w=2668,
        alpha_O=2.1,
        alpha_p=0
    )
    
    return exp.fat_tail_experiment(1000)

# Define the parameter space
problem = {
    'num_vars': 6,
    'names': [
        'number_of_traders', 'percent_rational', 'percent_risky',
        'high_lookback', 'high_risk', 'new_node_edges'],
    'bounds': [
        [50, 200],  # number_of_traders
        [0.05, 1.0],  # percent_rational
        [0.05, 1.0],  # percent_risky
        [5, 30],        # high_lookback
        [0.05, 0.20],  # high_risk
        [2, 10],  # new_node_edges
    ]
}

# Generate Sobol samples
param_values = sobol.sample(problem, 1024, calc_second_order=True)

results = []

start_time = time.time()
for i, params in enumerate(param_values):
    results.append(model(params))
    if i % 10 == 0:
        print(f"Progress: {i / len(param_values) * 100:.2f}%")

total_time = time.time() - start_time
print(f"Total time for all runs: {total_time / 3600:.2f} hours")

sobol_indices = sobol_analyze(problem, np.array(results))

print("Sobol Sensitivity Indices")
print("S1 (First order indices):")
print(sobol_indices['S1'])
print("ST (Total order indices):")
print(sobol_indices['ST'])

# Optionally, visualize the Sobol indices
plt.figure(figsize=(10, 6))
plt.bar(problem['names'], sobol_indices['S1'], align='center', alpha=0.7)
plt.xlabel('Parameters')
plt.ylabel('First Order Sensitivity Indices')
plt.title('First Order Sobol Sensitivity Indices')
plt.savefig('sobol_indices1.jpeg')
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(problem['names'], sobol_indices['ST'], align='center', alpha=0.7)
plt.xlabel('Parameters')
plt.ylabel('Total Order Sensitivity Indices')
plt.title('Total Order Sobol Sensitivity Indices')
plt.savefig('sobol_indices_T.jpeg')
plt.show()

