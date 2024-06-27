import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Experiment import Experiment
from tqdm import tqdm

import numpy as np

# Define a function for your model
def run_experiment(params):
    exp = Experiment(
        initial_price=0,
        time_steps=1000,
        network_type='barabasi',
        number_of_traders=params['number_of_traders'],
        percent_fund=0.5,
        percent_chartist=0.5,
        percent_rational=params['percent_rational'],
        percent_risky=params['percent_risky'],
        high_lookback=params['high_lookback'],
        low_lookback=1,
        high_risk=params['high_risk'],
        low_risk=0.01,
        new_node_edges=params['new_node_edges'],
        connection_probability=0.5,
        mu=0.01,
        beta=1,
        alpha_w=2668,
        alpha_O=2.1,
        alpha_p=0
    )
    return exp.fat_tail_experiment(1000)  # Assuming the Experiment class has a run method that returns results

# Default parameters
default_params = {
    'number_of_traders': 100,
    'percent_rational': 0.1,
    'percent_risky': 0.1,
    'high_lookback': 15,
    'high_risk': 0.2,
    'new_node_edges': 5
}

# Parameter to vary - number of traders
num_traders_range = np.arange(50, 201, 50)  # From 50 to 200 in steps of 10

# Record results
avg_result = []
for num in tqdm(num_traders_range):
    results = []
    for _ in range(30):
        params = default_params.copy()
        params['number_of_traders'] = num
        result = run_experiment(params)
        results.append(result)
    avg_result.append(np.mean(results))

plt.plot(num_traders_range, avg_result)
plt.savefig('num_traders_sensitivity.jpeg')
plt.show()