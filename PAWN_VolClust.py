import numpy as np
from SALib.sample import latin
from SALib.analyze import pawn
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm
from Experiment import *

# Define the model function
def model(params_chunk):
    """
    Evaluate the model for a chunk of parameter sets.

    Parameters:
    ----------
    params_chunk : list
        List of parameter sets.

    Returns:
    -------
    list
        List of volatility clustering results for each parameter set.
    """
    results = []
    for params in params_chunk:
        number_of_traders = int(params[0])
        if (number_of_traders % 2) != 0:
            number_of_traders += 1  # Increment to make even if odd

        exp = Experiment(
            initial_price=0,
            time_steps=500,
            network_type='barabasi',
            number_of_traders=number_of_traders,
            percent_fund=0.5,
            percent_chartist=0.5,
            percent_rational=params[1],
            percent_risky=params[2],
            high_lookback=int(params[3]),
            low_lookback=1,
            high_risk=params[4],
            low_risk=0.01,
            new_node_edges=int(params[5]),
            connection_probability=0.5,
            mu=params[6],
            beta=1,
            alpha_w=2668,
            alpha_O=2.1,
            alpha_p=0
        )
        market = exp.run_simulation()
        y2 = exp.analyze_volatility_clustering(market.prices)
        results.append(y2[0])
    return results

# Define the problem for sensitivity analysis
problem = {
    'num_vars': 7,
    'names': [
        'number_of_traders', 'percent_rational', 'percent_risky',
        'high_lookback', 'high_risk', 'new_node_edges',
        'mu'
    ],
    'bounds': [
        [50, 200],  # number_of_traders
        [0.05, 1.0],  # percent_rational
        [0.05, 1.0],  # percent_risky
        [5, 30],  # high_lookback
        [0.05, 0.20],  # high_risk
        [2, 10],  # new_node_edges
        [0.001, 0.1],  # mu
    ]
}

# Generate Latin Hypercube samples
N = 1000
param_values = latin.sample(problem, N)

# Parallel model evaluation with progress tracking
def parallel_model_evaluation(param_values, num_workers=8):
    """
    Evaluate the model in parallel.

    Parameters:
    ----------
    param_values : array
        Array of parameter sets.
    num_workers : int
        Number of parallel workers.

    Returns:
    -------
    array
        Array of volatility clustering results for all parameter sets.
    """
    chunks = np.array_split(param_values, num_workers)
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(model, chunks), total=num_workers))
    return np.concatenate(results)

if __name__ == '__main__':
    # Number of workers for parallel processing
    num_workers = mp.cpu_count()

    # Run the model in parallel
    Y = parallel_model_evaluation(param_values, num_workers)

    # Perform sensitivity analysis using PAWN method
    k = 15  # Number of bins
    S = pawn.analyze(problem, param_values, Y, k)

    # Extract PAWN sensitivity indices
    pawn_Si_mean = S['mean']
    pawn_Si_max = S['maximum']

    # Compute the critical value for the KS test
    alpha = 0.05  # significance level
    num_samples = len(param_values)
    critical_value = 1.36 / np.sqrt(num_samples)

    # Verify KS Statistics
    significant = {}
    for name, si_max in zip(problem['names'], pawn_Si_max):
        significant[name] = si_max > critical_value

    # Rank Factors
    factors = problem['names']
    sorted_max_ranking = sorted(zip(factors, pawn_Si_max), key=lambda x: x[1], reverse=True)

    # Unzip the sorted rankings
    sorted_factors_max, sorted_pawn_Si_max = zip(*sorted_max_ranking)

    # Plot results
    plt.figure(figsize=(12, 8))
    plt.bar(sorted_factors_max, sorted_pawn_Si_max, align='center', color='salmon', edgecolor='black')
    plt.axhline(critical_value, color='red', linestyle='--')
    plt.xlabel('Parameter')
    plt.ylabel('Sensitivity Index (maximum)')
    plt.tight_layout()
    plt.savefig('pawn_SA_vol_clustering.png')
    plt.show()

    # Display significance
    print("Significance Results:")
    for name in factors:
        print(f"{name}: {'Significant' if significant[name] else 'Not Significant'}")
