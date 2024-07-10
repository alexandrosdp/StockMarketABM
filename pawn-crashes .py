import numpy as np
from SALib.sample import latin
from SALib.analyze import pawn
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm
from Experiment import Experiment

# Purpose: to understand the impact of different parameters on the frequency of crashes
# perform the sensitivity analysis based on the number of crashes in one run per parameter set


def model(params_chunk):
    results = []
    for params in params_chunk:
        number_of_traders = int(params[0])
        if number_of_traders % 2 != 0:
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

        # Run the experiment once and count crashes
        crash_count, _ = exp.multiple_runs_crash(1)
        results.append(crash_count)
    return results

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

# Generate samples
N = 1000
param_values = latin.sample(problem, N)

# Parallel model evaluation with progress tracking


def parallel_model_evaluation(param_values, num_workers=4):
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

    # # Extract PAWN sensitivity indices
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
    # Sort factors by mean sensitivity indices
    #sorted_mean_ranking = sorted(zip(factors, pawn_Si_mean), key=lambda x: x[1], reverse=True)
    sorted_max_ranking = sorted(zip(factors, pawn_Si_max), key=lambda x: x[1], reverse=True)

    # Unzip the sorted rankings
    #sorted_factors_mean, sorted_pawn_Si_mean = zip(*sorted_mean_ranking)
    sorted_factors_max, sorted_pawn_Si_max = zip(*sorted_max_ranking)

    # Plot results
    

    # Plot maximum sensitivity indices
    plt.figure(figsize=(12, 8))
    plt.bar(sorted_factors_max, sorted_pawn_Si_max, align='center', color='salmon', edgecolor='black')
    plt.axhline(critical_value, color='red', linestyle='--')
    plt.xlabel('Parameter')
    plt.ylabel('Sensitivity Index (maximum)')
    plt.tight_layout()
    plt.savefig('pawn_SA_crashes.jpeg')
    plt.show()
    
    #plt.figure(figsize=(12, 8))
    #plt.scatter( , pawn_Si_max, color='skyblue', edgecolor='black')

    # Display significance
    print("Significance Results:")
    for name in factors:
        print(f"{name}: {'Significant' if significant[name] else 'Not Significant'}")
