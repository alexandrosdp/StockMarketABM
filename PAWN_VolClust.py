import numpy as np
from SALib.sample import latin
from SALib.analyze import pawn
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm
from Experiment import *

# Define the model function
def model(params_chunk):
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
            beta=params[7],
            alpha_w=params[8],
            alpha_O=params[9],
            alpha_p=0
        )
        market = exp.run_simulation()
        y2 = exp.analyze_volatility_clustering(market.prices)
        results.append(y2)
    return results

# Define the problem for sensitivity analysis
problem = {
    'num_vars': 10,
    'names': [
        'number_of_traders', 'percent_rational', 'percent_risky',
        'high_lookback', 'high_risk', 'new_node_edges',
        'mu', 'beta', 'alpha_w', 'alpha_O'
    ],
    'bounds': [
        [50, 200],  # number_of_traders
        [0.05, 1.0],  # percent_rational
        [0.05, 1.0],  # percent_risky
        [5, 30],  # high_lookback
        [0.05, 0.20],  # high_risk
        [2, 10],  # new_node_edges
        [0.001, 0.1],  # mu
        [0.1, 2.0],  # beta
        [1000, 5000],  # alpha_w
        [1.0, 3.0]  # alpha_O
    ]
}

# Generate Latin Hypercube samples
N = 1000
param_values = latin.sample(problem, N)

# Parallel model evaluation with progress tracking
def parallel_model_evaluation(param_values, num_workers=8):
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

    # Plot results
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)

    # Plot mean sensitivity indices
    axes[0].bar(problem['names'], S['mean'], align='center', color='skyblue', edgecolor='black')
    axes[0].set_ylabel('Sensitivity Index (mean)')
    axes[0].set_title('PAWN Sensitivity Analysis (Mean)')

    # Plot median sensitivity indices
    axes[1].bar(problem['names'], S['median'], align='center', color='salmon', edgecolor='black')
    axes[1].set_xlabel('Parameter')
    axes[1].set_ylabel('Sensitivity Index (median)')
    axes[1].set_title('PAWN Sensitivity Analysis (Median)')

    # Enhance the overall aesthetics
    for ax in axes:
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, color='gray', linestyle='--', linewidth=0.5)
        ax.xaxis.grid(False)

    plt.tight_layout()
    plt.savefig('pawn_SA_Vol.png')
    plt.show()
