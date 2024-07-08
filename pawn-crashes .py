import numpy as np
from SALib.sample import latin
from SALib.analyze import pawn
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm
from Experiment import Experiment

# Purpose: to understand the impact of different parameters on the frequency of crashes


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
            mu=0.01,
            beta=1,
            alpha_w=2668,
            alpha_O=2.1,
            alpha_p=0
        )

        crash_count, _ = exp.multiple_runs_crash(1)
        results.append(crash_count)
    return results


problem = {
    'num_vars': 6,
    'names': [
        'number_of_traders', 'percent_rational', 'percent_risky',
        'high_lookback', 'high_risk', 'new_node_edges'
    ],
    'bounds': [
        [50, 200],  # number_of_traders
        [0.05, 1.0],  # percent_rational
        [0.05, 1.0],  # percent_risky
        [5, 30],  # high_lookback
        [0.05, 0.20],  # high_risk
        [2, 10]  # new_node_edges
    ]
}

# Generate samples
N = 500
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

    # Plot results
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

    # Plot mean sensitivity indices
    axes[0].bar(problem['names'], S['mean'], align='center',
                color='skyblue', edgecolor='black')
    axes[0].set_ylabel('Sensitivity Index (mean)')
    axes[0].set_title('PAWN Sensitivity Analysis (Mean)- frequency of crashes')

    # Plot median sensitivity indices
    axes[1].bar(problem['names'], S['median'], align='center',
                color='salmon', edgecolor='black')
    axes[1].set_xlabel('Parameter')
    axes[1].set_ylabel('Sensitivity Index (median)')
    axes[1].set_title(
        'PAWN Sensitivity Analysis (Median)- frequency of crashes')

    # Enhance the overall aesthetics
    for ax in axes:
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, color='gray', linestyle='--', linewidth=0.5)
        ax.xaxis.grid(False)

    plt.tight_layout()
    plt.savefig('pawn_sensitivity_analysis- frequency of crashes.svg')
    plt.show()
