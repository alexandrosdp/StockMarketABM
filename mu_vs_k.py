import numpy as np
import matplotlib.pyplot as plt
from Experiment import Experiment
from tqdm import tqdm


if __name__ == "__main__":
    ks_mean = []
    ks_ci_lower = []
    ks_ci_upper = []
    index = np.arange(0.01, 0.1, 0.01)
    for mu1 in index:
        ks = []
        for _ in tqdm(range(5)):
            experiment = Experiment(initial_price=0,
                                time_steps=500,
                                network_type="barabasi",
                                number_of_traders=150,
                                percent_fund=0.50,
                                percent_chartist=0.50,
                                percent_rational=0.50,
                                percent_risky=0.050,
                                high_lookback=10,
                                low_lookback=1,
                                high_risk=0.50,
                                low_risk=0.10,
                                new_node_edges=5,
                                connection_probability=0.50,
                                mu=mu1,
                                beta=1,
                                alpha_w=2668,
                                alpha_O=2.1,
                                alpha_p=0
                                )
            market = experiment.run_simulation()
            ks.append(experiment.fat_tail_experiment(500, market.prices))
        ks = np.array(ks)
        ks_mean.append(np.mean(ks))
        ks_ci_lower.append(np.percentile(ks, 2.5))
        ks_ci_upper.append(np.percentile(ks, 97.5))
    plt.figure()
    plt.plot(index, ks_mean, label='Mean')
    plt.plot(index, ks_ci_lower, label='Lower CI')
    plt.plot(index, ks_ci_upper, label='Upper CI')
    plt.fill_between(index, ks_ci_lower, ks_ci_upper, alpha=0.2, label='95% CI')
    plt.xlabel('Mu')
    plt.ylabel('Kurtosis')
    plt.title('Kurtosis vs Mu')
    plt.legend()
    #plt.savefig('kurtosis_vs_mu.png')
    plt.show()
            
            
