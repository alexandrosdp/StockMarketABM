from Experiment import Experiment
import matplotlib.pyplot as plt
from tqdm import tqdm

# Initialize an Experiment object with specified parameters
experiment = Experiment(
    initial_price=0,
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
    mu=0.01,
    beta=1,
    alpha_w=2668,
    alpha_O=2.1,
    alpha_p=0
)

# List to store kurtosis values
ks = []

# Run the experiment multiple times and store the kurtosis values
for i in tqdm(range(500)):
    market = experiment.run_simulation()
    ks.append(experiment.fat_tail_experiment(500, market.prices))

# Plot the histogram of kurtosis values
plt.hist(ks, bins=50, density=True, alpha=0.8, color='b', edgecolor='black', linewidth=1.2)
plt.title('Kurtosis Distribution')
plt.xlabel('Kurtosis')
plt.ylabel('Frequency')
plt.savefig('kurtosis_distribution.jpeg')
plt.show()
