import numpy as np
import matplotlib.pyplot as plt
from Experiment import Experiment  # Assuming Experiment is a custom class in the Experiment module
from tqdm import tqdm  # For displaying progress bars

"""
This script runs a financial market simulation to study the effect of different values of mu on the kurtosis of the market prices.
It uses the `Experiment` class to set up and run the simulations, then calculates the mean and confidence intervals of the kurtosis.
Finally, it plots the results showing how kurtosis changes with different values of mu.
"""

# List to store mean kurtosis values for each mu
ks_mean = []

# List to store lower confidence interval values for kurtosis
ks_ci_lower = []

# List to store upper confidence interval values for kurtosis
ks_ci_upper = []

# Define the range of mu values to iterate over (0.01 to 0.09)
index = np.arange(0.01, 0.1, 0.01)

# Loop over each mu value
for mu1 in index:
    # Temporary list to store kurtosis values for the current mu
    ks = []
    
    # Run 5 simulations for each mu value
    for _ in tqdm(range(5)):
        # Initialize the experiment with the given parameters
        experiment = Experiment(
            initial_price=0,
            time_steps=500,
            network_type="barabasi",  # Type of network used in the simulation
            number_of_traders=150,  # Total number of traders
            percent_fund=0.50,  # Percentage of fundamentalist traders
            percent_chartist=0.50,  # Percentage of chartist traders
            percent_rational=0.50,  # Percentage of rational traders
            percent_risky=0.050,  # Percentage of traders taking high risks
            high_lookback=10,  # High lookback period for chartist traders
            low_lookback=1,  # Low lookback period for chartist traders
            high_risk=0.50,  # High risk factor for traders
            low_risk=0.10,  # Low risk factor for traders
            new_node_edges=5,  # Number of edges for new nodes in the network
            connection_probability=0.50,  # Probability of connection in the network
            mu=mu1,  # Mu value (variable parameter in this study)
            beta=1,  # Beta parameter for the experiment
            alpha_w=2668,  # Alpha_w parameter for the experiment
            alpha_O=2.1,  # Alpha_O parameter for the experiment
            alpha_p=0  # Alpha_p parameter for the experiment
        )
        
        # Run the simulation
        market = experiment.run_simulation()
        
        # Calculate the kurtosis for the market prices
        ks.append(experiment.fat_tail_experiment(500, market.prices))
    
    # Convert the kurtosis values to a numpy array for statistical calculations
    ks = np.array(ks)
    
    # Calculate and store the mean kurtosis
    ks_mean.append(np.mean(ks))
    
    # Calculate and store the 2.5 percentile (lower CI)
    ks_ci_lower.append(np.percentile(ks, 2.5))
    
    # Calculate and store the 97.5 percentile (upper CI)
    ks_ci_upper.append(np.percentile(ks, 97.5))

# Plotting the results
plt.figure()
plt.plot(index, ks_mean, label='Mean')  # Plot the mean kurtosis
plt.plot(index, ks_ci_lower, label='Lower CI')  # Plot the lower CI of kurtosis
plt.plot(index, ks_ci_upper, label='Upper CI')  # Plot the upper CI of kurtosis
plt.fill_between(index, ks_ci_lower, ks_ci_upper, alpha=0.2, label='95% CI')  # Fill the area between the CIs
plt.xlabel('Mu')  # Label for x-axis
plt.ylabel('Kurtosis')  # Label for y-axis
plt.title('Kurtosis vs Mu')  # Title of the plot
plt.legend()  # Display legend
plt.show()  # Show the plot
