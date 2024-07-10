import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from Experiment import Experiment
import numpy as np

# Set up the Streamlit interface
st.title('Stock Price Simulation Experiment')

st.write("""
# Interactive Simulation of Stock Prices
Adjust the parameters and run the simulation to see the results.
""")

# Sidebar for user inputs
st.sidebar.header('Simulation Parameters')
initial_price = st.sidebar.number_input('Initial Price', min_value=0.0, value=100.0)
time_steps = st.sidebar.number_input('Time Steps', min_value=1, max_value=1000, value=500)
network_type = st.sidebar.selectbox('Network Type', ['small_world', 'barabasi'])
number_of_traders = st.sidebar.number_input('Number of Traders', min_value=1, max_value=500, value=150)
percent_fund = st.sidebar.slider('Percent Fundamental Traders', min_value=0.0, max_value=1.0, value=0.5)
percent_chartist = st.sidebar.slider('Percent Chartist Traders', min_value=0.0, max_value=1.0, value=0.5)
percent_rational = st.sidebar.slider('Percent Rational Traders', min_value=0.0, max_value=1.0, value=0.5)
percent_risky = st.sidebar.slider('Percent Risky Traders', min_value=0.0, max_value=1.0, value=0.5)
high_lookback = st.sidebar.number_input('High Lookback Period', min_value=1, max_value=100, value=10)
low_lookback = st.sidebar.number_input('Low Lookback Period', min_value=1, max_value=100, value=1)
high_risk = st.sidebar.slider('High Risk', min_value=0.0, max_value=1.0, value=0.5)
low_risk = st.sidebar.slider('Low Risk', min_value=0.0, max_value=1.0, value=0.1)
new_node_edges = st.sidebar.number_input('New Node Edges', min_value=1, max_value=100, value=5)
connection_probability = st.sidebar.slider('Connection Probability', min_value=0.0, max_value=1.0, value=0.5)
mu = st.sidebar.number_input('Mu', min_value=0.0, value=0.01)



# Create an instance of the Experiment class with user inputs
experiment = Experiment(
    initial_price=initial_price,
    time_steps=time_steps,
    network_type=network_type,
    number_of_traders=number_of_traders,
    percent_fund=percent_fund,
    percent_chartist=percent_chartist,
    percent_rational=percent_rational,
    percent_risky=percent_risky,
    high_lookback=high_lookback,
    low_lookback=low_lookback,
    high_risk=high_risk,
    low_risk=low_risk,
    new_node_edges=new_node_edges,
    connection_probability=connection_probability,
    mu=mu, beta=1, alpha_w=2668, alpha_O=2.1, alpha_p=0
)

# Button to run the simulation
if st.sidebar.button('Run Simulation'):
    market = experiment.run_simulation()
    prices = market.prices
    df = pd.DataFrame({'Day': range(len(prices)), 'Price': prices})

    # Display results
    st.subheader('Simulated Stock Prices')
    st.line_chart(df.set_index('Day'))

    
    # Fat Tail Experiment
    st.subheader('Fat Tail Experiment')
    kurtosis_value = experiment.fat_tail_experiment(time_steps, prices, True)
    st.write(f"Kurtosis Value: {kurtosis_value}")

    # Volatility Clustering Analysis
    st.subheader('Volatility Clustering Analysis')
    experiment.analyze_volatility_clustering(prices)

    # Autocorrelation of Returns Analysis
    st.subheader('Autocorrelation of Returns Analysis')
    p_value = experiment.analyze_autocorrelation_of_returns(prices)
    st.write(f"p value: {p_value}")

    # Crash Experiment
    st.subheader('Crash Experiment')
    crash, drop_magnitude = experiment.crash_experiment()
    st.write(f"Number of Crashes Detected: {crash}")
    st.write(f"Drop Magnitude: {drop_magnitude}")

    

# Instructions
st.write("""
## Instructions:
- Use the sliders and input boxes on the left to adjust the simulation parameters.
- Click the 'Run Simulation' button to run the simulation and display the results.
""")
