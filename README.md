# Stock Price Simulation Model

This repository contains a stock price simulation model using agent-based modeling. The project includes various classes for different types of traders and a main experiment class to run and analyze simulations. An interactive web application built with Streamlit allows users to adjust parameters and run simulations to visualize and analyze results.

## Project Structure

- `experiment.py`: Contains the `Experiment` class for running simulations and analyzing results.
- `fundamental_trader.py`: Defines the `FundamentalTrader` class.
- `chartist.py`: Defines the `Chartist` class.
- `network.py`: Contains the `Network` class for creating and managing the trader network.
- `utils.py`: Utility functions used throughout the project.
- `simulate_network.py`: Contains the `Market` class that handles market dynamics.
- `requirements.txt`: Lists the required Python packages.
- `streamlit_app.py`: Streamlit application for interactive simulations.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/my_stock_simulation.git
   cd my_stock_simulation

### Create A Virtual Environment

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

### Install Required Packages

pip install -r requirements.txt

To run the interactive Streamlit application, use the following command:
streamlit run streamlit_app.py

