import numpy as np
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, sensitivity, information_cost, base_demand):
        self.sensitivity = sensitivity
        self.information_cost = information_cost
        self.base_demand = base_demand

    def calculate_demand(self, price, fundamental_value):
        raise NotImplementedError(
            "This method should be overridden by subclasses")


class Fundamentalist(Agent):
    def calculate_demand(self, price, fundamental_value):
        price_gap = fundamental_value - price
        return self.base_demand + price_gap / self.sensitivity - self.information_cost


class Chartist(Agent):
    def __init__(self, sensitivity, information_cost, base_demand, threshold):
        super().__init__(sensitivity, information_cost, base_demand)
        self.threshold = threshold
        self.last_price = None

    def calculate_demand(self, price, last_price):
        if self.last_price is not None and abs(price - self.last_price) > self.threshold:
            # Switching regime based on price threshold breach
            self.sensitivity = -self.sensitivity
        self.last_price = price
        trend = price - last_price
        return self.base_demand + trend * self.sensitivity - self.information_cost


class Market:
    def __init__(self, agents, initial_price, initial_fundamental_value, market_maker_effectiveness):
        self.agents = agents
        self.prices = [initial_price]
        self.fundamental_values = [initial_fundamental_value]
        self.volumes = []
        self.market_maker_effectiveness = market_maker_effectiveness

    def simulate(self, steps):
        for _ in range(steps):
            current_price = self.prices[-1]
            fundamental_value = self.fundamental_values[-1]
            # Ensure there's a previous price
            last_price = self.prices[-2] if len(
                self.prices) > 1 else self.prices[-1]
            demands = [agent.calculate_demand(current_price, fundamental_value if isinstance(
                agent, Fundamentalist) else last_price) for agent in self.agents]
            total_demand = sum(demands)
            volume = sum(abs(d) for d in demands)
            self.volumes.append(volume)
            new_price = current_price + self.market_maker_effectiveness * \
                total_demand  # Market maker adjusts prices
            self.prices.append(new_price)
            self.fundamental_values.append(
                fundamental_value * (1 + np.random.normal(0.01, 0.005)))  # Simulate economic cycles

        return self.prices, self.volumes


# Initialize agents and market
agents = [
    Fundamentalist(0.1, 0.02, 10),
    Chartist(0.05, 0.01, 5, 0.2),  # Added threshold for regime-switching
    Chartist(0.07, 0.01, 5, 0.2)
]
# Initial price, fundamental value, and market maker's effectiveness
market = Market(agents, 50, 105, 0.05)

# Simulate market dynamics
prices, volumes = market.simulate(500)  # Run the simulation for 500 time steps

# Plot results
fig, axs = plt.subplots(2, 1, figsize=(14, 12))
axs[0].plot(prices, label='Market Prices')
axs[0].set_title('Market Price Dynamics')
axs[0].set_ylabel('Price')
axs[0].legend()

axs[1].plot(volumes, label='Trading Volumes')
axs[1].set_title('Trading Volume Dynamics')
axs[1].set_ylabel('Volume')
axs[1].legend()

plt.tight_layout()
plt.show()
