import numpy as np
import matplotlib.pyplot as plt


class Agent:
    """Defines base agent behavior in the market simulation."""

    def __init__(self, trend_sensitivity, volume_sensitivity, information_cost):
        self.trend_sensitivity = trend_sensitivity
        self.volume_sensitivity = volume_sensitivity
        self.information_cost = information_cost

    def adjust_profit_expectations(self, current_price, interest_rate, recent_trend):
        adjustment = recent_trend * self.trend_sensitivity
        return (current_price - adjustment - self.information_cost) * (1 - interest_rate)

    def calculate_demand_based_on_trend(self, current_price, volume_trend):
        return self.base_demand + volume_trend * self.volume_sensitivity


class Fundamentalist(Agent):
    def __init__(self, growth_rate, fundamental_value, risk_aversion, information_cost, base_demand):
        super().__init__(0.05, 0.01, information_cost)  # Initialize the base class
        self.growth_rate = growth_rate
        self.fundamental_value = fundamental_value
        self.risk_aversion = risk_aversion
        self.base_demand = base_demand  # Initialize the base demand

    def calculate_demand(self, current_price):
        if current_price < self.fundamental_value:
            return (self.fundamental_value - current_price) / self.risk_aversion
        else:
            return (self.fundamental_value - current_price) * self.risk_aversion


class Chartist(Agent):
    def __init__(self, b, g, lambda1, information_cost, base_demand):
        super().__init__(b, g, information_cost)
        self.lambda1 = lambda1
        self.base_demand = base_demand  # Initialize the base demand

    def update_fundamental_value(self, prev_price):
        self.vt = (prev_price // self.lambda1) * \
            self.lambda1 + self.lambda1 / 2

    def calculate_expected_price(self, prev_price):
        if self.vt is None:
            self.update_fundamental_value(prev_price)
        return prev_price + self.b * (prev_price - self.vt)

    def calculate_demand(self, prev_price):
        expected_price = self.calculate_expected_price(prev_price)
        return self.g * (expected_price - prev_price)


class Market:
    """Market simulation handling multiple agents."""

    def __init__(self, agents, interest_rate, q, adjustment_speed):
        self.agents = agents
        self.interest_rate = interest_rate
        self.q = q
        self.adjustment_speed = adjustment_speed
        # Assuming all agents start with the same fundamental value
        self.prices = [agents[0].fundamental_value]
        self.volumes = []
        self.price_memory = []

    def update_market_fractions(self):
        recent_trend = (self.prices[-1] - self.prices[-2]
                        ) if len(self.prices) > 1 else 0
        profits = [agent.adjust_profit_expectations(
            self.prices[-1], self.interest_rate, recent_trend) for agent in self.agents]
        exp_profits = np.exp(self.q * np.array(profits))
        total_exp_profits = np.sum(exp_profits)
        self.market_fractions = exp_profits / total_exp_profits

    def compute_excess_demand(self):
        volume_trend = (
            self.volumes[-1] - self.volumes[-2]) if len(self.volumes) > 1 else 0
        demands = [agent.calculate_demand_based_on_trend(
            self.prices[-1], volume_trend) for agent in self.agents]
        total_demand = sum(demands)
        volume = sum(abs(demand) for demand in demands)
        self.volumes.append(volume)
        return total_demand

    def update_price(self):
        self.price_memory.append(self.prices[-1])
        excess_demand = self.compute_excess_demand()
        volume_factor = np.log1p(self.volumes[-1])
        new_price = self.prices[-1] + \
            self.adjustment_speed * excess_demand * volume_factor
        self.prices.append(new_price)


def simulate_market():
    agents = [
        # example base_demand value
        Fundamentalist(0.008, 100, 2, 5, base_demand=10),
        Chartist(1.2, 0.833, 10, 5, base_demand=5),
        Chartist(-0.7, 3.214, 10, 5, base_demand=5)
    ]

    market = Market(agents, 0.01, 1.5, 0.1)
    for _ in range(50):  # Run the simulation for 50 time steps
        market.update_market_fractions()
        market.update_price()
    return market.prices, market.volumes


prices, volumes = simulate_market()

# Plotting the results
fig, axs = plt.subplots(2, 1, figsize=(14, 10))
axs[0].plot(prices, label='Prices')
axs[0].set_title('Price Dynamics')
axs[0].set_ylabel('Price')
axs[0].legend()

axs[1].plot(volumes, label='Volumes')
axs[1].set_title('Volume Dynamics')
axs[1].set_ylabel('Volume')
axs[1].legend()

plt.tight_layout()
plt.show()
