
import math
from Fundamentalist import Fundamentalist
from Chartist import Chartist
import matplotlib.pyplot as plt


class Market:


    def __init__(self, fundamentalist, chartists, interest_rate, q, adjustment_speed, prices):

        self.fundamentalist = fundamentalist
        self.chartists = chartists
        self.interest_rate = interest_rate
        self.q = q # Intensity of choice
        self.adjustment_speed = adjustment_speed
        self.prices = prices # Initial price
        self.market_fractions = None

    def update_market_fractions(self):
        for chartist in self.chartists:
            chartist.update_fundamental_value(self.prices[-1])
        profits = [chartist.compute_expected_profit(self.interest_rate, self.prices[-1]) for chartist in self.chartists] + [self.fundamentalist.calculate_expected_profit(self.prices[-1], self.interest_rate)]
        exp_profits = [math.exp(self.q * profit) for profit in profits]
        sum_exp_profits = sum(exp_profits)
        self.market_fractions = [exp_profit / sum_exp_profits for exp_profit in exp_profits]

    def compute_excess_demand(self):
        """
        Computes the excess demand in the market

        """

        demands = [chartist.calculate_demand(self.prices[-1]) for chartist in self.chartists] + [self.fundamentalist.calculate_demand(self.prices[-1])]
        self.excess_demand = sum(demand * fraction for demand, fraction in zip(demands, self.market_fractions))

    def update_price(self):
        """
        Updates the market price based on excess demand
        """
        new_price = self.prices[-1] + self.adjustment_speed * self.excess_demand
        self.prices.append(new_price)

def run_simulation(initial_price, time_steps):

    fundamentalist = Fundamentalist(growth_rate=0.008, fundamental_value=50, risk_aversion=2, information_cost = 3)
    chartists = [Chartist(b=1.2, g=0.833, lambda1 = 13.1787),  # Trend follower
                Chartist(b=-0.7, g=3.214, lambda1 = 13.1787)]  # Contrarian
    prices = [initial_price]
    market = Market(fundamentalist, chartists, interest_rate=0.0001, q=0.9, adjustment_speed=1, prices=prices)
    
    # Simulate market updates
    for t in range(time_steps):  # Simulate for 10 periods

        fundamentalist.update_fundamental_value(time=t, s=25)
        market.update_market_fractions()
        market.compute_excess_demand()

        market.update_price()

    return market.prices

if __name__ == '__main__':
    initial_price = 50
    prices = run_simulation(initial_price, 1000)
    plt.plot(prices)
    plt.show()

