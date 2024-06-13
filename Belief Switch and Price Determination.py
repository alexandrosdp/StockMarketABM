#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[ ]:


class Fundamentalist:
    # ... (existing methods)

    def calculate_expected_profit(self, prev_price, current_price, r, C):
        """
        Computes the expected profit for fundamentalists
        """
        s_pt = abs((self.fundamental_value - self.prev_price) / (3 * self.fundamental_value))
        self.expected_profit = s_pt * abs(self.fundamental_value - (1 + interest_rate) * self.prev_price) - self.information_cost
        return self.expected_profit

    
class Chartist:
    # ... (existing methods)
    
    def compute_expected_profit(self, interest_rate, vt_minus_1):
        """
        Computes the expected profit for chartists
        """
        if self.b > 0:  # Trend followers
            self.expected_profit = abs(self.b * (self.prev_price - vt_minus_1) - interest_rate * self.prev_price)
        else:  # Contrarians
            self.expected_profit = abs(self.b * (self.prev_price - vt_minus_1) - interest_rate * self.prev_price)
        return self.expected_profit


# In[ ]:


class Market:
    def __init__(self, fundamentalist, chartists, interest_rate, q, adjustment_speed):
        self.fundamentalist = fundamentalist
        self.chartists = chartists
        self.interest_rate = interest_rate
        self.q = q # Intensity of choice
        self.adjustment_speed = adjustment_speed
        self.market_fractions = [1 / (len(chartists) + 1)] * (len(chartists) + 1)

    def update_market_fractions(self):
        profits = [chartist.expected_profit for chartist in self.chartists] + [self.fundamentalist.expected_profit]
        exp_profits = [math.exp(self.q * profit) for profit in profits]
        sum_exp_profits = sum(exp_profits)
        self.market_fractions = [exp_profit / sum_exp_profits for exp_profit in exp_profits]

    def compute_excess_demand(self):
        """
        Computes the excess demand in the market
        """
        demands = [chartist.calculate_demand(chartist.prev_price, chartist.vt) for chartist in self.chartists] + [self.fundamentalist.demand]
        self.excess_demand = sum(demand * fraction for demand, fraction in zip(demands, self.market_fractions))

    def update_price(self, prev_price):
        """
        Updates the market price based on excess demand
        """
        self.price = prev_price + self.adjustment_speed * self.excess_demand


# In[ ]:


if __name__ == '__main__':
    fundamentalist = Fundamentalist(growth_rate=0.008, fundamental_value=100, prev_price=199.001, risk_aversion=2)
    chartists = [Chartist(b=1.2, g=0.833, price_regimes=[(90, 110), (110, 130)]),  # Trend follower
                 Chartist(b=-0.7, g=3.214, price_regimes=[(90, 110), (110, 130)])]  # Contrarian

    market = Market(fundamentalist, chartists, interest_rate=0.0001, q=0.9, adjustment_speed=1)
    
    # Simulate market updates
    for t in range(10):  # Simulate for 10 periods
        fundamentalist.update_fundamental_value(time=t, s=25)
        for chartist in chartists:
            chartist.update_fundamental_value(chartist.prev_price)
            chartist.expected_price(chartist.prev_price)
        fundamentalist.compute_expected_profit(market.interest_rate)
        for chartist in chartists:
            chartist.compute_expected_profit(market.interest_rate, chartist.vt)
        market.update_market_fractions()
        market.compute_excess_demand()
        market.update_price(fundamentalist.prev_price)
        
        # Update previous prices for next iteration
        fundamentalist.prev_price = market.price
        for chartist in chartists:
            chartist.prev_price = market.price


# In[ ]:




