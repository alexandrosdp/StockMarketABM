
import math

class Chartist:
    
    def __init__(self, b, g, lambda1):
        """
        Initializes a new instance of the Chartist class.
        
        :param b: Sensitivity to the price deviation (positive for trend followers, negative for contrarians)
        :param g: Elasticity of the demand function
        :param price_regimes: List of tuples representing price regimes [(p_min1, p_max1), (p_min2, p_max2), ...]
        """
        self.b = b
        self.g = g
        self.lambda1 = lambda1
        self.demamd = None 
        self.vt = None  # Short-term fundamental value, initially unknown
        
        
    
    def update_fundamental_value(self, prev_price):
        """
        Updates the short-term fundamental value based on the last price and price regimes.
        
        :param prev_price: Last known price
        """
        self.vt = (math.floor(prev_price/self.lambda1 ) + math.ceil(prev_price/self.lambda1 )) * self.lambda1 / 2
    
        
    
    def expected_price(self, prev_price):
        """
        Calculate the expected price based on the last price and the current fundamental value.
        
        :param prev_price: Last known price
        :return: Expected price for the next period
        """
        if self.vt is None:
            self.update_fundamental_value(prev_price)
        return prev_price + self.b * (prev_price - self.vt)
    
    def calculate_demand(self, prev_price):
        """
        Calculates the chartist's demand based on price expectations and current price.
        
        :param prev_price: Last known price
        :return: Demand value
        """
        self.demand =  self.g * self.b * (prev_price - self.vt)

        return self.demand
    
    def compute_expected_profit(self, interest_rate, prev_price):
        """
        Computes the expected profit for chartists
        """

        self.expected_profit = abs(self.b * (prev_price - self.vt) - interest_rate * prev_price)
       
        return self.expected_profit




