
import math

class Chartist:
    
    def __init__(self, b, g, current_price):
        """
        Initializes a new instance of the Chartist class.
        
        :param b: Sensitivity to the price deviation (positive for trend followers, negative for contrarians)
        :param g: Elasticity of the demand function
        :param price_regimes: List of tuples representing price regimes [(p_min1, p_max1), (p_min2, p_max2), ...]
        """
        self.b = b
        self.g = g
        self.current_price = current_price
        self.vt = None  # Short-term fundamental value, initially unknown
        
        
    
    def update_fundamental_value(self, lambda1 ):
        """
        Updates the short-term fundamental value based on the last price and price regimes.
        
        :param pt_minus_1: Last known price
        """
        self.vt = (math.floor(self.current_price/lambda1 ) + math.ceil(self.current_price/lambda1 )) * lambda1 / 2
    
        
    
    def expected_price(self, pt_minus_1):
        """
        Calculate the expected price based on the last price and the current fundamental value.
        
        :param pt_minus_1: Last known price
        :return: Expected price for the next period
        """
        if self.vt is None:
            self.update_fundamental_value(pt_minus_1)
        return pt_minus_1 + self.b * (pt_minus_1 - self.vt)
    
    def calculate_demand(self, pt_minus_1, vt_minus_1):
        """
        Calculates the chartist's demand based on price expectations and current price.
        
        :param pt_minus_1: Last known price
        :return: Demand value
        """
        return self.g * self.b * (pt_minus_1 - vt_minus_1)




