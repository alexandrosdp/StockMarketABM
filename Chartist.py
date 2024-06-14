
class Chartist:
    
    def __init__(self, b, g, price_regimes):
        """
        Initializes a new instance of the Chartist class.
        
        :param b: Sensitivity to the price deviation (positive for trend followers, negative for contrarians)
        :param g: Elasticity of the demand function
        :param price_regimes: List of tuples representing price regimes [(p_min1, p_max1), (p_min2, p_max2), ...]
        """
        self.b = b
        self.g = g
        self.price_regimes = price_regimes
        self.vt = None  # Short-term fundamental value, initially unknown
    
    def update_fundamental_value(self, pt_minus_1):
        """
        Updates the short-term fundamental value based on the last price and price regimes.
        
        :param pt_minus_1: Last known price
        """
        for p_min, p_max in self.price_regimes:
            if p_min <= pt_minus_1 <= p_max:
                self.vt = (p_min + p_max) / 2
                break
    
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
    
    def compute_expected_profit(self, interest_rate, pt_minus_1, vt_minus_1):
        """
        Computes the expected profit for chartists
        """

        self.expected_profit = abs(self.b * (pt_minus_1 - vt_minus_1) - interest_rate * pt_minus_1)
       
        return self.expected_profit




