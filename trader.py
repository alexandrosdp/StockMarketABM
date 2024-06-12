

class Fundamentalist():

    """
    Description
    -----------
    Defines a class for a fundamentalist trader
    """

    def __init__(self, growth_rate, fundamental_value, prev_price):

        """
        Parameters
        -----------

        fundamental_value : float
        The fundamental value of the stock.
        prev_price : float
            The previous price of the stock.
        mt : float
            The lower bound of the fundamentalists price boundary.
        MT : float
            The upper bound of the fundamentalists price boundary.
        demand : float
            The calculated demand of the fundamentalist for a given time step.
        
        """

        self.growth_rate = growth_rate
        self.fundamental_value = fundamental_value
        self.prev_price = prev_price

        #To be initialised later 
        self.mt = 0
        self.MT = 0
        self.demand = 0
        
    def compute_price_boundaries(self):

        k = 2 #Preselected factor (Using value from paper)

        self.mt = (1/k)*self.fundamental_value 

        self.MT = k*self.fundamental_value
        

    def determine_chance_function(self):
        
        #Parameters that describe the sensitiveness of fundamentalists
        a = 1 #Using value from paper
        d = -0.3 #Using value from paper

        self.compute_price_boundaries() #Calculate values of mt and MT

        #Compute value of chance function (given by A)
        A = (a*(self.prev_price - self.mt)**d)*((self.MT - self.prev_price)**d)

        return A



    def determine_demand(self):

        price_zone = (self.mt,self.MT)

        A = self.determine_chance_function()

        if(price_zone[0] <= self.prev_price <= price_zone[1]): #Check if the global price is in the trader's price zone

            self.demand = (self.fundamental_value - self.prev_price)*A

        else:

            self.demand = 0

    def update_fundamental_value(self, time, s):

        # determine the cycle of the business growth
        i = time // (4 * s) + 1

        if time >= 4 * (i - 1) * s and time < (4 * i - 1) * s:
            self.fundamental_value = self.fundamental_value * (1 + self.growth_rate)
            # return self.growth_rate
        else:
            self.fundamental_value = self.fundamental_value * (1 - self.growth_rate/2)
            # return -(self.growth_rate/2)


if __name__ == '__main__':

   fundamentalist = Fundamentalist(growth_rate=0.008,fundamental_value=100, prev_price= 199.001) 
    
   fundamentalist.compute_price_boundaries()
   fundamentalist.determine_demand()

   fundamentalist.update_fundamental_value(time=0, s = 25)

   print(f"PRICE ZONE -- > ({fundamentalist.mt};{fundamentalist.MT})" ) 
   print(fundamentalist.demand)




        
        

        



