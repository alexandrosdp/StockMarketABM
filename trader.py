


class Fundamentalist():

    """
    Description
    -----------
    Defines a class for a fundamentalist trader
    """

    def __init__(self, fundamental_value, prev_price):

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
        d = 0.3 #Using value from paper

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


if __name__ == '__main__':

   fundamentalist = Fundamentalist(fundamental_value=100, prev_price= 110) 

   fundamentalist.compute_price_boundaries()
   fundamentalist.determine_demand()

   print(f"PRICE ZONE -- > ({fundamentalist.mt};{fundamentalist.MT})" ) 
   print(fundamentalist.demand)




        
        

        



