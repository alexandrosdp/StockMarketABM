


class Fundamentalist():

    def __init__(self, growth_rate, fundamental_value, prev_price):

        self.growth_rate = growth_rate
        self.fundamental_value = fundamental_value
        self.prev_price = prev_price

        #To be initialised later 
        self.mt = 0
        self.MT = 0
        self.demand = 0
        
    def compute_price_boundaries(self):

        k = 1.1 #Preselected factor

        self.mt = (1/k)*self.fundamental_value

        self.MT = k*self.fundamental_value
        

    def determine_chance_function(self):
        
        #Parameters that describe the sensitiveness of fundamentalists
        a = 0 
        d = 0

        self.compute_price_boundaries() #Calculate values of mt and MT

        #Compute value of chance function (given by A)
        A = (a(self.prev_price - self.mt)**d)*((self.MT - self.prev_price)**d)

        return A



    def determine_demand(self):

        price_zone = (self.mt,self.MT)

        if(self.prev_price in price_zone):

            self.demand = (self.fundamental_value - self.prev_price)()

        else:

            self.demand = 0

    def business_growth(self, time, s):

        # determine the cycle of the business growth
        i = time // (4 * s) + 1

        if time >= 4 * (i - 1) * s and time < (4 * i - 1) * s:
            self.fundamental_value = self.fundamental_value * (1 + self.growth_rate)
            # return self.growth_rate
        else:
            self.fundamental_value = self.fundamental_value * (1 - self.growth_rate/2)
            # return -(self.growth_rate/2)
        
        

        



