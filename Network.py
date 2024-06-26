
import random
import networkx as nx
import matplotlib.pyplot as plt
from Fundamentalist import Fundamentalist
from Chartist import Chartist
import numpy as np





class Network():

    def __init__(self, network_type, number_of_traders, percent_fund, percent_chartist, percent_rational = 0.50,percent_risky = 0.50, high_lookback = 5, low_lookback = 1, high_risk = 0.50, low_risk = 0.10,  new_node_edges = None, connection_probability = None):

        self.network_type = network_type
        self.number_of_traders = number_of_traders
        self.percent_fund = percent_fund
        self.percent_chartist = percent_chartist
        self.percent_rational = percent_rational
        self.percent_risky = percent_risky
        self.high_lookback = high_lookback 
        self.low_lookback = low_lookback
        self.high_risk = high_risk, 
        self.low_risk = low_risk

        self.connection_probability = connection_probability
        self.new_node_edges = new_node_edges
        self.network = None
        self.trader_dictionary = None


    def create_network(self):

        if self.network_type == "barabasi":
                
            self.network = nx.barabasi_albert_graph(n = self.number_of_traders,  m=self.new_node_edges)

        elif self.network_type == "erdos_renyi":

            self.network = nx.erdos_renyi_graph(n = self.number_of_traders, p = self.connection_probability)

        traders  = self.create_traders()

        for i, node in enumerate(self.network.nodes()):

            self.network.nodes[node]['trader'] = traders[i] #Assign a trader to each node

            self.trader_dictionary = {trader.node_number: trader for trader in traders}


        return self.network, self.trader_dictionary


    def display_network(self):
        """Plots the network structure with additional information from trader_dict."""
        # Loop over the nodes and link them to a trader
        for node in self.network.nodes():
     
            if node in self.trader_dictionary:
                self.network.nodes[node]['label'] = self.trader_dictionary[node].type[0]

        # Drawing the network
        pos = nx.spring_layout(self.network)  # positions for all nodes

        # nodes
        nx.draw_networkx_nodes(self.network, pos, node_color='lightblue')

        # edges
        nx.draw_networkx_edges(self.network, pos, edge_color='red')

        nx.draw_networkx_labels(self.network, pos, labels=nx.get_node_attributes(self.network, 'label'))
        
        plt.show()

    def get_neighbors(self, node_number):
        """Returns the neighbors of a node."""
        return list(self.network.neighbors(node_number))

    def create_traders(self):

        """Create a trader dictionary with a certain fraction of each type of trader"""

        num_fund = int(self.number_of_traders * self.percent_fund)
        num_chart = int(self.number_of_traders * self.percent_chartist)
        trader_types = ['fundamentalist'] * num_fund + ['chartist'] * num_chart 
        traders = [] 

        # generate the different fractions of traders
        Nc = 0
        Nf = 0 
        for i in range(self.number_of_traders):
            random.shuffle(trader_types)
            trader_type = trader_types.pop()
            if trader_type == 'fundamentalist':
                Nf += 1
                eta = 0.991
                alpha_w = 2668
                alpha_O = 2.1
                alpha_p = 0
                phi = np.abs(np.random.normal(1,0.5))
                #phi = 1.00
                sigma_f =  0.681
                pstar = 0.02
                lookback_period = self.high_lookback if Nf/num_fund < self.percent_rational else  self.low_lookback
                max_risk = self.high_risk if Nf/num_fund < self.percent_risky else  self.low_risk
                traders.append(Fundamentalist(i,eta,alpha_w,alpha_O,alpha_p,phi,sigma_f,pstar,lookback_period,max_risk))
            if trader_type == 'chartist':
                Nc += 1
                eta = 0.991
                chi = np.abs(np.random.normal(1.20,0.5))
                # chi = 1.20
                sigma_c = 1.724
                lookback_period = self.high_lookback if Nc/num_chart < self.percent_rational else  self.low_lookback
                max_risk =  self.high_risk if Nc/num_fund < self.percent_risky else  self.low_risk
                traders.append(Chartist(i, eta, chi,sigma_c,lookback_period, max_risk))
            # if trader_type == 'random trader':
            #     traders.append(RandomTrader(i))
        return traders

    

    

    




        