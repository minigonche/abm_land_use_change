# Model class and scripts for Land Use Agent Based Modeling
# based on MESA
import numpy as np
import itertools
from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

import random

from agents import Buyer, Seller, LandPatch
from city_generator import CityGenerator

from sklearn.metrics import pairwise_distances
from sklearn import metrics

import constants as cons

import json


def get_mean_bid(model):
    """
    Exctracts the mean bid f the model in the current epoch
    """
    if(len(model.current_bids)==0):
        return(0)

    return(int(np.mean(model.current_bids)))


def get_mean_amenities(model):
    """
    Returns the mean amenities value. Takes into account only
    land patches (not city ones)
    """
    total_amenities = 0
    for patch in model.land_patches:
        total_amenities += patch.amenities

    mean = total_amenities/len(model.land_patches)

    return(mean)

def get_mean_price(model):

    arr = [patch.sold_value for patch in model.city_patches if patch.sold_value > 0]    
    return(np.nanmean(arr))

def get_std_price(model):

    arr = [patch.sold_value for patch in model.city_patches if patch.sold_value > 0]    
    return(np.nanstd(arr))


def get_tda_information(model):

    tda_info = []

    for i in range(model.patches.shape[0]):
        for j in range(model.patches.shape[0]):
            patch = model.patches[i,j]            
            values = patch.to_dictionary()
            values['x'] = i
            values['y'] = j
            tda_info.append(values)


    return(json.dumps(tda_info, ensure_ascii=False))

def get_final_rich_percentage(model):

    final_type = np.array([ patch.sold_type for patch in model.city_patches if patch.sold_type != None])
    return(100*np.sum(final_type == cons.AGENT_TYPES.RICH)/len(final_type))


def get_final_poor_percentage(model):
    final_type = np.array([ patch.sold_type for patch in model.city_patches if patch.sold_type != None])
    return(100*np.sum(final_type == cons.AGENT_TYPES.POOR)/len(final_type))

def get_num_epochs(model):
    return(model.num_epochs)



#Indices are taken from:
# https://www.dartmouth.edu/~segregation/IndicesofSegregation.pdf
def get_index_of_disimilarity(model,  tract = 4):

    n,r,R,p,P = model.get_tract_info(tract)

    if(P == 0 or R==0):
        return(0)

    result = 0.5*np.sum(np.abs(r/R - p/P))
    return(result)

def get_index_of_exposure_interaction(model,  tract = 4):

    n,r,R,p,P = model.get_tract_info(tract)

    if(P == 0 or R==0):
        return(1)

    result = np.sum((p/P)*(r/(r+p)))

    return(result)


def percetnage_available_land(model):

    return(100*len(model.land_patches)/(model.width*model.height))

        


class LandModel(Model):



    """
    Main model
    """
    def __init__(self,
                 percentage_buyers, # Percentage of buyers to be added acording to the available land  
                 percentage_high_income, #Percentage that are high income
                 rich_market_reach, # The amount of properties the rich can bid
                 poor_market_reach, # The amount of properties the poor can bid
                 base_price, # The land base price
                 base_income, # The base income for all buyers
                 rich_poor_std, #Standard deviation for the rich and poor
                 num_cities, # The number of cities
                 b, # Constant for the wtp formula                 
                 urbanization_over_rural_roductivity, # Constant for the probability of selling
                 size, # tuple: (width, height)                                  
                 alpha = 0.5,# The alpha parameter (preference)
                 amenities = "uniform", #The amenities
                 rural_productivity = "uniform", # The rural productivity
                 buyer_mode =  cons.BID_SCHEME.BEST,
                 max_epochs = 10, #number of epochs
                 max_urbanization = np.inf # Maximum number of urbanization units
                 ): 


        position = "CENTRAL"

        #print('')
        #print('Rich: ' + str(rich_market_reach) + ' Poor: ' + str(poor_market_reach))

        #Variables for batch run
        self.num_epochs = 0
        self.max_epochs = max_epochs
        self.max_urbanization = max_urbanization
        self.running = True

        #Scheduler
        self.schedule = BaseScheduler(self)

        #Assigns variables to the model
        self.base_price = base_price
        self.base_income = base_income
        self.rich_poor_std = rich_poor_std

        self.rich_market_reach = rich_market_reach
        self.poor_market_reach = poor_market_reach

        self.b = b

        self.w2 = urbanization_over_rural_roductivity
        self.w1 = 1 - self.w2

        self.epyslon = 0

        self.alpha = alpha

        self.buyer_mode = buyer_mode

        #Sets up the grid

        #Size of Grid
        self.width = size[0]
        self.height = size[1]
        self.grid = MultiGrid(self.width, self.height, True) # Multigrid - Agents can share a cell

        #Assign the shapes
        self.grid_shape = (self.width, self.height)


        #Distribution variables
        self.amenities = self.get_distribution(amenities)
        self.rural_productivity = self.get_distribution(rural_productivity)

        #Starts the patches as an empty matix
        self.patches = np.empty((self.width, self.height), dtype = object)


        #Invokes the city generator
        # Checks if will import from raster
        if str(num_cities).upper() == 'RASTER':
            centers, city_fun = CityGenerator.import_from_raster(self.grid_shape)

        else:    
            # Handles the city creation (at random) and the corresponding centers
            centers, city_fun = CityGenerator.make_cities_fun(self.grid_shape, num_cities = num_cities, position = position)

        self.city_centers = centers

        #Calculates max city distance (for the city proximity function)
        self.max_city_distance = self.get_max_city_distance()

        #Starts land patches and city pathces (both are sets to guarantee O(1) lookup)
        self.land_patches = set()
        self.city_patches = set()


        #Iterates over each possible patch and assigns city or land
        for index in list(itertools.product(range(self.width), range(self.height))):

            patch = LandPatch("land_patch_" + str(index), self)
            patch.type = city_fun(index)

            if(patch.type == cons.LAND_TYPES.CITY):
                self.city_patches.add(patch)
                patch.convert_to_city()
            else:
                self.land_patches.add(patch)
                patch.convert_to_land()

            self.grid.place_agent(patch, index)
            self.patches[index] = patch
            patch.set_properties() # Initializes patch


        #The number of sellers will be the number of land patches
        num_sellers = len(self.land_patches)


        self.percentage_buyers = percentage_buyers
        if(percentage_buyers <= 0 or percentage_buyers > 100):
            raise ValueError('Percentage of buyers is incorrect: ' + str(percentage_buyers))

        # Calculates the number of buyers
        self.percentage_high_income = percentage_high_income
        self.num_buyers = np.round((self.percentage_buyers/100)*len(self.land_patches))
        

        # Sets up the number of poor and rich buyers
        self.num_rich_buyers = int(self.num_buyers*self.percentage_high_income/100)
        self.num_poor_buyers = int(self.num_buyers*(100- self.percentage_high_income)/100)

        #Number of Agents
        self.num_sellers = num_sellers
        self.num_buyers = self.num_rich_buyers + self.num_poor_buyers
        self.num_agents = self.num_sellers + self.num_buyers

        self.buyers = set()
        self.sellers = set()

        #To remove
        # At the end of each iteration will remove the sellers who sold
        self.to_remove = []

        # adds the sellers (one for each land patch)
        added_sellers = 0
        for pat in self.land_patches:

            seller = Seller("seller_" + str(added_sellers), self, pat)
            self.schedule.add(seller)

            #Adds the seller
            self.grid.place_agent(seller, pat.pos)
            self.sellers.add(seller)

            added_sellers += 1


        # Adds the buyers
        self.add_rich_buyers(self.num_rich_buyers)
        self.add_poor_buyers(self.num_poor_buyers)

        # The data collector
        # For plotting oprions
        self.current_bids = []

        self.datacollector = DataCollector(
                model_reporters={"Amenities": get_mean_amenities})




    def step(self):
        """
        Method that simulates the step of an epoch
        """

        #Scehdule
        #print('Step ' + str(self.num_epochs))

        #Calculates the sellers that will sell
        market = set()
        for seller in self.sellers:
            if(seller.will_sell()):
                market.add(seller)

        if(len(market) == 0):
            print('Nobody wants to sell')

        #All the buyer make their bids
        for buyer in self.buyers:
            buyer.make_bids(market, self.buyer_mode)


        #Asjust Epsylon
        self.epyslon = (len(self.buyers) - len(self.sellers))/(len(self.buyers) + len(self.sellers))

        #Randomly, buyers and sellers interact
        sellers = random.sample(self.sellers, len(self.sellers))
        for seller in sellers:
            sold = seller.sell()
            if(sold):
                #Removes seller
                self.grid._remove_agent(seller.pos, seller)
                self.sellers.remove(seller)

        self.current_bids = []

        #removes all buyers
        self.remove_all_buyers()

        #Updates the number of buyers for next epoch
        self.num_buyers = np.round((self.percentage_buyers/100)*len(self.land_patches))
        
        

        # Sets up the number of poor and rich buyers
        self.num_rich_buyers = int(self.num_buyers*self.percentage_high_income/100)
        self.num_poor_buyers = int(self.num_buyers*(100- self.percentage_high_income)/100)        
        self.num_buyers = self.num_rich_buyers + self.num_poor_buyers

        #Adds the new generation
        self.add_rich_buyers(self.num_rich_buyers)
        self.add_poor_buyers(self.num_poor_buyers)

        #Collects data
        self.datacollector.collect(self)

        #Adjust progress variables (for batch process)
        self.num_epochs += 1
        if(self.num_epochs == self.max_epochs or self.get_num_patches() == 0):
            self.running = False

        #Stops the process if the maximum urbanization limit is exceeded
        if(len(self.city_patches) >= self.max_urbanization):
            print(self.num_epochs)
            self.running = False



    def get_max_city_distance(self):
        """
        Calculates the max city distance available in the greed (recall there can
        be multiple cities)
        """
        indices = list(itertools.product(range(self.width),range(self.height)))
        positions = np.array(indices)
        centers = np.array(self.city_centers)

        dist_centers = metrics.pairwise.euclidean_distances(positions,centers)
        dist_centers = np.min(dist_centers, axis = 1)

        return(np.max(dist_centers))


    def get_available_pos(self):
        """
        Gets an available position (so agent don't end up in a city patch)
        Was design for when the agents bought around them.
        """

        if(self.land_patches):
            land_patch = random.sample(self.land_patches,1)[0]
            return(land_patch.pos)

        return(None)

    def get_available_positions(self, k, replace = False):

        choices = np.random.choice(list(self.land_patches), size = k, replace = replace)

        return([land_patch.pos for land_patch in choices])
        


    def get_num_patches(self):
        return(len(self.land_patches))



    def remove_all_buyers(self):
        """
        Removes all buyers from the model. At the end of every epoch,
        this method is called
        """
        for agent in self.buyers:
            self.grid._remove_agent(agent.pos, agent)
            self.schedule.remove(agent)

        self.buyers = set()


    def add_rich_buyers(self, num_rich_buyers):
        """
        Add rich buyers to the grid
        """
        indices = self.get_available_positions(num_rich_buyers)

        for i in range(num_rich_buyers):
            income = max(0,(self.base_income + np.abs(np.random.normal(loc = 0, scale = self.rich_poor_std))))
            buyer = Buyer("buyer_rich_" + str(i), self, income = income, market_reach = self.rich_market_reach, type = cons.AGENT_TYPES.RICH)
            self.schedule.add(buyer)

            #Gets an available position
            index = indices[i]

            #Adds the buyer
            self.grid.place_agent(buyer, index)
            self.buyers.add(buyer)


    def add_poor_buyers(self, num_poor_buyers):
        """
        Add poor buyers to the grid
        """
        indices = self.get_available_positions(num_poor_buyers)

        for i in range(num_poor_buyers):
            income = max(0,(self.base_income - np.abs(np.random.normal(loc = 0, scale = self.rich_poor_std))))
            buyer = Buyer("buyer_poor_" + str(i), self, income = income, market_reach = self.poor_market_reach, type = cons.AGENT_TYPES.POOR)
            self.schedule.add(buyer)

            #Gets an available position
            index = indices[i]

            #Adds the buyer
            self.grid.place_agent(buyer, index)

            self.buyers.add(buyer)



    def get_distribution(self, name):
        """
        Gets the corresponding function depending on the distribution
        """

        if("CONSTANT" in str(name).upper()):
            name = name.replace(" ","")
            values = name.split("=")
            if(len(values) != 2):
                raise ValueError("If function is constant, should specify its value")

            constant = float(values[1])
            return(lambda x: constant )        

        if(str(name).upper() == "UNIFORM"):
            return(lambda x: np.random.uniform())

        if(str(name).upper() == "RASTER_AMENITIES"):
            return(CityGenerator.amenities_from_raster(self.grid_shape))


        raise ValueError("No implementation for the distribution: "  + str(name))



    def get_tract_info(self, tract = 4):

        matrix = self.get_sold_type_matrix()
        array = get_tract_structure(matrix, tract)

        n = len(array)
        r = np.apply_along_axis(lambda x: np.sum(x == cons.AGENT_TYPES.RICH), 0, array)
        R = np.sum(matrix == cons.AGENT_TYPES.RICH)
        p = np.apply_along_axis(lambda x: np.sum(x == cons.AGENT_TYPES.POOR), 0, array)
        P = np.sum(matrix == cons.AGENT_TYPES.POOR)

        return(n,r,R,p,P)




    def get_sold_type_matrix(self):

        #Extract the matrix with the values
        matrix = np.array([[ self.patches[i,j].sold_type for j in range(self.patches.shape[1])] for i in range(self.patches.shape[0])])
        return(matrix)







        


def get_tract_structure(matrix, tract = 4):


    #Padds matrix        
    matrix = padd_matrix(matrix, tract)
    
    dim_1 = matrix.shape[0]
    dim_2 = matrix.shape[1]

    #Gets number of tracts
    num_tracts = int((matrix.shape[0]*matrix.shape[1])/tract**2)

    # Reshape magic
    matrix = matrix.reshape((int(dim_1/tract), tract, dim_2), order= 'A')
    matrix = np.array( [x.reshape((tract,dim_2), order = "C").reshape((tract**2,int(dim_2/tract)), order = "F").transpose() for x in matrix])
    matrix = matrix.reshape(num_tracts,tract**2)

    return(matrix)
    

def padd_matrix(matrix, tract):

    dim_1 = matrix.shape[0]
    dim_2 = matrix.shape[1]

    #Expands the grid so that it can be divisible by the required tract
    dim_1_res = np.mod(tract - np.mod(dim_1,tract),tract)
    dim_2_res = np.mod(tract - np.mod(dim_2,tract),tract)

    dim_1_pad = (int(np.floor(dim_1_res/2)), int(np.ceil(dim_1_res/2)))
    dim_2_pad = (int(np.floor(dim_2_res/2)), int(np.ceil(dim_2_res/2)))
    padding = (dim_1_pad,dim_2_pad)
    matrix = np.pad(matrix, padding, mode = "constant", constant_values = None)

    return(matrix)
