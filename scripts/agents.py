import numpy as np
import itertools
from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.space import MultiGrid

import random

from sklearn.metrics import pairwise_distances
from sklearn import metrics

import constants as cons

class Buyer(Agent):
    """
    Class that models the buyer agent
    """

    def __init__(self,
                unique_id, # Uniqu id for the agent
                model, # The model the agent is asign to
                income, # The buyers income
                market_reach, # The number of patches the agent can see
                type):# The type (rich or poor)

        #Calls superclass
        super().__init__(unique_id, model)

        #Assigns all variables
        self.type = type
        self.income = income
        self.market_reach = market_reach
        self.preference = self.model.alpha
        self.sold = False


    def ask_bid_price(self, land):
        """
        Calculates the wtp of the buyer
        """

        # If already sold return 0
        if(self.sold):
            return(0)

        alpha = self.preference
        utility =  (land.amenities**alpha)  +  (land.proximity_to_city**(1-alpha))
        b = self.model.b

        wtp = ((self.income)*utility)/(b**2 + utility)

        return(wtp)


    def sold(self):
        """
        Updates the buyers status so he can not purchase anymore land
        """
        self.sold = True


    def get_number_of_bids_to_make(self):
        """
        Gets the number of land patches the agent can see to make bids
        """
        return(self.market_reach)


    def make_bids(self, market, mode = cons.BID_SCHEME.BEST):
        """
        Makes bids to different patches. The current implementation is that
        the buyer bids a certain number of lands selected randomly
        """

        #max number of bids
        num_bids = self.get_number_of_bids_to_make()

        #Makes random bids. The number of bids is done according to parameter
        possible_sellers = random.sample(market, min(len(market),num_bids))


        if(mode == cons.BID_SCHEME.BEST):

            best_land = None
            quality = -1
            # Makes bids
            for seller in possible_sellers:
                land = seller.land_patch
                if(self.ask_bid_price(land) > quality):
                    quality = self.ask_bid_price(land)
                    best_land = land

            if(best_land != None):
                best_land.make_bid(self)

        else:
            # Makes bids
            for seller in possible_sellers:
                land = seller.land_patch
                land.make_bid(self)



    def __hash__(self):
        """
        Support Method for saving agents in sets
        """
        return(hash(self.unique_id))



class Seller(Agent):

    """
    Class that models the seller agent
    """

    def __init__(self,
                unique_id, # Unique id for the agent
                model, # The model the agent is asign to
                land_patch): # The land patch the agento owns

        #Calls super class
        super().__init__(unique_id, model)

        self.land_patch = land_patch

    def step(self):
        """
        Method that simultes the epoch of this agent.
        Currently, it only sells the property (if any bids where made)
        The rest is orchestrated by the model
        """
        self.sell()

    def will_sell(self):
        """
        Method that answers if the seller will sell the property
        """
        return(np.random.uniform() <= self.calculate_probability_to_sale())


    def calculate_probability_to_sale(self):
        """
        Calculates the probability that the agent will sell its property
        """

        w1 = self.model.w1
        w2 = self.model.w2
        urbanization = self.land_patch.calculate_urbanization()

        return(w1*(1-self.land_patch.rural_productivity) + w2*urbanization)

    def sell(self):

        """
        Interaction with the bids. The seller looks among the bids made to its
        property, selecs the highest and sells the property to the buyer
        (if the conditions are met). This method return true if the patch
        was sold, false if not
        """

        #finds the land patch
        patch = self.land_patch

        # Willignes to accept
        wta = patch.calculate_price(self)

        #The sellers that made the bids
        bidders = patch.bidders

        #Askprice adjusted to market
        pa = wta*(1 + self.model.epyslon)

        # IF there are any bids
        if(len(bidders) > 0):

            #bid prices
            wtps = [bidder.ask_bid_price(patch) for bidder in bidders]

            #Asjusted bids
            pbs = [wtp*(1 - self.model.epyslon) for wtp in wtps]

            #Adds bids to model
            self.model.current_bids = self.model.current_bids + wtps


            #Select prospect buyer
            i = np.argmax(pbs)
            pb = np.max(pbs)

            sell_property = False
            sell_price = -1

            if(pb > pa):#Case 1
                sell_property = True
                sell_price = pb
            elif(pb > 0.75*pa):#Case 2
                sell_property = True
                sell_price = (pa - pb)/2 + pb


            if(sell_property):
                #Current buyer
                buyer = bidders[i]

                #Sells the property
                patch.convert_to_city(buyer.type, sell_price, buyer)

                #Adds the buyer and the seller to be removed
                self.model.to_remove.append(buyer)
                buyer.sold = True

            return(True)

        return(False)


    def __hash__(self):
        """
        Support Method for saving agents in sets
        """
        return(hash(self.unique_id))



class LandPatch(Agent):

    
    def __init__(self,
                unique_id, # Unique model id
                model): # The model where the agent is deployes


        super().__init__(unique_id, model)

        #Atributes
        self.type = cons.LAND_TYPES.LAND
        self.bidders = []
        self.sold_type = None
        self.sold_value = -1
        self.buyer = None
        self.rural_productivity = -1
        self.proximity_to_city = 1
        self.amenities = -1


    
    """ Proximity to city. Starts at 1 and is calculated when the whole
    grid is formed """    
    def calculate_proximity_to_city(self):
        """
        Calculates the proximity to the city
        """
        centers = self.model.city_centers

        dist_centers = metrics.pairwise.euclidean_distances([[self.pos[0], self.pos[1]]],centers)
        dist_centers = np.min(dist_centers, axis = 1)[0]

        return(1 - dist_centers/self.model.max_city_distance)


    def set_properties( self, proximity_to_city = None, amenities = None, rural_productivity = None):
        """
        Sets the properties of the land patch. This method is created so it can
        be called once the model is completely set up (so it can take into account
        other information about the model if necesary)
        """
        #amenities
        self.amenities = amenities
        if(self.amenities is None):            
            self.amenities = self.model.amenities(self)
            

        #Rural Productivity
        self.rural_productivity = rural_productivity
        if(self.rural_productivity is None):
            self.rural_productivity = self.model.rural_productivity(self)


        #Proximity to the city
        self.proximity_to_city = proximity_to_city
        if(self.proximity_to_city is None):
            self.proximity_to_city = self.calculate_proximity_to_city()


    def step(self):
        """
        Method that simultes the epoch of this agent.
        Currently, it only resets its bids.
        The rest is orchestrated by the model
        """
        #Resets the bidders
        self.bidders = []
        return


    def calculate_price(self, seller):
        """
        Calculates the price of the land based on its attributes
        """

        if(self.type == cons.LAND_TYPES.CITY):
            return np.inf

        base_price = self.model.base_price*(0.5*self.amenities + 0.5*self.proximity_to_city)

        # Now checks price against neighbor properties sold
        neighbors = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=True)

        sold_value = 0
        for pos in neighbors:
            patch = self.model.patches[pos]
            if(patch.type == cons.LAND_TYPES.CITY):
                sold_value = max(sold_value, patch.sold_value)


        return(max(base_price, sold_value))


    def make_bid(self, buyer):
        """
        Receive a bid made by a buyer
        """
        self.bidders.append(buyer)


    def convert_to_city(self, sold_type = None, sold_value = -1, buyer = None):
        """
        Converts the patch to city
        """
        self.sold_type = sold_type # The type that bought the land (rich or poor)
        self.sold_value = sold_value # The amount that was paid for the land
        self.type = cons.LAND_TYPES.CITY
        self.model.land_patches.discard(self)
        self.model.city_patches.add(self)
        self.buyer = buyer

    def convert_to_land(self):
        """
        Converts the patch to land
        """
        self.type = cons.LAND_TYPES.LAND
        self.model.land_patches.add(self)
        self.model.city_patches.discard(self)


    def calculate_urbanization(self):
        """
        Calculates the urbanization value, based on its neighbors
        """
        neighbors = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=True)

        urban = 0
        for pos in neighbors:
            patch = self.model.patches[pos]
            if(patch.type == cons.LAND_TYPES.CITY):
                urban += 1

        return(urban/len(neighbors))


    def to_dictionary(self):

        
        response = {}
        response['type'] = self.type
        response['sold_type'] = self.sold_type
        response['sold_value'] = self.sold_value
        response['rural_productivity'] = self.rural_productivity
        response['proximity_to_city'] = self.proximity_to_city
        response['amenities'] = self.amenities

        response['income_of_buyer'] = -1
        if(self.buyer is not None):
            response['income_of_buyer'] = self.buyer.income


        return(response)



    def __hash__(self):
        """
        Support Method for saving agents in sets
        """
        return(hash(self.unique_id))
