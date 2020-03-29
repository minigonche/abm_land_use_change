from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule
from model import LandModel
from agents import Buyer, Seller, LandPatch
from colors import rgb, hex
import seaborn as sns
import numpy as np

import constants as cons

colors = ['#' + str(rgb(int(255*arr[0]), int(255*arr[1]), int(255*arr[2])).hex)  for arr in sns.light_palette("green")]


def agent_portrayal(agent):

    if agent is None:
        return

    portrayal = {}

    #if type(agent) is Buyer:
    #    portrayal["Shape"] = "circle"
    #    portrayal['Color'] = "red"
    #    portrayal["Layer"] = 2
    #    portrayal["Filled"] = "true"
    #    portrayal["r"] = 0.2

    #Doesn't draw the sellers
    #elif type(agent) is Seller:
    #    portrayal["Shape"] = "circle"
    #    portrayal['Color'] = "red"
    #    portrayal["Layer"] = 1
    #    portrayal["Filled"] = "true"
    #    portrayal["r"] = 0.5

    if type(agent) is LandPatch:
        print(agent.sold_type)
        if(agent.sold_type != None):
            if(agent.sold_type == cons.AGENT_TYPES.RICH):
              portrayal["Color"] = ["#fff200", "#fff200", "#fff200"]
            else:
              portrayal["Color"] = ["#0c00ff", "#0c00ff", "#0c00ff"]
        elif agent.type == cons.LAND_TYPES.CITY:
            portrayal["Color"] = ["#A9A9A9", "#DCDCDC", "#474646"]
        else:
            portrayal["Color"] = colors[int(np.floor(agent.amenities *(len(colors)-1)))]
        portrayal["Shape"] = "rect"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["w"] = 1
        portrayal["h"] = 1

    return portrayal


width = 50
height = 50

total_squares = width*height

grid = CanvasGrid(agent_portrayal, width, height, 500, 500)

#n_sellers = UserSettableParameter('slider', "Patches for Sale", int(0.5*total_squares), 1, total_squares, 1)

num_cities = UserSettableParameter('slider', "Number of Cities", 4, 1, 10, 1)

percentage_buyers = UserSettableParameter('slider', "Percentage of Buyers", 10, 1, 100, 1)

percentage_high_income = UserSettableParameter('slider', "Proportion High Income", 50, 0, 100, 1)

rich_market_reach = UserSettableParameter('slider', "Rich Market Reach", 10, 1, 50, 1)

poor_market_reach = UserSettableParameter('slider', "Poor Market Reach", 10, 1, 50, 1)

base_price = UserSettableParameter('slider', "Base Price", 1000, 0, 10000, 10)

base_income = UserSettableParameter('slider', "Base Income", 1000, 0, 10000, 10)

rich_poor_std = UserSettableParameter('slider', "Rich Poor STD", 200, 0, 1000, 10)

b = UserSettableParameter('slider', "b", 1, 0, 10, 0.1)

urbanization_over_rural_roductivity = UserSettableParameter('slider', "Urbanization over Rural Productivity", 0.5, 0, 1, 0.05)



chart = ChartModule([{"Label": "Oferta Promedio",
                      "Color": "Black"}],
                    data_collector_name='datacollector')

server = ModularServer(LandModel,
                       [grid, chart],
                       "Land Model",
                       {"percentage_buyers" :percentage_buyers, 
                         "percentage_high_income":percentage_high_income, 
                         "rich_market_reach":rich_market_reach, 
                         "poor_market_reach":poor_market_reach,
                         "base_price":base_price,
                         "base_income":base_income,
                         "rich_poor_std":rich_poor_std, 
                         "num_cities":num_cities,
                         "b":b, 
                         "urbanization_over_rural_roductivity":urbanization_over_rural_roductivity,                          
                         "size":(width,height)})
