#Batch run for the land model use
from mesa.batchrunner import BatchRunner

from model import *



size = (200,200)
num_cities = 'raster'
percentage_buyers = 0.115
percentage_high_income = 50
rich_market_reach = 25
poor_market_reach = 25
base_price = 1000
base_income = 1000
rich_poor_std = 200
b = 1
urbanization_over_rural_roductivity = 0.5
alpha = 0.5
amenities = "RASTER_AMENITIES"
rural_productivity = "uniform"
max_epochs = np.inf
max_urbanization = 3045

fixed_params = {"percentage_buyers" :percentage_buyers, 
                         "percentage_high_income":percentage_high_income, 
                         "rich_market_reach":rich_market_reach, 
                         "poor_market_reach":poor_market_reach,
                         "base_price":base_price,
                         "base_income":base_income,
                         "rich_poor_std":rich_poor_std, 
                         "num_cities":num_cities,
                         "b":b, 
                         "urbanization_over_rural_roductivity":urbanization_over_rural_roductivity,                          
                         "size":size,
                         #"alpha":alpha,
                         "amenities":amenities,
                         "rural_productivity":rural_productivity,
                         "max_epochs":max_epochs,
                         "max_urbanization":max_urbanization
                         }

variable_params = { 
                        #"percentage_buyers" :[15,50,85], 
                         #"percentage_high_income":[15,50,85], 
                         #"rich_market_reach":[5,10,25], 
                         #"poor_market_reach":[5,10,25],
                         #"base_price":[500,1000,2000],
                         #"base_income":[500,1000,2000],
                         #"rich_poor_std":[50,200,500], 
                         #"num_cities":[1,2,4],                         
                         #"urbanization_over_rural_roductivity":[0.15,0.5,0.85],                          
                         "alpha":[0.1,0.5,0.9]
                         #"max_epochs":[10,25,50]
                   
           }


batch_run = BatchRunner(LandModel,
                        fixed_parameters=fixed_params,
                        variable_parameters=variable_params,
                        iterations=50,
                        max_steps=10,
                        model_reporters={"mean_amenities": get_mean_amenities,
                                         "mean_price": get_mean_price,
                                         "std_price": get_std_price,
                                         "tda_info": get_tda_information,
                                         "dissimilarity_index": get_index_of_disimilarity,
                                         "interaction_exposure_index": get_index_of_exposure_interaction,
                                         "percetnage_available_land":percetnage_available_land,
                                         "final_rich_percentage":get_final_rich_percentage,
                                         "final_poor_percentage":get_final_poor_percentage,
                                         "num_epochs":get_num_epochs})


batch_run.run_all()


run_data = batch_run.get_model_vars_dataframe()

sin_tda =  run_data[run_data.columns.difference(['tda_info'])] 
tda =  run_data[['Run','tda_info']] 

sin_tda.to_excel('output.xlsx')
tda.to_csv('tda.csv')





