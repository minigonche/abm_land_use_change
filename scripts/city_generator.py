
import numpy as np
import itertools
import random

import constants as con

from sklearn.cluster import KMeans


import pickle


city_file = 'rasters/bogota_city-1990.pkl'
amenities_file = 'rasters/bogota_amenities-1990.pkl'
sensibility = 0.45
num_city_centers = 5

def save_pickle(obj, file):
        with open(file,'wb') as f:
            pickle.dump(obj, f)


    
def load_pickle(file):
    with open(file,'rb') as f:
        x = pickle.load(f)
        return(x)


class CityGenerator(object):
    """
    Static class in charge of creating city layouts. All
    """
    @staticmethod
    def create_center(grid_shape, position = 'CENTRAL'):

        if(position == 'CENTRAL'):
            x_cen , y_cen = grid_shape[0]/2, grid_shape[1]/2
        elif(position == "RANDOM"):
            x_cen , y_cen = np.random.randint(grid_shape[0]), np.random.randint(grid_shape[1])
        else:
            raise ValueError("Position: " + str(position) + " Not Supported")

        return((x_cen, y_cen))


    @staticmethod
    def make_circular_city(grid_shape, x_cen, y_cen, percentage = 0.3):

        fun = lambda x,y : (x - x_cen)**2 + (y - y_cen)**2 <= (percentage*grid_shape[0]/2)**2

        def get_type(ind):
            if(fun(ind[0], ind[1])):
                return(con.LAND_TYPES.CITY)
            return(con.LAND_TYPES.LAND)

        return(get_type)



    @staticmethod
    def make_cities_fun(grid_shape, num_cities = 3, position = 'RANDOM', ):

        _functions = []
        _centers = []
        for i in range(num_cities):
            center = CityGenerator.create_center(grid_shape = grid_shape,
                                           position = position)
            _centers.append(center)
            _functions.append(CityGenerator.make_circular_city(grid_shape = grid_shape,
                                           x_cen = center[0],
                                           y_cen = center[1]))
                                           #percentage = np.random.randint(40)/100))
        def fun(ind):
            for _f in _functions:
                if(_f(ind) == con.LAND_TYPES.CITY):
                    return(con.LAND_TYPES.CITY)
            return(con.LAND_TYPES.LAND)

        return(_centers, fun)


    @staticmethod
    def import_from_raster(grid_shape):

        
        percentile = 90

        array = load_pickle(city_file)

        array = CityGenerator.down_size(array, grid_shape)

        # Defines the function
        def fun(ind):
            if(array[ind] >= sensibility):
                 return(con.LAND_TYPES.CITY)

            return(con.LAND_TYPES.LAND)

        #  Gets the city centers
        per = np.percentile(array.flatten(), percentile)
        points = []

        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if(array[i,j] >= per):
                    points.append([i,j])
                    

        kmeans = KMeans(n_clusters=num_city_centers, random_state=0).fit(points)
        _centers = [ (int(x[0]),int(x[1])) for x in kmeans.cluster_centers_]

        return(_centers, fun)



    @staticmethod
    def amenities_from_raster(grid_shape):

        amen_array = load_pickle(amenities_file)

        amen_array = CityGenerator.down_size(amen_array, grid_shape)

        # Defines the function
        def amenities_fun(patch):            
            return(amen_array[patch.pos])

        return(amenities_fun)
        


    @staticmethod
    def padd_matrix(matrix, pixel_size):

        dim_1 = matrix.shape[0]
        dim_2 = matrix.shape[1]

        #Expands the grid so that it can be divisible by the required tract
        dim_1_res = np.mod(pixel_size[0] - np.mod(dim_1,pixel_size[0]),pixel_size[0])
        dim_2_res = np.mod(pixel_size[1] - np.mod(dim_2,pixel_size[1]),pixel_size[1])

        dim_1_pad = (int(np.floor(dim_1_res/2)), int(np.ceil(dim_1_res/2)))
        dim_2_pad = (int(np.floor(dim_2_res/2)), int(np.ceil(dim_2_res/2)))
        padding = (dim_1_pad,dim_2_pad)
        matrix = np.pad(matrix, padding, mode = "constant", constant_values = np.nan)
                    
        return(matrix)


    @staticmethod
    def aggregate(matrix, pixel_size = (2,2), nan_value = 0):
            
        #Padds matrix        
        matrix = CityGenerator.padd_matrix(matrix, pixel_size)
        
        #Replaces NAn
        matrix[np.isnan(matrix)] = nan_value

        dim_1 = matrix.shape[0]
        dim_2 = matrix.shape[1]

        pixel_1 = pixel_size[0]
        pixel_2 = pixel_size[1]

        #Reshape Magic
        matrix = matrix.reshape((int(dim_1/pixel_1),pixel_1, dim_2), order= 'A')
        matrix = np.array( [x.transpose().reshape((int(dim_2/pixel_2), pixel_1*pixel_2), order = 'C') for x in matrix])
        
        matrix = np.nanmean(matrix, axis=2)
        
        return(matrix)


    @staticmethod
    def down_size(matrix, size = (100,100), nan_value = 0):
        
        #Replaces NAn
        matrix[np.isnan(matrix)] = nan_value
        
        original_dim = matrix.shape
        
        if(size[0] > original_dim[0] or size[1] > original_dim[1]):
            raise ValueError("Target size can't be larger than original size. Original Size: " + str(original_dim) + ' , Target Size: ' + str(size) )
        
        pixel_size = (int(np.floor(original_dim[0]/size[0])), int(original_dim[1]/size[1]))
        
        result = CityGenerator.aggregate(matrix, pixel_size)
        
        final_size = result.shape
        
        dim_1_start = int((final_size[0] - size[0])/2)
        dim_2_start = int((final_size[1] - size[1])/2)
        
        result = result[dim_1_start:(dim_1_start + size[0]),dim_2_start:(dim_2_start + size[1])]
        
        return(result)



