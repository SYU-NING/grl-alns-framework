import math
import numpy as np

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def categorical_sample(self, prob_n, np_random):
    '''sample from categorical distribution. Each row specifies class probabilities
    '''
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.random()).argmax()

def maxabs_scale(values):
    maxv = max(values)
    return [v / maxv for v in values]

def compute_distance_matrix(loc_x, loc_y, nodes, **kwargs):
    '''gives the whole matrix, compute_distance in function gives a specified pair
    '''
    distance_matrix = {(i, j): np.sqrt((loc_x[i] - loc_x[j])**2 + (loc_y[i] - loc_y[j])**2)
                       for i in nodes for j in nodes}
    
    if "local_random" in kwargs:
        noise_lb, noise_ub = 0.8, 1.2
        # noise_lb, noise_ub = kwargs["noise_lb"], kwargs["noise_ub"]
        local_random = kwargs["local_random"]
        distance_matrix = {key: value * local_random.uniform(noise_lb, noise_ub) for key, value in distance_matrix.items()}

    return distance_matrix

def compute_arc_length(lat1, lon1, lat2, lon2):
    ''' 
    Compute the gepgrahical distance between two nodes given in latitudes and longitudes 
    '''
    # Convert coordinates from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Compute the differences in latitudes and longitudes
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = math.sin(dlat/2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Radius of the Earth in kilometers
    earth_radius = 6371

    # Compute the arc length
    arc_length = earth_radius * c

    return arc_length

def compute_arc_length_matrix(latitude, longitude, nodes):
    '''gives the whole matrix, compute geographical distance using lat and long for any node pair'''

    distance_matrix = {(i, j): compute_arc_length(latitude[i], longitude[i], latitude[j], longitude[j])
                       for i in nodes for j in nodes}
    return distance_matrix

def normalize_list(lst):
    """ Returns a normalized list with values between [0,1] """
    min_val, max_val = min(lst), max(lst)
    return [(val - min_val) / (max_val - min_val) for val in lst]

def normalize_dict_values(dictionary):
    norm_values = normalize_list(list(dictionary.values()))
    norm_dict = {}
    for i, key in enumerate(dictionary.keys()):
        norm_dict[key] = norm_values[i]
    return norm_dict



def same_tour_frequency(customers):
    '''record for every pair of customers the number of times they appear inside same tour '''
    customer_pairs = {(i, j): 0 for i in customers for j in customers if i != j}
    return customer_pairs


def smallest_removal_gain(customers):
    '''record smallest removal gain for every customer node '''
    customer_node = {i: float('inf') for i in customers}
    return customer_node


def non_removal_count(customers):
    '''record non-removal frequency for every customer node '''
    customer_node = {i: 0 for i in customers}
    return customer_node


def smallest_travel_distance(customers):
    '''record smallest travelling distance between every pair of customers '''
    customer_pairs = {(i, j): float('inf') for i in customers for j in customers if i < j}
    return customer_pairs


def round_to_nearest_multiple_of_10(number):
    rounded_number = round(number/10) *10
    return float(rounded_number)
