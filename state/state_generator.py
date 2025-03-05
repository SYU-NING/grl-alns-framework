import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from state.alns_state import RoutingInstanceInfo, ALNSState, INITIAL_TOUR_FRACTION
from utils.general_utils import maxabs_scale, chunks, round_to_nearest_multiple_of_10

import xml.etree.ElementTree as ET


# each unserved customer causes this penalty:
nonserve_penalty = 1000

base_vehicle_hiring_cost = 100#00

# time window [a,b], arrival before a or after b will cause this penalty per unit
# original: 20
exceed_tw_penalty = 0.1

# setting to 1/4 of number of customer nodes for now,
# which aligns with the destroy scales used.
# original comment: should be around 1/10 of total population, or the same scale as destroy operator scale!
forbidden_random_num_of_iterations_nodes_fraction = 0.25

# 20 nodes with 100 capacity used in experiments -> 5 capacity per node
# obeying the same multiplier for larger instances.
solomon_capacity_per_node = 5

gehring_capacity_per_node = 1

desmet_capacity_per_node = 10   # TODO: need to fine-tune value


class StateGenerator(object):

    def generate_many(self, problem_variant, random_seeds, num_customers, instance_name="default", **kwargs):
        instances = []

        chunk_size = math.ceil(len(random_seeds) / len(num_customers))
        seed_chunks = list(chunks(random_seeds, chunk_size))
        for i, chonk in enumerate(seed_chunks):
            cust_number = num_customers[i]
            for seed in chonk:
                kwargs_cp = deepcopy(kwargs)
                kwargs_cp['cust_number'] = cust_number

                state = self.generate_alns_state(problem_variant, seed, instance_name=instance_name, **kwargs_cp)
                state.ds_chunk_size = chunk_size
                state.ds_seed_chunks = seed_chunks
                state.ds_num_customers = num_customers
                instances.append(state)

        return instances


class SolomonStateGenerator(StateGenerator):
    name = "solomon"

    ''' Solomon instances (max 100 customers):
        Defines the initial setup for a state, pass on specific parameter values to ALNSState located in alns_state.py
        HardcodedALNSStateGenerator --> VRPInstanceInfo --> ALNSState
    '''

    default_customer_number = 30
    instance_name = "O101200"    

    def generate_alns_state(self, problem_variant, random_seed, instance_name=instance_name, **kwargs):
        cust_number = kwargs['cust_number'] if 'cust_number' in kwargs else self.default_customer_number
        
        #----------------------------------------------------------------------------
        data_dir = Path(__file__).parents[1] / "data" / "solomon"
        inst_filename = f"{instance_name[0:4]}-{instance_name[4:]}"
        file_name = str(data_dir / f"{inst_filename}.txt")

        data = pd.read_csv(file_name, skiprows=8, header = None, sep='\s{2,}', engine='python')
        data.columns = ['cust_index', 'loc_x', 'loc_y', 'demand', 'ready_time', 'due_date', 'service_time']
        data = data.head(cust_number + 1)
        #----------------------------------------------------------------------------
        
        # cannot use maxabs scale independently for time windows and service time...
        orig_tw_start = data.ready_time.tolist()[:cust_number + 1]
        orig_tw_end = data.due_date.tolist()[:cust_number + 1]
        orig_service_time = data.service_time.tolist()[:cust_number + 1]


        loc_x = data.loc_x.tolist()[:cust_number + 1]
        loc_y = data.loc_y.tolist()[:cust_number + 1]
        demand = data.demand.tolist()[:cust_number + 1]
        tw_start = orig_tw_start
        tw_end = orig_tw_end
        service_time = orig_service_time


        if problem_variant == "TSP":
            depot_number = 1
            facility_opening_cost = None
            facility_capacity = None
            vehicle_type_number = 1
            vehicle_hiring_cost = base_vehicle_hiring_cost
            vehicle_fleetsize = None
            vehicle_capacity = sum(demand) + 100
            loc_x = loc_x
            loc_y = loc_y
            demand = demand

        elif problem_variant == "CVRP":
            depot_number = 1
            facility_opening_cost = None
            facility_capacity = None
            vehicle_type_number = 1
            # OLA results: vehicle_hiring_cost = 100
            vehicle_hiring_cost = base_vehicle_hiring_cost
            vehicle_fleetsize = None
            vehicle_capacity = cust_number * solomon_capacity_per_node
            loc_x = loc_x
            loc_y = loc_y
            demand = demand

        elif problem_variant == "VRPTW":
            depot_number = 1
            facility_opening_cost = None
            vehicle_type_number = 1
            vehicle_hiring_cost = base_vehicle_hiring_cost
            vehicle_fleetsize = None
            facility_capacity = None
            # vehicle_capacity = sum(demand) + 1
            vehicle_capacity = cust_number * solomon_capacity_per_node
            loc_x = loc_x
            loc_y = loc_y
            demand = demand

            ## heterogenous-vrp requirements:
        # 1) initial tours must be all served (smallest fleetsize sufficiently enough)
        # 2) due to economic-of-scale, larger vehicles have cheaper unit hiring cost, reflected in the initial parameter definition.
        elif problem_variant == "HVRP":
            depot_number = 1
            facility_opening_cost = None
            facility_capacity = None

            ## sample 1 works!
            vehicle_type_number = 2
            vehicle_hiring_cost = [base_vehicle_hiring_cost, base_vehicle_hiring_cost + (base_vehicle_hiring_cost / 5)]
            vehicle_fleetsize = [int(np.ceil(cust_number / 4))+5, 2]  #[13, 2]
            vehicle_capacity = [cust_number * solomon_capacity_per_node, (cust_number + 15) * solomon_capacity_per_node] #[150, 225]

            ## sample 2 works! 
            # vehicle_hiring_cost = [100, 120]
            # vehicle_fleetsize = [1, int(np.ceil(cust_number / 4))+5]  # [1, 13]
            # vehicle_capacity = [cust_number * solomon_capacity_per_node, (cust_number + 15) * solomon_capacity_per_node] #[150, 225]

            ## sample 3 works
            # vehicle_hiring_cost = [80, 100, 120]
            # vehicle_fleetsize = [3, 1, int(np.ceil(cust_number / 4))+5] 
            # vehicle_capacity = [(cust_number - 15) * solomon_capacity_per_node, cust_number * solomon_capacity_per_node, (cust_number + 15) * solomon_capacity_per_node]

            loc_x = loc_x
            loc_y = loc_y
            demand = demand 

        elif problem_variant == "LRP":
            depot_number = 3  #int(input("input number of depots = 1 if not LRP, else put an integer >= 2"))
            additional_depot_loc = [(25.0,25.0),(25.0,75.0),(75.0,25.0),(75.0,75.0)]  #manually define 4 facilities on plot with Solomon instance
            facility_opening_cost = [600, 1000, 100, 1000]
            facility_capacity = [1300, 1600, 1300, 1600]

            vehicle_type_number = 1
            vehicle_hiring_cost = base_vehicle_hiring_cost
            vehicle_fleetsize = None
            vehicle_capacity = cust_number * solomon_capacity_per_node
            facility_loc_x, facility_loc_y = map(list, zip(*additional_depot_loc[:depot_number-1]))
            loc_x = loc_x[:1] + facility_loc_x + loc_x[1:]
            loc_y = loc_y[:1] + facility_loc_y + loc_y[1:]
            demand = demand[:1] + [0]*(depot_number-1) + demand[1:]


        # random_proportion = kwargs['random_proportion'] if 'random_proportion' in kwargs else None

        #----------------------------------------------------------------------------
        ## Pass specific values to RoutingInstanceInfo object:
        instance_info = RoutingInstanceInfo(
            problem_variant,
            data_file_name=self.name,
            instance_name=instance_name,
            depot_number=depot_number,
            cust_number=cust_number,
            facility_opening_cost=facility_opening_cost,     # LRP only
            facility_capacity=facility_capacity,             # LRP only
            vehicle_hiring_cost=vehicle_hiring_cost,
            vehicle_capacity=vehicle_capacity,
            vehicle_type_number=vehicle_type_number,         # HVRP only
            vehicle_fleetsize=vehicle_fleetsize,             # HVRP only
            nonserve_penalty=nonserve_penalty,
            loc_x=loc_x,
            loc_y=loc_y,
            latitude = None,  
            longitude = None,
            edge_weight_matrix = None,
            demand=demand,
            tw_start=tw_start,                               # VRPTW only
            tw_end=tw_end,                                   # VRPTW only
            service_time=service_time,                       # VRPTW only
            exceed_tw_penalty=exceed_tw_penalty,             # VRPTW only
            forbidden_random_num_of_iterations=int(cust_number * forbidden_random_num_of_iterations_nodes_fraction)
        )
        return ALNSState(instance_info, random_seed, **kwargs) #, random_proportion=random_proportion



class GehringHombergerStateGenerator(StateGenerator):
    name = "gehring_homberger"

    ''' 
    C1_02_01.xml = Clustered, 200 nodes, not yet know what the last digit stands for
    R1_06_01.xml = Random, 600 nodes
    O1_08_01.xml = Mixed clustered-random, 800 nodes

    data download from: http://www.vrp-rep.org/datasets/item/2014-0018.html
    '''

    default_customer_number = 200
    instance_name = "O1_02_01"    

    def generate_alns_state(self, problem_variant, random_seed, instance_name=instance_name, **kwargs):
        cust_number = kwargs['cust_number'] if 'cust_number' in kwargs else self.default_customer_number

        # Load the XML data from the file
        file_path =  Path(__file__).parents[1] / "data" / "gehring" / f"{instance_name}.xml"
        with open(file_path, 'r') as file:
            xml_data = file.read()

        # Parse the XML data
        root = ET.fromstring(xml_data)

        # Extract the desired information
        nodess = root.findall('.//node')
        node_ids = [int(node.get('id')) for node in nodess][:cust_number + 1]
        loc_x = [float(node.find('cx').text) for node in nodess][:cust_number + 1]
        loc_y = [float(node.find('cy').text) for node in nodess][:cust_number + 1]

        vehicle_profile = root.find('.//vehicle_profile')
        vehicle_type = vehicle_profile.get('type')
        original_vehicle_capacity = float(root.find('.//vehicle_profile/capacity').text)
        vehicle_fleetsize = vehicle_profile.get('number')
        max_travel_time = float(root.find('.//vehicle_profile/max_travel_time').text)

        requests = root.findall('.//request')
        tw_start = [0] + [float(request.find('tw/start').text) for request in requests][:cust_number]
        tw_end   = [max_travel_time] + [float(request.find('tw/end').text) for request in requests][:cust_number]
        demand   = [0] + [float(request.find('quantity').text) for request in requests][:cust_number]
        service_time = [0] + [float(request.find('service_time').text) for request in requests][:cust_number]

        ## default parameter values unless updated by specific vrp variant: 
        depot_number = 1
        facility_opening_cost = None
        facility_capacity = None

        vehicle_type_number = 1
        vehicle_hiring_cost = base_vehicle_hiring_cost
        vehicle_capacity = original_vehicle_capacity
        # vehicle_capacity = cust_number * gehring_capacity_per_node


        ## parameters to be updated in specific variant: 
        if problem_variant == "TSP":
            vehicle_capacity = sum(demand) + 100

        elif problem_variant == "HVRP":
            vehicle_type_number = 2
            vehicle_hiring_cost = [base_vehicle_hiring_cost * 0.5, base_vehicle_hiring_cost * 1.2]
            
            # define capacity so that approx. 90% of tours would use larger vehicles, but smaller vehicle is much cheaper, meaning the algorithm needs to make improvement
            average_tour_demand = sum(demand) * INITIAL_TOUR_FRACTION
            lb_capacity = round_to_nearest_multiple_of_10(average_tour_demand * 0.9 ) 
            ub_capacity = round_to_nearest_multiple_of_10(average_tour_demand * 1.8 )
            vehicle_capacity = [lb_capacity, ub_capacity]

            # define fleetsize so that smallest vehicles cannot cover all demands, and larger vehicles can cover all demand but at a more expensive hiring cost.
            lb_fleetsize = max(int((sum(demand)/lb_capacity) * 1.1), 1/INITIAL_TOUR_FRACTION)
            ub_fleetsize = max(int((sum(demand)/ub_capacity) * 2), 1/INITIAL_TOUR_FRACTION+1)
            vehicle_fleetsize = [lb_fleetsize, ub_fleetsize]
            
            # print()
            # print('-----------------')
            # print(f"vehicle_capacity = {lb_capacity, ub_capacity}, total demand={sum(demand)}, average_tour_demand={average_tour_demand}")
            # print(f"fleetsize = {lb_fleetsize, ub_fleetsize}, covered={lb_fleetsize*lb_capacity}, {ub_capacity*ub_fleetsize}")
            # print('-----------------')
            # print()

            if lb_fleetsize*lb_capacity + ub_capacity*ub_fleetsize < sum(demand):
                raise ValueError(f"total fleet of vehicles cannot cover the complete customer demands {sum(demand)}")
        
        elif problem_variant == "LRP":
            depot_number = 3 
            additional_depot_loc = [(max(loc_x)/2, min(loc_y)),
                                    (max(loc_x)/2, max(loc_y)),
                                    (min(loc_x), max(loc_y)/2),
                                    (max(loc_x), max(loc_y)/2)]
            
            ## capacity ratio = 1:2:3:4 for specific number of depots
            facility_capacity = [sum(demand)*1.5 * i / sum(range(1, depot_number + 1)) for i in range(1, depot_number + 1)]    
            
            ## the i-th facility open cost has a discount of (1 - (i - 1) * 0.05) due to economic of scale: 
            ## Shunee's assumption: open_cost_per_ratio = approx 1000 per facility for 100 customers instance size
            open_cost_per_ratio = (sum(demand) * base_vehicle_hiring_cost) / (vehicle_capacity * 2)
            discount = 0.05
            facility_opening_cost = [open_cost_per_ratio * i * (1 - (i - 1) * discount) for i in range(1, depot_number + 1)]

            facility_loc_x, facility_loc_y = map(list, zip(*additional_depot_loc[:depot_number-1]))
            loc_x = loc_x[:1] + facility_loc_x + loc_x[1:]
            loc_y = loc_y[:1] + facility_loc_y + loc_y[1:]
            demand = demand[:1] + [0]*(depot_number-1) + demand[1:]


        #----------------------------------------------------------------------------
        ## Pass specific values to RoutingInstanceInfo object:
        instance_info = RoutingInstanceInfo(
            problem_variant,
            data_file_name=self.name,
            instance_name=instance_name,
            depot_number=depot_number,
            cust_number=cust_number,
            facility_opening_cost=facility_opening_cost,     # LRP only
            facility_capacity=facility_capacity,             # LRP only
            vehicle_hiring_cost=vehicle_hiring_cost,
            vehicle_capacity=vehicle_capacity,
            vehicle_type_number=vehicle_type_number,         # HVRP only
            vehicle_fleetsize=vehicle_fleetsize,             # HVRP only
            nonserve_penalty=nonserve_penalty,
            loc_x=loc_x,
            loc_y=loc_y,
            latitude = None,  
            longitude = None,
            edge_weight_matrix = None,
            demand=demand,
            tw_start=tw_start,                               # VRPTW only
            tw_end=tw_end,                                   # VRPTW only
            service_time=service_time,                       # VRPTW only
            exceed_tw_penalty=exceed_tw_penalty,             # VRPTW only
            forbidden_random_num_of_iterations=int(cust_number * forbidden_random_num_of_iterations_nodes_fraction)
        )
        return ALNSState(instance_info, random_seed, **kwargs)


class DeSmetStateGenerator(StateGenerator):
    name = "de_smet"

    ''' 
  * DATA FORMAT: 
    (node index, coordination in lat and long, city district)

  * SHUNEE'S VRP VARIANT SELECTION: (just record the type, can go for larger size)
    belgium-n50-k10    --> TSP, CVRP, HVRP  - (note: we should compute arc length using lat long instead of euclidean distance using x y!)
    belgium-d2-n50-k10 --> LRP
  X belgium-tw-n50-k10 --> VRPTW  - (note: we need to remove 100 from all the tw data!)

  * COMPLETE FILE NAME DESCRIPTION: 
    belgium-n50-k10 : 1 default depots, 50 nodes
    belgium-d2-n50-k10 : 2 depots, 50 nodes
    belgium-road-km-d2-n50-k10 : 2 depots, 50 nodes, edge wegiht equal distance given in matrix 
    belgium-road-time-d2-n50-k10 : 2 depots, 50 nodes, edge wegiht equals time given in matrix 
    belgium-road-time-tw-d2-n50-k10 : 2 depots, 50 nodes, edge wegiht equals time given in matrix, customer time window given
    belgium-tw-n50-k10 : 1 default depots, 50 nodes, customer time windows given
    belgium-tw-d2-n50-k10 : 2 depots, 50 nodes, customer time windows given

    data download from: http://www.vrp-rep.org/datasets/item/2017-0001.html
    '''

    def generate_alns_state(self, problem_variant, random_seed, **kwargs):
        
        ## dataset for different variants: 
        if problem_variant in {"TSP", "CVRP", "HVRP"}:
            instance_name = "belgium-n50-k10" 
        
        elif problem_variant == "VRPTW":
            raise ValueError("Shunee's note: de Smet dataset provided time window dont seem to work with its edge weight matrix! ")
            instance_name = "belgium-tw-n50-k10"   # time window doesnt fit with computed customer arrival time, abandon this instance type, use below! 
            # instance_name = "belgium-road-time-tw-d2-n50-k10"  # strange time window, not fitting well 
        
        elif problem_variant == "LRP":
            instance_name = "belgium-d2-n50-k10"
        

        file_path =  Path(__file__).parents[1] / "data" / "desmet" / f"{instance_name}.vrp"

        with open(file_path, 'r') as file:
            
            node_id, latitude, longitude, demand = [], [], [], []
            tw_start, tw_end, service_time = [], [], []
            depot_number = 0
            edge_weight_matrix = []

            for line in file:
                if line.startswith('DIMENSION:'):
                    depot_and_customer_number = int(line.split(':')[1].strip())
                
                elif line.startswith('CAPACITY:'):
                    original_vehicle_capacity = int(line.split(':')[1].strip())
                
                elif line.startswith("NODE_COORD_SECTION"):
                    # Start reading node coordinates
                    for _ in range(depot_and_customer_number):
                        node_data = file.readline().strip().split()
                        node_id.append(int(node_data[0]))
                        latitude.append(float(node_data[1]))
                        longitude.append(float(node_data[2]))

                elif line.startswith('EDGE_WEIGHT_SECTION'):
                    for _ in range(depot_and_customer_number):
                        edge_weights_line = next(file)
                        edge_weight_matrix.append(list(map(float, edge_weights_line.strip().split())))
 
                elif line.startswith("DEMAND_SECTION"):
                    # Start reading demand section
                    for _ in range(depot_and_customer_number):
                        demand_data = file.readline().strip().split()
                        if problem_variant != "VRPTW":
                            demand.append(int(demand_data[1]))
                        else:
                            # demand.append(int(demand_data[1]))
                            # tw_start.append(int(demand_data[2]))
                            # tw_end.append(int(demand_data[3]))
                            # service_time.append(int(demand_data[4]))
                            tw_start.append(int(demand_data[2])/100 - 200)  # self-tuned to fit the actual tw computation range
                            tw_end.append(int(demand_data[3])/100 - 200)
                            service_time.append(int(demand_data[4])/100)

                elif line.startswith("DEPOT_SECTION"):
                    # Count the number of depots until -1 is encountered
                    for line in file:
                        line = line.strip()
                        if line == "-1":
                            break
                        depot_number += 1

        loc_x, loc_y = self.convert_lat_long_to_xy(latitude, longitude)
        customer_number = depot_and_customer_number - depot_number


        ## default parameter values unless updated by specific vrp variant: 
        facility_opening_cost = None
        facility_capacity = None

        vehicle_type_number = 1
        vehicle_hiring_cost = base_vehicle_hiring_cost
        vehicle_capacity = original_vehicle_capacity
        # vehicle_capacity = cust_number * desmet_capacity_per_node 
        vehicle_fleetsize = math.ceil(sum(demand)/original_vehicle_capacity) + 2


        ## parameters to be updated in specific variant: 
        if problem_variant == "TSP":
            vehicle_capacity = sum(demand) + 10

        elif problem_variant == "HVRP":
            vehicle_type_number = 2
            vehicle_hiring_cost = [base_vehicle_hiring_cost, base_vehicle_hiring_cost + (base_vehicle_hiring_cost / 5)]
            vehicle_fleetsize = [int(np.ceil(customer_number / 4))+5, 2]  #[13, 2]
            vehicle_capacity = [original_vehicle_capacity, int(original_vehicle_capacity * 1.5)]
            # vehicle_capacity = [cust_number * desmet_capacity_per_node, (cust_number + 15) * desmet_capacity_per_node] #[150, 225]

        elif problem_variant == "LRP":
            ## capacity ratio = 1:2:3:4 for specific number of depots
            facility_capacity = [sum(demand)*1.5 * i / sum(range(1, depot_number + 1)) for i in range(1, depot_number + 1)]    
            
            ## the i-th facility open cost has a discount of (1 - (i - 1) * 0.05) due to economic of scale: 
            ## Shunee's assumption: open_cost_per_ratio = approx 1000 per facility for 100 customers instance size
            open_cost_per_ratio = (sum(demand) * base_vehicle_hiring_cost) / (vehicle_capacity * 2)
            discount = 0.05
            facility_opening_cost = [open_cost_per_ratio * i * (1 - (i - 1) * discount) for i in range(1, depot_number + 1)]

            # facility_opening_cost = [800, 1000, 800, 1000, 800, 1000, 800, 1000, 800, 1000][:depot_number]
            # facility_capacity = [1300, 1600, 1300, 1600, 1300, 1600, 1300, 1600, 1300, 1600][:depot_number]


        #----------------------------------------------------------------------------
        ## Pass specific values to RoutingInstanceInfo object:
        instance_info = RoutingInstanceInfo(
            problem_variant,
            data_file_name=self.name,
            instance_name=instance_name,
            depot_number=depot_number,
            cust_number=customer_number,
            facility_opening_cost=facility_opening_cost,     # LRP only
            facility_capacity=facility_capacity,             # LRP only
            vehicle_hiring_cost=vehicle_hiring_cost,
            vehicle_capacity=vehicle_capacity,
            vehicle_type_number=vehicle_type_number,         # HVRP only
            vehicle_fleetsize=vehicle_fleetsize,             # HVRP only
            nonserve_penalty=nonserve_penalty,
            loc_x=loc_x, 
            loc_y=loc_y,
            latitude = latitude,  
            longitude = longitude,
            edge_weight_matrix = edge_weight_matrix,
            demand=demand,
            tw_start=tw_start,                               # VRPTW only
            tw_end=tw_end,                                   # VRPTW only
            service_time=service_time,                       # VRPTW only
            exceed_tw_penalty=exceed_tw_penalty,             # VRPTW only
            forbidden_random_num_of_iterations=int(customer_number * forbidden_random_num_of_iterations_nodes_fraction)
        )
        return ALNSState(instance_info, random_seed, **kwargs)  # , random_proportion=random_proportion

    def convert_lat_long_to_xy(self, latitudes, longitudes):
        x_coordinates = []
        y_coordinates = []

        # Earth radius in kilometers
        earth_radius = 6371

        # BRUSSEL - Reference latitude and longitude (in degrees)
        ref_latitude = 50.8427501 
        ref_longitude = 4.3515499
        ref_latitude_rad = math.radians(ref_latitude)
        ref_longitude_rad = math.radians(ref_longitude)

        # Convert latitude and longitude to radians
        for latitude, longitude in zip(latitudes, longitudes):
            latitude_rad = math.radians(latitude)
            longitude_rad = math.radians(longitude)
            
            # Calculate x and y coordinates
            x = earth_radius * (longitude_rad - ref_longitude_rad) * math.cos(ref_latitude_rad)
            y = earth_radius * (latitude_rad - ref_latitude_rad)
            x_coordinates.append(x)
            y_coordinates.append(y)
        
        return x_coordinates, y_coordinates

