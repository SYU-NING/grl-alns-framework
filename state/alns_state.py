import random
from copy import deepcopy

import xxhash

from utils.general_utils import *


## If CHOSEN_METHOD = "hybrid", define proportion of randomly-generated nodes:
TSP_RANDOM_PROP = 0.2
CVRP_RANDOM_PROP = 0.5
HVRP_RANDOM_PROP = 0.25
VRPTW_RANDOM_PROP = 0.25
LRP_RANDOM_PROP = 0.25

## Fraction of customers in the initial random tours (yields tours with size 4 for 20 nodes, etc.)
INITIAL_TOUR_FRACTION = 0.2
#===============================


class RoutingInstanceInfo:
    ''' 
    Store instance-based information defined prior to the learning process and will not be modified at any state 
    '''
    def __init__(self, problem_variant, data_file_name, instance_name, depot_number, cust_number, 
                 vehicle_type_number, vehicle_capacity, nonserve_penalty,
                 loc_x, loc_y, latitude, longitude, edge_weight_matrix, demand, service_time, 
                 forbidden_random_num_of_iterations, **kwargs):

        ## define which problem we are dealing with:
        self.problem_variant = problem_variant
        self.data_file_name = data_file_name

        self.instance_name = instance_name
        self.depot_number = depot_number
        self.cust_number = cust_number
        self.nonserve_penalty = nonserve_penalty


        self.facility_opening_cost = kwargs['facility_opening_cost'] if 'facility_opening_cost' in kwargs else None
        self.facility_capacity = kwargs['facility_capacity'] if 'facility_capacity' in kwargs else None

        self.vehicle_type_number = vehicle_type_number
        self.vehicle_hiring_cost = kwargs['vehicle_hiring_cost'] if 'vehicle_hiring_cost' in kwargs else None
        self.vehicle_fleetsize = kwargs['vehicle_fleetsize'] if 'vehicle_fleetsize' in kwargs else None
        self.vehicle_capacity = vehicle_capacity
        
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.demand = demand 
        self.normalized_demand = normalize_list(demand)
        self.service_time = service_time

        self.latitude = latitude
        self.longitude = longitude

        self.edge_weight_matrix = edge_weight_matrix

        self.tw_start = kwargs['tw_start'] if 'tw_start' in kwargs else None
        self.tw_end = kwargs['tw_end'] if 'tw_end' in kwargs else None
        self.exceed_tw_penalty = kwargs['exceed_tw_penalty'] if 'exceed_tw_penalty' in kwargs else None

        self.forbidden_random_num_of_iterations = forbidden_random_num_of_iterations




class ALNSState(object):
    ''' 
    Receives a list of parameter values from and store inside the defined state object
    '''
    def __init__(self, instance_info, random_seed, **kwargs):
        
        # fetch defined parameter values from state_generator.py
        self.inst = instance_info   

        # define random generator with seed
        self.random_seed = random_seed
        self.local_random = random.Random()
        self.local_random.seed(self.random_seed)

        # define sets using values defined in state_generator.py 
        inst = self.inst
        self.depot = [i for i in range(inst.depot_number)]
        self.customers = [i + inst.depot_number for i in range(inst.cust_number)]
        self.nodes = self.depot + self.customers
        self.num_nodes = len(self.nodes)
        self.service_time = inst.service_time

        if self.inst.data_file_name == "de_smet":
            if self.inst.problem_variant != "VRPTW":  # use road time distance matrix given in dataset 
                self.distance_matrix = self.inst.edge_weight_matrix
            else: # compute using lat and long
                self.distance_matrix = compute_arc_length_matrix(inst.latitude, inst.longitude, self.nodes)
        else: 
            # compute arc distance matrix using x y coordinates (with noise for medium-quality tours):
            self.initial_distance_matrix = compute_distance_matrix(inst.loc_x, inst.loc_y, self.nodes, local_random=self.local_random) 
            
            # compute arc distance matrix using x y coordinates (without noise):
            self.distance_matrix = compute_distance_matrix(inst.loc_x, inst.loc_y, self.nodes) 

        self.dist_min, self.dist_max = min(self.distance_matrix.values()), max(self.initial_distance_matrix.values())

        # generate initial solution to kick start improvement process
        in_tour = [True] * (inst.depot_number + inst.cust_number)
        self.node_status = in_tour
        self.customer_pool = []


        self.vrw_experiment_setting = 'o'


        ## define a list of dictionaries for historical-based destroy operators classes:
        self.same_route_frequency = same_tour_frequency(self.customers)
        self.smallest_removal_gain = smallest_removal_gain(self.customers)
        self.first_time_employ = True
        self.non_removal_count = non_removal_count(self.customers)
        self.smallest_travel_distance = smallest_travel_distance(self.customers)


        # self.random_proportion = kwargs['random_proportion'] if 'random_proportion' in kwargs else None


        #----------------------------------------------------
        ##Customer visiting sequence: (1) random (2) greedy clarks-wright (3) a mixed random-greedy approach

        initial_tour = kwargs.pop("initial_tour_method")
        proportion = self.get_initialization_proportion(self.inst.problem_variant, initial_tour)
        
        ## (1) generate bad quality tours (random):
        if initial_tour == "random":
            if self.inst.problem_variant == "TSP":
                self.alns_list = self.generate_random_tsp_grand_tour(self.customers)
            elif self.inst.problem_variant == "LRP":
                self.alns_list = self.generate_initial_tour_random(self.customers)
                self.customer_depot_allocation = self.generate_depot_allocation(self.alns_list) 
            else:
                self.alns_list = self.generate_initial_tour_random(self.customers)


        ## (2) generate good quality tours (clarks & wright):
        elif initial_tour == "clark_wright":
            if self.inst.problem_variant == "LRP":
                self.alns_list, self.customer_depot_allocation = self.generate_initial_tour_clark_wright(self.customers)
            else:
                # print("-------------------")
                # print(f"random_seed={self.random_seed}")
                # for item, value in vars(self.inst).items():
                #     print(f"{item}: {value}")
                # print(f"initial_distance_matrix={self.initial_distance_matrix}")
                
                self.alns_list = self.generate_initial_tour_clark_wright(self.customers)
                
                # print(f'self.alns_list={self.alns_list}')
                # print("-------------------")


        ## (3) generate average quality tours (mixed random + clarks):
        elif initial_tour == "hybrid":
            ## divide into (x)% random + (1-x)% clark wright: 
            customer_set_random = self.local_random.sample(self.customers, int(proportion * len(self.customers))) 
            customer_set_cw = [x for x in self.customers if x not in customer_set_random]
            
            # print(f"random len={len(customer_set_random)}, {customer_set_random}")
            # print(f"clarks len={len(customer_set_cw)}, {customer_set_cw}")
            
            if self.inst.problem_variant == "TSP":
                alns_list_random = self.generate_random_tsp_grand_tour(customer_set_random)
                alns_list_cw = self.generate_initial_tour_clark_wright(customer_set_cw)
                self.alns_list = [alns_list_random[0] + alns_list_cw[0]] if alns_list_cw != [] else alns_list_random
            
            elif self.inst.problem_variant == "LRP":
                alns_list_random = self.generate_initial_tour_random(customer_set_random)
                allocation_random = self.generate_depot_allocation(alns_list_random)
                alns_list_cw, allocation_cw = self.generate_initial_tour_clark_wright(customer_set_cw)
                self.alns_list = alns_list_random + alns_list_cw
                self.customer_depot_allocation = {**allocation_random, **allocation_cw}
            else:
                alns_list_random = self.generate_initial_tour_random(customer_set_random)
                alns_list_cw = self.generate_initial_tour_clark_wright(customer_set_cw)
                self.alns_list = alns_list_random + alns_list_cw
        else:
            raise ValueError("tour initialization must be random, clark_wright or hybrid")
        
        #----------------------------------------------------

        ## for HVRP, track individual tour's service vehicle type: 
        if self.inst.problem_variant == "HVRP":
            self.cust_vehicle_type = self.generate_smallest_vehicle_capcity()
    
        ## for VRPTW, track individual node's time window violation using the arrival time: 
        if self.inst.problem_variant == "VRPTW":
            self.time_window_violation = self.generate_tw_violation(self.alns_list)

    def get_normalized_distance(self, i, j):
        val = self.distance_matrix[(i, j)]
        normalized = (val - self.dist_min) / (self.dist_max - self.dist_min)
        return normalized

    #----------------------------------------------------
    #----------------------------------------------------

    @staticmethod
    def get_initialization_proportion(variant, initial_tour_method): #random_proportion
        ''' 
        "initial_tour_method" CAN CHOOSE FROM = "random", "clark_wright", "hybrid"
            random = all nodes are randomly grouped into chunks to construct bad quality initial tours 
            clark_wright = all nodes are grouped greedily to construct good quality initial tours 
            hybrid = select a proportion of nodes to follow random, with the remaining using Clarks-Wright 
        '''

        if initial_tour_method == 'hybrid':
            if variant == "TSP": # not too far away from optimality but still allow a long enough process of operator selections to train the model.
                proportion = TSP_RANDOM_PROP
            if variant == "CVRP":
                proportion = CVRP_RANDOM_PROP
            if variant == "HVRP":
                proportion = HVRP_RANDOM_PROP
            if variant == "LRP":
                proportion = LRP_RANDOM_PROP
            if variant == "VRPTW": # adapt CW solution without introducing any extra randomness
                proportion = VRPTW_RANDOM_PROP
        else:
            proportion = None

        return proportion


    def generate_initial_tour_random(self, customer_set, **kwargs):
        ''' method 1. randomly generate initial solution (state) as the beginning of improvement process.
        '''
        # initial_tour_size = int(len(customer_set) * INITIAL_TOUR_FRACTION)
        average_tour_length = int(len(self.customers) * INITIAL_TOUR_FRACTION)
        # print(f'initial_tour_size={initial_tour_size}, len(customer_set)={len(customer_set)}, average_tour_length={average_tour_length}')
        
        custs_cp = deepcopy(customer_set)
        self.local_random.shuffle(custs_cp)
        return list(chunks(custs_cp, average_tour_length))


    def generate_random_tsp_grand_tour(self, customer_set): 
        ''' For TSP environment: randomly come up with a visiting sequence 
            to include all nodes in a single grand tour start and end at the depot.
        '''
        custs = deepcopy(customer_set)
        self.local_random.shuffle(custs)
        return [custs]


    def generate_initial_tour_clark_wright(self, customer_set, **kwargs):
        ''' method 2. Clarks & Wright construction algorithm to form initial solution 
        Procedure: 
            i)   to start with, we assume pendulum tours for every customer nodes to the depot;
            ii)  for each pair of edge nodes (i,j), which is the starting and ending node pair from a tour,
                 let saving = d_{0,i} + d_{0,j} - d_{i,j}, which is the reduced travel distance if replace
                 two individual tours by a larger round tour;
            iii) for loop: each iteration select the largest saving from the list, check feasibility (vehicle capacity),
                 if feasible then merge. Continue until no positive saving exists/no edge node exists/all merges become infeasible
        Note:
            - for LRP, we first apply a bin-packing algorithm of all customers to the nearest feasible facility, then perform separate Clarks-wright within each facility
            - for HVRP, we assume all vehicles taking the smallest capacity, start from this initial solution 
            - for VRPTW, we first rank all edges according to their distance-based savings, afterwards use time window penalty to determine whether merge the edge or not
        '''

        if self.inst.problem_variant == "LRP":
            ## bin-packing each customers into the nearest facility with remaining capacity: 
            facility_allocation = {depot: [] for depot in self.depot}
            customer_depot_allocation = {}
            depot_remaining_capacity = {depot: self.inst.facility_capacity[depot] for depot in self.depot}
            
            # allocate customer to depot before forming depot-based tours:
            for cust in customer_set: 
                nearest_depot = sorted(self.depot, key=lambda x: self.initial_distance_matrix[(cust, x)])
                depot = next((depot for depot in nearest_depot if self.inst.demand[cust] <= depot_remaining_capacity[depot]), None)
                
                # every customer can be assigned to at least one depot:
                if depot is not None:
                    facility_allocation[depot].append(cust)
                    depot_remaining_capacity[depot] -= self.inst.demand[cust]
                    
                    # creat customer-depot allocation plan, no need for "generate_depot_allocation" function! 
                    customer_depot_allocation[cust] = depot
                else:
                    raise ValueError(f"Facility capacity is defined too small, customer = {cust} cannot fit into anywhere.")

            ## Create tours with the subset of customers allocated to each facility: 
            alns_list, closed_facilities = [], []
            for facility, allocated_custs in facility_allocation.items():
                if allocated_custs == []:
                    closed_facilities.append(facility)
                tours = clark_wright_algorithm(allocated_custs, facility, self.initial_distance_matrix, self.inst)
                alns_list += tours
            
            ## assign node status for depot:
            opened_facilities = list(set(self.depot) - set(closed_facilities))
            for i in closed_facilities:
                self.node_status[i] = False 
            
            # print(f"facility_allocation={facility_allocation}")
            # print(f"opened_facilities={opened_facilities}")
            # print(f"closed_facilities={closed_facilities}")

            return alns_list, customer_depot_allocation

        elif self.inst.problem_variant == "HVRP":
            # if all vehicles (have the smallest capacity) cannot serve all customers, create error: 
            if sum(self.inst.demand) > sum([self.inst.vehicle_capacity[i] * self.inst.vehicle_fleetsize[i] for i in range(self.inst.vehicle_type_number)]):
                raise ValueError("The total fleet of vehicles cannot serve all customers! ")
            
            elif sum(self.inst.demand) > min(self.inst.vehicle_capacity) * sum(self.inst.vehicle_fleetsize):
                raise ValueError("The total fleet of vehicles, if assume the smallest capacity, cannot serve all customers! ")

            else: 
                # form alsn tours under homogenous vehicle assumption:
                return clark_wright_algorithm(customer_set, self.depot[0], self.initial_distance_matrix, self.inst)
            
        else:
            return clark_wright_algorithm(customer_set, self.depot[0], self.initial_distance_matrix, self.inst)


    def generate_smallest_vehicle_capcity(self):
        ''' Find the smallest vehicle capacity available to cover the given tour's accumulated demand. 
            Keep a tracking dictionary of the vehicle type to serve each customer.
        '''
        vehicle_type = {}
        fleetsize = deepcopy(self.inst.vehicle_fleetsize)
        # print(f"demand={self.inst.demand}")
        # print(f"fleetsize={fleetsize}")

        for tour in self.alns_list:
            acc_demand = sum([self.inst.demand[c] for c in tour])

            # find the vehicle capacities that are still available: 
            available_capacity = [capacity for vehicle_index, capacity in enumerate(self.inst.vehicle_capacity) if fleetsize[vehicle_index] > 0]
            
            # amongst all available capacities, find the smallest feasible one to serve the tour: 
            smallest_feasible_capacity = min((capacity for _, capacity in enumerate(available_capacity) if capacity >= acc_demand), default=None) 
            
            if max(available_capacity) < acc_demand: 
                raise ValueError(f"the largest vehicle capacity [{max(available_capacity)}] still cannot cover this tour with demand [{acc_demand}]")
            if smallest_feasible_capacity == None: 
                raise ValueError(f"need to increase the fleetsize of large-scale vehicles. This tour needs a larger vehicle as it exceeds the capacity of any small-scale vehicle")
            if fleetsize[self.inst.vehicle_capacity.index(smallest_feasible_capacity)] == 0: 
                raise ValueError(f"the number of vehicle type [{self.inst.vehicle_capacity.index(smallest_feasible_capacity)}] with capacity [{smallest_feasible_capacity}] is defined too little! ")
            
            # update vehicle fleetsize for the newly employed type: 
            fleetsize[self.inst.vehicle_capacity.index(smallest_feasible_capacity)] -= 1
            
            # update assigned vehicle capacity info for all customers within this tour: 
            for cust in tour:
                vehicle_type[cust] = smallest_feasible_capacity
        
            # print(f"tour={tour}, acc_demand={acc_demand}")
            # print(f"available_capacity={available_capacity}, smallest_feasible_capacity={smallest_feasible_capacity}")
            # print(f"fleetsize[{self.inst.vehicle_capacity.index(smallest_feasible_capacity)}]={fleetsize[self.inst.vehicle_capacity.index(smallest_feasible_capacity)]}")
            # print(f"vehicle_type={vehicle_type}")
            # print()
        return vehicle_type
    

    def bin_packing_facility_allocation(self):
        ''' (FUNCTION NOT USED!) 
            Assume uncapacitated facility, allocate all customer nodes to their nearest facility, then start clarks wright 
            to form tours for each facility and add all tours to the alns_list as the initial solution 
        '''
        nodes = np.array([[self.inst.loc_x[i], self.inst.loc_y[i]] for i in self.customers])
        centers = np.array([[self.inst.loc_x[i], self.inst.loc_y[i]] for i in self.depot])
        
        # Calculate Euclidean distance between nodes and centers
        distances = np.linalg.norm(nodes[:, np.newaxis] - centers, axis=-1)
        
        # Assign each node to the nearest center
        node_allocations = np.argmin(distances, axis=-1)

        # create a dictionary to start the facility-node allocation: 
        customer_depot_allocation = {i: [] for i in set(node_allocations)}
        for node, depot in enumerate(node_allocations):
            customer_depot_allocation[depot].append(node)
        
        return customer_depot_allocation
    

    def generate_depot_allocation(self, customer_list):
        ''' randomly allocate tours to any existing depot. The depots not being allocated are set to closed.
        '''
        ## (1) random allocation: might experience facility overcapacity! 
        # depot_allocation = [self.local_random.choice(self.depot) for _ in range(len(alns_list))]

        ## (2) bin-packing allocation: iteratively assign each tour to the smallest index facility with available capacity: 
        facility_capacity = self.inst.facility_capacity.copy()
        depot_allocation = []
        for tour in customer_list:
            tour_demand = sum([self.inst.demand[c] for c in tour])
            for depot in self.depot:
                if tour_demand <= facility_capacity[depot]:
                    allocated_facility = depot
                    break
                elif depot == self.depot[-1]:
                    raise ValueError(f"Facility capacity for depot = {depot} need to set larger! ")
            depot_allocation.append(allocated_facility)
            facility_capacity[allocated_facility] -= tour_demand

        ## assign customers from the tour to the depot: 
        customer_depot_allocation = {cust: depot_allocation[i] for i, tour in enumerate(customer_list) for cust in tour}
        
        ## assign node status for depot:
        closed_depot_list = list(set(self.depot) - set(depot_allocation))
        for i in closed_depot_list:
            self.node_status[i] = False 

        # print(f"alns = {alns_list}, depot_allocation = {depot_allocation}, self.node_status = {self.node_status}")
        # print(f"opened depots = {set(customer_depot_allocation.values())}, customer_depot_allocation = {customer_depot_allocation}")
        
        return customer_depot_allocation
    
    
    def generate_tw_violation(self, alns_list): 
        time_window_violation = {}

        for tour in alns_list:
            full_tour = self.depot + tour
            node2_arrival_time_forward  = [0] * len(full_tour)
            node2_arrival_time_backward = [0] * len(full_tour)
            forward_tw_violation  = [0] * len(full_tour)
            backward_tw_violation = [0] * len(full_tour)   

            for i, (node1, node2) in enumerate(zip(full_tour[:-1], full_tour[1:])):
                rev_node1 = full_tour[::-1][i-1]
                rev_node2 = full_tour[::-1][i]

                # track the arrival time at each end node, with both directions of the tour: 
                node2_arrival_time_forward[i + 1]  = node2_arrival_time_forward[i] + self.inst.service_time[node1] + self.initial_distance_matrix[(node1, node2)]
                node2_arrival_time_backward[i + 1] = node2_arrival_time_backward[i] + self.inst.service_time[rev_node1] + self.initial_distance_matrix[(rev_node1, rev_node2)]

                forward_tw_violation[i + 1]  = compute_violated_time(node2, node2_arrival_time_forward[i + 1], self.inst)
                backward_tw_violation[i + 1] = compute_violated_time(rev_node2, node2_arrival_time_backward[i + 1], self.inst)

            # compute the aggregated tw violation, determine if forward or backward: 
            tw_violations = backward_tw_violation[::-1] if sum(backward_tw_violation) < sum(forward_tw_violation) else forward_tw_violation
            for index, cust in enumerate(tour):
                time_window_violation[cust] = tw_violations[index]
                
        return time_window_violation
    

#----------------------------------------------------
# non-class global methods:    
            
def clark_wright_algorithm(customer_set, depot, distance_matrix, inst):
    safety_count = 0
    tours = [[x] for x in customer_set]
    edge_nodes = customer_set  #no need for deepcopy, as only edge nodes can form pendulum tours that need to be checked!  

    # compute the saving for merging each pair of nodes, define edge node set 
    saving_record = clark_wright_compute_saving(edge_nodes, depot, distance_matrix)  
    no_more_feasible_merge = False

    # define vehicle capacity based on problem variant: 
    vehicle_capacity = min(inst.vehicle_capacity) if inst.problem_variant == "HVRP" else inst.vehicle_capacity


    while True and (safety_count <= 1000): 
        
        ## check for endless loop in case it exists: 
        if safety_count >= 1000:
            raise ValueError("endless loop occurs in clarks-wright algorithm") 
        else:
            safety_count += 1 


        ## check if no merging is feasible or all arcs have been merged: 
        if (saving_record == {}) or (no_more_feasible_merge == True): 
            
            ## check if all nodes are inside, if not, add pendulum tour 
            for cust in customer_set:
                if any(cust in tour for tour in tours)==False:
                    tours += [cust]
            
            ## if noise factor causes multiple tours in TSP, merge tours into one: 
            if inst.problem_variant == "TSP":
                tours = [[item for sublist in tours for item in sublist]]
            break  # break while-loop
        
        
        ## iterate from arcs with the max saving for each iteration: 
        else:
            for new_arc in saving_record.items():  
                customer1, customer2 = new_arc[0][0], new_arc[0][1]   # new_arc=((65, 71), 79.2) takes the first position to extract arc index
                old_tour1 = tours[[customer1 in tour for tour in tours].index(True)]
                old_tour2 = tours[[customer2 in tour for tour in tours].index(True)]

                # check if tour is feasible (not exceed capacity), we create a new tour by merging two old ones:
                if sum([inst.demand[c] for c in old_tour1 + old_tour2]) > vehicle_capacity:      
                    no_more_feasible_merge = True
            
                    # if this new_arc is the last item inside saving_record, break to avoid going through for-loop again due to while-loop: (added just in case, but line above can guarantee to break already!)
                    if new_arc == list(saving_record.items())[-1]:
                        break
               
                else:
                    no_more_feasible_merge = False

                    # attach the two tours in the correct way:
                    if customer1 == old_tour1[-1] and customer2 == old_tour2[0]:
                        new_tour = old_tour1 + old_tour2
                    elif customer1 == old_tour1[0] and customer2 == old_tour2[0]:
                        new_tour = old_tour1[::-1] + old_tour2
                    elif customer1 == old_tour1[-1] and customer2 == old_tour2[-1]:
                        new_tour = old_tour1 + old_tour2[::-1]
                    elif customer1 == old_tour1[0] and customer2 == old_tour2[-1]:
                        new_tour = old_tour1[::-1] + old_tour2[::-1]
                        
                    if inst.problem_variant == "VRPTW":
                        best_saving = 0
                        penalty_old_tour1 = min(total_time_window_penalty(distance_matrix, depot, old_tour1, inst), total_time_window_penalty(distance_matrix, depot, old_tour1[::-1], inst)) 
                        penalty_old_tour2 = min(total_time_window_penalty(distance_matrix, depot, old_tour2, inst), total_time_window_penalty(distance_matrix, depot, old_tour2[::-1], inst))
                        penalty_new_tour = min(total_time_window_penalty(distance_matrix, depot, new_tour, inst), total_time_window_penalty(distance_matrix, depot, new_tour[::-1], inst))

                        # compute the current saving from time window penalty and distance: 
                        current_saving = (penalty_old_tour1 + penalty_old_tour2 - penalty_new_tour) + saving_record[new_arc[0]]  # new_arc = ((65, 71), 79.29538482039659) 

                        # # check if the total time window violation is not increasing, or else this merge is not worthwhile: 
                        if current_saving > best_saving: 
                            best_saving = current_saving
                            best_tours = new_tour
                        
                        if best_saving == 0:
                            # avoid the endless loop created by "continue" that will go through the non-saving arc set over and over again! 
                            if new_arc == list(saving_record.items())[-1]:
                                no_more_feasible_merge == True
                            else:
                                continue  # go to the next pair of nodes to check saving, since this merge will worsen the time window! 

                    
                    # update 1: tour records - remove two old tours, add new merged tour: 
                    tours.remove(old_tour1)
                    tours.remove(old_tour2)
                    tours += [new_tour]

                    # update 2: find the edge nodes set, only those nodes are mergable with others:
                    edge_nodes = list(set([tour[n] for n in (0,-1) for tour in tours]))

                    # update 3: remove nonfeasible arcs from the saving dictionary that doesnt contain edge node:
                    remove_key = []
                    # remove all arcs with either end being a non-edge node:
                    for node_pair, _ in saving_record.items():  
                        if node_pair[0] not in edge_nodes or node_pair[1] not in edge_nodes: 
                            remove_key += [node_pair]
                    # remove all start-end node pairs belonging to the same tour. Link those nodes will form illegal subtours: 
                    for tour in tours:                  
                        if (tour[0], tour[-1]) in saving_record.keys():
                            remove_key += [(tour[0], tour[-1])]
                        if (tour[-1], tour[0]) in saving_record.keys():
                            remove_key += [(tour[-1], tour[0])]

                    for key in remove_key:
                        saving_record.pop(key)
                    
                    break  # break for-loop, start iterating over "saving" dictionary from the start
    
            ## terminate the while loop correctly and avoid the endless loop issue:
            if no_more_feasible_merge: 
                break
    
    return tours 


def clark_wright_compute_saving(edge_nodes, depot, distance_matrix):
    ''' compute the saving for all edge node pairs and return a dictionary in descending order with
        only positive savings included.
    '''
    # we can only merge the edge nodes to form larger tours:
    edge_node_pairs = [(c1,c2) for c1 in edge_nodes for c2 in edge_nodes if c1<c2] 

    # compute the saving associated with each potential merge: 
    matrix = {arc: distance_matrix[(depot, arc[0])] + distance_matrix[(depot, arc[1])] - distance_matrix[(arc[0], arc[1])] for arc in edge_node_pairs}
    
    # only interested in the positive savings: 
    matrix_positive = dict((key, value) for key, value in matrix.items() if value > 0)
    
    return dict(sorted(matrix_positive.items(), key=lambda item: item[1], reverse=True))


def total_time_window_penalty(distance_matrix, depot, tour, inst):
    total_time_window_penalty = 0
    full_tour = [depot] + tour
    for i, (node1, node2) in enumerate(zip(full_tour[:-1], full_tour[1:])):
        end_node_arrival_time = distance_matrix[(node1, node2)] + inst.service_time[node1]
        total_time_window_penalty += compute_violated_time(node2, end_node_arrival_time, inst)
    return total_time_window_penalty


def compute_violated_time(cust, arrival_time, inst):
    ''' wish the arrival time t lay within time window [a,b] for the Solomon instance
        a < t < b        -->  no violation, return (t - a)
        t < a or t > b   -->  violation = (a - t) * penalty
    '''
    time_window_start, time_window_end = inst.tw_start[cust], inst.tw_end[cust]

    if arrival_time < time_window_start:
        return time_window_start - arrival_time
    elif arrival_time > time_window_end:
        return arrival_time - time_window_end
    else:  
        return 0

def get_state_hash(state, size=64):
    if size == 32:
        hash_instance = xxhash.xxh32()
    elif size == 64:
        hash_instance = xxhash.xxh64()
    else:
        raise ValueError("only 32 or 64-bit hashes supported.")

    for tour in state.alns_list:
        hash_instance.update(np.array(tour))

    graph_hash = hash_instance.intdigest()
    return graph_hash

