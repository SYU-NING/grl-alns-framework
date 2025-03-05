from abc import ABC, abstractmethod
from operators.functions import *
from copy import deepcopy

from environment.objective_functions import compute_tour_time_window_violation


class Operator(ABC):
    ''' Basic functions required for the destroy and repair operators '''
    
    def __init__(self, fixed_scale_pc, random_seed):
        self.fixed_scale_pc = fixed_scale_pc
        self.random_seed = random_seed
        self.local_random = random.Random()
        self.local_random.seed(self.random_seed)

    def __str__(self):
        return "-".join([self.name, str(self.fixed_scale_pc)])

    def __repr__(self):
        return self.__str__()

    def get_operator_scale(self, state):
        return int((state.inst.cust_number * self.fixed_scale_pc) / 100)
        # return self.fixed_scale_pc

    @abstractmethod
    def apply_op(self, state, **kwargs):
        pass

    def post_destroy_operator(self, state):
        if state.inst.problem_variant == "VRPTW": 
            state.time_window_violation = generate_time_window_violation(state, removed=True)


    def post_repair_operator(self, state):
        # HVRP - check if any vehicle type exceed fleetsize:
        if state.inst.problem_variant == "HVRP": 

            for tour in state.alns_list:
                if len(set([state.cust_vehicle_type[c] for c in tour])) > 1:
                    raise ValueError("Not all customer has the same tour vehicle capacity! ")
            
            current_fleetsize = compute_current_fleetsize(state)
            for capcity_index, capacity in enumerate(state.inst.vehicle_capacity): 
                if current_fleetsize.count(capacity) > state.inst.vehicle_fleetsize[capcity_index]:
                    raise ValueError(f"Error: total fleetsize for vehicle with capacity {capacity} is only {state.inst.vehicle_fleetsize[capcity_index]}, which cannot cover {current_fleetsize.count(capacity)} appearing in the current solution! ")
        
        # LRP - check if invalid depot allocation, overcapacitated depot
        elif state.inst.problem_variant == "LRP":
            if {-1} <= set(state.customer_depot_allocation.values()):
                raise ValueError(f"Repaired solution still contains customer(s) unallocated to any depot! ")
            
            opened_facilities = set(list(state.customer_depot_allocation.values()))
            opened_facilities = {x for x in opened_facilities if x != -1}
            closed_facilities = set(state.depot) - opened_facilities
        
            for facility in closed_facilities:
                if state.node_status[facility] == True: 
                    raise ValueError("Repaired operator failed to update this depot as it is supposed to be closed")
            
            for facility in opened_facilities:
                if state.node_status[facility] == False: 
                    raise ValueError("Repaired operator failed to update this depot as it is supposed to be open")
                
                allocated_demand = sum([state.inst.demand[k] for k, v in state.customer_depot_allocation.items() if v == facility])
                if allocated_demand > state.inst.facility_capacity[facility]:
                    raise ValueError(f"Repaired solution has overcapacitated depot = {facility}, allocated demand = {allocated_demand}, capacity = {state.inst.facility_capacity[facility]}")
        
        elif state.inst.problem_variant == "VRPTW": 
            state.time_window_violation = generate_time_window_violation(state, removed=False)

        else:
            pass


    def get_available_customers(self, state):
        return [customer for customer in state.customers if state.node_status[customer] is True]


    def remove_customers(self, state, customers_to_remove):
        ''' reset customer node status, customer-to-depot allocation, alns_list visiting sequence 
        '''
        state.customer_pool += customers_to_remove

        # update customer node status in feature vector
        for c in customers_to_remove:
            state.node_status[c] = False
            
            # update customer to depot allocation and depot's node status:  
            if state.inst.problem_variant == "LRP":
                state.customer_depot_allocation[c] = -1 
                self.update_facility_opening_status(state)

        # update ALNS list to exclude removed nodes. 
        for tour_index, tour in enumerate(deepcopy(state.alns_list)):    #DEBUGGED: must have deepcopy, or once a pendulum tour removed, tour index will jump by 1!!
            state.alns_list[tour_index][:] = [i for i in tour if i not in customers_to_remove][:]
        state.alns_list = [tour for tour in state.alns_list if tour != []]        


    def update_facility_opening_status(self, state):
        '''update depot opening status inside boolean list node_status '''
        # if state.inst.problem_variant == "LRP":
        #     opened_facilities = set(value for value in state.customer_depot_allocation.values() if value != -1)
        #     closed_facilities = set(state.depot) - opened_facilities  
        #     for depot in opened_facilities:
        #         state.node_status[depot] = True
        #     for depot in closed_facilities:
        #         state.node_status[depot] = False
        if state.inst.problem_variant == "LRP":
            updated_depot_status = [True if i in set(state.customer_depot_allocation.values()) else False for i in state.depot]
            state.node_status = updated_depot_status + state.node_status[state.inst.depot_number:]
        

    def update_vehicle_capacity_status(self, state, cust, best_position): 
        '''update the vehicle capacity record within state.cust_vehicle_type'''

        if state.inst.problem_variant == "HVRP":
            if best_position == "pendulum-tour": 
                state.cust_vehicle_type[cust] = smallest_available_capacity(state, [cust]) #smallest vehicle available!
            else:
                tour = state.alns_list[best_position[0]]
                old_capacity = current_tour_assigned_capacity(state, best_position[0])
                updated_capacity = smallest_available_capacity(state, tour, current_capacity=old_capacity)
                for cust in state.alns_list[best_position[0]]:
                    state.cust_vehicle_type[cust] = updated_capacity


    def deviated_routes_node_greedy_removal(self, state, deviated_routes):
        ''' input sorted routes in acsending/descending order, for each route, use greedy removal gain to remove nodes iteratively 
            appear in: RouteDeviateDestroy, CapacityDeviateDestroy, GreedyCapacityDestroyMax, GreedyCapacityDestroyMin '''
        
        destroy_quota = deepcopy(self.get_operator_scale(state))
        for tour in deviated_routes:
            ## find the most deviated route and compute all its nodes' removal gains (acsending order):
            removal_gain_dict = compute_removal_gain(state, [tour])

            ## remove nodes w.r.t removal gain:
            removal_scale = min(destroy_quota, len(tour))
            worst_customers = sorted(removal_gain_dict, key=removal_gain_dict.get, reverse=True)[:removal_scale]
            self.remove_customers(state, worst_customers)

            ## remove customers until destroy scale is met:
            destroy_quota -= removal_scale
            if destroy_quota == 0:
                break

    
    def deviated_routes_node_sequantial_removal(self, state, deviated_routes):
        ''' input sorted routes in acsending/descending order, for each route, sequentially remove nodes using their indice 
            appear in: GreedyRouteDestroyMax, GreedyRouteDestroyMin'''
        
        customers_to_remove = []
        total_removed_number = len(customers_to_remove)

        for selected_route in deviated_routes:
            ## (1) if removal with this route just meet the criteria:
            if total_removed_number + len(selected_route) == self.get_operator_scale(state):
                customers_to_remove += selected_route
                total_removed_number += len(selected_route)
                self.remove_customers(state, selected_route)
                break
            ## (2) if new route still isnt enough:
            elif total_removed_number + len(selected_route) < self.get_operator_scale(state):
                customers_to_remove += selected_route
                total_removed_number += len(selected_route)
                self.remove_customers(state, selected_route)
                continue
            ## (3) if new route is more than enough:
            else:
                remain_quota = self.get_operator_scale(state) - total_removed_number
                customers_to_remove += selected_route[:remain_quota]
                total_removed_number += remain_quota
                self.remove_customers(state, selected_route[:remain_quota])

            if len(customers_to_remove) == self.get_operator_scale(state):
                break


#--------------------------------------------------
# repair operator related functions: 
# -------------------------------------------------

    def insert_customers_to_best_position(self, state, customers_to_insert, **kwargs):
        ''' insert customer nodes back to the alns solution at the best position 
        '''
        ## define whether a perturbation factor is considered or not for distance computation:
        distance_matrix = kwargs.get("perturbated_distance_matrix") if "perturbated_distance_matrix" in kwargs else state.distance_matrix

        for cust in customers_to_insert:
            ## Receive the best insertion position for the selected customer
            best_position, _ = find_best_node_insertion_distance(state, cust, distance_matrix)

            ## Update the visiting sequence record, creating pendulum tour if new customer doesnt fit capacity constraint
            if state.inst.problem_variant == "LRP":
                if best_position[0] == "pendulum-tour": 
                    state.alns_list.append([cust])
                    state.node_status[best_position[1]] = True
                    state.customer_depot_allocation[cust] = best_position[1]
                else:
                    state.alns_list[best_position[0]][best_position[1]:best_position[1]] = [cust]
                    new_tour_with_cust = [sublist for sublist in state.alns_list if cust in sublist][0]
                    state.customer_depot_allocation[cust] = next((d for d in {state.customer_depot_allocation[c] for c in new_tour_with_cust if c != cust} if d != -1), None)
            
            # non-LRP instances: 
            else:
                if best_position == "pendulum-tour": 
                    state.alns_list.append([cust])
                    self.update_vehicle_capacity_status(state, cust, best_position)
                else:
                    state.alns_list[best_position[0]][best_position[1]:best_position[1]] = [cust]
                    self.update_vehicle_capacity_status(state, cust, best_position)

            ## Update node status in feature vector
            state.node_status[cust] = True
            self.update_facility_opening_status(state)
        return state
    


    def insert_customers_to_random_position(self, state, customers_to_insert):
        ''' insert customer nodes back to the alns solution at a random first-fitting tour and a random edge position 
        '''
        for cust in customers_to_insert:
            if state.inst.problem_variant == "LRP":
                opened_facilities, closed_facilities, opened_facility_with_capacity = find_depot_status(state, cust)
            else:
                opened_facility_with_capacity = None 

            # randomly shuffle the tours for the customer to be inserted into: 
            shuffled_tour_index = [t for t in range(len(state.alns_list))]
            self.local_random.shuffle(shuffled_tour_index)

            # find a random insertion position for this customer "cust": 
            for counter, tour_index in enumerate(shuffled_tour_index):
                
                tour = state.alns_list[tour_index]

                # check if insertion into "tour" satisfies both vehicle and facility capacities? 
                facility_capacity_satisfied, vehicle_capacity_satisfied = check_capacity_satisfaction(state, cust, tour, tour_index, opened_facility_with_capacity)
                
                # (1) Insert to existing tour, if both capacities satisfied: 
                if vehicle_capacity_satisfied and facility_capacity_satisfied:
                    random_position = self.local_random.choice(range(len(tour)+1))
                    state.alns_list[tour_index][random_position:random_position] = [cust]
                    self.update_vehicle_capacity_status(state, cust, (tour_index, random_position))
                    # current_fleetsize = compute_current_fleetsize(state)
                    
                    if state.inst.problem_variant == "LRP":
                        facility_index = list(set([state.customer_depot_allocation[t] for t in tour if state.customer_depot_allocation[t] != -1]))[0]
                        state.customer_depot_allocation[cust] = facility_index
                        
                    break

                # (2) Create a pendulum tour: 
                if counter == len(shuffled_tour_index) - 1:
                    state.alns_list.append([cust])
                    self.update_vehicle_capacity_status(state, cust, "pendulum-tour")
                    # current_fleetsize = compute_current_fleetsize(state)
                    
                    if state.inst.problem_variant == "LRP":
                        # possible that any Facility-related destroy closes necessary facility, causing the exisitng ones to be incapable of serving the whole customer set! In this case, open a new cheapest facility! 
                        if opened_facility_with_capacity != []:
                            random_facility = self.local_random.choice(opened_facility_with_capacity)
                            state.customer_depot_allocation[cust] = random_facility
                        # otherwise, find the cheapest closed depot to open and assign this pendulum tour there:  
                        else: 
                            cheapest_facility = min(closed_facilities, key=lambda x: state.inst.facility_opening_cost[x])
                            state.node_status[cheapest_facility] = True
                            state.customer_depot_allocation[cust] = cheapest_facility

            ## Update customer node status in feature vector:
            state.node_status[cust] = True

        return state



    def sort_customers_by_minimum_global_insertion(self, state, customers_to_insert):
        ''' For deep repair operators to insert the customers according to their global insertion costs 
        '''
        ## Find the global best acsending insertion cost list: version 2
        min_insertion_cost_dict = {cust: min(find_best_node_insertion_distance(state, cust, state.distance_matrix)[1].values()) for cust in customers_to_insert}
        
        ## Find the global best acsending insertion cost list: version 1
        # min_insertion_cost_dict = {}
        # for cust in customers_to_insert:
        #     best_insertion_position, insertion_cost_dict = find_best_node_insertion_distance(state, cust, state.distance_matrix) 
        #     min_insertion_cost_dict[cust] = insertion_cost_dict[best_insertion_position]
        
        ## sort customers based on insertion costs from smallest to largest:
        sorted_customers_to_insert = sorted(min_insertion_cost_dict, key=lambda x: min_insertion_cost_dict[x])
        return sorted_customers_to_insert


    def select_customers_from_pool(self, state):
        ''' select B_r customers from customer pool to be inserted back to the tour '''

        # if len(state.customer_pool) > self.get_operator_scale(state):
        #     print(f"This round, customer pool contains {len(state.customer_pool)} customers! ")
        #     # insert_list = self.local_random.sample(state.customer_pool, self.get_operator_scale(state))

        insert_list = deepcopy(state.customer_pool)
        state.customer_pool = [c for c in state.customer_pool if c not in insert_list]
        return insert_list




    # ----------------------------------
    # LOCAL SEARCH FUNCTIONS: 
    # ----------------------------------

    def aggregating_traveltime_of_a_tour(self, route, state):
        ''' return total traversal time of a single complete (depot included) route.
        '''
        total_traveltime = 0
        for index in range(len(route) - 1):
            total_traveltime += state.distance_matrix[(route[index], route[index+1])]
        # Depot already added in some local search methods, but not in others; distance of (0,0) is 0 anyway
        total_traveltime += state.distance_matrix[(state.depot[0], route[0])]
        total_traveltime += state.distance_matrix[(route[-1], state.depot[0])]
        return total_traveltime


    def swap(self, tour, i, j):
        ''' Swaps the elements in position i and j from the given tour.
        '''
        new_tour = deepcopy(tour)
        cust1, cust2 = new_tour[i], new_tour[j]
        new_tour[i], new_tour[j] = cust2, cust1
        return new_tour


    def find_first_improving_solution(self, tour, state):
        ''' Once find an improvement for a tour, apply the swap and terminate
        '''
        current_tour = deepcopy(tour)
        current_tour_length = self.aggregating_traveltime_of_a_tour(current_tour, state)
       # print("Cost current solution ", current_tour_length)

        found_improving_solution = False
        for (i, j) in [(a, b) for a in range(len(current_tour)) for b in range(len(current_tour)) if a > b]:
            new_tour = self.swap(current_tour, i, j)
            new_tour_length = self.aggregating_traveltime_of_a_tour(new_tour, state)

            if new_tour_length < current_tour_length:
       #         print("Found improving solution with cost", new_tour_length)
                current_tour = new_tour
                found_improving_solution = True
                break  # When the first improving solution is found it breaks the loop so that it stops looking for new solutions
        return found_improving_solution, current_tour


    def two_opt_single_tour(self, route, state):
        ''' Take in a single tour, exchange two non-adjacent arcs and improve the tours until no improvement can be found.
            Reference: https://github.com/RyanRizzo96/Vehicle-Routing-Problem-2-Opt-/blob/master/Haversine_working.py
        '''
        best_route = route
        improved = True
        best_route_length = self.aggregating_traveltime_of_a_tour(route, state)

        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route)):
                    if j - i == 1:
                        continue
                    new_route = route[:]
                    new_route[i:j] = route[j - 1:i - 1:-1]        # this is the 2-opt since j >= i we use -1
                    new_route_length = self.aggregating_traveltime_of_a_tour(new_route, state)
                    if new_route_length < best_route_length:
                        best_route = new_route
                        best_route_length = new_route_length
                        #print("best", best_route_length)
                        improved = True
        return best_route


    def two_opt_single_tour_tw(self, route, state):
        best_route = route
        improved = True
        # best_route_length = self.aggregating_time_violation_of_a_tour(route, state)
        best_route_length = compute_tour_time_window_violation(state, route, state.distance_matrix)
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route)):
                    if j - i == 1:
                        continue
                    new_route = route[:]
                    new_route[i:j] = route[j - 1:i - 1:-1]        # this is the 2-opt since j >= i we use -1
                    # new_route_length = self.aggregating_time_violation_of_a_tour(new_route, state)
                    new_route_length = compute_tour_time_window_violation(state, new_route, state.distance_matrix)
                    if new_route_length < best_route_length:
                        best_route = new_route
                        best_route_length = new_route_length
                        #print("best", best_route_length)
                        improved = True
        return best_route