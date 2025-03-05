import numpy as np
import random
import math
from copy import deepcopy

from environment.objective_functions import compute_cust_time_window_violation, compute_tour_time_window_violation


def smallest_feasible_capacity(state, tour):
    ''' return the smallest FEASIBLE capacity that can serve a given tour, assuming INFINITE the fleetsize availability '''
    acc_demand = sum([state.inst.demand[c] for c in tour])
    for capacity in state.inst.vehicle_capacity:
        if capacity >= acc_demand:
            return capacity
    raise ValueError(f"tour = {tour} cannot fit into any vehicle capacity!")


def compute_current_fleetsize(state):
    ''' given the current tours, return the list of vehicle capacity for each tour '''
    return [list(set([state.cust_vehicle_type[c] for c in tour]))[0] for tour in state.alns_list]


def current_tour_assigned_capacity(state, tour_index):
    ''' find the current tour's assigned vehicle capacity'''
    return list(set([state.cust_vehicle_type[c] for c in state.alns_list[tour_index]]))[0]


def smallest_available_capacity(state, tour, **kwargs):
    ''' return the smallest AVAILABLE capacity that can serve a given tour, assuming FINITE fleetsize availability. '''
    
    # for another customer to be inserted to tour: 
    insert_cust = kwargs["insert_cust"] if "insert_cust" in kwargs else 0 
    current_capacity = kwargs["current_capacity"] if "current_capacity" in kwargs else 0
    
    # if we are simply updating the existing tour then this tour should be included while counting:
    itself_included = 1 if "included" in kwargs else 0
    
    # compute the list of vehicle capacities for each tour in alns_list:
    acc_demand = sum([state.inst.demand[c] for c in tour]) + state.inst.demand[insert_cust]
    current_fleetsize = compute_current_fleetsize(state)
    
    # find the first (smallest) capacity from all available vehicle capacities (including the current one used by "tour"! )
    for i, capacity in enumerate(state.inst.vehicle_capacity):
        if capacity >= acc_demand:
            if state.inst.vehicle_fleetsize[i] - current_fleetsize.count(capacity) + (capacity == current_capacity) + itself_included > 0: 
                # print(f"state.inst.vehicle_fleetsize[{i}]={state.inst.vehicle_fleetsize[i]}, current_fleetsize.count({capacity})={current_fleetsize.count(capacity)}")
                # print(f"({capacity} == {current_capacity})={capacity == current_capacity}, itself_included={itself_included} ")
                return capacity
            
    raise ValueError(f"no available vehicle to hold customer and tour! vehicle capacity checked failed ")



def compute_vehicle_downgrade_gain(state, cust_to_be_removed, tour_index):
    tour = state.alns_list[tour_index]
    new_tour = [c for c in tour if c != cust_to_be_removed]
    
    ## find the current smallest vehicle capacity for the old tour, and the new available capcity for the new tour: 
    old_capacity = current_tour_assigned_capacity(state, tour_index)
    new_capacity = smallest_available_capacity(state, new_tour, current_capacity=old_capacity)
    
    ## 1) if old tour contains only one customer, the removal cost is the whole hiring cost:
    if new_tour == []:
        return state.inst.vehicle_hiring_cost[state.inst.vehicle_capacity.index(old_capacity)]
    
    ## 2) if no available alternative capacity exists, or if new tour not small enough to fit into a smaller capacity, 
        # then the new tour resumes the old tour's capacity, which is found using "smallest_feasible_capacity" function: 
    if new_capacity == None: 
        new_capacity = old_capacity
        
    if (old_capacity - new_capacity) > 0:
        old_capacity_index = state.inst.vehicle_capacity.index(old_capacity)
        new_capacity_index = state.inst.vehicle_capacity.index(new_capacity)
        return state.inst.vehicle_hiring_cost[old_capacity_index] - state.inst.vehicle_hiring_cost[new_capacity_index]
    else:
        return 0 


def compute_facility_closing_gain(state, cust):
    # find the index of the closed depot by removing open depots from the depot list: 
    closed_depot_original = list(set(state.depot) - set([value for _, value in state.customer_depot_allocation.items()])) 
    closed_depot_updated = list(set(state.depot) - set([value for key, value in state.customer_depot_allocation.items() if key != cust])) 
    closed_depot = [x for x in closed_depot_updated if x not in closed_depot_original]
    if closed_depot != []: 
        return sum([state.inst.facility_opening_cost[i] for i in closed_depot])
    else:
        return 0


def compute_removal_gain(state, alns_list):
    ''' input the ALNS tour lists and output the removal gain dictionary including all nodes
    '''
    removal_gain_dict = {}
    for tour_index, tour in enumerate(alns_list):
        
        for cust_index, cust in enumerate(tour):
            removal_gain = 0
            previous_cust = tour[cust_index - 1] if cust_index != 0 else 0
            next_cust = tour[cust_index + 1] if cust_index != len(tour) - 1 else 0

            # compute the distance-based removal gain:
            removal_gain += triangular_travel_difference(previous_cust, cust, next_cust, state.distance_matrix)


            # non-tsp variants: vehicle type update cost / facility closing cost / time window violation cost: 
            if state.inst.problem_variant == "CVRP":
                removal_gain += state.inst.vehicle_hiring_cost if len(tour) == 1 else 0 
        
            elif state.inst.problem_variant == "LRP":
                removal_gain += compute_facility_closing_gain(state, cust)
                removal_gain += state.inst.vehicle_hiring_cost if len(tour) == 1 else 0 
                
            elif state.inst.problem_variant == "HVRP":
                removal_gain += compute_vehicle_downgrade_gain(state, cust, tour_index)
            
            elif state.inst.problem_variant == "VRPTW":
                new_tour = tour[:cust_index] + tour[cust_index + 1:]
                # compute the total time violation cost for the tour (both forward and backward directions):
                new_tour_violation_forward = compute_tour_time_window_violation(state, new_tour, state.distance_matrix)
                new_tour_violation_backward = compute_tour_time_window_violation(state, new_tour[::-1], state.distance_matrix)
                # print(f"new_tour={new_tour}, new_tour_violation_forward={new_tour_violation_forward}, new_tour_violation_backward={new_tour_violation_backward}")
                
                # if forward direction is better: 
                if new_tour_violation_forward < new_tour_violation_backward:
                    old_tour_violation_forward = compute_tour_time_window_violation(state, tour, state.distance_matrix)
                    removal_gain += state.inst.exceed_tw_penalty * max(0, old_tour_violation_forward - new_tour_violation_forward)
                    # print(f'forward better: old - new = {old_tour_violation_forward} - {new_tour_violation_forward} = {old_tour_violation_forward - new_tour_violation_forward}')
                else: # if backward direction better: 
                    old_tour_violation_backward = compute_tour_time_window_violation(state, tour[::-1], state.distance_matrix)
                    removal_gain += state.inst.exceed_tw_penalty * max(0, old_tour_violation_backward - new_tour_violation_backward)
                    # print(f'backward better: old - new = {old_tour_violation_backward} - {new_tour_violation_backward} = {old_tour_violation_backward - new_tour_violation_backward}')
            
            # store the (customer, removal gain) pair in dictionary
            removal_gain_dict[cust] = removal_gain
    return removal_gain_dict


def rescale_values(deviated_routes):
    ''' find each tour length/demand difference from the average tour. Applied in RouteDeviateDestroy and CapacityDeviateDestroy'''
    average = np.average(list(deviated_routes.values()))
    for key in deviated_routes.keys():
        deviated_routes[key] = abs(deviated_routes[key] - average)
    return deviated_routes

def sort_deviated_route_avg(state, operator_name):
    ''' returns a list of tours in descending order regarding how different the length/demand is from the average.
        appear in: RouteDeviateDestroy 
    '''
    routes = {}
    if operator_name == "D-CapacityDeviate":
        for tour_index, tour in enumerate(state.alns_list):
            routes[tour_index] = sum([state.inst.demand[c] for c in tour])
    elif operator_name == "D-RouteDeviate":
        for tour_index, tour in enumerate(state.alns_list):
            travel_distance = state.distance_matrix[(0, tour[0])] + state.distance_matrix[(tour[-1], 0)]
            travel_distance += sum([state.distance_matrix[(cust, tour[cust_index + 1])] for cust_index, cust in enumerate(tour[:-1])])
            routes[tour_index] = travel_distance
    deviated_routes = rescale_values(routes)
    
    deviated_route_sorted_dict = dict(sorted(deviated_routes.items(), reverse=True, key=lambda item: item[1])) # max to min
    deviated_route_sorted_list = [state.alns_list[index] for index in list(deviated_route_sorted_dict.keys())]
    # deviated_route_sorted = [state.alns_list[index] for index in list(dict(sorted(deviated_route.items(), reverse=True, key=lambda item: item[1])).keys())]
    return deviated_route_sorted_list

def sort_deviated_route_capacity(state, operator_name):
    ''' returns a list of tours in descending order regarding how utilised each tour is.
        appear in: CapacityDeviateDestroy 
    '''
    deviated_routes = {}
    for tour_index, tour in enumerate(state.alns_list):
        
        # IF HVRP, find how much free space is left for each tour, given a minimum feasible vechicle capacity:
        if state.inst.problem_variant == "HVRP":
            cum_demand = sum([state.inst.demand[c] for c in tour])
            deviated_routes[tour_index] = min((num for num in state.inst.vehicle_capacity if num >= cum_demand), default=None) - cum_demand
        else:
            deviated_routes[tour_index] = state.inst.vehicle_capacity - sum([state.inst.demand[c] for c in tour])

    deviated_route_sorted_dict = dict(sorted(deviated_routes.items(), reverse=True, key=lambda item: item[1])) # max to min
    deviated_route_sorted_list = [state.alns_list[index] for index in list(deviated_route_sorted_dict.keys())]

    return deviated_route_sorted_list

def accumulate_tour_demand(state):
    routes = {}
    for tour_index, tour in enumerate(state.alns_list):
        routes[tour_index] = sum([state.inst.demand[c] for c in tour])
    return routes

def accumulate_tour_distance_length(state):
    routes = {}
    for tour_index, tour in enumerate(state.alns_list):
        travel_distance = state.distance_matrix[(0, tour[0])] + state.distance_matrix[(tour[-1], 0)]
        travel_distance += sum([state.distance_matrix[(cust, tour[cust_index + 1])] for cust_index, cust in enumerate(tour[:-1])])
        routes[tour_index] = travel_distance
    return routes

def sort_deviated_route(state, operator_name):
    ''' returns a list of tours in ascending / descending order regarding the length/demand of each tour.
    '''
    if operator_name == "D-MaxCapacityRoute":
        routes = accumulate_tour_demand(state)
        deviated_route_sorted_dict = dict(sorted(routes.items(), reverse=True, key=lambda item: item[1]))  # max to min
    elif operator_name == "D-MinCapacityRoute":
        routes = accumulate_tour_demand(state)
        deviated_route_sorted_dict = dict(sorted(routes.items(), key=lambda item: item[1]))  # min to max
    elif operator_name == "D-MaxRoute":
        routes = accumulate_tour_distance_length(state)
        deviated_route_sorted_dict = dict(sorted(routes.items(), reverse=True, key=lambda item: item[1]))  # max to min
    elif operator_name == "D-MinRoute":
        routes = accumulate_tour_distance_length(state)
        deviated_route_sorted_dict = dict(sorted(routes.items(), key=lambda item: item[1]))  # min to max
        #print("deviated_route_sorted_dict = ", deviated_route_sorted_dict)
        #print("min to max")
    else:
        print("This function is not implemented to run the selected destroy operator")
    
    deviated_route_sorted_list = [state.alns_list[index] for index in list(deviated_route_sorted_dict.keys())]
    return deviated_route_sorted_list



# -------------------------------------------------
# repair operator related functions: 
# -------------------------------------------------

def first_nonzero_reverse(lst): 
    ''' for an available fleetsize list "lst", find the index of the first element in reversed lst 
        and return this index in the original list.'''
    return len(lst) - lst[::-1].index(next(filter(lambda x: x != 0, lst[::-1]))) - 1


def first_nonzero(lst): 
    ''' for an available fleetsize list "lst", find the index of the first element in original order lst and return this index.'''
    return lst[:].index(next(filter(lambda x: x != 0, lst[:])))


def compute_remaining_fleetsize(state, tour_index):
    ''' return the list of remaining available set of vehicles from smallest to largest capacities. 
        find the remaining fleetsize for each vehicle type, including the one currently using (can look for vehicle itself)! '''
    current_fleetsize = compute_current_fleetsize(state) 
    return [state.inst.vehicle_fleetsize[i] - current_fleetsize.count(num) + (state.inst.vehicle_capacity[i] == current_fleetsize[tour_index]) 
            for i, num in enumerate(state.inst.vehicle_capacity)]
    # remaining_fleetsize = []
    # for i, num in enumerate(state.inst.vehicle_capacity):
    #     included = 0
    #     if state.inst.vehicle_capacity[i] == current_fleetsize[tour_index]: 
    #         included = 1
    #     print(f"state.inst.vehicle_fleetsize[{i}]={state.inst.vehicle_fleetsize[i]}, current_fleetsize.count({num})={current_fleetsize.count(num)}, current_fleetsize[{tour_index}]={current_fleetsize[tour_index]}, included = {included}")
    #     remaining_fleetsize.append( state.inst.vehicle_fleetsize[i] - current_fleetsize.count(num) + included )
    # return remaining_fleetsize
   


def compute_remaining_additional_fleetsize(state):
    ''' return the list of remaining available set of vehicles from smallest to largest capacities. 
        find the remaining fleetsize for each vehicle type, NOT including the ones currently using! '''
    current_fleetsize = compute_current_fleetsize(state) 
    return [state.inst.vehicle_fleetsize[i] - current_fleetsize.count(num) for i, num in enumerate(state.inst.vehicle_capacity)]


def check_capacity_satisfaction(state, cust, tour, tour_index, opened_facility_with_capacity):
    facility_capacity_satisfied = True
        
    # Can this inserted customer fit into the (largest available) vehicle capacity?
    if state.inst.problem_variant == "HVRP":
        remain_fleetsize = compute_remaining_fleetsize(state, tour_index)
        vehicle_capacity = state.inst.vehicle_capacity[first_nonzero_reverse(remain_fleetsize)]
    else:
        vehicle_capacity = state.inst.vehicle_capacity
        
        # Can this inserted customer fit into this tour and assigned to the corresponding depot? 
        if state.inst.problem_variant == "LRP":
            facility_for_this_tour = set([state.customer_depot_allocation[t] for t in tour if state.customer_depot_allocation[t] != -1])
            
            facility_capacity_satisfied = True if facility_for_this_tour <= set(opened_facility_with_capacity) and (bool(facility_for_this_tour) != False) else False
            # print(f"--facility_capacity_satisfied = {facility_capacity_satisfied}, because ({facility_for_this_tour}) is in ({set(opened_facility_with_capacity)})")
            
            if len(list(facility_for_this_tour)) > 1: 
                raise ValueError(f"Error! One tour has two facility allocation! ")
        
    # Continue only when if the existing largest capacity can serve this tour plus the new customer (NOT necessarily the tightest capacity!):
    vehicle_capacity_satisfied = (sum([state.inst.demand[c] for c in tour]) + state.inst.demand[cust] <= vehicle_capacity)
    return facility_capacity_satisfied, vehicle_capacity_satisfied 



def find_depot_status(state, cust):
    opened_facilities = set(list(state.customer_depot_allocation.values()))
    opened_facilities = {x for x in opened_facilities if x != -1}
    closed_facilities = set(state.depot) - opened_facilities
    
    opened_facility_with_capacity = []
    for facility in opened_facilities:
        allocated_cust = [k for k, v in state.customer_depot_allocation.items() if v == facility]
        allocated_demand = sum([state.inst.demand[c] for c in allocated_cust])
        # print(f"allocated_demand = {allocated_demand}, state.inst.demand[{cust}]={state.inst.demand[cust]}, state.inst.facility_capacity[{facility}]={state.inst.facility_capacity[facility]}")
        if allocated_demand + state.inst.demand[cust] <= state.inst.facility_capacity[facility]:
            opened_facility_with_capacity.append(facility)

    return opened_facilities, closed_facilities, opened_facility_with_capacity


def triangular_travel_difference(prev_cust, cust, next_cust, distance_matrix):
    ''' compute the extra gain/lost by travelling the two edges of a triangular tour '''
    return distance_matrix[(prev_cust, cust)] + distance_matrix[(cust, next_cust)] - distance_matrix[(prev_cust, next_cust)]


def find_best_node_insertion_distance(state, insert_cust, distance_matrix, **kwargs): 
    ''' For greedy-based operators. Insert a single node back to the tours into the best position (geographical distance)
        arc = the index of arc inside a tour, e.g. tour = [5,6,7,8], arc = 0 --> (0,5), arc = 1 --> (5,6), arc = 4 --> (8,0)
    '''
    insertion_cost_dict = {}

    if state.inst.problem_variant == "LRP":
        _, closed_facilities, opened_facility_with_capacity = find_depot_status(state, insert_cust)
    else:
        opened_facility_with_capacity = None 

    # find the best insertion position for this customer "cust": 
    for tour_index, tour in enumerate(state.alns_list):
        # check if insertion into "tour" satisfies both vehicle and facility capacities? 
        facility_capacity_satisfied, vehicle_capacity_satisfied = check_capacity_satisfaction(state, insert_cust, tour, tour_index, opened_facility_with_capacity)
        
        # if both capacities satisfied, insert into least-distance position: 
        if vehicle_capacity_satisfied and facility_capacity_satisfied: 

            for arc in range(len(tour)+1):
                insertion_cost = 0
                previous_cust = tour[arc-1] if arc != 0 else 0
                next_cust = tour[arc] if arc != len(tour) else 0

                # compute the insertion additional travel cost (replaced all state.distance_matrix by input distance_matrix)
                insertion_cost += triangular_travel_difference(previous_cust, insert_cust, next_cust, distance_matrix)

                # (1) additional hiring cost due to vehicle capacity increase: 
                if state.inst.problem_variant == "HVRP":
                    ## find the current smallest vehicle capacity for the old tour, and the new available capcity for the new tour: 
                    old_capacity = current_tour_assigned_capacity(state, tour_index)
                    updated_capacity = smallest_available_capacity(state, tour, insert_cust=insert_cust, current_capacity=old_capacity)
                    
                    ## if no available larger vehicle exists, or if new tour not small enough to fit into a smaller capacity, the new tour takes the old capacity: 
                    if (updated_capacity - old_capacity) > 0:
                        old_capacity_index = state.inst.vehicle_capacity.index(old_capacity)
                        new_capacity_index = state.inst.vehicle_capacity.index(updated_capacity)
                        insertion_cost += (state.inst.vehicle_hiring_cost[new_capacity_index] - state.inst.vehicle_hiring_cost[old_capacity_index])

                # (2) additional time window violation costs: 
                elif state.inst.problem_variant == "VRPTW":
                # Note: the time window of all customers after the insert cust will change due to the insertion, compute the excess tw violation cost by taking
                # the difference between the original tour time violation and updated tour time violation; Both forward and backward time window computations 
                # should be considered as the recorded alns tour [1,2,3] might be visiting as [3,2,1] in the optimal solution. 
                    new_tour = tour[:arc] + [insert_cust] + tour[arc:]
                    
                    # compute the new tour's forward and backward time window violation (with the inserted "cust" inside the tour): 
                    new_tour_violation_forward = compute_tour_time_window_violation(state, new_tour, distance_matrix)
                    new_tour_violation_backward = compute_tour_time_window_violation(state, new_tour[::-1], distance_matrix)  
                    
                    # check which order suits the time window better, forward or backward?
                    # only introduce additional violation cost if new_tour_violation > old_tour_violation, as a worse violation should be penalised but a better one should not. 
                    if new_tour_violation_forward < new_tour_violation_backward:
                        old_tour_violation_forward = compute_tour_time_window_violation(state, tour, distance_matrix)
                        insertion_cost += state.inst.exceed_tw_penalty * max(0, new_tour_violation_forward - old_tour_violation_forward)
                    else:
                        old_tour_violation_backward = compute_tour_time_window_violation(state, tour[::-1], distance_matrix)
                        insertion_cost += state.inst.exceed_tw_penalty * max(0, new_tour_violation_backward - old_tour_violation_backward)
                
                # store insertion cost for each position inside a dictionary: 
                insertion_cost_dict[(tour_index, arc)] = insertion_cost


    # if create a pendulum tour, add the smallest available vehicle's hiring cost to the total insertion cost:
    if state.inst.problem_variant == "CVRP":
        insertion_cost_dict["pendulum-tour"] = state.inst.vehicle_hiring_cost + distance_matrix[(insert_cust, 0)] + distance_matrix[(0, insert_cust)]
    
    elif state.inst.problem_variant == "HVRP":
        vehicle_index = next((i for i, num in enumerate(compute_remaining_additional_fleetsize(state)) if num), None)
        vehicle_hiring_cost = state.inst.vehicle_hiring_cost[vehicle_index]
        insertion_cost_dict["pendulum-tour"] = vehicle_hiring_cost + distance_matrix[(insert_cust, 0)] + distance_matrix[(0, insert_cust)]
    
    elif state.inst.problem_variant == "LRP":
        all_open_facility_full = True if opened_facility_with_capacity == [] else False
        # print(f"all_open_facility_full={all_open_facility_full}, opened_facility_with_capacity={opened_facility_with_capacity}")
        
        if all_open_facility_full == False:
            for facility in opened_facility_with_capacity:
                insertion_cost_dict[("pendulum-tour", facility)] = state.inst.vehicle_hiring_cost + distance_matrix[(insert_cust, facility)] + distance_matrix[(facility, insert_cust)]
                # print(f"opened facility={facility}, insertion_cost_dict[(pendulum-tour, facility)]={state.inst.vehicle_hiring_cost + distance_matrix[(cust, facility)] + distance_matrix[(facility, cust)]}")
        else:
            for facility in closed_facilities:
                insertion_cost_dict[("pendulum-tour", facility)] = state.inst.vehicle_hiring_cost + state.inst.facility_opening_cost[facility] + distance_matrix[(insert_cust, facility)] + distance_matrix[(facility, insert_cust)]
                # print(f"closed facility={facility}, insertion_cost_dict[(pendulum-tour, facility)]={state.inst.vehicle_hiring_cost + distance_matrix[(cust, facility)] + distance_matrix[(facility, cust)]}")
    
    elif state.inst.problem_variant == "VRPTW":
        insert_cost = state.inst.vehicle_hiring_cost + distance_matrix[(insert_cust, 0)] + distance_matrix[(0, insert_cust)]
        # In pendulum tour, tw violation is still involved since starting time from depot can be adjusted. Nevertheles pendulum tours are not preferred as hiring cost is still high..
        tw_violation = state.inst.exceed_tw_penalty * compute_cust_time_window_violation(state, insert_cust, distance_matrix[(0, insert_cust)])
        insertion_cost_dict["pendulum-tour"] = insert_cost + tw_violation
    

    # find the minimal insertion cost position
    best_insertion_position = min(insertion_cost_dict, key=insertion_cost_dict.get)
    return best_insertion_position, insertion_cost_dict

    
    
def compute_regret_value(state, k_value, cust):
        ''' Re-order customers in the order of their regret values, before inserting them back to their best positions.
        '''
        ## Compute the insertion costs for all possible position for a customer
        best_position, regret_dict = find_best_node_insertion_distance(state, cust, state.distance_matrix)

        ## Compute additional objs caused by best and k-best insertion, get regret value (=cost difference bettween best and k-th best)
        # Additional objective value caused by inserting customer into best position:
        best = {best_position: regret_dict[best_position]}   
        # Inserting customer into k-best position. Notice that k-value=2 means returning 2nd smallest, which is 1st in Python language:
        k_best = {key: value for key, value in regret_dict.items() if value in sorted(regret_dict.values(), reverse=False)[k_value-1:k_value]}
        
        # In case there is multiple positions with same insertion cost, we always select the first position.
        if len(k_best) > 1:
            k_best.pop(best_position, None)
            k_best = {list(k_best.keys())[0]: list(k_best.values())[0]}

        ## If the only feasible insertion is creating a pendulum tour due to vehicle capacity, we may well insert this node the last!
        # a.k.a, let the regret value be zero. 
        if list(best.keys())==["pendulum-tour"] or list(best.keys())[0][0]=="pendulum-tour":
            regret_value = 0
        elif len(k_best) == 0: 
            regret_value = list(best.values())[0]
        else:
            regret_value = list(k_best.values())[0] - list(best.values())[0]
        return regret_value


def check_even_or_odd(operator_scale):
    if operator_scale % 2 == 0:
        extra_removal = False
        destroy_scale = deepcopy(operator_scale/2)
    else:
        extra_removal = True
        destroy_scale = (operator_scale - 1)/2
    return extra_removal, int(destroy_scale)



def generate_time_window_violation(state, **kwargs):
    ''' Compute the time violation of each node's arrival time t from its time window [a, b]. 
        Assume a tour [n1, n2, n3, n4], its true travel direction could be either forward or backward. Compare the violation of both, 
        take the direction with the smaller violation cost and assign all customers within this tour to be forward or backward; 
    '''
    time_window_violation = {}

    for tour in state.alns_list:
        full_tour = state.depot + tour
        node2_arrival_time_forward  = [0] * len(full_tour)
        node2_arrival_time_backward = [0] * len(full_tour)
        forward_tw_violation  = [0] * len(full_tour)
        backward_tw_violation = [0] * len(full_tour)

        # track the arrival time at each end node, with forward and backwards directions of the tour: 
        # e.g. full_tour = [0, 7, 8, 5, 6], forward node2 = [7, 8, 5, 6], backward node2 = [6, 5, 8, 7], node1 = departure node, node2 = arrival node
        for i, (node1, node2) in enumerate(zip(full_tour[:-1], full_tour[1:])):
            rev_node1 = full_tour[::-1][i-1]
            rev_node2 = full_tour[::-1][i]
            
            # track the arrival time at each end node, with both directions of the tour: 
            node2_arrival_time_forward[i + 1] = node2_arrival_time_forward[i] + state.inst.service_time[node1] + state.distance_matrix[(node1, node2)]
            node2_arrival_time_backward[i + 1] = node2_arrival_time_backward[i] + state.inst.service_time[rev_node1] + state.distance_matrix[(rev_node1, rev_node2)]

            forward_tw_violation[i + 1] = compute_cust_time_window_violation(state, node2, node2_arrival_time_forward[i + 1])
            backward_tw_violation[i + 1] = compute_cust_time_window_violation(state, rev_node2, node2_arrival_time_backward[i + 1])

        # compute the aggregated tw violation, determine if forward or backward: 
        tw_violations = backward_tw_violation[::-1] if sum(backward_tw_violation) < sum(forward_tw_violation) else forward_tw_violation
        for index, cust in enumerate(tour):
            time_window_violation[cust] = tw_violations[index]

    # if part of the solution is destroyed, record violation as "na":                
    removed = kwargs["removed"] if "removed" in kwargs else False
    if removed == True:
        removed_customers = [c for c in state.customers if c not in list(time_window_violation.keys())]
        for cust in removed_customers:
            time_window_violation[cust] = 'na'

    return time_window_violation



def compute_lat_long_haversine_arc_length(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    # Horizontal arc length (longitude)
    horizontal_distance = R * math.cos(lat1_rad) * delta_lon

    # Vertical arc length (latitude)
    vertical_distance = R * delta_lat

    return horizontal_distance, vertical_distance


