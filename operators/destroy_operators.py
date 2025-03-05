from operator import itemgetter

from operators.functions import *
from operators.base_op import Operator
from state.alns_state import clark_wright_algorithm


class NoOpDestroy(Operator):
    name = "D-NoOp"

    ''' This operator does nothing.
        Useful in case the solution is already good and should not be changed further.
    '''

    def apply_op(self, state, **kwargs):
        return state


class RandomDestroy(Operator):
    name = "D-Random"

    ''' Random destroy operator randomly selects q available customers, removes them from their tours, 
        and adds them into the customer pool.
    '''
    
    def apply_op(self, state, **kwargs): 
        available_customers = self.get_available_customers(state)
        customers_to_remove = self.local_random.sample(available_customers, self.get_operator_scale(state))
        self.remove_customers(state, customers_to_remove)
        self.post_destroy_operator(state)
        return state


class GreedyDestroy(Operator):
    name = "D-Greedy"

    ''' Greedy destroy operator selects the top-ranking q customers with the highest removal gain 
        (wrt geographical distance), removes from their tours, adds into customer pool. The removal gain
        is the cost difference when this customer is in the allocated tour, and when the customer is removed.
    '''
    
    def find_worst_q_customers(self, state):
        removal_gain_dict = compute_removal_gain(state, state.alns_list)
        worst_customers = sorted(removal_gain_dict, key=removal_gain_dict.get, reverse=True)[:self.get_operator_scale(state)]
        return worst_customers


    def apply_op(self, state, **kwargs):
        customers_to_remove = self.find_worst_q_customers(state)
        self.remove_customers(state, customers_to_remove)
        self.post_destroy_operator(state)
        return state


class ShawDestroy(Operator):
    name = "D-Shaw"

    ''' Shaw operator introduced in Shaw 1998 paper, where:
        proximity = geographical distance + served by same vehicle + customer demand difference + service start time difference '''

    def served_by_same_vehicle_dictionary(self, state):
        ''' return all customer pairs as keys and binary result (same tour or not) as values '''
        nums = sorted(set(num for sublst in state.alns_list for num in sublst))
        combos = {(num1, num2): any(num1 in sublst and num2 in sublst for sublst in state.alns_list)
                for i, num1 in enumerate(nums) for num2 in nums[i+1:]}
        combos.update({(num2, num1): value for (num1, num2), value in combos.items()})
        return combos
    
    def compute_shaw_proximity(self, state, same_vehicle_status_dict, customer, cust): 
        ''' Shaw destroy operator relateness computation as in Demir 2012 paper '''
        alpha = 0.75
        beta = 0.1
        gamma = 0.1
        delta = 0.5
        same_vehicle = -1 if same_vehicle_status_dict[(cust, customer)] == True else 1

        geo_distance = state.get_normalized_distance(cust, customer)
        demand_difference = abs(state.inst.normalized_demand[customer] - state.inst.normalized_demand[cust])
        if state.inst.problem_variant == "VRPTW":
            start_time_difference = abs(state.inst.tw_start[customer] - state.inst.tw_start[cust])
        else:
            start_time_difference = 0
        
        #R = alpha*geo_distance + beta*start_time_difference + gamma*same_vehicle + delta*demand_difference
        #print(f"customer={customer}, cust={cust}, sameV={same_vehicle}, geo_distance={geo_distance}, demandDiff={demand_difference}, R={R}")
        
        return alpha*geo_distance + beta*start_time_difference + gamma*same_vehicle + delta*demand_difference


    def apply_op(self, state, **kwargs):
        # randomly select the first removal from available in-tour customers
        available_customers = self.get_available_customers(state)
        cust = self.local_random.sample(available_customers, 1)[0] 
    
        # if two customers belong to the same tour? dictionary with binary value and customer pair key
        same_vehicle = self.served_by_same_vehicle_dictionary(state)

        # find the (q-1) customers with most similar proximity (lower the better): 
        shaw_relateness = {i: self.compute_shaw_proximity(state, same_vehicle, i, cust) for i in available_customers if i != cust}
        nearest_cust = list(dict(sorted(shaw_relateness.items(), key=itemgetter(1))[:self.get_operator_scale(state) - 1]).keys()) 
        customers_to_remove = nearest_cust + [cust]

        self.remove_customers(state, customers_to_remove)
        self.post_destroy_operator(state)

        # print(f"ShawDestroy class - alns_list={state.alns_list}, len(state.alns_list)={len(state.alns_list)}")
        # print(f"ShawDestroy class - customer-depot = {state.customer_depot_allocation}")
        # print(f"ShawDestroy class - state.node_status = {state.node_status}")
        return state 



# ---------------------------------------------
# Non-generic Operators - Related-based
# ---------------------------------------------

class RelatedDestroy(Operator):
    name = "D-Related"

    ''' This destroy operator first randomly selects and removes a single customer and then the (q-1) geographically nearest customers
        from their allocated tours, and place them inside the customer pool waiting to be re-allocated by the upcoming selected repair operator.

        Note: "nearest" here means tour-wise distance, not geographical distance
    '''

    def apply_op(self, state, **kwargs):
        # randomly select the first removal from available in-tour customers
        available_customers = self.get_available_customers(state)
        cust = self.local_random.sample(available_customers, 1)[0]

        # find the (q-1) geographically nearest customers
        distance = {i:  state.distance_matrix[(i, cust)] for i in available_customers if i != cust}
        nearest_cust = list(dict(sorted(distance.items(), key=itemgetter(1))[:self.get_operator_scale(state) - 1]).keys()) 
        customers_to_remove = nearest_cust + [cust]

        self.remove_customers(state, customers_to_remove)
        self.post_destroy_operator(state)
        return state


class RelatedDemandDestroy(Operator):
    name = "D-RelatedDemand" 
    
    ''' Shaw variation, proximity = customer demand '''

    def apply_op(self, state, **kwargs):
        # randomly select the first removal from available in-tour customers
        available_customers = self.get_available_customers(state)
        cust = self.local_random.sample(available_customers, 1)[0] 
    
        # find the (q-1) customers with most similar proximity (lower the better): 
        demand_proximity = {i: abs(state.inst.demand[i] - state.inst.demand[cust]) for i in available_customers if i != cust}
        
        nearest_cust = list(dict(sorted(demand_proximity.items(), key=itemgetter(1))[:self.get_operator_scale(state) - 1]).keys()) 
        customers_to_remove = nearest_cust + [cust]

        self.remove_customers(state, customers_to_remove)
        self.post_destroy_operator(state)
        return state


class NodeNeighbourhoodDestroy(Operator):
    name = "D-NodeNeighbour"

    ''' Inspired by Demir, Emec, Alinaghian (Node Neighbourhood Removal): related-based, distance-based
        Selects a random node and removes a set of nodes lays within a rectangular perimeter around the selected node. 
    '''

    def compute_distance_rectangular(self, state, available_customers, cust):
        distance = {}
        
        if state.inst.data_file_name == "de_smet":
            centre_x, centre_y = state.inst.latitude[cust], state.inst.longitude[cust]
            for customer in available_customers:
                cust_x, cust_y = state.inst.latitude[customer], state.inst.longitude[customer]
                horizontal_dist, vertical_dist = compute_lat_long_haversine_arc_length(centre_x, centre_y, cust_x, cust_y)
                distance[customer] = max(horizontal_dist, vertical_dist)
        else:
            centre_x, centre_y = state.inst.loc_x[cust], state.inst.loc_y[cust]
            for customer in available_customers:
                cust_x, cust_y = state.inst.loc_x[customer], state.inst.loc_y[customer]
                distance[customer] = max(abs(cust_x - centre_x), abs(cust_y - centre_y))

        nearest_cust = list(dict(sorted(distance.items(), key=itemgetter(1))[:self.get_operator_scale(state) - 1]).keys())
        return nearest_cust


    def apply_op(self, state, **kwargs):
        ## Randomly select the first removal from available in-tour nodes
        available_customers = self.get_available_customers(state)
        cust = self.local_random.sample(available_customers, 1)[0]

        ## Find remaining (q-1) nodes - apply specific destroy method based on vrp environment/instance:
        # if state.inst.problem_variant != ProblemVariant.CVRP:
        #     raise ValueError("This operator is not compatible outside of CVRP. Check config.")

        nearest_cust = self.compute_distance_rectangular(state, [x for x in available_customers if x != cust], cust)

        customers_to_remove = nearest_cust + [cust]
        self.remove_customers(state, customers_to_remove)
        self.post_destroy_operator(state)
        return state


class ZoneDestroy(Operator):
    name = "D-Zones"

    ''' Inspired by Emec 2017 (Zone Removal): related-based, distance-based
        Removes a cluster of related nodes defined using Cartesian coordinate. Norm 1 . '''

    def compute_distance_cartesian(self, state, available_customers, cust):
        distance = {}

        if state.inst.data_file_name == "de_smet":
            centre_x, centre_y = state.inst.latitude[cust], state.inst.longitude[cust]
            for customer in available_customers:
                cust_x, cust_y = state.inst.latitude[customer], state.inst.longitude[customer]
                horizontal_dist, vertical_dist = compute_lat_long_haversine_arc_length(centre_x, centre_y, cust_x, cust_y)
                distance[customer] = horizontal_dist + vertical_dist
        else:
            centre_x, centre_y = state.inst.loc_x[cust], state.inst.loc_y[cust]
            for customer in available_customers:
                cust_x, cust_y = state.inst.loc_x[customer], state.inst.loc_y[customer]
                distance[customer] = abs(cust_x - centre_x) + abs(cust_y - centre_y)

        nearest_custs = list(dict(sorted(distance.items(), key=itemgetter(1))[:self.get_operator_scale(state) - 1]).keys())
        return nearest_custs

    def apply_op(self, state, **kwargs):

        ## randomly select the first removal from available in-tour nodes
        available_customers = self.get_available_customers(state)
        cust = self.local_random.sample(available_customers, 1)[0]

        ## find remaining (q-1) nodes - apply specific destroy method based on vrp environment/instance:
        # if state.inst.problem_variant != ProblemVariant.CVRP:
        #     raise ValueError("This operator is not compatible outside of CVRP. Check config.")

        nearest_custs = self.compute_distance_cartesian(state, [x for x in available_customers if x != cust], cust)

        customers_to_remove = nearest_custs + [cust]
        self.remove_customers(state, customers_to_remove)
        self.post_destroy_operator(state)
        return state


class PairDestroy(Operator):
    name = "D-Pair"

    ''' Inspired by Mancini 2016 Pair Removal: related-based, distance-based
        Randomly selects half set of the customers to remove together with their geographically 
        closest customers which are not already removed
    '''

    def find_geographic_closest_pair(self, state, node):
        remain_available_customers = self.get_available_customers(state)
        distance = {}
        for customer in remain_available_customers:
            distance[customer] = state.distance_matrix[(node, customer)]
        nearest_cust = list(dict(sorted(distance.items(), key=itemgetter(1))[:1]).keys())
        self.remove_customers(state, nearest_cust)


    def apply_op(self, state, **kwargs):
        extra_removal, destroy_scale = check_even_or_odd(self.get_operator_scale(state))

        ## select half the removal quota:
        available_customers = self.get_available_customers(state)
        if self.get_operator_scale(state) != 1:
            half_customers_to_remove = self.local_random.sample(available_customers, destroy_scale)
            self.remove_customers(state, half_customers_to_remove)

            ## find the closest pairs:
            for node in half_customers_to_remove:
                self.find_geographic_closest_pair(state, node)

        ## if destroy scale is odd, need to remove 1 more:
        if extra_removal == True:
            available_customers = self.get_available_customers(state)
            last_customers_to_remove = self.local_random.sample(available_customers, 1)
            self.remove_customers(state, last_customers_to_remove)

        self.post_destroy_operator(state)
        return state


class RouteNeighbourhoodDestroy(Operator):
    name = "D-RouteNeighbour"

    ''' Inspired by Emec 2017 Route Neighbourhood Removal: related-based, distance-based
        Randomly selects a pair of routes and removes pairs of customers with the smallest repaired distance. 
        Note: adjusted to the current function in case the route length is smaller than destroy scale! 
    '''

    def remove_pairs_with_smallest_distance(self, state, destroy_pair):
        ''' find the pair of customers with minimum distance between the routes '''
        for _ in range(destroy_pair):
            ## if there are at least 2 routes, we find the best pair of customers to remove
            if len(state.alns_list) >= 2:
                route_1, route_2 = self.local_random.sample(state.alns_list, 2)
                pair_distances = [(c1, c2, state.distance_matrix[(c1, c2)]) for c1 in route_1 for c2 in route_2]
                customer_1, customer_2, _ = min(pair_distances, key=lambda x: x[2])
                removed_customers = [customer_1, customer_2]
                
            ## if only 1 route remains, we randomly select 2 customers to remove
            else:
                available_customers = self.get_available_customers(state)
                removed_customers = self.local_random.sample(available_customers, 2)

            self.remove_customers(state, removed_customers)


    def apply_op(self, state, **kwargs):
        ## destroy scale
        extra_removal, destroy_pair = check_even_or_odd(self.get_operator_scale(state))

        ## select route pairs and remove customer pairs with smallest distance:
        self.remove_pairs_with_smallest_distance(state, destroy_pair)

        ## if destroy scale is odd, need to remove 1 more:
        if extra_removal == True:
            available_customers = self.get_available_customers(state)
            last_customers_to_remove = self.local_random.sample(available_customers, 1)
            self.remove_customers(state, last_customers_to_remove)

        self.post_destroy_operator(state)
        return


class ClusterDestroy(Operator):
    name = "D-Cluster"

    ''' Inspired by Pisinger 2007 : related-based, distance-based
        Removes clusters of related requests from a few (or one) routes. As a motivation, consider a route where the requests 
        are grouped into two geographical clusters. When removing requests from such a route it is often important to 
        remove one of these clusters entirely. (Kruskal’s algorithm applied to form min spanning tree) '''

    def min_spanning_tree(self, sorted_edges, removed_customers, operator_scale):
        for edge in sorted_edges:
            removed_customers += list(edge)
            removed_customers = list(set(removed_customers))

            if len(removed_customers) == operator_scale:
                return removed_customers
            elif operator_scale == 1:
                return [removed_customers[0]]
            elif len(removed_customers) > operator_scale:
                extra = len(removed_customers) - operator_scale
                return removed_customers[:-extra]
        return removed_customers


    def cluster_version_1(self, state):
        edge_dict = {}
        for tour in deepcopy(state.alns_list):
            for cust_index, cust in enumerate(tour[:-1]):
                edge_dict[(cust, tour[cust_index + 1])] = state.distance_matrix[(cust, tour[cust_index + 1])]
        sorted_edges = sorted(edge_dict, key=edge_dict.get)
        removed_customers = self.min_spanning_tree(sorted_edges, [], self.get_operator_scale(state))
        return removed_customers


    def cluster_version_2(self, state): 
        edge_dict, removed_customers = {}, []
        all_routes = deepcopy(state.alns_list)
        keep_removal = True

        if keep_removal == True: 
            random_route = self.local_random.sample(all_routes, 1)
            all_routes.remove(random_route)
            for cust_index, cust in enumerate(random_route[:-1]):
                edge_dict[(cust, random_route[cust_index + 1])] = state.distance_matrix[(cust, random_route[cust_index + 1])]
            sorted_edges = sorted(edge_dict, key=edge_dict.get)
            removed_customers.append(self.min_spanning_tree(sorted_edges, removed_customers, self.get_operator_scale(state)))
            if len(removed_customers) == self.get_operator_scale(state):
                keep_removal = False


    def apply_op(self, state, **kwargs):
        ## Version 1: apply Kruskal's algorithm on the whole solution (multiple routes): 
        removed_customers = self.cluster_version_1(state)
        
        ## Version 2: apply Kruskal's algorithm sequencially on each route until destroy scale is met: 
        #removed_customers = self.cluster_version_2(state)
        
        self.remove_customers(state, removed_customers)
        self.post_destroy_operator(state)
        return state


class HistoricalPairDestroy(Operator):
    name = "D-HistoricalPair"

    ''' Related-based: (from historical request-pair removal), historical record-based
        Compute the number of times the any two nodes x and y have been served by the same vehicle. 
        This is used as the related weight in Shaw removal.
    '''

    def update_same_tour_frequency(self, state):
        ''' updated record of how many times any pair of customers appear in the same route '''
        for tour in deepcopy(state.alns_list):
            for cust in tour:
                custs_same_route = [c for c in tour if c != cust]
                for other in custs_same_route:
                    state.same_route_frequency[(cust, other)] += 1


    def apply_op(self, state, **kwargs):
        ## update historical same-tour frequency:
        self.update_same_tour_frequency(state)

        ## randomly select the first removal from available in-tour nodes
        available_customers = self.get_available_customers(state)
        cust = self.local_random.sample(available_customers, 1)[0]

        ## use frequency to determine how close each pair of customers is:
        frequency = {i: state.same_route_frequency[(cust, i)] for i in available_customers if i != cust}
        nearest_cust = list(dict(sorted(frequency.items(), key=itemgetter(1), reverse=True)[:self.get_operator_scale(state) - 1]).keys())  # largest element showing first!

        customers_to_remove = nearest_cust + [cust]
        self.remove_customers(state, customers_to_remove)
        self.post_destroy_operator(state)
        return state


class FacilitySwapDestroy(Operator):
    name = "D-FacilitySwap"

    ''' Exchnages the status of two facilities that are geographically nearst. Or randomly select an closing depot, 
        change status with another nearest closed depot.  
        Note: if all depots are opened, no depot status is swapped! '''
    
    def apply_op(self, state, **kwargs):
        
        # randomly select an opened depot: 
        opened_depot_list = list({v for v in state.customer_depot_allocation.values()})
        depot_to_close = self.local_random.choice(opened_depot_list)
        
        closed_depot_list = list(set(state.depot) - set(state.customer_depot_allocation.values()))
        
        # cannot swap if all depots are currently open: 
        if closed_depot_list == []:
            return state
        else: 
            # sort out the closed depot according to geographical distance:
            depot_to_open_list = sorted(closed_depot_list, key=lambda x: state.distance_matrix[(depot_to_close, x)])
            for depot_to_open in depot_to_open_list:
                allocated_demand = sum([state.inst.demand[k] for k, v in state.customer_depot_allocation.items() if v == depot_to_close])
                
                # swap the two facilities status only if the new one has enough capacity! 
                if allocated_demand <= state.inst.facility_capacity[depot_to_open]:
                    state.customer_depot_allocation = {i: depot_to_open 
                                                    if state.customer_depot_allocation[i] == depot_to_close 
                                                    else state.customer_depot_allocation[i] for i in state.customers}
                    
                    state.node_status[depot_to_open] = True
                    state.node_status[depot_to_close] = False
                    break

            return state



class RelatedTimeDestroy(Operator):
    name = "D-RelatedTime"

    ''' randomly select a customer and remove customers with the most similar time window starting time '''
    
    def apply_op(self, state, **kwargs):
        # randomly select the first removal from available in-tour customers
        available_customers = self.get_available_customers(state)
        cust = self.local_random.sample(available_customers, 1)[0] 

        # find the (q-1) customers with most similar tw starting time (lower the difference the better): 
        tw_start_diff = {i: abs(state.inst.tw_start[i] - state.inst.tw_start[cust]) for i in available_customers if i != cust}
        nearest_cust = list(dict(sorted(tw_start_diff.items(), key=itemgetter(1))[:self.get_operator_scale(state) - 1]).keys()) 
        customers_to_remove = nearest_cust + [cust]

        self.remove_customers(state, customers_to_remove)
        self.post_destroy_operator(state)

        return state 



class RelatedDurationDestroy(Operator):
    name = "D-RelatedDuration"

    ''' randomly select a customer and remove customers with the most similar service time '''
    
    def apply_op(self, state, **kwargs):
        # randomly select the first removal from available in-tour customers
        available_customers = self.get_available_customers(state)
        cust = self.local_random.sample(available_customers, 1)[0] 

        # find the (q-1) customers with most similar proximity (lower the better): 
        service_diff = {i: abs(state.inst.service_time[i] - state.inst.service_time[cust]) for i in available_customers if i != cust}
        nearest_cust = list(dict(sorted(service_diff.items(), key=itemgetter(1))[:self.get_operator_scale(state) - 1]).keys()) 
        customers_to_remove = nearest_cust + [cust]

        self.remove_customers(state, customers_to_remove)
        self.post_destroy_operator(state)

        return state 



# ---------------------------------------------
# Non-generic Operators - Greedy-based
# ---------------------------------------------

class RouteDeviateDestroy(Operator):
    name = "D-RouteDeviate"

    ''' Inspired by Demir 2012 Neighbourhood Removal: greedy-based
        removes a set of nodes from routes which distances deviate the most from the average route length. '''
    
    def apply_op(self, state, **kwargs):
        ## sort the list of tours wrt deviation difference from the average in descending order
        deviated_routes = sort_deviated_route_avg(state, self.name)
        
        ## for each route, sequentially remove nodes using their indice (greedy):
        self.deviated_routes_node_greedy_removal(state, deviated_routes)

        self.post_destroy_operator(state)
        return state


class GreedyRouteDestroyMax(Operator):
    name = "D-MaxRoute"

    ''' Inspired by Demir 2012: greedy-based
        Removes the worst route with respect to parametrized criteria (longest/shortest distance, heaviest/lightest carrying weight, etc).  '''

    def apply_op(self, state, **kwargs):
        ## sort the list of tours wrt total length in descending order:
        deviated_routes = sort_deviated_route(state, self.name)
        
        ## for each route, sequentially remove nodes using their indice
        self.deviated_routes_node_sequantial_removal(state, deviated_routes)
        
        self.post_destroy_operator(state)
        return state


class GreedyRouteDestroyMin(Operator):
    name = "D-MinRoute"

    ''' Inspired by Demir 2012: greedy-based
        Removes the worst route with respect to parametrized criteria (longest/shortest distance, heaviest/lightest carrying weight, etc).  '''

    def apply_op(self, state, **kwargs):
        ## sort the list of tours wrt total length in descending order:
        deviated_routes = sort_deviated_route(state, self.name)

        ## for each route, sequentially remove nodes using their indice
        self.deviated_routes_node_sequantial_removal(state, deviated_routes)

        self.post_destroy_operator(state)
        return state


class CapacityDeviateDestroy(Operator):
    name = "D-CapacityDeviate"

    ''' Variation of RouteDeviateDestroy: greedy-based, utility-based
        removes a set of nodes from routes which aggregated demand deviate the most from their associated vehicle capacity. '''

    def apply_op(self, state, **kwargs):
        
        ## sort the list of tours wrt deviation difference from the average in descending order - Two options: 
        
        #deviated_routes = sort_deviated_route_avg(state, self.name)
        deviated_routes = sort_deviated_route_capacity(state, self.name)

        ## for each route, sequentially remove nodes using their indice (greedy):
        self.deviated_routes_node_greedy_removal(state, deviated_routes)

        self.post_destroy_operator(state)
        return state
    

class GreedyCapacityDestroyMax(Operator):
    name = "D-MaxCapacityRoute"

    ''' Inspired by Demir 2012: greedy-based
        Removes the worst route with respect to parametrized criteria (longest/shortest distance, heaviest/lightest carrying weight, etc).  '''

    def apply_op(self, state, **kwargs):
        ## sort the list of tours wrt total length in descending order:
        deviated_routes = sort_deviated_route(state, self.name)
        
        #print(f"route ranking = {deviated_routes}")

        ## for each route, sequentially remove nodes using their indice (greedy):
        self.deviated_routes_node_greedy_removal(state, deviated_routes)

        self.post_destroy_operator(state)
        return state


class GreedyCapacityDestroyMin(Operator):
    name = "D-MinCapacityRoute"

    ''' Inspired by Demir 2012: greedy-based
        Removes the worst route with respect to parametrized criteria (longest/shortest distance, heaviest/lightest carrying weight, etc).  '''

    def apply_op(self, state, **kwargs):
        ## sort the list of tours wrt total length in descending order:
        deviated_routes = sort_deviated_route(state, self.name)
        
        ## for each route, sequentially remove nodes using their indice (greedy):
        self.deviated_routes_node_greedy_removal(state, deviated_routes)

        self.post_destroy_operator(state)
        return state


class HistoricalKnowledgeDestroy(Operator):
    name = "D-HistKnowledge"

    ''' Greedy-based: (from Historical knowledge node removal)
        Look at the historical records, removes a set of nodes with the highest deviations from their historical records of distances from their preceding and following nodes; 
        First iteration define all nodes removal gain = inf, every iteration update and record the best (smallest) removal gain, select nodes with largest deviations to remove. 
    '''
    
    def apply_op(self, state, **kwargs):
        
        ## Compute every node's removal gain for this ALNS solution:
        removal_gain_dict = compute_removal_gain(state, state.alns_list)

        ## Update the smallest removal gain global dictionary for every node:
        for c in state.customers:
            state.smallest_removal_gain[c] = min(state.smallest_removal_gain[c], removal_gain_dict[c])

        ## Select removed customers with largest removal gain from the history:
        if state.first_time_employ == True: 
            # Separate first iteration from others, since the deviations from float(inf) is still inf.
            state.first_time_employ = False
            customers_to_remove = sorted(state.smallest_removal_gain, key=removal_gain_dict.get, reverse=True)[:self.get_operator_scale(state)]
        else:
            # Record each node's removal gain deviations = (current round removal gain) - (historical smallest removal gain)
            deviation_dict = {c: removal_gain_dict[c] - state.smallest_removal_gain[c] for c in state.customers}
            customers_to_remove = sorted(deviation_dict, key=removal_gain_dict.get, reverse=True)[:self.get_operator_scale(state)]

        self.remove_customers(state, customers_to_remove)
        self.post_destroy_operator(state)
        return state


class SameFacilityDestroy(Operator):
    name = "D-SameFacility"

    ''' Greedy-based: 
        Remove customers with the largest removal gain from the same facility. 
    '''

    def apply_op(self, state, **kwargs):

        # find the smallest facility (easier to shut down):
        frequency_dict = {v: list(state.customer_depot_allocation.values()).count(v) for v in set(state.customer_depot_allocation.values())}
        smallest_depot = min(frequency_dict, key=frequency_dict.get)

        # find the set of customers assigned to this facility: 
        customers_assigned_to_depot = [k for k, v in state.customer_depot_allocation.items() if v == smallest_depot]
        routes_assigned_to_depot = [tour for tour in state.alns_list if any(customer in customers_assigned_to_depot for customer in tour)]
        
        # find the q customers with the highest removal gain and remove them:
        self.deviated_routes_node_greedy_removal(state, routes_assigned_to_depot)
        
        # find the q customers with the highest removal gain and remove them:
        # removal_gain_dict = compute_removal_gain(state, routes_assigned_to_depot)
        # customers_to_remove = sorted(removal_gain_dict, key=removal_gain_dict.get, reverse=True)
        # if len(customers_to_remove) > self.get_operator_scale(state):
        #     customers_to_remove = customers_to_remove[:self.get_operator_scale(state)]
        # self.remove_customers(state, customers_to_remove)

        self.post_destroy_operator(state)
        return state



class GreedyTimeDestroy(Operator):
    name = "D-GreedyTime"

    ''' remove customers with the most different/violated starting time windows computed from the trip '''
    
    def apply_op(self, state, **kwargs):

        time_window_violation = generate_time_window_violation(state, removed=False)

        # find the q customers with most similar proximity (lower the better): 
        customers_to_remove = list(dict(sorted(time_window_violation.items(), key=itemgetter(1), reverse=True)[:self.get_operator_scale(state)]).keys()) 

        self.remove_customers(state, customers_to_remove)
        self.post_destroy_operator(state)

        return state 




# ---------------------------------------------
# Non-generic Operators - Random-based
# ---------------------------------------------

class RandomRouteDestroy(Operator):
    name = "D-RandomRoute"

    ''' Inspired by Demir 2012 Route Removal: random-based
        Randomly removes a complete route from the solution '''

    def apply_op(self, state, **kwargs):
        customers_to_remove = []
        total_removed_number = len(customers_to_remove)

        while True:
            selected_route = self.local_random.sample(state.alns_list, 1)[0]

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

        self.post_destroy_operator(state)
        return state


class ForbiddenRandomDestroy(Operator):
    name = "D-ForbiddenRandom"

    ''' Randomly removes a set of requests that weren’t removed for more than certain number of times in the previous iterations
    '''

    def apply_op(self, state, **kwargs):
        ## Select those customers that are not removed for a certain number of times:
        customers = list(dict((k, v) for k, v in state.non_removal_count.items() if v >= state.inst.forbidden_random_num_of_iterations).keys())

        ## If total removal not reaching quota, random select the remaining removals:
        if len(customers) >= self.get_operator_scale(state):
            customers_to_remove = self.local_random.sample(customers, self.get_operator_scale(state))
        else:
            all_available_customers = self.get_available_customers(state)
            remaining_quota = self.get_operator_scale(state) - len(customers)
            customers_to_remove = customers + self.local_random.sample([i for i in all_available_customers if i not in customers], remaining_quota)

        ## Update the forbidden list for non-removed nodes:
        for c in [i for i in state.customers if i not in customers_to_remove]:
            state.non_removal_count[c] += 1

        ## Update forbidden list for removed nodes:
        for c in customers_to_remove:
            state.non_removal_count[c] = 0

        self.remove_customers(state, customers_to_remove)
        self.post_destroy_operator(state)
        return state


class FacilityCloseDestroy(Operator):
    name = "D-FacilityClose"

    ''' Randomly choose an open facility and close it. Move all the allocated customesr to the customer pool for reinsertion. 
            Note: apply this only when the combined facility capacity is large enough to hold all customers! 
                  also need to bypass destroy size as the removal may not follow destroy scale! '''

    def apply_op(self, state, **kwargs):
        
        opened_depot_list = list({v for v in state.customer_depot_allocation.values()})
        self.local_random.shuffle(opened_depot_list)
        
        # randomly select an opened depot to close: 
        for depot in opened_depot_list: 
            remaining_opened_facilities = list(set(opened_depot_list) - {depot})
            remaining_capacity = sum([state.inst.facility_capacity[f] for f in remaining_opened_facilities])
            
            # can only close this facility if all customer demand can be served by existing open facilities:
            if sum(state.inst.demand) <= remaining_capacity: 
                customers_to_remove = [k for k, v in state.customer_depot_allocation.items() if v == depot]
                self.remove_customers(state, customers_to_remove)
                self.post_destroy_operator(state)
                return state
            else: 
                return state 
        
        
class FacilityOpenDestroy(Operator):
    name = "D-FacilityOpen"

    ''' randomly choose a closed facility and open it. Generate tour(s) using the q closest customers.
        Note: we don't use repair oeprators to insert the customers back to tours since the other tours can begin from 
        an existing opened depot not the newly opened one.  '''

    def remove_customers_lrp_version(self, state, customers_to_remove, depot_to_open):

        # remove customers from alns_list: 
        for tour_index, tour in enumerate(deepcopy(state.alns_list)):    
            state.alns_list[tour_index][:] = [i for i in tour if i not in customers_to_remove][:]
        state.alns_list = [tour for tour in state.alns_list if tour != []]
        
        # reset new depot status, reassign the removed customer to depot allocation:
        state.node_status[depot_to_open] = True
        for c in customers_to_remove:
            state.customer_depot_allocation[c] = depot_to_open


    def apply_op(self, state, **kwargs):

        close_depot_list = list(set(state.depot) - set(state.customer_depot_allocation.values()))

        if close_depot_list == []:
            return state
        else:
            # randomly open a closed depot
            depot_to_open = self.local_random.choice(close_depot_list)

            # find the q nearest customers and remove them from alns_list, reset their node status: 
            distance = {c: state.distance_matrix[(depot_to_open, c)] for c in state.customers}
            customers_to_remove = sorted(distance, key=distance.get)[:self.get_operator_scale(state)]
            
            if sum([state.inst.demand[c] for c in customers_to_remove]) <= state.inst.facility_capacity[depot_to_open]:
                # if facility capacity is satisfied, remove those customers from the original ALNS: 
                self.remove_customers_lrp_version(state, customers_to_remove, depot_to_open)
                
                # there is a chance that another depot lost all its allocated customers, so need to update depot status:
                self.update_facility_opening_status(state)
                
                # apply clarks and wright heuristic to form new tours starting from this new depot: 
                tours = clark_wright_algorithm(customers_to_remove, depot_to_open, state.distance_matrix, state.inst)
                
                state.alns_list += tours
                
            self.post_destroy_operator(state)

        return state





# are state.customer_depot_allocation and depot's node_status (for both customers and depots) updated after each destroy? 
