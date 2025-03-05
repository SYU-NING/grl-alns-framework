from abc import ABC

class ObjectiveFunction:

    def compute(self, state):
        return self.compute_obj_function_values(state)[self.name]


    @staticmethod
    def compute_obj_function_values(state):
        ''' Returns the total objective function for a ALNS solution '''
        objective = {}
        total_travel_cost = 0
        total_hiring_cost = 0
        total_exceed_time = 0
        facility_opening_cost = 0

        ## (1) Total travel cost:
        for tour in state.alns_list:
            full_tour = state.depot + tour + state.depot
            for i in range(len(full_tour) - 1):
                node1, node2 = full_tour[i], full_tour[i+1]
                total_travel_cost += state.distance_matrix[(node1, node2)]


        ## (2) Total vehicle hiring cost:
        if state.inst.problem_variant == "HVRP":
            current_fleetsize = [list(set([state.cust_vehicle_type[c] for c in tour]))[0] for tour in state.alns_list]
            total_hiring_cost = sum(state.inst.vehicle_hiring_cost[state.inst.vehicle_capacity.index(tour_capacity)] 
                                    for tour_capacity in current_fleetsize)
        else:
            total_hiring_cost = len(state.alns_list) * state.inst.vehicle_hiring_cost
        

        ## (3) Facility opening cost: 
        if state.inst.problem_variant == "LRP":
            opened_facilities = set(d for d in state.customer_depot_allocation.values() if d != -1)
            facility_opening_cost += sum([state.inst.facility_opening_cost[facility] for facility in opened_facilities]) 


        ## (4) Customer nonservice penalty:
        nonserve = len(state.customer_pool) * state.inst.nonserve_penalty
        
        
        ## (5) Total time window violations:
        if state.inst.problem_variant == "VRPTW":  
            # total_exceed_time += sum(list(state.time_window_violation.values()))
            total_exceed_time += sum([v for v in state.time_window_violation.values() if isinstance(v, (int, float))])


        ## Objective Value: 
        objective[TraversalCost.name] = total_travel_cost
        objective[HiringCost.name] = total_hiring_cost
        objective[NonservicePenalty.name] = nonserve
        objective[TotalObjective.name] = total_travel_cost + total_hiring_cost + nonserve
     
        if state.inst.problem_variant == "VRPTW":
            tw_violation_cost = total_exceed_time * state.inst.exceed_tw_penalty
            objective[TimeWindowExceedPenalty.name] = tw_violation_cost
            objective[TotalObjective.name] += tw_violation_cost
        elif state.inst.problem_variant == "LRP":
            objective[FacilityCost.name] = facility_opening_cost
            objective[TotalObjective.name] += facility_opening_cost

        return objective


class TraversalCost(ObjectiveFunction):
    name = "traversal_cost"


class HiringCost(ObjectiveFunction):
    name = "hiring_cost"


class NonservicePenalty(ObjectiveFunction):
    name = "nonservice_penalty"


class TotalObjective(ObjectiveFunction):
    name = "total_objective"

class TimeWindowExceedPenalty(ObjectiveFunction):
    name = "timewindow_penalty"

class FacilityCost(ObjectiveFunction):
    name = "facility_opening_cost"



def compute_cust_time_window_violation(state, cust, arrival_time):
    ''' wish the arrival time t lay within time window [a,b] for the Solomon instance
        a < t < b        -->  no violation, return (t - a)
        t < a or t > b   -->  violation = (a - t) * penalty
    '''
    time_window_start, time_window_end = state.inst.tw_start[cust], state.inst.tw_end[cust]

    if arrival_time < time_window_start:
        return time_window_start - arrival_time
    elif arrival_time > time_window_end:
        return arrival_time - time_window_end
    else:  
        return 0  
    

def compute_tour_time_window_violation(state, tour, distance_matrix):
    ''' for each customer within a tour, compute the cumulated time violation for all customers 
        with their arrival time t lay within time window [a,b] for the Solomon instance.
            a < t < b        -->  no violation, return (t - a)
            t < a or t > b   -->  violation = (a - t) * penalty
    '''
    arrival_time = 0
    violated_time = 0
    for previous_cust, cust in zip(state.depot + tour[:-1], tour):
        arrival_time += state.inst.service_time[previous_cust] + distance_matrix[(previous_cust, cust)]
        time_window_start, time_window_end = state.inst.tw_start[cust], state.inst.tw_end[cust]
        if arrival_time < time_window_start:
            violated_time += (time_window_start - arrival_time)
        elif arrival_time > time_window_end:
            violated_time += (arrival_time - time_window_end)

    return violated_time