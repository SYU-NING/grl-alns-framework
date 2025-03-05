from operators.base_op import Operator
from operators.functions import *


class RandomRepair(Operator):
    name = "R-Random"

    ''' Insert a single node back to the tours into a random position (that not necessarily improve the solution compared to before) '''

    def apply_op(self, state, **kwargs):
        insert_list = self.select_customers_from_pool(state)
        self.insert_customers_to_random_position(state, insert_list)
        self.post_repair_operator(state)
        return state


class GreedyRepair(Operator):
    name = "R-Greedy"

    ''' Insert a single node back to the tours into the best position (geographical distance, service time...) '''

    def apply_op(self, state, **kwargs):
        insert_list = self.select_customers_from_pool(state)
        self.insert_customers_to_best_position(state, insert_list)
        self.post_repair_operator(state)
        return state


class GreedyRepairPerturbation(Operator):
    name = "R-GreedyNoise"

    ''' Insert a single node back to the tours into the best position computed with a perturbation factor. 
        The perturbation step involves adding some random noise to the insertion of elements, in order to diversify the search and avoid getting stuck in local optima. '''

    def noise_value(self, distance_matrix):
        perturbated_distance_matrix = deepcopy(distance_matrix)
        for key in distance_matrix:
            perturbated_distance_matrix[key] = distance_matrix[key] * self.local_random.uniform(0.8, 1.2)
        return perturbated_distance_matrix

    def apply_op(self, state, **kwargs):
        insert_list = self.select_customers_from_pool(state)
        perturbated_distance_matrix = self.noise_value(state.distance_matrix)
        self.insert_customers_to_best_position(state, insert_list, perturbated_distance_matrix=perturbated_distance_matrix)
        self.post_repair_operator(state)
        return state


class DeepGreedyRepair(Operator):
    name = "R-GreedyDeep"

    ''' Instead of inserting the first customer i from D into the current solution like the greedy insertion, 
        it inserts the customer i having the minimum global cost position. '''

    def apply_op(self, state, **kwargs):
        insert_list = self.select_customers_from_pool(state)

        # Reorder the insertion list to be from smallest insertion cost to largest:
        resorted_insert_list = self.sort_customers_by_minimum_global_insertion(state, insert_list)

        self.insert_customers_to_best_position(state, resorted_insert_list)
        self.post_repair_operator(state)
        return state

class Regret2Repair(Operator):
    name = 'R-Regret2'
    default_k_value = 2

    ''' Nodes are re-inserted back to their best position in the order of maximal regret value, which is the 
        cost difference between best and k-th best insertions. Customers with a high regret value should be 
        inserted first. '''
    
    def apply_op(self, state, **kwargs):
        k_value = kwargs.get('k_value', self.default_k_value)

        insert_list = self.select_customers_from_pool(state)

        # compute the regret value for each selected node
        regret = [(cust, compute_regret_value(state, k_value, cust)) for cust in insert_list]

        # rank nodes according to their regret values
        ordered_insert_list = sorted(regret, key = lambda x: x[1], reverse=True)
        customers_to_insert = [c[0] for c in ordered_insert_list]

        self.insert_customers_to_best_position(state, customers_to_insert)
        self.post_repair_operator(state)
        return state


class Regret3Repair(Operator):
    name = 'R-Regret3'
    default_k_value = 3
    
    def apply_op(self, state, **kwargs):
        k_value = kwargs.get('k_value', self.default_k_value)

        insert_list = self.select_customers_from_pool(state)
        
        regret = [(cust, compute_regret_value(state, k_value, cust)) for cust in insert_list]
        ordered_insert_list = sorted(regret, key = lambda x: x[1], reverse=True)
        customers_to_insert = [c[0] for c in ordered_insert_list]
 
        self.insert_customers_to_best_position(state, customers_to_insert)
        self.post_repair_operator(state)
        return state


class Regret4Repair(Operator):
   name = 'R-Regret4'
   default_k_value = 4
    
   def apply_op(self, state, **kwargs):
        k_value = kwargs.get('k_value', self.default_k_value)

        insert_list = self.select_customers_from_pool(state)
        regret = [(cust, compute_regret_value(state, k_value, cust)) for cust in insert_list]
        ordered_insert_list = sorted(regret, key = lambda x: x[1], reverse=True)
        
        customers_to_insert = [c[0] for c in ordered_insert_list]
 
        self.insert_customers_to_best_position(state, customers_to_insert)
        self.post_repair_operator(state)
        return state