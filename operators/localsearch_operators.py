from operators.base_op import Operator
from operators.functions import *

from copy import deepcopy


class TwoOptLocalSearch(Operator):
    name = "L-TwoOpt"

    def apply_op(self, state, **kwargs):
        ''' Exchange two arcs inside a tour, and check if the solution is better off.
            A complete 2-opt local search will compare every possible valid combination of the swapping mechanism.
            Website: https://en.wikipedia.org/wiki/2-opt
        '''
        for tour in deepcopy(state.alns_list): 
            need_improvement = True
            best_tour = state.depot + tour + state.depot    # complete the tour with depot
            iteration = 0

            while need_improvement and (iteration < 500):   # Apply 2-opt on each tour until no improvement can be made:
                new_tour = self.two_opt_single_tour(best_tour, state)
                if new_tour == best_tour:       # no better tour found
                    need_improvement = False
                else:                           # still needs improvement
                    need_improvement = True
                    best_tour = new_tour[:]
            
            state.alns_list.remove(tour)        # remove old tour
            state.alns_list.append([x for x in best_tour if x not in state.depot])   # add the two-opt improved tour
        return state


class SwapLocalSearch(Operator):
    name = "L-Swap"

    def apply_op(self, state, **kwargs):
        ''' Apply swap until no improvement found in the current ALNS list
        '''
        for tour in deepcopy(state.alns_list):  # requires deepcopy or alns_list will be altered during for-loop
            current_tour = state.depot + tour + state.depot
            need_improvement = True
            iteration = 0

            while need_improvement and (iteration < 500):
                need_improvement = False
                iteration += 1
                found_improving_solution, new_tour = self.find_first_improving_solution(current_tour, state)
                if found_improving_solution:
                    need_improvement = found_improving_solution
                    current_tour = new_tour

            state.alns_list.remove(tour)
            state.alns_list.append([x for x in current_tour if x not in state.depot])

        return state



        

