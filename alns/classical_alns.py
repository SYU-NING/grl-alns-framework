import math
import warnings
from copy import deepcopy
import pandas as pd

from alns.rl.dqn.dqn_agent import BaselineDQNAgent, ProbabilisticBaselineDQNAgent
from environment.objective_functions import ObjectiveFunction
from operators.functions import *


class ClassicalALNS(object):
    '''
    Methods input:
        ALNS_List     = all customers visiting sequences from the single depot, e.g. [[1,2],[3]] records two tours "0->1->2->0" and "0->3->0"
        Customer_Pool = initial list of q customers that have been removed from the complete visiting plan
        destroy_scale = destroy phase's destroy scale, how many nodes to be removed by a destroy operator
        Feature_Vector= list consists of all node-based features/attributes for every customer nodes (also the depot)
    Methods output:
        ALNS_List     = visiting sequences after deconstruction with incomplete visiting plan 
        Customer_Pool = updated list of q+B_d customers that have been removed from the complete visiting plan
        Feature_Vector= updated list consists of all node-based features/attributes for every customer nodes (also the depot)
    '''

    def __init__(self, op_selection_agent, random_seed, use_localsearch=False):
        
        self.op_selection_agent = op_selection_agent

        self.use_localsearch = use_localsearch

        # Random seed settings: 
        self.random_seed = random_seed
        self.local_random = random.Random()
        self.local_random.seed(self.random_seed)

        # Classical ALNS parameters:
        self.counter = 0                # counting the current total iterations (outter + inner)
        self.operator_counter = 0       # to track the rows of alns_record dataframe, since each iteration could give 2 or 3 rows depending on local search
        self.Lambda = 0.3               # only proceed good candidate with solution value within (1+Lambda)% to local search stage
        self.nonchange_iteration = 200  # for the stopping criteria, determine whether the current best solution is

        # Simulated Annealing parameters:
        self.cooling_coef = 0.998       # cooling coefficient, cool down temprature by this rate after each segment
        self.initial_t = 100            # initial tempreture. Greater the value, more likely the solution gonna get accepted (greater the likelkhood of moving araound the search space)

        #RW weight segment parameter:
        self.weight_segment = self.op_selection_agent.weight_segment if hasattr("op_selection_agent", "weight_segment") else 200



    def setup_alns(self, starting_state, outer_its_per_customer, inner_its):
        self.outer_iteration = int(outer_its_per_customer)    # total outter iterations, should be changed into a while loop
        # self.outer_iteration = int(starting_state.inst.cust_number * outer_its_per_customer)    # total outter iterations, should be changed into a while loop
        self.inner_iteration = inner_its    # total inner iterations - a segment, SA tempreture changed

        self.current_t = self.initial_t     # the initial value of the current tempreture should be the same as the initial_t
        self.best_obj_record = []           # store only best solution after a segment. Format = (ALNS.counter, best_obj)
        self.best_obj_record_frequency = []           
        
        # Compute initial objective from initial graph (or first improvement too great if define best_obj = inf): 
        initial_objective = ObjectiveFunction.compute_obj_function_values(starting_state)['total_objective'] + 1
        self.new_obj = initial_objective + 100.
        self.best_obj = initial_objective + 100.
        self.previous_segment_best_obj = initial_objective + 100.


        self.alns_record = pd.DataFrame(columns=["total_objective",
                                                 "traversal_cost",
                                                 "hiring_cost",
                                                 "nonservice_penalty",
                                                 "selected_operator",
                                                 "ALNS_list_storage"])


    def simulated_annealing_acceptor(self, new_obj):
        ''' Prob(accept new solution) = exp( -(new - old)/current_tempreture )
                * when new <= old: P >= 1, acceptance rate 100% 
                * when new > old, new not as good as old, accept only with probability P, where 0 < P < 1. 
                * higher temprature, larger P
        '''
        #if the current obj value is better(smaller), accept it

        if new_obj < self.best_obj:
            accept_status = "accepted"                                     # accept the better solution new_obj
            credit_status = "better_accept"
        else: # if not, accept it with a probability 
            p = math.exp(- (new_obj - self.best_obj) /self.current_t)
            if self.local_random.random() < p:                                        # make a decision to accept the worse solution or not
                accept_status = "accepted"                                 # accept the worse solution new_obj
                credit_status = "worse_accept"   
            else: 
                accept_status = "not_accepted"                             # refuse the worse solution new_obj
                credit_status = "worse_refuse"          

        return credit_status



    def update_alns_record(self, state, operator, state_obj_value):
        record = deepcopy(state_obj_value)
        record["selected_operator"] = str(operator)
        record["ALNS_list_storage"] = deepcopy(state.alns_list)

        if state.inst.problem_variant == "LRP":
            record["cust_depot_allocation"] = deepcopy(state.customer_depot_allocation)
        
        elif state.inst.problem_variant == "HVRP":
            record["cust_vehicle_type"] = deepcopy(state.cust_vehicle_type)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.alns_record = self.alns_record.append(record, ignore_index=True)


    
    def findout_which_experiment_setting(self, starting_state):
        ''' This function is under classical RW vs victor RW comparison experiment, 
            Aim to test how much difference can be caused by the three changes applied from CRW -> VRW 
        '''
        if starting_state.vrw_experiment_setting == 'o':
            experiment1 = 'classical_discrete'
            experiment2 = 'classical_reward_per_pair'
        elif starting_state.vrw_experiment_setting == 'a':
            experiment1 = 'continuous'
            experiment2 = 'classical_reward_per_pair'
        elif starting_state.vrw_experiment_setting == 'b':
            experiment1 = 'classical_discrete'
            experiment2 = 'reward_per_segment_discrete_score'
        elif starting_state.vrw_experiment_setting == 'c':
            experiment1 = 'classical_discrete'
            experiment2 = 'classical_reward_per_pair'
        elif starting_state.vrw_experiment_setting == 'ab':
            experiment1 = 'continuous'
            experiment2 = 'reward_per_segment_best'
        elif starting_state.vrw_experiment_setting == 'abc':
            experiment1 = 'continuous'
            experiment2 = 'reward_per_segment_best'
        else: 
            print('Error - Not part of the experiment!')
        #   print('setting-1 = {}, setting-2 = {}'.format(experiment1, experiment2))
        return experiment1, experiment2



    def run_master_loop(self, starting_state, outer_its_per_customer, inner_its):
        
        self.setup_alns(starting_state, outer_its_per_customer, inner_its)
        state = starting_state

        # Test CRW vs VRW, aim to prive that those adjustments to fit MDP framework wouldnt drag down RW performance
        experiment1, experiment2 = self.findout_which_experiment_setting(starting_state)

        self.op_selection_agent.inform_alns_starting(
            total_steps=(self.outer_iteration * self.inner_iteration * 2),
            starting_state=starting_state,
            starting_obj_value=self.new_obj
        )

        is_baseline_dqn = issubclass(type(self.op_selection_agent), BaselineDQNAgent) or issubclass(type(self.op_selection_agent), ProbabilisticBaselineDQNAgent)

        for outer_iter in range(self.outer_iteration):      #before total time runs out!
            for inner_iter in range(self.inner_iteration):  #before changing the cooling factor!!
                # print(f"executing step {self.operator_counter}")

                ###(1) Choose the operators using a Roulette Wheel mechanism selection based on past success:
                destroy_op = self.op_selection_agent.select_destroy_operator(state, search_step=self.operator_counter)      # return the operator class!
                repair_op = self.op_selection_agent.select_repair_operator(state, search_step=self.operator_counter+1)
                localsearch_op = self.op_selection_agent.operator_library.get_available_local_search_ops()[0]

                # print()
                # print(f"round {self.counter} ---------------")
                # print(f"operator pair = {destroy_op} & {repair_op}")
            
                ###(2) Apply (destroy, repair) pair. If solution is promising, apply Local Search to further improve:
                # (2a) apply Destroy Operator (directly on the corresponding operator class, weeee!):
                destroy_op.apply_op(state)
                state_obj_value = ObjectiveFunction.compute_obj_function_values(state)
                self.update_alns_record(state, destroy_op, state_obj_value)

                if is_baseline_dqn:
                    self.op_selection_agent.after_operator_applied(destroy_op, t=self.operator_counter, state=state, obj_value=state_obj_value["total_objective"])

                
                # (2b) apply Repair Operator:
                repair_op.apply_op(state)
                state_obj_value = ObjectiveFunction.compute_obj_function_values(state)
                self.update_alns_record(state, repair_op, state_obj_value)

                if is_baseline_dqn:
                    self.op_selection_agent.after_operator_applied(repair_op, t=self.operator_counter + 1, state=state, obj_value=state_obj_value["total_objective"])

                # print()
                # print(f"{state.alns_list}, len(state.alns_list)={len(state.alns_list)}")
                # print(f"customer-depot = {state.customer_depot_allocation}")
                # print(f"state.node_status = {state.node_status}")
                # print(f"tour type = {state.cust_vehicle_type}")
                # print(f"state.customer_pool={state.customer_pool}")
                # print()

                # (2c) apply Local Search only if the current post-repair solution is good enough:
                if self.use_localsearch:
                    obj_before_localsearch = self.alns_record.iloc[-1].total_objective
                    if obj_before_localsearch <= (1 + self.Lambda) * self.best_obj:
                        localsearch_op.apply_op(state)
                        state_obj_value = ObjectiveFunction.compute_obj_function_values(state)
                        self.update_alns_record(state, localsearch_op, state_obj_value)


                self.counter += 1  # track how many rounds of destroy-repair have been applied
                self.operator_counter += 2 if self.use_localsearch == False else 3  # track how many operators have been applied

                ###(3) Simulated Annealing Acceptor:
                self.new_obj = self.alns_record.iloc[-1].total_objective         # fetch the newly repaired solution
                credit_status = self.simulated_annealing_acceptor(self.new_obj)  # check if the new solution is better/worse, also if accepted or not

                ###(4) Adaptive weight adjustment for Roulette Wheel:
                if not is_baseline_dqn:
                    self.op_selection_agent.after_operator_applied(destroy_op,
                                                                   counter=self.counter,
                                                                   credit_status=credit_status,
                                                                   new_obj=self.new_obj,
                                                                   best_obj=self.best_obj,
                                                                   previous_segment_best_obj=self.previous_segment_best_obj,
                                                                   experiment1=experiment1,
                                                                   experiment2=experiment2)
                    self.op_selection_agent.after_operator_applied(repair_op,
                                                                   counter=self.counter,
                                                                   credit_status=credit_status,
                                                                   new_obj=self.new_obj,
                                                                   best_obj=self.best_obj,
                                                                   previous_segment_best_obj=self.previous_segment_best_obj,
                                                                   experiment1=experiment1,
                                                                   experiment2=experiment2)

                ## Update best solution record:
                if credit_status == "better_accept" or self.counter == 1:   # update the current record if a new solution is accepted or if the first iteration gives worse solution than initial alns_list
                    self.best_obj = deepcopy(self.new_obj)                  # since new solution is accepted and is better, update the record
                    self.best_obj_record_frequency.append([self.operator_counter, self.best_obj, 0])   # record best found solution from this segment, which might be better or worse
                else:
                    self.best_obj_record_frequency[-1][-1] += 1             # count the number of times 
                
                self.best_obj_record.append([self.counter, self.best_obj])

                if self.counter % self.weight_segment == 0:
                    self.previous_segment_best_obj = deepcopy(self.best_obj)

                
            ###(5) Update information after a segment is finished # Shunee: doublecheck if the N-iterations are for SA and RW sements
            self.current_t *= self.cooling_coef                      # update temperature for Simulated Annealing

            ###(6) Stopping Criteria: if the best solution is unchanged within consecutive N iterations 
            if self.best_obj_record_frequency[-1][-1] >= self.nonchange_iteration:   # if the best found solution remains unchanged for a while, stop the search, break the loop, horray! 
                #print(self.best_obj_record_frequency[-1][-1])
                #print("Terminate Reason 1: best solution unchanged within consecutive ", self.nonchange_iteration, " iterations.")
                break    

            if outer_iter == self.outer_iteration - 1:
                #print("Terminate Reason 2: outer iteration reached.")
                break

