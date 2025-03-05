import random
import torch
import numpy as np
from copy import deepcopy

from operators.localsearch_operators import *


class OperatorSelector(object):
    is_mdp_based = False

    '''base vanilla version of the selector mechanism '''

    def __init__(self, operator_library, random_seed, **kwargs):
        self.operator_library = operator_library
        self.set_random_seeds(random_seed)


    def set_random_seeds(self, random_seed):
        '''Ramdom Seed "Generator" '''
        self.random_seed = random_seed
        self.local_random = random.Random()
        self.local_random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)

        np.seterr('raise')

    def select_destroy_operator(self, state, **kwargs):
        pass

    def select_repair_operator(self, state, **kwargs):
        pass

    def after_operator_applied(self, operator, **kwargs):
        pass

    def inform_alns_starting(self, **kwargs):
        pass


class RandomOperatorSelector(OperatorSelector):
    algorithm_name = "random"
    is_trainable = False
    requires_hyperopt = False
    requires_tune = False


    def __init__(self, operator_library, random_seed, **kwargs):
        super().__init__(operator_library, random_seed, **kwargs)

        self.weight_segment = 15
        self.reaction_factor = 0.75

    def select_destroy_operator(self, state, **kwargs):
        all_ops = self.operator_library.get_available_destroy_ops()
        return self.local_random.choice(all_ops)

    def select_repair_operator(self, state, **kwargs):
        all_ops = self.operator_library.get_available_repair_ops()
        return self.local_random.choice(all_ops)

    def after_operator_applied(self, operator, **kwargs):
        # do nothing ¯\_(ツ)_/¯
        pass


class RouletteWheelOperatorSelector(OperatorSelector):
    algorithm_name = "classicrw"
    is_trainable = False
    requires_hyperopt = False
    requires_tune = True

    def __init__(self, operator_library, random_seed, **kwargs):
        super().__init__(operator_library, random_seed, **kwargs)


        self.RW_scoring = {'better_accept': 4,     # operator's credit score +4 if algorithm accepts a better solution generated
                           'worse_accept':  2,     # add smaller credit score if algorithm accepts a worse solution generated
                           'worse_refuse':  0}     # no credit score for rejecting a generated solution which is worse

        # performances
        all_destroy = [str(op) for op in self.operator_library.get_available_destroy_ops()]
        all_repair = [str(op) for op in self.operator_library.get_available_repair_ops()]
        
        self.Score_d = dict(zip(all_destroy, [[0, 0, 1] for x in range(len(all_destroy))]))
        self.Score_r = dict(zip(all_repair, [[0, 0, 1] for x in range(len(all_repair))]))


    def select_operator_by_score(self, score_to_use):
        '''Select a random destroy/repair operator based on their associated probabilities '''
        performance_score = deepcopy(score_to_use)
        
        total_score = sum([X[2] for X in list(performance_score.values())])

        try:
            operator_probabilities = [x / total_score for x in np.cumsum([X[2] for X in list(performance_score.values())])]
        except FloatingPointError:
            # self.logger.info(f"caught FP error in RW-MDP. continuing with uniform probs.")
            # self.logger.info(f"performance scores were: {performance_score}")
            operator_probabilities = [(i + 1) / len(performance_score) for i in range(len(performance_score))]

        operator_probabilities.insert(0, 0)
        random_prob = self.local_random.random()
        for i in range(len(operator_probabilities) - 1):
            if operator_probabilities[i] < random_prob and random_prob <= operator_probabilities[i + 1]:
                operator_index = i + 1

        selected_op_str = list(score_to_use.keys())[operator_index - 1]
        return selected_op_str


    def select_destroy_operator(self, state, **kwargs):
        ## List out all destroy operator options:
        score_to_use = self.Score_d
        
        ## Select a destroy operator: 
        selected_op_str = self.select_operator_by_score(score_to_use)

        ## Make sure that the selected destroy scale matches the repair scale later: 
        # state.match_destroy_scale = int(selected_op_str[-2:].replace('-', ''))
        
        return self.operator_library.get_destroy_op(selected_op_str)  # use operator name in string format to get corresponding operator class


    def select_repair_operator(self, state, **kwargs):
        ## List out all repair operators (with scale matching the destroy scale before): 
        # score_to_use = {key: self.Score_r[key] for key in self.Score_r.keys() if int(key[-2:].replace('-',''))==state.match_destroy_scale}
        score_to_use = self.Score_r
        
        ## Select a repair operator: 
        selected_op_str = self.select_operator_by_score(score_to_use)

    #    print(selected_op_str, type(selected_op_str))
    #    print(self.operator_library.get_repair_op(selected_op_str), type(self.operator_library.get_repair_op(selected_op_str)))
        return self.operator_library.get_repair_op(selected_op_str)


    def credit_score_compute(self, credit_status, method, **kwargs):
        ''' * method 1: classical RW with discrete scores
                The added score is pre-fixed (0,2,4) score. Based on quality of solution from each operator pair, 
                add relevant discrete score to the selected operator's existing score within a segment. 
            * method 2: Victor's RW with continuous scores
                The added score is computed from the improvement in objective (old - new), where:
                best_obj = historical best-found objective (not yet updated by new_obj in this iteration)
                new_obj = new objective derived from the current operator pair
        '''
        discount = 10

        if method == 'classical_discrete': # method 1
            credit_score = self.RW_scoring[credit_status] 

        elif method == 'continuous': # method 2
            new_obj, best_obj = kwargs.get('new_obj'), kwargs.get('best_obj')
            if credit_status == 'better_accept': 
                credit_score = best_obj - new_obj
            elif credit_status == 'worse_accept': 
                credit_score = abs(best_obj - new_obj)/discount    # reduce the magnitude to avoid very bad solution to affect score
                #credit_score = 0                                  # worse solution then no reward
            else:
                credit_score = 0
        return credit_score


    def normalisation(self, previous_segment_best_obj, current_segment_best_obj):
        improvement = previous_segment_best_obj - current_segment_best_obj
        return round((improvement + 2000 )/ (current_segment_best_obj - 1000), 2)


    def updated_weight_compute(self, method, weight_previous_segment, total_score_this_segment, total_attempts_this_segment, **kwargs):
        ''' 
        CRW: method == 'classical_reward_per_pair': 
            Reward for improvement brought by every operator pair. Reward values are discretised. When RW weight update at the end 
            of the segment, we use the total accumulated scores of the whole segment to update each operator's weight; 
        VRW: method == 'reward_per_segment_best'
            Reward only once for every segment. The reward value equals to the objective function difference, here we use best_obj 
            found in this segment - best_obj in previous history. Every operator applied within the segment will be given the same 
            amount of reward to update its segmental weight.
        Note: 
        1. Reward can be discrete values manually defined as 0,2,4 in classical ALNS, or continous as the difference in 
           objective function value;
        2. Notice: if segment improvement is really huge (segment_credit_score = previous_segment_best_obj - current_segment_best_obj >> 0), 
           then updated_weight will be enormous. This will drag the segment weight to 0 in the later stage. Therefore, 
           we need the function "normalisation" to rescale the RW update algorithm. 
        '''

        if method == 'classical_reward_per_pair':                    # refer to experiment "1" (classical RW)
            updated_weight = (1 - self.reaction_factor) * weight_previous_segment + self.reaction_factor * (total_score_this_segment / total_attempts_this_segment)
        else:
            previous_segment_best_obj, current_segment_best_obj = kwargs.get('previous_segment_best_obj'), kwargs.get('current_segment_best_obj')
            if method == 'reward_per_segment_best':                   # refer to experiment "2ab"           
                segment_credit_score = self.normalisation(previous_segment_best_obj, current_segment_best_obj) 
            elif method == 'reward_per_segment_discrete_score':       # refer to experiment "2b"
                segment_last_score = kwargs.get('segment_last_score')
                segment_credit_score = segment_last_score                        
            
            if segment_credit_score < 0:
                raise ValueError('The current best within current segment becomes larger than the previous best! ERROR')
            
            updated_weight = (1 - self.reaction_factor) * weight_previous_segment + self.reaction_factor * (segment_credit_score / total_attempts_this_segment)
            
            #print('updated_weight = ', updated_weight)
            #print('weight_previous_segment={}, segment_credit_score={}, total_attempts_this_segment={}, segment_credit/total_attempts={}'.format(weight_previous_segment,segment_credit_score,total_attempts_this_segment, segment_credit_score / total_attempts_this_segment))
        return updated_weight  



    def after_operator_applied(self, operator, **kwargs):
        ''' experiment 1: do we consider discretised manual-defined operator score (CRW) or continuous score derived from objective function (VRW)?  
                          choices = 'classical_discrete' or 'continuous'
            experiment 2: how frequent should each operator be rewarded, for every time applied (CRW) or for every segment (VRW)?  
                          choices = 'classical_reward_per_pair', 'reward_per_segment_best', 'reward_per_segment_discrete_score'
        ## Potential issues: 
            If we apply 15 operators AND with VRW, we need to add the same credit score to all those applied within the segment!! 
            Not only the last pair!!! This problem doesnt exist for CRW since the reward is updated every operator iteration! 
        
        '''
        counter, credit_status = kwargs.get('counter'), kwargs.get('credit_status')
        experiment1, experiment2 = kwargs.get('experiment1'), kwargs.get('experiment2')
        new_obj, best_obj = kwargs.get('new_obj'), kwargs.get('best_obj')
        
        ## credit score for each applied operator during this segment (for updating weight): 
        credit_score = self.credit_score_compute(credit_status, experiment1, new_obj=new_obj, best_obj=best_obj)
        #credit_score = self.RW_scoring[credit_status]  # add this amount, which is based on solution quality, to the selected operator's existing score for each inner iteration
        
        ## Select the score library for the input destroy/repair operator:
        operator_string = str(operator)
        score_to_update = (self.Score_d if operator_string[0] == 'D' else self.Score_r)


        ## DURING the sement: counting and recording performance data
        if counter % self.weight_segment != 0: 
            score_to_update[operator_string][0] += credit_score  # update the selected destroy operator's total score received within this segment
            score_to_update[operator_string][1] += 1             # update number of times this destroy operator is called within this segment
        
        ## END of segment: compute each operator's weight for this segment using the total score and number of attempts recorded
        else:
            for key, value in score_to_update.items():
                total_score_this_segment    = value[0]  #if in VRW only used to track if this operator is applied during segment or not. Not used for weight adjustment 
                total_attempts_this_segment = value[1]
                weight_previous_segment     = value[2]
                ## Update weight for all operators applied during this segment: (if VRW then update with the same score)
                if total_score_this_segment != 0 and total_attempts_this_segment != 0: 
                    previous_segment_best_obj = kwargs.get('previous_segment_best_obj')
                    current_segment_best_obj = kwargs.get('best_obj')
                    updated_weight = self.updated_weight_compute(experiment2,
                                                                 weight_previous_segment,
                                                                 total_score_this_segment,
                                                                 total_attempts_this_segment,
                                                                 segment_last_obj=new_obj,
                                                                 segment_last_score=credit_score,
                                                                 previous_segment_best_obj=previous_segment_best_obj,
                                                                 current_segment_best_obj=current_segment_best_obj)
                    score_to_update[key][2] = round(updated_weight, 3)
                score_to_update[key][0] = 0
                score_to_update[key][1] = 0


    def setup(self, options, hyperparams):
        self.weight_segment = hyperparams['weight_segment']
        self.reaction_factor = hyperparams['reaction_factor']

    def get_default_hyperparameters(self):
        hyperparams = {
                       'weight_segment': 20,
                       'reaction_factor': 0.95,
                       }
        return hyperparams


def create_hardcoded_scale_selector(base_class, hardcoded_scale):
    new_name = base_class.__name__ + f"S{hardcoded_scale}"
    newclass = type(new_name, (base_class,), {})
    setattr(newclass, "hardcoded_scale", hardcoded_scale)
    setattr(newclass, "algorithm_name", base_class.algorithm_name + f"_s{hardcoded_scale}")
    return newclass

