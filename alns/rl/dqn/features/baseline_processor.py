from copy import deepcopy

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data

from state.alns_state import get_state_hash
from utils.general_utils import maxabs_scale


class BaselineFeatureSpec(object):
    def __init__(self, name, datatype, feat_dim):
        self.name = name
        self.datatype = datatype
        self.feat_dim = feat_dim

class BaselineFeatureProcessor(object):
    def __init__(self, hyperparams, max_nodes, problem_variant, operator_library):
        super().__init__()
        self.hyperparams = hyperparams
        self.max_nodes = max_nodes
        self.problem_variant = problem_variant
        self.operator_library = operator_library

        self.construct_action_int_mappings()
        self.all_feats = self.get_feature_specs()

        self.known_representations = {}


    def get_feature_specs(self):
        feats = [
            BaselineFeatureSpec("cost_improvement", "int", 1), # Current solution improved: 0 or 1
            BaselineFeatureSpec("cost_difference_best", "float", 1), # % difference between objective values of current and best solutions (-1 if current is <= 0)
            BaselineFeatureSpec("cost_of_best", "float", 1), # Cost of best solution
            BaselineFeatureSpec("is_current_best", "int", 1), # Current equal to best (0 or 1)
            BaselineFeatureSpec("cost", "float", 1), # The cost of the current solution
            BaselineFeatureSpec("iteration", "float", 1), # Iteration number
            BaselineFeatureSpec("stagnation_count", "float", 1), # Number of iterations without improving best found solution
            BaselineFeatureSpec("unseen", "float", 1), # 1 if solution has not been prev encountered, 0 otherwise.
            BaselineFeatureSpec("was_changed", "int", 1), # 1 if solution was changed from the previous, 0 otherwise
            BaselineFeatureSpec("last_action_sign", "int", 1), # 1 if solution was changed from the previous, 0 otherwise,
            BaselineFeatureSpec("last_action", "int", self.num_possible_actions)
        ]


        return feats


    def reset(self, for_mdp=True, **kwargs):
        if for_mdp:
            self.env = kwargs.pop("env")
            budget = 2 * self.env.initial_budget
            state_list = self.env.state_list
            obj_function_values = self.env.objective_function_values[0, :]
            last_operators = self.env.last_applied_actions
        else:
            budget = kwargs.pop("total_steps")
            state_list = [kwargs.pop("starting_state")]
            obj_function_values = np.array([kwargs.pop("starting_obj_value")])
            last_operators = [kwargs.pop("operator")]


        self.t = 0
        self.costs_array = np.full(((budget) + 1, len(state_list)),
                               fill_value=float("inf"),
                               dtype=np.float32)
        self.costs_array[0, :] = np.copy(obj_function_values)
        self.best_costs_array = np.copy(self.costs_array[0, :])
        self.best_costs_timesteps = np.zeros_like(self.best_costs_array)

        self.seen_solutions = set()

        # print(f"at reset, list has size {len(env.state_list)}")

        for i, state in enumerate(state_list):
            state_repr = self.get_state_representation(state, i, last_operators, initial=True)
            state_hash = get_state_hash(state)
            self.known_representations[state_hash] = state_repr
            self.seen_solutions.add(state_hash)

    def construct_action_int_mappings(self):
        self.action_int_mappings = {}
        all_ops = deepcopy(self.operator_library.unique_ops_destroy)
        all_ops.extend(self.operator_library.unique_ops_repair)

        for i, op in enumerate(all_ops):
            self.action_int_mappings[op] = i
        self.action_int_mappings[None] = len(all_ops)

        self.num_possible_actions = len(all_ops) + 1

    def update_representations(self, for_mdp=True, **kwargs):
        self.t = kwargs.pop("t")
        if for_mdp:
            self.env = kwargs.pop("env")
            state_list = self.env.state_list
            obj_function_values = self.env.get_objective_function_values(state_list)
            last_operators = self.env.last_applied_actions
        else:
            state_list = [kwargs.pop("state")]
            obj_function_values = np.array([kwargs.pop("obj_value")])
            last_operators = [kwargs.pop("operator")]


        self.costs_array[self.t+1, :] = obj_function_values
        self.best_costs_array = np.min(self.costs_array, axis=0)
        self.best_costs_timesteps = np.argmin(self.costs_array, axis=0)

        for i, state in enumerate(state_list):
            state_repr = self.get_state_representation(state, i, last_operators, initial=False)
            state_hash = get_state_hash(state)
            self.known_representations[state_hash] = state_repr
            self.seen_solutions.add(state_hash)


    def get_mlp_max_input_size(self):
        input_size = 0
        input_size += sum([f.feat_dim for f in self.all_feats])
        return input_size


    def get_states_representations(self, state_list, **kwargs):
        # is_offline = kwargs.get("offline", False)
        repr_list = []
        for i, state in enumerate(state_list):
            state_hash = get_state_hash(state)
            state_repr = self.known_representations[state_hash]
            repr_list.append(state_repr)

        return repr_list

    def get_state_representation(self, state, i, last_operators, initial=False):
        repr_size = self.get_mlp_max_input_size()
        repr_out = np.zeros(repr_size, dtype=np.float32)
        starting_pos = 0

        for gf in self.all_feats:
            if gf.name != "last_action":
                if gf.name == "cost_improvement":
                     feat_val = 0. if initial else self.costs_array[self.t, i] - self.costs_array[self.t + 1, i]
                elif gf.name == "cost_difference_best":
                    feat_val = 0. if initial else self.costs_array[self.t + 1, i] - self.best_costs_array[i]
                elif gf.name == "cost_of_best":
                    feat_val = self.best_costs_array[i]
                elif gf.name == "is_current_best":
                    feat_val  = 1. if initial else self.costs_array[self.t + 1, i] == self.best_costs_array[i]
                elif gf.name == "cost":
                    feat_val = self.costs_array[self.t, i] if initial else self.costs_array[self.t + 1, i]
                elif gf.name == "iteration":
                    feat_val = 0 if initial else self.t + 1
                elif gf.name == "stagnation_count":
                    feat_val = self.t + 1 - self.best_costs_timesteps[i]
                elif gf.name == "unseen":
                    state_hash = get_state_hash(state)
                    feat_val = 0. if state_hash in self.seen_solutions else 1.
                elif gf.name == "was_changed":
                    if initial or self.t == 0:
                        feat_val = 0.
                    else:
                        feat_val = (self.costs_array[self.t + 1, i] != self.costs_array[self.t - 1, i])
                elif gf.name == "last_action_sign":
                    if initial or self.t == 0:
                        feat_val = 0.
                    else:
                        feat_val = (self.costs_array[self.t + 1, i] > self.costs_array[self.t - 1, i])

                if "cost" in gf.name:
                    feat_val /= 1000

            else:
                if last_operators[i] is not None:
                    action_key = last_operators[i].name
                else:
                    action_key = None

                last_action_int = self.action_int_mappings[action_key]
                feat_val = F.one_hot(torch.LongTensor([last_action_int]), num_classes=self.num_possible_actions)

            repr_out[starting_pos: starting_pos + gf.feat_dim] = feat_val
            starting_pos += gf.feat_dim




        out_tensor = torch.from_numpy(repr_out)
        return out_tensor


    def populate_features(self, state, out_tensor):
        starting_pos = 0
        for gf in self.all_feats:
            starting_pos += gf.feat_dim

        return starting_pos








