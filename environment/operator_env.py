from enum import Enum

import numpy as np
import math
from copy import deepcopy

from environment.objective_functions import ObjectiveFunction
from operators.destroy_operators import NoOpDestroy


class EnvPhase(Enum):
    APPLY_DESTROY = 0
    APPLY_REPAIR = 1

    @staticmethod
    def get_num_phases():
        return len([e.value for e in EnvPhase])


class OperatorEnv(object):
    def __init__(self, problem_variant, operator_library, seed, initial_budget, use_localsearch=False):

        self.problem_variant = problem_variant
        self.ol = operator_library
        self.seed = seed

        self.initial_budget = initial_budget
        self.use_localsearch = use_localsearch

        self.reward_eps = 1e-4
        self.reward_scale_multiplier = 0.1


    def setup(self, state_list, training=False, **kwargs):
        self.state_list = state_list
        self.timestep = 0

        self.training = training

        force_budget = kwargs['force_budget'] if 'force_budget' in kwargs else None


        for state in self.state_list:
            state.env_phase = EnvPhase.APPLY_DESTROY

            if force_budget is None:
                state.destroy_budget = self.initial_budget
                state.repair_budget = self.initial_budget
            else:
                state.destroy_budget = force_budget
                state.repair_budget = force_budget

        self.exhausted_budgets = np.zeros(len(self.state_list), dtype=np.bool)
        self.objective_function_values = np.zeros((2, len(self.state_list)), dtype=np.float)

        initial_values = self.get_objective_function_values(self.state_list)
        self.objective_function_values[0, :] = initial_values

        self.rewards = np.zeros(len(state_list), dtype=np.float)

        self.last_applied_actions = [None] * len(self.state_list)



    def step(self, actions):
        for i in range(len(self.state_list)):
            if not self.exhausted_budgets[i]:
                state = self.state_list[i]
                self.state_list[i] = self.apply_action(state, actions[i])
                self.last_applied_actions[i] = actions[i]

                if state.destroy_budget == 0 and state.repair_budget == 0:
                    self.exhausted_budgets[i] = True
                    objective_function_value = self.get_objective_function_value(state)
                    self.objective_function_values[-1, i] = objective_function_value
                    reward = self.reward_scale_multiplier * (self.objective_function_values[0, i] - objective_function_value)
                    if abs(reward) < self.reward_eps:
                        reward = 0.

                    self.rewards[i] = reward

        self.timestep += 1


    def get_valid_actions(self, state):
        if state.env_phase == EnvPhase.APPLY_DESTROY:
            if state.destroy_budget == 0:
                return []

            available_ops = self.ol.get_available_destroy_ops()
            return available_ops
        elif state.env_phase == EnvPhase.APPLY_REPAIR:
            if state.repair_budget == 0:
                return []

            available_ops = self.ol.get_available_repair_ops()
            return available_ops


    def apply_action(self, state, action, in_place=True):
        if not in_place:
            state_ref = deepcopy(state)
        else:
            state_ref = state

        if state_ref.env_phase == EnvPhase.APPLY_DESTROY:
            action.apply_op(state_ref)
            state_ref.destroy_budget -= 1
            state_ref.env_phase = EnvPhase.APPLY_REPAIR

        elif state_ref.env_phase == EnvPhase.APPLY_REPAIR:
            if len(state_ref.customer_pool) == 0:
                # no-op operator was applied; do nothing.
                pass
            else:
                action.apply_op(state_ref)

            if state_ref.inst.problem_variant == "TSP":
                tours = state_ref.alns_list
                if len(tours) > 1:
                    print("Illegal state: found >1 tour for TSP.")
                    print(f"Offending operator: {action}")
                    raise ValueError("TSP instance not allowed >1 tour.")


            if self.use_localsearch:
                ls_op = self.ol.get_available_local_search_ops()[0]
                ls_op.apply_op(state_ref)

            state_ref.repair_budget -= 1

            state_ref.env_phase = EnvPhase.APPLY_DESTROY

        return state_ref


    def is_terminal(self):
        return np.all(self.exhausted_budgets)

    def get_objective_function_value(self, state):
        obj_function_value = ObjectiveFunction.compute_obj_function_values(state)["total_objective"]
        return obj_function_value

    def get_objective_function_values(self, states):
        return np.array([self.get_objective_function_value(g) for g in states])

    def get_initial_values(self):
        return self.objective_function_values[0, :]

    def get_final_values(self):
        return self.objective_function_values[-1, :]

    def clone_state(self, indices=None):
        if indices is None:
            return deepcopy(self.state_list)
        else:
            cp_g_list = []

            for i in indices:
                cp_g_list.append(deepcopy(self.state_list[i]))

            return cp_g_list
