import math
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path

import numpy as np

from alns.op_selector import OperatorSelector
from environment.operator_env import EnvPhase
from experiment_storage.file_paths import FilePaths


class MDPAgent(OperatorSelector):
    is_mdp_based = True

    def __init__(self, operator_library, random_seed, **kwargs):
        super().__init__(operator_library, random_seed, **kwargs)

        self.env = kwargs.get("env", None)
        if self.env is None:
            raise ValueError(f"MDPAgent must be created with an <<env>> argument -- the training environment for the agent.")

        self.track_operators = kwargs.get("track_operators", False)
        if self.track_operators:
            self.op_tracking_dir = kwargs.get("op_tracking_dir", None)
            self.model_identifier_prefix = kwargs.get("model_identifier_prefix", None)
            self.setup_operators_tracking_file()

    def post_env_setup(self):
        pass

    def post_actions_applied(self, t):
        pass

    def setup_operators_tracking_file(self):
        op_tracking_filename = self.op_tracking_dir / FilePaths.construct_op_tracking_file_name(self.model_identifier_prefix)
        op_tracking_file = Path(op_tracking_filename)
        if op_tracking_file.exists():
            op_tracking_file.unlink()
        self.op_tracking_out = open(op_tracking_filename, 'a')
        self.op_tracking_out.write("tour_id,timestep,action,total_objective\n")

    @abstractmethod
    def make_actions(self, t, **kwargs):
        pass

    def run_env_loop(self, instance_list, training, force_budget=None, make_action_kwargs=None):
        self.env.setup(instance_list, training=training, force_budget=force_budget)
        self.post_env_setup()

        t = 0
        while not self.env.is_terminal():
            action_kwargs = (make_action_kwargs or {})
            list_at = self.make_actions(t, **action_kwargs)

            if self.track_operators:
                self.log_tracked_operators(t, list_at)

            # print(f"making actions at time {t}.")
            # print(f"at step {t} agent picked actions {list_at}")

            self.env.step(list_at)
            self.post_actions_applied(t)

            t += 1

        # print(f"ran for {t} steps.")

    def log_tracked_operators(self, t, list_at):
        entries = []
        for i, state in enumerate(self.env.state_list):
            value_at_state = self.env.get_objective_function_value(state)
            entries.append(f"{state.random_seed},{t},{list_at[i]},{value_at_state}\n")
        self.op_tracking_out.writelines(entries)

    def eval(self, instance_list, validation=False):
        eval_insts = [deepcopy(g) for g in instance_list]
        per_size = self.eval_separate_per_size(eval_insts)
        if not validation:
            validation_results = [e['perf'] for e in per_size]
        else:
            validation_results = [e['perf'] / (e['cust_number'] * math.log(e['cust_number'])) for e in per_size]

        return np.mean(validation_results)

    def eval_separate_per_size(self, instance_list):
        perfs_per_size = []

        num_customers = instance_list[0].ds_num_customers
        chunk_size = instance_list[0].ds_chunk_size

        idx = 0

        for cust_number in num_customers:
            instances_with_number = instance_list[idx: idx + chunk_size]
            self.run_env_loop(instances_with_number, training=False)

            chunk_initial_vals = self.env.get_initial_values()
            chunk_final_vals = self.env.get_final_values()

            mean_decrease = np.mean(chunk_initial_vals - chunk_final_vals)
            perfs_per_size.append({"cust_number": cust_number, "perf": mean_decrease})

            idx += chunk_size


        return perfs_per_size


    def make_random_actions(self, t, **kwargs):
        acts = []
        for i, state in enumerate(self.env.state_list):
            if state.env_phase == EnvPhase.APPLY_DESTROY:
                all_ops = self.operator_library.get_available_destroy_ops()
                act = self.local_random.choice(all_ops)
            elif state.env_phase == EnvPhase.APPLY_REPAIR:
                all_ops = self.operator_library.get_available_repair_ops()
                act = self.local_random.choice(all_ops)

            acts.append(act)

        return acts

    def setup(self, options, hyperparams):
        pass

    def get_default_hyperparameters(self):
        return {}

    def train(self, train_instances, validation_instances, max_steps, **kwargs):
        raise NotImplementedError("Method not implemented yet.")

    def select_destroy_operator(self, state, **kwargs):
        raise NotImplementedError("Method not implemented yet.")

    def select_repair_operator(self, state, **kwargs):
        raise NotImplementedError("Method not implemented yet.")

    def after_operator_applied(self, operator, **kwargs):
        raise NotImplementedError("Method not implemented yet.")


class RandomMDPAgent(MDPAgent):
    algorithm_name = "randommdp"

    is_deterministic = False
    is_trainable = False
    requires_hyperopt = False
    requires_tune = False


    def make_actions(self, t, **kwargs):
        return self.make_random_actions(t, **kwargs)

    def select_destroy_operator(self, state, **kwargs):
        state.env_phase = EnvPhase.APPLY_DESTROY
        all_ops = self.operator_library.get_available_destroy_ops()
        destroy_op = self.local_random.choice(all_ops)

        return destroy_op

    def select_repair_operator(self, state, **kwargs):
        all_ops = self.operator_library.get_available_repair_ops()
        repair_op = self.local_random.choice(all_ops)
        return repair_op

    def after_operator_applied(self, operator, **kwargs):
        pass



