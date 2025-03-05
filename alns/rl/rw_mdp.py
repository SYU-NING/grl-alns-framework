import pickle
from copy import deepcopy

import numpy as np

from alns.rl.mdp_agent import MDPAgent
from alns.rl.pytorch_agent import PyTorchAgent
from tqdm import tqdm

from environment.operator_env import EnvPhase


class VictorRouletteWheelMDPAgent(PyTorchAgent):
    algorithm_name = "vrwmdp"


    is_deterministic = False
    is_trainable = True
    requires_hyperopt = True
    requires_tune = False


    def __init__(self, operator_library, random_seed, **kwargs):
        super().__init__(operator_library, random_seed, **kwargs)

        all_destroy = [str(op) for op in self.operator_library.get_available_destroy_ops()]
        all_repair = [str(op) for op in self.operator_library.get_available_repair_ops()]

        self.Score_d = dict(zip(all_destroy, [[0, 0, 1] for x in range(len(all_destroy))]))
        self.Score_r = dict(zip(all_repair, [[0, 0, 1] for x in range(len(all_repair))]))

    def make_actions(self, t, **kwargs):
        if t % EnvPhase.get_num_phases() == 0:
            self.stored_acts = {}
            for env_phase in [EnvPhase.APPLY_DESTROY, EnvPhase.APPLY_REPAIR]:
                self.stored_acts[env_phase] = [None for _ in range(len(self.env.state_list))]

        max_score_flag = kwargs['max_score'] if 'max_score' in kwargs else False
        # print(f"Got max_score flag as {max_score_flag}")
        acts = []
        for i, state in enumerate(self.env.state_list):
            if state.env_phase == EnvPhase.APPLY_DESTROY:
                destroy_op_str = self.select_operator_by_score(self.Score_d, max_score=max_score_flag)
                future_destroy_op = self.operator_library.get_destroy_op(destroy_op_str)
                self.stored_acts[EnvPhase.APPLY_DESTROY][i] = future_destroy_op

                ## repair op needs to match scale of destroy...
                scale_pc = self.operator_library.fixed_scale_pc
                repair_op_str = self.select_operator_by_score(self.Score_r, max_score=max_score_flag)
                future_repair_op = self.operator_library.get_repair_op(repair_op_str)
                self.stored_acts[EnvPhase.APPLY_REPAIR][i] = future_repair_op

                act = future_destroy_op

            elif state.env_phase == EnvPhase.APPLY_REPAIR:
                act = self.stored_acts[state.env_phase][i]

            acts.append(act)

        return acts

    def select_operator_by_score(self, score_to_use, filter_to_scale=None, max_score=False):
        '''Select a random destroy/repair operator based on their associated probabilities '''
        performance_score = deepcopy(score_to_use)
        if filter_to_scale is not None:
            performance_score = {k: v for k, v in performance_score.items() if self.operator_library.scale_from_op_str(k) == filter_to_scale}

        total_score = sum([X[2] for X in list(performance_score.values())])

        if max_score:
            op_scores = [X[2] for X in list(performance_score.values())]
            max_score_idx = np.argmax(op_scores)
            operator_index = max_score_idx
        else:
            try:
                operator_probabilities = [x / total_score for x in np.cumsum([X[2] for X in list(performance_score.values())])]
            except FloatingPointError:
                # self.logger.info(f"caught FP error in RW-MDP. continuing with uniform probs.")
                # self.logger.info(f"performance scores were: {performance_score}")
                operator_probabilities = [(i + 1) / len(performance_score) for i in range(len(performance_score))]

            operator_probabilities.insert(0, 0)
            prob_2 = self.local_random.random()
            for i in range(len(operator_probabilities) - 1):
                if operator_probabilities[i] < prob_2 and prob_2 <= operator_probabilities[i + 1]:
                    operator_index = i
                    break
        selected_op_str = list(performance_score.keys())[operator_index]
        #     print(selected_op_str)
        return selected_op_str

    def train(self, train_g_list, validation_g_list, max_steps, **kwargs):
        self.setup_graphs(train_g_list, validation_g_list)
        self.setup_sample_idxes(len(train_g_list))

        self.setup_histories_file()

        pbar = tqdm(range(max_steps + 1), unit='steps', disable=None)

        self.operator_pairs_counter = 0
        for self.step in pbar:
            print(f"executing step {self.step}")
            # print(f"d scores {self.Score_d}")
            # print(f"r scores {self.Score_r}")
            self.run_simulation()
            self.check_validation_loss(self.step, max_steps)

            should_stop = self.check_stopping_condition(self.step, max_steps)
            if should_stop:
                break

    def run_simulation(self):
        selected_idx = self.advance_pos_and_sample_indices()

        for i, idx in enumerate(selected_idx):
            self.env.setup([self.train_g_list[idx]], training=True)
            self.post_env_setup()

            destroy_used, repair_used = [], []
            t = 0
            while not self.env.is_terminal():
                list_at = self.make_actions(t, max_score=False)
                self.env.step(list_at)
                selected_action = list_at[0]
                reward = self.env.rewards[0]
                t += 1

                if self.env.state_list[0].env_phase == EnvPhase.APPLY_DESTROY:
                    repair_used.append(selected_action)
                    self.operator_pairs_counter += 1
                elif self.env.state_list[0].env_phase == EnvPhase.APPLY_REPAIR:
                    destroy_used.append(selected_action)

            # print(self.Score_d)
            # print(self.Score_r)
            if reward < 0.:
                reward = 0.
            # print(f"updating with {reward}")
            # print(f"updating at step {self.step}")
            for _ in range(self.batch_size):
                for destroy_op, repair_op in zip(destroy_used, repair_used):
                    self.update_scores_with_reward(destroy_op, reward)
                    self.update_scores_with_reward(repair_op, reward)

    def update_scores_with_reward(self, applied_op, reward):
        operator_string = str(applied_op)
        score_to_update = (self.Score_d if operator_string[0] == 'D' else self.Score_r)
        if self.operator_pairs_counter % self.weight_segment != 0:
            score_to_update[operator_string][0] += reward  # update the selected destroy operator's total score received within this segment
            score_to_update[operator_string][1] += 1
        else:
            for key, value in score_to_update.items():
                total_score_this_segment = value[0]
                total_attempts_this_segment = value[1]
                weight_previous_segment = value[2]
                if total_score_this_segment != 0 and total_attempts_this_segment != 0:  # Shunee: doublecheck if same to ALNS literature?
                    updated_weight = (1 - self.reaction_factor) * weight_previous_segment + self.reaction_factor * (total_score_this_segment / total_attempts_this_segment)
                    score_to_update[key][2] = round(updated_weight, 3)
                score_to_update[key][0] = 0
                score_to_update[key][1] = 0

    def save_model_checkpoints(self):
        model_dir = self.checkpoints_path / self.model_identifier_prefix
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{self.algorithm_name}_agent.pkl"
        with open(model_path, 'wb') as fh:
            scores = (self.Score_d, self.Score_r, self.best_validation_changed_step)
            pickle.dump(scores, fh)
            # print(self.Score_d)
            # print(self.Score_r)

    def restore_model_from_checkpoint(self):
        model_path = self.checkpoints_path / self.model_identifier_prefix / f"{self.algorithm_name}_agent.pkl"
        with open(model_path, 'rb') as fh:
            self.Score_d, self.Score_r, at_step = pickle.load(fh)
            print(f"loaded scores from step {at_step}")
            print(self.Score_d)
            print(self.Score_r)

    def patch_scores(self, d_score, r_score):
        # originally used to patch the scale of operators
        # in their corresponding keys
        # in case VRW is used on a larger problem instance.
        # e.g., {D-Random-4: 0.5} -> {D-Random-6: 0.5}
        patched_d = dict()
        patched_r = dict()
        for k, v in d_score.items():
            oproot = k.split("-")[:-1]
            new_k = "-".join(oproot + [str(self.operator_library.max_scale)])
            patched_d[new_k] = v
        for k, v in r_score.items():
            oproot = k.split("-")[:-1]
            new_k = "-".join(oproot + [str(self.operator_library.max_scale)])
            patched_r[new_k] = v
        return patched_d, patched_r

    def select_destroy_operator(self, state, **kwargs):
        state.env_phase = EnvPhase.APPLY_DESTROY
        destroy_op_str = self.select_operator_by_score(self.Score_d)
        destroy_op = self.operator_library.get_destroy_op(destroy_op_str)
        return destroy_op

    def select_repair_operator(self, state, **kwargs):
        repair_op_str = self.select_operator_by_score(self.Score_r) #, filter_to_scale=state.selected_scale)
        repair_op = self.operator_library.get_repair_op(repair_op_str)
        return repair_op

    def after_operator_applied(self, operator, **kwargs):
        pass

    def post_env_setup(self):
        pass

    def setup(self, options, hyperparams):
        super().setup(options, hyperparams)
        self.weight_segment = hyperparams['weight_segment']
        self.reaction_factor = hyperparams['reaction_factor']

        if self.restore_model:
            self.restore_model_from_checkpoint()


    def get_default_hyperparameters(self):
        hyperparams = {'weight_segment': 200,
                       'reaction_factor': 0.5,
                       }
        return hyperparams

class MaxScoreVictorRouletteWheelMDPAgent(VictorRouletteWheelMDPAgent):
    algorithm_name = "msvrwmdp"

    def eval(self, instance_list, make_action_kwargs=None):
        base_kwargs = {} if make_action_kwargs is None else deepcopy(make_action_kwargs)
        base_kwargs["max_score"] = True
        return super().eval(instance_list, base_kwargs)
