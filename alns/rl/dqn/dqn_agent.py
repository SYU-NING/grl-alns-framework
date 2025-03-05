import math

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from alns.rl.dqn.features.baseline_processor import BaselineFeatureProcessor
from alns.rl.dqn.features.instance_processor import InstanceFeatureProcessor
from alns.rl.dqn.q_net import NStepQNet
from alns.rl.dqn.nstep_replay_mem import NstepReplayMem
from alns.rl.pytorch_agent import PyTorchAgent
from environment.operator_env import EnvPhase
from utils.config_utils import get_device_placement
from torch.nn import functional as F

class DQNAgent(PyTorchAgent):
    algorithm_name = "dqn"

    is_deterministic = False
    is_trainable = True
    requires_hyperopt = True
    requires_tune = False

    def setup(self, options, hyperparams):
        super().setup(options, hyperparams)
        self.setup_feat_processor()
        self.setup_nets()
        self.take_snapshot()

    def train(self, train_g_list, validation_g_list, max_steps, **kwargs):
        self.setup_graphs(train_g_list, validation_g_list)
        self.setup_sample_idxes(len(train_g_list))

        self.setup_mem_pool(max_steps, self.hyperparams['mem_pool_to_steps_ratio'])
        self.setup_histories_file()
        self.setup_training_parameters(max_steps)

        # tqdm_disable = None
        tqdm_disable = False
        pbar = tqdm(range(self.burn_in), unit='batch', disable=tqdm_disable)
        for p in pbar:
            with torch.no_grad():
                self.run_simulation()
        pbar = tqdm(range(max_steps + 1), unit='steps', disable=tqdm_disable)
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        for self.step in pbar:
            with torch.no_grad():
                self.run_simulation()
            if self.step % self.net_copy_interval == 0:
                self.take_snapshot()
            self.check_validation_loss(self.step, max_steps)

            cur_time, list_st, list_at, list_rt, list_s_primes, list_term = self.mem_pool.sample(
                batch_size=self.batch_size)
            list_target = torch.Tensor(list_rt)
            if get_device_placement() == 'GPU':
                list_target = list_target.cuda()

            cleaned_sp = []
            nonterms = []
            for i in range(len(list_st)):
                if not list_term[i]:
                    cleaned_sp.append(list_s_primes[i])
                    nonterms.append(i)

            if len(cleaned_sp):
                q_t_plus_1 = self.old_net((cur_time + 1) % self.n_steps, cleaned_sp, offline=True)
                q_rhs = self.old_net.get_max_qvals((cur_time + 1) % self.n_steps, q_t_plus_1)
                list_target[nonterms] = q_rhs

            # print(f"Q targets: {list_target}")
            list_target = Variable(list_target.view(-1, 1))
            q_sa = self.net(cur_time % self.n_steps, list_st, for_actions=list_at, offline=True)

            loss = F.mse_loss(q_sa, list_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description('exp: %.5f, loss: %0.5f' % (self.eps, loss))

            should_stop = self.check_stopping_condition(self.step, max_steps)
            if should_stop:
                break


    def setup_nets(self):
        # self.logger.info(f"agent received hyps as {self.hyperparams}")
        self.n_steps = 2
        self.net = NStepQNet(self.feat_processor, self.hyperparams, self.env.ol, self.env.problem_variant)
        self.old_net = NStepQNet(self.feat_processor, self.hyperparams, self.env.ol, self.env.problem_variant)
        if get_device_placement() == 'GPU':
            self.net = self.net.cuda()
            self.old_net = self.old_net.cuda()
        if self.restore_model:
            self.restore_model_from_checkpoint()

    def setup_feat_processor(self):
        max_nodes = self.options['max_nodes']
        self.feat_processor = InstanceFeatureProcessor(self.hyperparams, max_nodes, self.env.problem_variant, self.env.ol)

    def setup_mem_pool(self, num_steps, mem_pool_to_steps_ratio):
        exp_replay_size = int(num_steps * mem_pool_to_steps_ratio)
        self.mem_pool = NstepReplayMem(memory_size=exp_replay_size, n_steps=2)

    def setup_training_parameters(self, max_steps):
        self.learning_rate = self.hyperparams['learning_rate']
        self.eps_start = self.hyperparams['epsilon_start']

        eps_step_denominator = self.hyperparams['eps_step_denominator'] if 'eps_step_denominator' in self.hyperparams else 2
        self.eps_step = max_steps / eps_step_denominator
        self.eps_end = 0.1
        self.burn_in = self.hyperparams['burn_in']
        self.net_copy_interval = 50

    def finalize(self):
        pass

    def take_snapshot(self):
        self.old_net.load_state_dict(self.net.state_dict())

    def make_actions(self, t, **kwargs):
        greedy = kwargs['greedy'] if 'greedy' in kwargs else True
        if greedy:
            return self.do_greedy_actions(t)
        else:
            self.eps = self.eps_end + max(0., (self.eps_start - self.eps_end)
                                          * (self.eps_step - max(0., self.step)) / self.eps_step)
            if self.local_random.random() < self.eps:
                exploratory_actions = self.make_random_actions(t, **kwargs)
                return exploratory_actions
            else:
                greedy_acts = self.do_greedy_actions(t)
                return greedy_acts


    def do_greedy_actions(self, time_t):
        mdp_substep = time_t % self.n_steps

        cur_states = self.env.state_list
        # all_valid_acts = [self.env.get_valid_actions(s) for s in cur_states]
        est_q_values = self.net(mdp_substep, cur_states)
        raw_acts = self.net.greedy_actions(mdp_substep, est_q_values)

        greedy_acts = []
        for i, state in enumerate(cur_states):
            if time_t % self.n_steps == 0:
                act = self.env.ol.get_destroy_op(raw_acts[i] + f"-{self.env.ol.fixed_scale_pc}")

            elif time_t % self.n_steps == 1:
                act = self.env.ol.get_repair_op(raw_acts[i] + f"-{self.env.ol.fixed_scale_pc}")

            greedy_acts.append(act)
        return greedy_acts

    def run_simulation(self):
        selected_idx = self.advance_pos_and_sample_indices()
        self.env.setup([self.train_g_list[idx] for idx in selected_idx], training=True)
        self.post_env_setup()

        final_st = [None] * len(selected_idx)
        final_acts = [None] * len(selected_idx)

        t = 0
        while not self.env.is_terminal():
            list_at = self.make_actions(t, greedy=False)

            non_exhausted_before, = np.where(~self.env.exhausted_budgets)
            list_st = self.env.clone_state(non_exhausted_before)
            self.env.step(list_at)
            self.post_actions_applied(t)

            non_exhausted_after, = np.where(~self.env.exhausted_budgets)
            exhausted_after, = np.where(self.env.exhausted_budgets)

            nonterm_indices = np.flatnonzero(np.isin(non_exhausted_before, non_exhausted_after))
            nonterm_st = [list_st[i] for i in nonterm_indices]
            nonterm_at = [list_at[i] for i in non_exhausted_after]
            rewards = np.zeros(len(nonterm_at), dtype=np.float)
            nonterm_s_prime = self.env.clone_state(non_exhausted_after)

            now_term_indices = np.flatnonzero(np.isin(non_exhausted_before, exhausted_after))
            # term_st = [list_st[i] for i in now_term_indices]
            term_st = self.env.clone_state(now_term_indices)
            for i in range(len(term_st)):
                g_list_index = non_exhausted_before[now_term_indices[i]]

                final_st[g_list_index] = term_st[i]
                final_acts[g_list_index] = list_at[g_list_index]

            if len(nonterm_at) > 0:
                self.mem_pool.add_list(nonterm_st, nonterm_at, rewards, nonterm_s_prime, [False] * len(nonterm_at), t % self.n_steps)

            t += 1

        final_at = list(final_acts)
        rewards = self.env.rewards
        final_s_prime = None
        self.mem_pool.add_list(final_st, final_at, rewards, final_s_prime, [True] * len(final_at), (t - 1) % self.n_steps)

    def sample_act_index_with_softmax(self, scores, temp):
        scores_with_temp = (scores / temp).squeeze()
        probs = F.softmax(scores_with_temp, dim=0).detach().numpy()

        idx_range = np.arange(start=0, stop=scores.shape[1])

        # numpy, confusingly, errors when sampling from a one-item array...
        if len(idx_range) == 1:
            return 0

        act_idx = np.random.choice(idx_range, p=probs)
        return act_idx

    def select_destroy_operator(self, state, **kwargs):
        mdp_substep = 0
        state.env_phase = EnvPhase.APPLY_DESTROY

        est_q_values = self.net(mdp_substep, [state])
        op_root_name = self.net.greedy_actions(mdp_substep, est_q_values)[0]
        destroy_op = self.env.ol.get_destroy_op(op_root_name + f"-{self.env.ol.fixed_scale_pc}")

        return destroy_op

    def select_repair_operator(self, state, **kwargs):
        mdp_substep = 1

        self.net.eval()
        est_q_values = self.net(mdp_substep, [state])
        op_root_name = self.net.greedy_actions(mdp_substep, est_q_values)[0]
        repair_op = self.env.ol.get_repair_op(op_root_name + f"-{self.env.ol.fixed_scale_pc}")

        return repair_op


    def get_default_hyperparameters(self):
        hyperparams = {'learning_rate': 0.0001,
                       'epsilon_start': 1,
                       'mem_pool_to_steps_ratio': 1,
                       'first_hidden_size': 512,
                       'onehot_tour_idxes': False,
                       'scale_repeat': 100,
                       'eps_step_denominator': 10,
                       'burn_in': 100,
                       }
        return hyperparams

    def after_operator_applied(self, operator, **kwargs):
        pass


class BaselineDQNAgent(DQNAgent):
    algorithm_name = "dqnbaseline"

    def setup_feat_processor(self):
        max_nodes = self.options['max_nodes']
        self.feat_processor = BaselineFeatureProcessor(self.hyperparams, max_nodes, self.env.problem_variant, self.env.ol)

    def post_env_setup(self):
        self.feat_processor.reset(for_mdp=True, env=self.env)

    def post_actions_applied(self, t):
        self.feat_processor.update_representations(for_mdp=True, env=self.env, t=t)

    def after_operator_applied(self, operator, **kwargs):
        self.feat_processor.update_representations(for_mdp=False, operator=operator, **kwargs)

    def inform_alns_starting(self, **kwargs):
        self.feat_processor.reset(for_mdp=False, operator=None, **kwargs)

class ProbabilisticDQNAgent(DQNAgent):
    algorithm_name = "dqnprob"

    is_deterministic = False
    is_trainable = True
    requires_hyperopt = False
    requires_tune = True

    def select_destroy_operator(self, state, **kwargs):
        search_step = kwargs.pop('search_step')

        mdp_substep = 0
        state.env_phase = EnvPhase.APPLY_DESTROY

        if search_step < self.num_mdp_timesteps:
            # act greedy.
            self.net.eval()
            est_q_values = self.net(mdp_substep, [state])
            op_root_name = self.net.greedy_actions(mdp_substep, est_q_values)[0]
        else:
            act_idx = self.sample_act_index_with_softmax(self.destroy_operator_scores, self.alns_temp)
            op_root_name = self.net.get_action_by_index(mdp_substep, act_idx)

        destroy_op = self.env.ol.get_destroy_op(op_root_name + f"-{self.env.ol.fixed_scale_pc}")

        return destroy_op


    def select_repair_operator(self, state, **kwargs):
        search_step = kwargs.pop('search_step')
        mdp_substep = 1

        if search_step < self.num_mdp_timesteps:
            # act greedy.
            self.net.eval()
            est_q_values = self.net(mdp_substep, [state])
            op_root_name = self.net.greedy_actions(mdp_substep, est_q_values)[0]
        else:
            act_idx = self.sample_act_index_with_softmax(self.repair_operator_scores, self.alns_temp)
            op_root_name = self.net.get_action_by_index(mdp_substep, act_idx)

        repair_op = self.env.ol.get_repair_op(op_root_name + f"-{self.env.ol.fixed_scale_pc}")

        return repair_op

    def setup(self, options, hyperparams):
        super().setup(options, hyperparams)
        self.num_mdp_timesteps = options['num_mdp_timesteps']
        self.alns_temp = hyperparams['alns_temp']


    def compute_dual_policy_scores(self, training_graphs):
        self.net.eval()

        destroy_operator_occurences = np.zeros(len(self.env.ol.unique_ops_destroy))
        repair_operator_occurences = np.zeros(len(self.env.ol.unique_ops_repair))

        self.env.setup(training_graphs, training=False)
        self.post_env_setup()

        t = 0
        while not self.env.is_terminal():
            print(f"computing dual policy scores at {t}")
            list_at = self.make_actions(t)
            mdp_substep = t % self.n_steps
            mappped_actions = [self.net.list_mod[mdp_substep].action_to_encoding[op.name] for op in list_at]

            occ = destroy_operator_occurences if mdp_substep == 0 else repair_operator_occurences
            for a in mappped_actions:
                occ[a] += 1

            self.env.step(list_at)
            self.post_actions_applied(t)
            t += 1

        self.destroy_operator_scores = torch.unsqueeze(torch.from_numpy(destroy_operator_occurences / np.sum(destroy_operator_occurences)), 0)
        self.repair_operator_scores = torch.unsqueeze(torch.from_numpy(repair_operator_occurences / np.sum(repair_operator_occurences)), 0)


class ProbabilisticBaselineDQNAgent(ProbabilisticDQNAgent, BaselineDQNAgent):
    algorithm_name = "dqnbaselineprob"

    is_deterministic = False
    is_trainable = True
    requires_hyperopt = False
    requires_tune = True
