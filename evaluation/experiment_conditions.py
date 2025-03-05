from copy import deepcopy

from alns.op_selector import RouletteWheelOperatorSelector, RandomOperatorSelector
from alns.rl.dqn.dqn_agent import DQNAgent, ProbabilisticDQNAgent, BaselineDQNAgent, ProbabilisticBaselineDQNAgent
from alns.rl.hybrid_agent import HybridAgent
from alns.rl.mdp_agent import RandomMDPAgent
from alns.rl.rw_mdp import VictorRouletteWheelMDPAgent, MaxScoreVictorRouletteWheelMDPAgent
from environment.objective_functions import TotalObjective
from evaluation.eval_utils import get_model_seed, construct_network_seeds
from operators.destroy_operators import *
from operators.operator_library import OperatorLibrary
from operators.repair_operators import *
from state.state_generator import SolomonStateGenerator

known_cmd_kwargs = ["problem_variant", "fixed_scale_pc",
                    "budget", "alns_outer_its_per_customer", "alns_inner_its"]


class ExperimentConditions(object):
    def __init__(self, problem_variant, instance_name, vars_dict):
        self.gen_params = {}
        self.problem_variant = problem_variant
        self.instance_name = instance_name

        for k, v in vars_dict.items():
            if k in known_cmd_kwargs:
                setattr(self, k, v)

        self.track_operators = False
        self.eval_on_train = False

        self.initial_tour_method = "random"
        # self.initial_tour_method = "clark_wright"
        # self.initial_tour_method = "hybrid"

        self.num_customers_train = [20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.num_customers_validate = [20, 30, 40, 50, 100]
        self.num_customers_test = [20, 30, 40, 50, 60, 70, 80, 90, 100]

        self.tours_per_size = 128

        self.alns_outer_multipliers = [1]

        self.objective_functions = [
            TotalObjective
        ]

        self.network_generators = [
            SolomonStateGenerator
        ]

        self.use_localsearch = False

        self.which_destroy = self.configure_destroy_operators(vars_dict)
        self.which_repair = self.configure_repair_operators(vars_dict)

        self.hyps_chunk_size = 1
        self.seeds_chunk_size = 1

    def get_seeds_as_tuple(self):
        return self.train_seeds, self.validation_seeds, self.test_seeds

    def __str__(self):
        as_dict = deepcopy(self.__dict__)
        del as_dict["all_agents"]
        del as_dict["objective_functions"]
        del as_dict["network_generators"]
        return str(as_dict)

    def __repr__(self):
        return self.__str__()

    def configure_destroy_operators(self, vars_dict):
        compatible = OperatorLibrary.get_compatible_destroy_operators(self.problem_variant)
        return compatible

    def configure_repair_operators(self, vars_dict):
        compatible = OperatorLibrary.get_compatible_repair_operators(self.problem_variant)
        return compatible


class MainExperimentConditions(ExperimentConditions):
    max_steps = 100  # used: 50000

    def __init__(self, problem_variant, instance_name, vars_dict):
        super().__init__(problem_variant, instance_name, vars_dict)

        self.experiment_params = {
            'train_tours': self.tours_per_size * len(self.num_customers_train),
            'validation_tours': int(self.tours_per_size / 2) * len(self.num_customers_validate),
            'test_tours': self.tours_per_size * len(self.num_customers_test),
            'num_runs': 1,  # used: 10
        }

        self.experiment_params['model_seeds'] = [get_model_seed(run_num) for run_num in
                                                 range(self.experiment_params['num_runs'])]

        self.train_seeds, self.validation_seeds, self.test_seeds = construct_network_seeds(
            self.eval_on_train,
            self.experiment_params['train_tours'],
            self.experiment_params['validation_tours'],
            self.experiment_params['test_tours'])

        self.all_agents = [
            DQNAgent,
            BaselineDQNAgent,
            #
            VictorRouletteWheelMDPAgent,
            RandomMDPAgent,
            #
            ProbabilisticDQNAgent,
            ProbabilisticBaselineDQNAgent,
            RouletteWheelOperatorSelector,
            HybridAgent,
        ]

        self.agent_budgets = {
            TotalObjective.name: {
                DQNAgent.algorithm_name: self.max_steps,
                BaselineDQNAgent.algorithm_name: self.max_steps,
                VictorRouletteWheelMDPAgent.algorithm_name: self.max_steps,
            },
        }

        self.hyperparam_grids = self.create_hyperparam_grids()
        print(self.hyperparam_grids)

    def create_hyperparam_grids(self):
        hyperparam_grid_base = {
            DQNAgent.algorithm_name: {
                "learning_rate": [0.0001],  # used: [0.01, 0.005, 0.001, 0.0005],
                "use_rounding": [False],

                "arch": ["gat"],
                "lf_dim": [32],
                "num_hidden_layers": [3],
                'inst_size_repeat': [4],

                'epsilon_start': [1],
                'mem_pool_to_steps_ratio': [1],  # used: [0.03]

                'onehot_tour_idxes': [False],
                'eps_step_denominator': [10],
                'burn_in': [10]  # used: [int(len(self.num_customers_train) * self.tours_per_size)]
            },

            BaselineDQNAgent.algorithm_name: {
                "learning_rate": [0.0001],  # used: [0.01, 0.005, 0.001, 0.0005],
                "use_rounding": [False],
                "arch": ["mlp"],
                'first_hidden_size': [64],
                "num_hidden_layers": [3],
                'inst_size_repeat': [4],

                'epsilon_start': [1],
                'mem_pool_to_steps_ratio': [1],  # used: [0.03]

                'onehot_tour_idxes': [False],
                'eps_step_denominator': [10],
                'burn_in': [10]  # used: [int(len(self.num_customers_train) * self.tours_per_size)]
            },
            VictorRouletteWheelMDPAgent.algorithm_name: {
                "weight_segment": [200],  # used: [20, 40, 160, 200],
                'reaction_factor': [0.99]  # used: [0.9, 0.95, 0.99]
            },

            ProbabilisticDQNAgent.algorithm_name: {
                "alns_temp": [0.01, 0.05]  # , 0.1, 0.25, 0.5, 1, 5]
            },

            ProbabilisticBaselineDQNAgent.algorithm_name: {
                "alns_temp": [0.01, 0.05]  # used:  [0.01, 0.05, 0.1, 0.25, 0.5, 1, 5],
            },

            RouletteWheelOperatorSelector.algorithm_name: {
                "weight_segment": [20, 75],  # used: [20, 75, 200],
                'reaction_factor': [0.9]  # used: [0.9, 0.95, 0.99]
            },
        }
        hyperparam_grids = {}
        for f in self.objective_functions:
            hyperparam_grids[f.name] = deepcopy(hyperparam_grid_base)

        return hyperparam_grids


def get_conditions_for_experiment(which, problem_variant, instance_name, cmd_args):
    vars_dict = vars(cmd_args)
    if which == 'main':
        cond = MainExperimentConditions(problem_variant, instance_name, vars_dict)
    else:
        raise ValueError(f"experiment {which} not recognized!")
    return cond


def get_default_options(file_paths):
    options = {"log_progress": True,
               "log_filename": str(file_paths.construct_log_filepath()),
               "log_tf_summaries": True,
               "random_seed": 42,
               "file_paths": file_paths,
               "restore_model": False, }
    return options
