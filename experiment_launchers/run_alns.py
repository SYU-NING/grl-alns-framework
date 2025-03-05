import os
import sys
from pathlib import Path

sys.path.append("../")




sys.path.append(str(Path(__file__).parent.parent))


import numpy as np

from alns.rl.pytorch_agent import PyTorchAgent
from alns.rl.rw_mdp import VictorRouletteWheelMDPAgent
from alns.op_selector import RouletteWheelOperatorSelector, RandomOperatorSelector
from alns.classical_alns import ClassicalALNS
from alns.rl.dqn.dqn_agent import DQNAgent, BaselineDQNAgent, ProbabilisticDQNAgent, ProbabilisticBaselineDQNAgent
from alns.rl.mdp_agent import RandomMDPAgent, MDPAgent
from environment.operator_env import OperatorEnv
from experiment_storage.file_paths import FilePaths
from operators.operator_library import OperatorLibrary
from state.state_generator import SolomonStateGenerator

def get_options(file_paths, example_state, budget):
    options = {"max_nodes":  example_state.num_nodes,
               "log_progress": True,
               "log_filename": str(file_paths.construct_log_filepath()),
               "log_tf_summaries": True,
               "random_seed": 42,
               "file_paths": file_paths,
               "restore_model": False,
               "num_mdp_timesteps": budget * 2
    }
    return options

def main():
    # seed = 42

    num_seeds = 1
    seeds = list(range(126, 126 + num_seeds))

    budget = 10
    max_steps = 100
    num_tours = 16

    initial_tour_method = "random"
    # initial_tour_method = "clark_wright"

    # problem_variant = "TSP"
    problem_variant = "CVRP"
    # problem_variant = "HVRP"
    # problem_variant = "VRPTW"
    # problem_variant = "LRP"

    # instance_name = "C101200"
    # instance_name = "R101200"
    instance_name = "O101200"

    cust_number = [20, 30]

    which_destroy = OperatorLibrary.get_compatible_destroy_operators(problem_variant)
    which_repair = OperatorLibrary.get_compatible_repair_operators(problem_variant)

    for seed in seeds: 
        alns_avgs, alns_mins = [], []

        # gen = HardcodedStateGenerator()
        # instance_name = "default"

        # train_state_list = gen.generate_many(list(range(0,num_tours)))
        # val_state_list = gen.generate_many(list(range(num_tours, num_tours * 2)))
        # test_state_list = gen.generate_many(list(range(num_tours * 2,num_tours * 3)))

        gen = SolomonStateGenerator()

        train_state_list = gen.generate_many(problem_variant, list(range(0, num_tours)), cust_number, instance_name=instance_name, cust_number=cust_number, initial_tour_method=initial_tour_method)
        val_state_list = gen.generate_many(problem_variant, list(range(num_tours, num_tours * 2)), cust_number, instance_name=instance_name, cust_number=cust_number, initial_tour_method=initial_tour_method)
        test_state_list = gen.generate_many(problem_variant, list(range(num_tours * 2, num_tours * 3)), cust_number, instance_name=instance_name, cust_number=cust_number, initial_tour_method=initial_tour_method)

        use_localsearch = False

        agent_class = DQNAgent
        # agent_class = ProbabilisticDQNAgent
        # agent_class = BaselineDQNAgent
        # agent_class = ProbabilisticBaselineDQNAgent
        # agent_class = RandomMDPAgent

        # agent_class = VictorRouletteWheelMDPAgent
        # agent_class = RouletteWheelOperatorSelector

        fixed_scale_pc = 10


        ol = OperatorLibrary(fixed_scale_pc=fixed_scale_pc, random_seed=seed, which_destroy=which_destroy, which_repair=which_repair)
        env = OperatorEnv(problem_variant, ol, seed, budget, use_localsearch=use_localsearch)

        if issubclass(agent_class, MDPAgent):
            agent = agent_class(ol, seed, env=env)
        else:
            agent = agent_class(ol, random_seed=seed)

        data_dir = os.getenv("ATES_EXPERIMENT_DATA_DIR")
        fp = FilePaths(data_dir, 'development', setup_directories=True)
        options = get_options(fp, train_state_list[0], budget)

        if issubclass(type(agent), PyTorchAgent):
            hyperparams = agent.get_default_hyperparameters()
            hyperparams['eps_step_denominator'] = 2
            hyperparams['learning_rate'] = 0.001
            hyperparams['first_hidden_size'] = 64
            hyperparams['scale_repeat'] = 100
            hyperparams['inst_size_repeat'] = 1
            hyperparams['onehot_tour_idxes'] = False
            hyperparams['use_rounding'] = False

            hyperparams['epsilon_start'] = 0.1
            hyperparams['burn_in'] = 50
            hyperparams['alns_temp'] = 10

            hyperparams['arch'] = 'gat'
            hyperparams['lf_dim'] = 32
            hyperparams['num_hidden_layers'] = 3


        if issubclass(type(agent), PyTorchAgent):
            options['restore_model'] = True
            agent.setup(options, hyperparams)

            if agent.algorithm_name in [ProbabilisticDQNAgent.algorithm_name, ProbabilisticBaselineDQNAgent.algorithm_name]:
                agent.compute_dual_policy_scores(train_state_list)

        else:
            agent.setup(options, agent.get_default_hyperparameters())

        # run classical ALNS:
        #
        for test_state in test_state_list:
            alns_inst = ClassicalALNS(agent, random_seed=seed, use_localsearch=use_localsearch)
            alns_inst.run_master_loop(test_state, outer_its_per_customer=10, inner_its=4)
            record = alns_inst.alns_record

            post_repair_record = record[record['nonservice_penalty'] == 0]
            avg_obj = post_repair_record['total_objective'].mean()
            min_obj = alns_inst.best_obj

            alns_avgs.append(avg_obj)
            alns_mins.append(min_obj)

        print(f"ALNS avg objective values: {alns_avgs}")
        print(f"ALNS min objective values: {alns_mins}")

        print(f"mean ALNS avg objective values: {np.mean(alns_avgs)}")
        print(f"mean ALNS min objective values: {np.mean(alns_mins)}")


if __name__ == '__main__':
    main()

