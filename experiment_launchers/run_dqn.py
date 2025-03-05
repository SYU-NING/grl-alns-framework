import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


import numpy as np

from alns.rl.rw_mdp import VictorRouletteWheelMDPAgent
from alns.op_selector import RouletteWheelOperatorSelector, RandomOperatorSelector
from alns.classical_alns import ClassicalALNS
from alns.rl.dqn.dqn_agent import DQNAgent, ProbabilisticDQNAgent, BaselineDQNAgent
from alns.rl.mdp_agent import RandomMDPAgent
from environment.operator_env import OperatorEnv
from experiment_storage.file_paths import FilePaths
from operators.operator_library import OperatorLibrary
from state.state_generator import SolomonStateGenerator

def get_options(file_paths, example_state):
    options = {"max_nodes":  example_state.num_nodes,
               "log_progress": True,
               "log_filename": str(file_paths.construct_log_filepath()),
               "log_tf_summaries": True,
               "random_seed": 42,
               "file_paths": file_paths,
               "restore_model": False}

    return options

def main():
    num_seeds = 1
    seeds = list(range(43, 43 + num_seeds))

    budget = 10
    max_steps = 1000

    # problem_variant = "TSP"
    problem_variant = "CVRP"
    # problem_variant = "VRPTW"
    # problem_variant = "HVRP"
    # problem_variant = "LRP"

    use_localsearch=False

    agent_perfs = []
    baseline_perfs = []

    for seed in seeds:
        # gen = HardcodedStateGenerator()
        # train_state_list = gen.generate_many(list(range(0,num_tours)))
        # val_state_list = gen.generate_many(list(range(num_tours, num_tours * 2)))
        # test_state_list = gen.generate_many(list(range(num_tours * 2,num_tours * 3)))

        gen = SolomonStateGenerator()
        instance_name = "O101200"

        # num_tours = 128
        # cust_number = 20
        # train_state_list = gen.generate_many(list(range(0, num_tours)), instance_name=instance_name, cust_number=cust_number)
        # val_state_list = gen.generate_many(list(range(num_tours, num_tours * 2)), instance_name=instance_name, cust_number=cust_number)
        # test_state_list = gen.generate_many(list(range(num_tours * 2, num_tours * 3)), instance_name=instance_name, cust_number=cust_number)

        initial_tour_method = "random"
        # initial_tour_method = "clark_wright"

        num_customers_train = [20, 30, 40, 50, 60, 70, 80, 90, 100]
        num_customers_val = [20, 30, 40, 50, 100]
        num_customers_test = [20, 30, 40, 50, 60, 70, 80, 90, 100]

        num_tours_train = 128
        num_tours_val = 128
        num_tours_test = 128

        train_state_list = gen.generate_many(problem_variant, list(range(0, num_tours_train)), num_customers_train, instance_name=instance_name, initial_tour_method=initial_tour_method)
        val_state_list = gen.generate_many(problem_variant, list(range(num_tours_train, num_tours_train + num_tours_val)), num_customers_val, instance_name=instance_name, initial_tour_method=initial_tour_method)
        test_state_list = gen.generate_many(problem_variant, list(range(num_tours_train + num_tours_val, num_tours_train + num_tours_val + num_tours_test)), num_customers_test, instance_name=instance_name, initial_tour_method=initial_tour_method)

        fixed_scale_pc = 10
        which_destroy = OperatorLibrary.get_compatible_destroy_operators(problem_variant, include_noop=True)
        which_repair = OperatorLibrary.get_compatible_repair_operators(problem_variant)


        ol = OperatorLibrary(fixed_scale_pc=fixed_scale_pc,
                             random_seed=seed, which_destroy=which_destroy, which_repair=which_repair)

        env = OperatorEnv(problem_variant, ol, seed, budget, use_localsearch=use_localsearch)

        baseline_agent = RandomMDPAgent(ol, seed, env=env)
        baseline_perf = baseline_agent.eval(test_state_list)
        baseline_perfs.append(baseline_perf)

        # agent = DQNAgent(ol, seed, env=env)
        # agent = BaselineDQNAgent(ol, seed, env=env)
        agent = VictorRouletteWheelMDPAgent(ol, seed, env=env)

        data_dir = os.getenv("ATES_EXPERIMENT_DATA_DIR")

        fp = FilePaths(data_dir, 'development', setup_directories=True)
        options = get_options(fp, train_state_list[0])
        options['num_mdp_timesteps'] = budget * 2

        hyperparams = agent.get_default_hyperparameters()

        # hyperparams['weight_segment'] = 200
        # hyperparams['reaction_factor'] = 0.99

        hyperparams['eps_step_denominator'] = 2
        hyperparams['learning_rate'] = 0.001
        hyperparams['first_hidden_size'] = 64
        hyperparams['onehot_tour_idxes'] = False
        # hyperparams['epsilon_start'] = 0.1

        hyperparams['use_rounding'] = False

        hyperparams['burn_in'] = 50
        hyperparams['arch'] = 'gat'
        hyperparams['lf_dim'] = 32
        hyperparams['num_hidden_layers'] = 3
        hyperparams['scale_repeat'] = 100
        hyperparams['inst_size_repeat'] = 1
        hyperparams['alns_temp'] = 1


        agent.setup(options, hyperparams)
        agent.train(train_state_list, val_state_list, max_steps)

        final_perf = agent.eval(test_state_list)
        agent_perfs.append(final_perf)


    print(f"random baseline performances on test set: {baseline_perfs}")
    print(f"random baseline mean performance on test set: {np.mean(baseline_perfs)}")

    print(f"DQN agent performances on test set: {agent_perfs}")
    print(f"DQN agent mean performance on test set: {np.mean(agent_perfs)}")


if __name__ == '__main__':
    main()

