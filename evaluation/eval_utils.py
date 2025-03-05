from itertools import product
import numpy as np

from utils.config_utils import local_np_seed


def generate_search_space(parameter_grid,
                          random_search=False,
                          random_search_num_options=20,
                          random_search_seed=42):
    combinations = list(product(*parameter_grid.values()))
    search_space = {i: combinations[i] for i in range(len(combinations))}

    if random_search:
        if not random_search_num_options > len(search_space):
            reduced_space = {}
            with local_np_seed(random_search_seed):
                random_indices = np.random.choice(len(search_space), random_search_num_options, replace=False)
                for random_index in random_indices:
                    reduced_space[random_index] = search_space[random_index]
            search_space = reduced_space
    return search_space


def construct_search_spaces(experiment_conditions):
    parameter_search_spaces = {}
    objective_functions = experiment_conditions.objective_functions

    for obj_fun in objective_functions:
        parameter_search_spaces[obj_fun.name] = {}
        for agent in experiment_conditions.all_agents:
            if agent.requires_hyperopt or agent.requires_tune:
                if agent.algorithm_name in experiment_conditions.hyperparam_grids[obj_fun.name]:
                    agent_grid = experiment_conditions.hyperparam_grids[obj_fun.name][agent.algorithm_name]
                    combinations = list(product(*agent_grid.values()))
                    search_space = {}
                    for i in range(len(combinations)):
                        k = str(i)
                        v = dict(zip(list(agent_grid.keys()), combinations[i]))
                        search_space[k] = v
                    parameter_search_spaces[obj_fun.name][agent.algorithm_name] = search_space

    return parameter_search_spaces

def construct_network_seeds(eval_on_train, num_train_graphs, num_validation_graphs, num_test_graphs):
    if not eval_on_train:
        base_offset = 0
        validation_seeds = list(range(base_offset, base_offset + num_validation_graphs))
        test_seeds = list(range(base_offset + num_validation_graphs, base_offset + num_validation_graphs + num_test_graphs))
        offset = base_offset + num_validation_graphs + num_test_graphs
        train_seeds = list(range(offset, offset + num_train_graphs))
    else:
        assert num_train_graphs == num_validation_graphs == num_test_graphs, "If using eval_on_train, number of graphs should be the same."
        base_offset = 0
        validation_seeds = list(range(base_offset, base_offset + num_validation_graphs))
        test_seeds = list(range(base_offset, base_offset + num_test_graphs))
        train_seeds = list(range(base_offset, base_offset + num_train_graphs))
    return train_seeds, validation_seeds, test_seeds

def get_model_seed(run_number):
    return int(run_number * 42)

def get_run_number(model_seed):
    return int(model_seed / 42)

def find_max_property_in_list(instance_list, fn_to_apply):
    max_prop = float("-inf")
    for inst in instance_list:
        max_prop = max(max_prop, fn_to_apply(inst))

    return max_prop

def find_max_nodes(instance_list):
    return find_max_property_in_list(instance_list, lambda g: g.num_nodes)

