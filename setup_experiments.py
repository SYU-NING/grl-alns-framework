import argparse
import pickle
import random
import traceback
from copy import copy, deepcopy
from datetime import datetime

from alns.op_selector import create_hardcoded_scale_selector
from alns.rl.dqn.dqn_agent import *
from alns.rl.hybrid_agent import HybridAgent
from alns.rl.rw_mdp import VictorRouletteWheelMDPAgent
from experiment_storage.file_paths import FilePaths
from experiment_storage.storage import EvaluationStorage

from evaluation.eval_utils import construct_search_spaces, generate_search_space
from evaluation.experiment_conditions import get_conditions_for_experiment
from tasks import OptimizeHyperparamsTask, EvaluateTask, ALNSTask, TuneALNSTask

from utils.config_utils import get_logger_instance
from utils.general_utils import chunks


def setup_hyperopt_part(experiment_conditions, parent_dir, existing_experiment_id, hyps_chunk_size, seeds_chunk_size):

    experiment_started_datetime = datetime.now()
    started_str = experiment_started_datetime.strftime(FilePaths.DATE_FORMAT)
    started_millis = experiment_started_datetime.timestamp()

    experiment_id = existing_experiment_id
    file_paths = FilePaths(parent_dir, experiment_id)
    storage = EvaluationStorage(file_paths)

    parameter_search_spaces = construct_search_spaces(experiment_conditions)

    storage.insert_experiment_details(
        experiment_conditions,
        started_str,
        started_millis,
        parameter_search_spaces)

    logger = get_logger_instance(str(file_paths.construct_log_filepath()))
    setup_hyperparameter_optimisations(storage,
                                       file_paths,
                                       experiment_conditions,
                                       experiment_id,
                                       hyps_chunk_size,
                                       seeds_chunk_size)
    logger.info(
        f"{datetime.now().strftime(FilePaths.DATE_FORMAT)} Completed setting up hyperparameter optimization tasks.")


def setup_eval_part(experiment_conditions, parent_dir, existing_experiment_id, reinsert_details=True):
    experiment_id = existing_experiment_id
    file_paths = FilePaths(parent_dir, experiment_id, setup_directories=False)
    storage = EvaluationStorage(file_paths)

    logger = get_logger_instance(str(file_paths.construct_log_filepath()))
    eval_tasks = construct_eval_tasks(experiment_id,
                                            file_paths,
                                            experiment_conditions,
                                            storage,
                                            reinsert_details)

    logger.info(f"have just setup {len(eval_tasks)} evaluation tasks.")
    storage.store_tasks(eval_tasks, "eval")



def setup_tune_part(experiment_conditions, parent_dir, existing_experiment_id, hyps_chunk_size,
                                       seeds_chunk_size):
    experiment_id = existing_experiment_id

    experiment_started_datetime = datetime.now()
    started_str = experiment_started_datetime.strftime(FilePaths.DATE_FORMAT)
    started_millis = experiment_started_datetime.timestamp()

    file_paths = FilePaths(parent_dir, experiment_id)
    storage = EvaluationStorage(file_paths)

    parameter_search_spaces = construct_search_spaces(experiment_conditions)

    storage.insert_experiment_details(
        experiment_conditions,
        started_str,
        started_millis,
        parameter_search_spaces)


    logger = get_logger_instance(str(file_paths.construct_log_filepath()))
    construct_tune_tasks(storage,
                                       file_paths,
                                       experiment_conditions,
                                       experiment_id,
                                       hyps_chunk_size,
                                       seeds_chunk_size)


def construct_tune_tasks(storage,
                                       file_paths,
                                       experiment_conditions,
                                       experiment_id,
                                       hyps_chunk_size,
                                       seeds_chunk_size):
    experiment_params = experiment_conditions.experiment_params
    model_seeds = experiment_params['model_seeds']

    tune_tasks = []

    start_task_id = 1

    optimal_opselec_hyperparams = storage.retrieve_optimal_hyperparams(experiment_id, False, task_type="hyperopt")

    for network_generator in experiment_conditions.network_generators:
        for obj_fun in experiment_conditions.objective_functions:
            for agent in experiment_conditions.all_agents:

                if agent.requires_tune:
                    if agent.algorithm_name in experiment_conditions.hyperparam_grids[obj_fun.name]:
                        agent_param_grid = experiment_conditions.hyperparam_grids[obj_fun.name][agent.algorithm_name]

                        if agent.algorithm_name == ProbabilisticDQNAgent.algorithm_name:
                            parent_class = DQNAgent
                            setting = (network_generator.name, obj_fun.name, DQNAgent.algorithm_name)
                            best_opselec_hyperparams, best_opselec_hyperparams_id = optimal_opselec_hyperparams[setting]
                        elif agent.algorithm_name == ProbabilisticBaselineDQNAgent.algorithm_name:
                            parent_class = BaselineDQNAgent
                            setting = (network_generator.name, obj_fun.name, BaselineDQNAgent.algorithm_name)
                            best_opselec_hyperparams, best_opselec_hyperparams_id = optimal_opselec_hyperparams[setting]
                        else:
                            parent_class = None
                            best_opselec_hyperparams, best_opselec_hyperparams_id = ({}, '0')

                        local_tasks = create_tune_tasks(
                                start_task_id,
                                agent,
                                obj_fun,
                                network_generator,
                                best_opselec_hyperparams,
                                best_opselec_hyperparams_id,
                                experiment_conditions,
                                storage,
                                file_paths,
                                agent_param_grid,
                                model_seeds,
                                experiment_id,
                                hyps_chunk_size,
                                seeds_chunk_size,
                                parent_class)
                        tune_tasks.extend(local_tasks)
                        start_task_id += len(local_tasks)


    logger = get_logger_instance(str(file_paths.construct_log_filepath()))
    logger.info(f"created {len(tune_tasks)} tune tasks.")
    storage.store_tasks(tune_tasks, "tune")


def create_tune_tasks(start_task_id,
                                     agent,
                                     objective_function,
                                     network_generator,
                                    best_opselec_hyperparams,
                                    best_opselec_hyperparams_id,
                                     experiment_conditions,
                                     storage,
                                     file_paths,
                                     parameter_grid,
                                     model_seeds,
                                     experiment_id,
                                     hyps_chunk_size,
                                     seeds_chunk_size,
                                    parent_class):
    parameter_keys = list(parameter_grid.keys())
    local_tasks = []
    search_space = list(generate_search_space(parameter_grid).items())

    search_space_chunks = list(chunks(search_space, hyps_chunk_size))
    model_seed_chunks = list(chunks(model_seeds, seeds_chunk_size))

    print(search_space_chunks)
    print(model_seed_chunks)

    additional_opts = {}

    task_id = start_task_id
    for ss_chunk in search_space_chunks:
        for ms_chunk in model_seed_chunks:
            local_tasks.append(TuneALNSTask(task_id,
                                                           agent,
                                                           objective_function,
                                                           network_generator,
                                                           best_opselec_hyperparams,
                                                           best_opselec_hyperparams_id,
                                                           experiment_conditions,
                                                           storage,
                                                           parameter_keys,
                                                           ss_chunk,
                                                           ms_chunk,
                                                           parent_class=parent_class,
                                                           additional_opts=additional_opts))
            task_id += 1

    return local_tasks

def setup_alns_part(experiment_conditions, parent_dir, existing_experiment_id):
    experiment_id = existing_experiment_id

    experiment_started_datetime = datetime.now()
    started_str = experiment_started_datetime.strftime(FilePaths.DATE_FORMAT)
    started_millis = experiment_started_datetime.timestamp()

    file_paths = FilePaths(parent_dir, experiment_id)
    storage = EvaluationStorage(file_paths)

    parameter_search_spaces = construct_search_spaces(experiment_conditions)

    storage.insert_experiment_details(
        experiment_conditions,
        started_str,
        started_millis,
        parameter_search_spaces)


    logger = get_logger_instance(str(file_paths.construct_log_filepath()))
    alns_tasks = construct_alns_tasks(experiment_id,
                                            file_paths,
                                            experiment_conditions,
                                            storage)

    logger.info(f"have just setup {len(alns_tasks)} alns tasks.")
    storage.store_tasks(alns_tasks, "alns")

def construct_alns_tasks(experiment_id, file_paths, original_experiment_conditions, storage):

    experiment_conditions = deepcopy(original_experiment_conditions)
    logger = get_logger_instance(str(file_paths.construct_log_filepath()))

    tasks = []
    task_id = 1

    try:
        hyps_search_spaces = storage.get_experiment_details(experiment_id)["parameter_search_spaces"]
        optimal_hyperparams = storage.retrieve_optimal_hyperparams(experiment_id, False, "hyperopt")
        try:
            optimal_hyperparams_tune = storage.retrieve_optimal_hyperparams(experiment_id, False, "tune")
            optimal_hyperparams.update(optimal_hyperparams_tune)
        except KeyError:
            logger.warn("could not find tune hyperparameters.")

    except (KeyError, ValueError):
        logger.warn("no hyperparameters retrieved as no configured agents require them.")
        logger.warn(traceback.format_exc())
        hyps_search_spaces = None
        optimal_hyperparams = {}

    print("optimal hyps are")
    print(optimal_hyperparams)
    for network_generator in experiment_conditions.network_generators:
        for objective_function in experiment_conditions.objective_functions:

            additional_opts = {}
            model_seeds = experiment_conditions.experiment_params['model_seeds']
            model_seed_chunks = list(chunks(model_seeds, experiment_conditions.seeds_chunk_size))

            for agent in experiment_conditions.all_agents:
                setting = (network_generator.name, objective_function.name, agent.algorithm_name)

                # if agent.algorithm_name == DQNAgent.algorithm_name:
                #     continue # continuing, as will use probabilistic policy with selectable temp.

                if agent.requires_tune:
                    best_alns_hyperparams, best_alns_hyperparams_id = optimal_hyperparams[setting]
                else:
                    best_alns_hyperparams, best_alns_hyperparams_id = ({}, '0')

                if agent.algorithm_name == HybridAgent.algorithm_name:
                    standard_dqn_setting = (network_generator.name, objective_function.name, DQNAgent.algorithm_name)
                    best_opselec_hyperparams, best_opselec_hyperparams_id = optimal_hyperparams[standard_dqn_setting]

                    standard_vrw_setting = (network_generator.name, objective_function.name, VictorRouletteWheelMDPAgent.algorithm_name)
                    best_hybrid_hyperparams, best_hybrid_hyperparams_id = optimal_hyperparams[standard_vrw_setting]

                    for model_seed_chunk in model_seed_chunks:
                        tasks.append(ALNSTask(task_id,
                                              agent,
                                              objective_function,
                                              network_generator,
                                              best_opselec_hyperparams,
                                              best_opselec_hyperparams_id,
                                              best_alns_hyperparams,
                                              best_alns_hyperparams_id,
                                              experiment_conditions,
                                              storage,
                                              model_seed_chunk,
                                              additional_opts=additional_opts,
                                              best_hybrid_hyperparams=best_hybrid_hyperparams,
                                              best_hybrid_hyperparams_id=best_hybrid_hyperparams_id
                                              ))
                        task_id += 1

                else:

                    if agent.is_trainable:
                        # agent went through the training stage.
                        if agent.algorithm_name == ProbabilisticDQNAgent.algorithm_name:
                            standard_dqn_setting = (network_generator.name, objective_function.name, DQNAgent.algorithm_name)
                            best_opselec_hyperparams, best_opselec_hyperparams_id = optimal_hyperparams[standard_dqn_setting]
                        elif agent.algorithm_name == ProbabilisticBaselineDQNAgent.algorithm_name:
                            standard_dqn_setting = (network_generator.name, objective_function.name, BaselineDQNAgent.algorithm_name)
                            best_opselec_hyperparams, best_opselec_hyperparams_id = optimal_hyperparams[standard_dqn_setting]
                        else:
                            best_opselec_hyperparams, best_opselec_hyperparams_id = optimal_hyperparams[setting]
                    else:
                        # agent didn't -- don't need these params.
                        best_opselec_hyperparams, best_opselec_hyperparams_id = ({}, '0')


                    if agent.algorithm_name == ProbabilisticDQNAgent.algorithm_name:
                        parent_class = DQNAgent
                    elif agent.algorithm_name == ProbabilisticBaselineDQNAgent.algorithm_name:
                        parent_class = BaselineDQNAgent
                    else:
                        parent_class = None

                    for model_seed_chunk in model_seed_chunks:
                        tasks.append(ALNSTask(task_id,
                                              agent,
                                              objective_function,
                                              network_generator,
                                              best_opselec_hyperparams,
                                              best_opselec_hyperparams_id,
                                              best_alns_hyperparams,
                                              best_alns_hyperparams_id,
                                              experiment_conditions,
                                              storage,
                                              model_seed_chunk,
                                              parent_class=parent_class,
                                              additional_opts=additional_opts
                                              ))
                        task_id += 1

                # if agent.is_deterministic and model_seed > 0:
                #     # deterministic agents only need to be evaluated once as they involve no randomness.
                #     break

    return tasks


def construct_eval_tasks(experiment_id,
                         file_paths,
                         original_experiment_conditions,
                         storage,
                         reinsert_details):

    experiment_conditions = deepcopy(original_experiment_conditions)
    logger = get_logger_instance(str(file_paths.construct_log_filepath()))

    if reinsert_details:
        parameter_search_spaces = construct_search_spaces(experiment_conditions)

        experiment_started_datetime = datetime.now()
        started_str = experiment_started_datetime.strftime(FilePaths.DATE_FORMAT)
        started_millis = experiment_started_datetime.timestamp()
        storage.insert_experiment_details(
            experiment_conditions,
            started_str,
            started_millis,
            parameter_search_spaces)

    tasks = []
    task_id = 1

    try:
        hyps_search_spaces = storage.get_experiment_details(experiment_id)["parameter_search_spaces"]
        optimal_hyperparams = storage.retrieve_optimal_hyperparams(experiment_id, False)
    except (KeyError, ValueError):
        logger.warn("no hyperparameters retrieved as no configured agents require them.")
        logger.warn(traceback.format_exc())
        hyps_search_spaces = None
        optimal_hyperparams = {}

    seeds_chunk_size = experiment_conditions.seeds_chunk_size

    for network_generator in experiment_conditions.network_generators:
        for objective_function in experiment_conditions.objective_functions:
            for agent in experiment_conditions.all_agents:
                if (not agent.is_mdp_based) or agent.algorithm_name in [ProbabilisticDQNAgent.algorithm_name, ProbabilisticBaselineDQNAgent.algorithm_name, HybridAgent.algorithm_name]:
                    continue

                additional_opts = {}

                hyperparams_needed = agent.requires_hyperopt

                setting = (network_generator.name, objective_function.name, agent.algorithm_name)

                if not hyperparams_needed:
                    hyp_space = {'0': {}}
                    best_hyperparams, best_hyperparams_id =  ({}, '0')
                else:
                    if setting in optimal_hyperparams:
                        hyp_space = hyps_search_spaces[objective_function.name][agent.algorithm_name]
                        best_hyperparams, best_hyperparams_id = optimal_hyperparams[setting]
                    else:
                        hyp_space = {'0': {}}
                        best_hyperparams, best_hyperparams_id = ({}, '0')

                print(f"agent {agent.algorithm_name}: going with hyps {best_hyperparams}")
                model_seeds = experiment_conditions.experiment_params['model_seeds']
                model_seed_chunks = list(chunks(model_seeds, seeds_chunk_size))

                for model_seed_chunk in model_seed_chunks:
                # if agent.is_deterministic and model_seed > 0:
                #     # deterministic agents only need to be evaluated once as they involve no randomness.
                #     break

                    tasks.append(EvaluateTask(task_id,
                                            agent,
                                            objective_function,
                                            network_generator,
                                            hyp_space,
                                            best_hyperparams,
                                            best_hyperparams_id,
                                            experiment_conditions,
                                            storage,
                                            model_seed_chunk,
                                            additional_opts=additional_opts
                                            ))
                    task_id += 1

    return tasks


def setup_hyperparameter_optimisations(storage,
                                       file_paths,
                                       experiment_conditions,
                                       experiment_id,
                                       hyps_chunk_size,
                                       seeds_chunk_size):
    experiment_params = experiment_conditions.experiment_params
    model_seeds = experiment_params['model_seeds']

    hyperopt_tasks = []

    start_task_id = 1
    for network_generator in experiment_conditions.network_generators:
        for obj_fun in experiment_conditions.objective_functions:
            for agent in experiment_conditions.all_agents:
                if not agent.is_mdp_based:
                    continue

                if agent.requires_hyperopt:
                    if agent.algorithm_name in experiment_conditions.hyperparam_grids[obj_fun.name]:
                        agent_param_grid = experiment_conditions.hyperparam_grids[obj_fun.name][agent.algorithm_name]

                        local_tasks = construct_parameter_search_tasks(
                                start_task_id,
                                agent,
                                obj_fun,
                                network_generator,
                                experiment_conditions,
                                storage,
                                file_paths,
                                agent_param_grid,
                                model_seeds,
                                experiment_id,
                                hyps_chunk_size,
                                seeds_chunk_size)
                        hyperopt_tasks.extend(local_tasks)
                        start_task_id += len(local_tasks)


    logger = get_logger_instance(str(file_paths.construct_log_filepath()))
    logger.info(f"created {len(hyperopt_tasks)} hyperparameter optimisation tasks.")
    storage.store_tasks(hyperopt_tasks, "hyperopt")


def construct_parameter_search_tasks(start_task_id,
                                     agent,
                                     objective_function,
                                     network_generator,
                                     experiment_conditions,
                                     storage,
                                     file_paths,
                                     parameter_grid,
                                     model_seeds,
                                     experiment_id,
                                     hyps_chunk_size,
                                     seeds_chunk_size):
    parameter_keys = list(parameter_grid.keys())
    local_tasks = []
    search_space = list(generate_search_space(parameter_grid).items())

    search_space_chunks = list(chunks(search_space, hyps_chunk_size))
    model_seed_chunks = list(chunks(model_seeds, seeds_chunk_size))

    print(search_space_chunks)
    print(model_seed_chunks)

    additional_opts = {}

    task_id = start_task_id
    for ss_chunk in search_space_chunks:
        for ms_chunk in model_seed_chunks:
            local_tasks.append(OptimizeHyperparamsTask(task_id,
                                                           agent,
                                                           objective_function,
                                                           network_generator,
                                                           experiment_conditions,
                                                           storage,
                                                           parameter_keys,
                                                           ss_chunk,
                                                           ms_chunk,
                                                           additional_opts=additional_opts))
            task_id += 1

    return local_tasks


def get_base_opts():
    return {}

def main():
    parser = argparse.ArgumentParser(description="Setup tasks for experiments.")
    parser.add_argument("--experiment_part", required=True, type=str,
                        help="Whether to run hyperparameter optimisation or evaluation.",
                        choices=["hyperopt", "eval", "tune", "alns"])

    parser.add_argument("--which", required=True, type=str,
                        help="Which experiment to run",
                        choices=["main", "topvar"])

    parser.add_argument("--parent_dir", type=str, help="Root path for storing experiment data.")
    parser.add_argument("--experiment_id", required=True, help="experiment id to use")

    parser.add_argument("--problem_variant", type=str, help="Problem variant to apply framework to.",
                        choices=["TSP", "CVRP", "VRPTW", "HVRP", "LRP"])

    parser.add_argument("--instance_name", type=str, required=True, help="Underlying graph topology name.")

    parser.add_argument("--fixed_scale_pc", type=float, help="Percentage of customer nodes to use as scale for the operators.")

    parser.add_argument("--budget", type=int, help="Budget of destroy-operator pairs.")

    parser.add_argument("--alns_outer_its_per_customer", type=float, help="Number of ALNS outer iterations per customer node.")
    parser.add_argument("--alns_inner_its", type=int, help="Number of ALNS inner iterations.")


    args = parser.parse_args()

    experiment_conditions = get_conditions_for_experiment(args.which, args.problem_variant, args.instance_name, args)

    if args.experiment_part == "hyperopt":
        setup_hyperopt_part(experiment_conditions, args.parent_dir, args.experiment_id, experiment_conditions.hyps_chunk_size, experiment_conditions.seeds_chunk_size)
    elif args.experiment_part == "eval":
        setup_eval_part(experiment_conditions, args.parent_dir, args.experiment_id)
    if args.experiment_part == "tune":
        setup_tune_part(experiment_conditions, args.parent_dir, args.experiment_id, experiment_conditions.hyps_chunk_size, experiment_conditions.seeds_chunk_size)
    elif args.experiment_part == "alns":
        setup_alns_part(experiment_conditions, args.parent_dir, args.experiment_id)


if __name__ == "__main__":
    main()
