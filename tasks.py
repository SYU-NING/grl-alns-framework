import argparse
import time
from copy import deepcopy

import traceback
import dill
import numpy as np

from alns.classical_alns import ClassicalALNS
from alns.rl.dqn.dqn_agent import ProbabilisticDQNAgent, DQNAgent, ProbabilisticBaselineDQNAgent
from alns.rl.hybrid_agent import HybridAgent
from alns.rl.rw_mdp import VictorRouletteWheelMDPAgent
from environment.operator_env import OperatorEnv
from evaluation.eval_utils import find_max_nodes
from experiment_storage.file_paths import FilePaths
from operators.operator_library import OperatorLibrary
from utils.config_utils import get_logger_instance


class OptimizeHyperparamsTask(object):
    def __init__(self,
                 task_id,
                 agent,
                 objective_function,
                 network_generator,
                 experiment_conditions,
                 storage,
                 parameter_keys,
                 search_space_chunk,
                 model_seeds_chunk,
                 train_kwargs=None,
                 additional_opts=None
                 ):
        self.task_id = task_id
        self.agent = agent
        self.objective_function = objective_function
        self.network_generator = network_generator
        self.experiment_conditions = experiment_conditions
        self.storage = storage
        self.parameter_keys = parameter_keys
        self.search_space_chunk = search_space_chunk
        self.model_seeds_chunk = model_seeds_chunk
        self.train_kwargs = train_kwargs
        self.additional_opts = additional_opts

    def run(self):
        log_filename = str(self.storage.file_paths.construct_log_filepath())
        logger = get_logger_instance(log_filename)

        for hyperparams_id, combination in self.search_space_chunk:
            hyperparams = {}

            for idx, param_value in enumerate(tuple(combination)):
                param_key = self.parameter_keys[idx]
                hyperparams[param_key] = param_value

            logger.info(f"executing with hyps {hyperparams}")

            for model_seed in self.model_seeds_chunk:
                logger.info(f"executing for seed {model_seed}")
                exp_copy = deepcopy(self.experiment_conditions)

                model_identifier_prefix = self.storage.file_paths.construct_model_identifier_prefix(self.agent.algorithm_name,
                                                                                       self.objective_function.name,
                                                                                       self.network_generator.name,
                                                                                       model_seed, hyperparams_id)

                gen_instance = self.network_generator()
                train_graphs, validation_graphs = gen_instance.generate_many(exp_copy.problem_variant, exp_copy.train_seeds, exp_copy.num_customers_train, instance_name=exp_copy.instance_name,
                                                                                          initial_tour_method=exp_copy.initial_tour_method), \
                    gen_instance.generate_many(exp_copy.problem_variant, exp_copy.validation_seeds, exp_copy.num_customers_validate, instance_name=exp_copy.instance_name, initial_tour_method=exp_copy.initial_tour_method)
                ol = OperatorLibrary(fixed_scale_pc=exp_copy.fixed_scale_pc,
                                     random_seed=model_seed,
                                     which_destroy=exp_copy.which_destroy, which_repair=exp_copy.which_repair)
                env = OperatorEnv(exp_copy.problem_variant, ol, model_seed, exp_copy.budget, use_localsearch=exp_copy.use_localsearch)
                agent_instance = self.agent(ol, model_seed, env=env)

                run_options = {}
                run_options["max_nodes"] = find_max_nodes(train_graphs + validation_graphs)
                run_options["random_seed"] = model_seed
                run_options["file_paths"] = self.storage.file_paths
                run_options["log_progress"] = True

                log_filename = str(self.storage.file_paths.construct_log_filepath())
                run_options["log_filename"] = log_filename
                run_options["model_identifier_prefix"] = model_identifier_prefix
                run_options["log_tf_summaries"] = False

                run_options["restore_model"] = False

                run_options.update((self.additional_opts or {}))

                try:
                    agent_instance.setup(run_options, hyperparams)
                    if self.agent.is_trainable:
                        max_steps = exp_copy.agent_budgets[self.objective_function.name][self.agent.algorithm_name]

                        agent_train_kwargs =  (self.train_kwargs or {})
                        agent_instance.train(train_graphs, validation_graphs, max_steps, **agent_train_kwargs)

                    average_reward = agent_instance.eval(validation_graphs, validation=False)
                    self.storage.write_hyperopt_results(model_identifier_prefix, average_reward, task_type="hyperopt")

                    agent_instance.finalize()
                except BaseException:
                    logger.warn("got exception while training & evaluating agent")
                    logger.warn(traceback.format_exc())


class EvaluateTask(object):
    def __init__(self,
                 task_id,
                 agent,
                 objective_function,
                 network_generator,
                 hyp_space,
                 best_hyperparams,
                 best_hyperparams_id,
                 experiment_conditions,
                 storage,
                 model_seeds_chunk,
                 additional_opts=None):
        self.task_id = task_id
        self.agent = agent
        self.objective_function = objective_function
        self.network_generator = network_generator

        self.hyp_space = hyp_space
        self.best_hyperparams = best_hyperparams
        self.best_hyperparams_id = best_hyperparams_id

        self.experiment_conditions = experiment_conditions
        self.storage = storage
        self.model_seeds_chunk = model_seeds_chunk
        self.additional_opts = additional_opts

    def run(self):
        log_filename = str(self.storage.file_paths.construct_log_filepath())
        logger = get_logger_instance(log_filename)
        local_results = []

        for hyps_id, hyps in self.hyp_space.items():
            #hyps_id = int(hyps_id_str)

            for model_seed in self.model_seeds_chunk:
                exp_copy = deepcopy(self.experiment_conditions)


                gen_instance = self.network_generator()
                test_graphs = gen_instance.generate_many(exp_copy.problem_variant, exp_copy.test_seeds, exp_copy.num_customers_test, instance_name=exp_copy.instance_name, initial_tour_method=exp_copy.initial_tour_method)

                ol = OperatorLibrary(fixed_scale_pc=exp_copy.fixed_scale_pc,
                                     random_seed=model_seed,
                                     which_destroy=exp_copy.which_destroy, which_repair=exp_copy.which_repair)
                env = OperatorEnv(exp_copy.problem_variant, ol, model_seed, exp_copy.budget, use_localsearch=exp_copy.use_localsearch)


                model_identifier_prefix = self.storage.file_paths.construct_model_identifier_prefix(self.agent.algorithm_name,
                                                                                       self.objective_function.name,
                                                                                       self.network_generator.name,
                                                                                       model_seed,
                                                                                       hyps_id)

                agent_instance = self.agent(ol, model_seed, env=env,
                                            track_operators=exp_copy.track_operators,
                                            op_tracking_dir=self.storage.file_paths.op_tracking_dir,
                                            model_identifier_prefix=model_identifier_prefix)

                run_options = {}
                run_options["max_nodes"] = find_max_nodes(test_graphs)
                run_options['random_seed'] = model_seed
                run_options["restore_model"] = True

                run_options["model_identifier_prefix"] = model_identifier_prefix
                run_options["file_paths"] = self.storage.file_paths
                run_options["log_progress"] = True
                run_options["log_filename"] = log_filename

                run_options.update((self.additional_opts or {}))

                if agent_instance.is_trainable:
                    agent_instance.setup(run_options, hyps)

                time_started_seconds = time.time()
                perfs_per_size = agent_instance.eval_separate_per_size(test_graphs)
                time_ended_seconds = time.time()
                durations_ms = (time_ended_seconds - time_started_seconds) * 1000

                for entry in perfs_per_size:
                    result_row = {}
                    result_row['initial_tour_method'] = exp_copy.initial_tour_method
                    result_row['cust_number'] = entry['cust_number']
                    result_row['network_generator'] = self.network_generator.name
                    result_row['objective_function'] = self.objective_function.name

                    result_row['algorithm'] = self.agent.algorithm_name
                    result_row['agent_seed'] = model_seed
                    result_row['instance_name'] = exp_copy.instance_name
                    result_row['cummulative_reward'] = entry['perf']

                    result_row['duration_ms'] = durations_ms / len(test_graphs)

                    result_row['hyps_id'] = hyps_id
                    result_row['is_best_hyps'] = (hyps_id == self.best_hyperparams_id)

                    local_results.append(result_row)

                if agent_instance.is_trainable:
                    agent_instance.finalize()

        self.storage.write_results("eval", local_results, self.task_id)

class TuneALNSTask(object):
    def __init__(self,
                 task_id,
                 agent,
                 objective_function,
                 network_generator,
                 best_opselec_hyperparams,
                 best_opselec_hyperparams_id,
                 experiment_conditions,
                 storage,
                 parameter_keys,
                 search_space_chunk,
                 model_seeds_chunk,
                 parent_class=None,
                 additional_opts=None
                 ):


        self.task_id = task_id
        self.agent = agent
        self.objective_function = objective_function
        self.network_generator = network_generator

        self.best_opselec_hyperparams = best_opselec_hyperparams
        self.best_opselec_hyperparams_id = best_opselec_hyperparams_id


        self.experiment_conditions = experiment_conditions
        self.storage = storage
        self.parameter_keys = parameter_keys
        self.search_space_chunk = search_space_chunk
        self.model_seeds_chunk = model_seeds_chunk

        self.opselec_parent_class = parent_class
        self.additional_opts = additional_opts

    def run(self):
        log_filename = str(self.storage.file_paths.construct_log_filepath())
        logger = get_logger_instance(log_filename)

        for hyperparams_id, combination in self.search_space_chunk:
            hyperparams = {}

            for idx, param_value in enumerate(tuple(combination)):
                param_key = self.parameter_keys[idx]
                hyperparams[param_key] = param_value

            logger.info(f"executing with hyps {hyperparams}")

            for model_seed in self.model_seeds_chunk:
                logger.info(f"executing for seed {model_seed}")
                exp_copy = deepcopy(self.experiment_conditions)

                model_identifier_prefix = self.storage.file_paths.construct_model_identifier_prefix(self.agent.algorithm_name,
                                                                                       self.objective_function.name,
                                                                                       self.network_generator.name,
                                                                                       model_seed, hyperparams_id)

                gen_instance = self.network_generator()
                train_graphs, validation_graphs, test_graphs = gen_instance.generate_many(exp_copy.problem_variant, exp_copy.train_seeds, exp_copy.num_customers_train, instance_name=exp_copy.instance_name,
                                                                                          initial_tour_method=exp_copy.initial_tour_method), \
                    gen_instance.generate_many(exp_copy.problem_variant, exp_copy.validation_seeds, exp_copy.num_customers_validate, instance_name=exp_copy.instance_name, initial_tour_method=exp_copy.initial_tour_method), \
                    gen_instance.generate_many(exp_copy.problem_variant, exp_copy.test_seeds, exp_copy.num_customers_test, instance_name=exp_copy.instance_name, initial_tour_method=exp_copy.initial_tour_method)

                ol = OperatorLibrary(fixed_scale_pc=exp_copy.fixed_scale_pc,
                                     random_seed=model_seed,
                                     which_destroy=exp_copy.which_destroy, which_repair=exp_copy.which_repair)
                env = OperatorEnv(exp_copy.problem_variant, ol, model_seed, exp_copy.budget, use_localsearch=exp_copy.use_localsearch)

                if self.agent.is_mdp_based:
                    agent_instance = self.agent(ol, model_seed, env=env)
                else:
                    agent_instance = self.agent(ol, random_seed=model_seed)

                if self.agent.is_mdp_based and self.agent.is_trainable:
                    run_options = {}
                    run_options["max_nodes"] = find_max_nodes(train_graphs + validation_graphs + test_graphs)
                    run_options['random_seed'] = model_seed
                    run_options["restore_model"] = True
                    run_options['num_mdp_timesteps'] = exp_copy.budget * 2
                    parent_model_prefix = self.storage.file_paths.construct_model_identifier_prefix(self.opselec_parent_class.algorithm_name,
                                                                                                    self.objective_function.name,
                                                                                                    self.network_generator.name,
                                                                                                    model_seed,
                                                                                                    self.best_opselec_hyperparams_id)
                    run_options["model_identifier_prefix"] = parent_model_prefix
                    run_options["file_paths"] = self.storage.file_paths
                    run_options["log_progress"] = True
                    run_options["log_filename"] = log_filename

                    run_options.update((self.additional_opts or {}))

                    joined_hyps = deepcopy(self.best_opselec_hyperparams)
                    joined_hyps.update(hyperparams)

                    agent_instance.setup(run_options, joined_hyps)

                else:
                    agent_instance.setup({}, hyperparams)

                if agent_instance.algorithm_name in [ProbabilisticDQNAgent.algorithm_name, ProbabilisticBaselineDQNAgent.algorithm_name]:
                    agent_instance.compute_dual_policy_scores(train_graphs)

                validation_set_objs = []
                for state_num, validation_state in enumerate(validation_graphs):

                    if state_num % 10 == 1:
                        logger.info(f"doing tour number {state_num} / {len(validation_graphs)}.")

                    alns_inst = ClassicalALNS(agent_instance, random_seed=model_seed, use_localsearch=exp_copy.use_localsearch)
                    alns_inst.run_master_loop(validation_state, outer_its_per_customer=exp_copy.alns_outer_its_per_customer, inner_its=exp_copy.alns_inner_its)

                    alns_tune_val = alns_inst.best_obj # post_repair_record['total_objective'].mean()
                    validation_set_objs.append(alns_tune_val)

                avg_val = -np.mean(np.array(validation_set_objs))
                self.storage.write_hyperopt_results(model_identifier_prefix, avg_val, task_type="tune")


class ALNSTask(object):
    def __init__(self,
                 task_id,
                 agent,
                 objective_function,
                 network_generator,
                 best_opselec_hyperparams,
                 best_opselec_hyperparams_id,
                 best_alns_hyperparams,
                 best_alns_hyperparams_id,
                 experiment_conditions,
                 storage,
                 model_seeds_chunk,
                 parent_class=None,
                 additional_opts=None,
                 best_hybrid_hyperparams=None,
                 best_hybrid_hyperparams_id=None):

        self.task_id = task_id
        self.agent = agent
        self.objective_function = objective_function
        self.network_generator = network_generator

        self.best_opselec_hyperparams = best_opselec_hyperparams
        self.best_opselec_hyperparams_id = best_opselec_hyperparams_id

        self.best_alns_hyperparams = best_alns_hyperparams
        self.best_alns_hyperparams_id = best_alns_hyperparams_id

        self.best_hybrid_hyperparams = best_hybrid_hyperparams
        self.best_hybrid_hyperparams_id = best_hybrid_hyperparams_id

        self.experiment_conditions = experiment_conditions
        self.storage = storage
        self.model_seeds_chunk = model_seeds_chunk

        self.parent_class = parent_class

        self.additional_opts = additional_opts

    def run(self):
        log_filename = str(self.storage.file_paths.construct_log_filepath())
        logger = get_logger_instance(log_filename)
        local_results = []

            #hyps_id = int(hyps_id_str)

        for model_seed in self.model_seeds_chunk:
            exp_copy = deepcopy(self.experiment_conditions)
            logger.info(f"running ALNS with model seed {model_seed} for agent {self.agent.algorithm_name}. ALNS iters: outer per cust {exp_copy.alns_outer_its_per_customer} / inner {exp_copy.alns_inner_its}.")
            gen_instance = self.network_generator()
            train_graphs, validation_graphs, test_graphs = gen_instance.generate_many(exp_copy.problem_variant, exp_copy.train_seeds, exp_copy.num_customers_train, instance_name=exp_copy.instance_name,
                                                                                      initial_tour_method=exp_copy.initial_tour_method), \
                gen_instance.generate_many(exp_copy.problem_variant, exp_copy.validation_seeds, exp_copy.num_customers_validate, instance_name=exp_copy.instance_name, initial_tour_method=exp_copy.initial_tour_method), \
                gen_instance.generate_many(exp_copy.problem_variant, exp_copy.test_seeds, exp_copy.num_customers_test, instance_name=exp_copy.instance_name, initial_tour_method=exp_copy.initial_tour_method)

            ol = OperatorLibrary(fixed_scale_pc=exp_copy.fixed_scale_pc,
                                 random_seed=model_seed,
                                 which_destroy=exp_copy.which_destroy, which_repair=exp_copy.which_repair)

            env = OperatorEnv(exp_copy.problem_variant, ol, model_seed, exp_copy.budget, use_localsearch=exp_copy.use_localsearch)


            if self.agent.algorithm_name == HybridAgent.algorithm_name:
                agent_instance = self.setup_hybrid_agent(exp_copy, ol, model_seed, env, train_graphs, validation_graphs, test_graphs)

            else:
                if self.agent.is_mdp_based:
                    agent_instance = self.agent(ol, model_seed, env=env)
                else:
                    agent_instance = self.agent(ol, random_seed=model_seed)

                if self.agent.is_mdp_based and self.agent.is_trainable:
                    run_options = {}
                    run_options["max_nodes"] = find_max_nodes(train_graphs + validation_graphs + test_graphs)
                    run_options['random_seed'] = model_seed
                    run_options["restore_model"] = True
                    run_options['num_mdp_timesteps'] = exp_copy.budget * 2

                    relevant_alg_name = self.parent_class.algorithm_name if self.parent_class is not None else self.agent.algorithm_name

                    parent_model_prefix = self.storage.file_paths.construct_model_identifier_prefix(relevant_alg_name,
                                                                                           self.objective_function.name,
                                                                                           self.network_generator.name,
                                                                                           model_seed,
                                                                                           self.best_opselec_hyperparams_id)
                    run_options["model_identifier_prefix"] = parent_model_prefix
                    run_options["file_paths"] = self.storage.file_paths
                    run_options["log_progress"] = True
                    run_options["log_filename"] = log_filename

                    run_options.update((self.additional_opts or {}))

                    joined_hyps = deepcopy(self.best_opselec_hyperparams)
                    joined_hyps.update(self.best_alns_hyperparams)

                    agent_instance.setup(run_options, joined_hyps)

                elif self.agent.requires_tune:
                    agent_instance.setup({}, self.best_alns_hyperparams)

                if agent_instance.algorithm_name in [ProbabilisticDQNAgent.algorithm_name, ProbabilisticBaselineDQNAgent.algorithm_name]:
                    agent_instance.compute_dual_policy_scores(train_graphs)

            if exp_copy.track_operators:
                alns_records = {}

            for alns_outer_multiplier in exp_copy.alns_outer_multipliers:
                outer_its = int(exp_copy.alns_outer_its_per_customer * alns_outer_multiplier)
                logger.info(f"running with outer its {outer_its}")
                for state_num, test_state in enumerate(test_graphs):

                    if state_num % 10 == 1:
                        logger.info(f"doing tour number {state_num} / {len(test_graphs)}.")

                    alns_inst = ClassicalALNS(agent_instance, random_seed=model_seed, use_localsearch=exp_copy.use_localsearch)


                    time_started_seconds = time.time()
                    alns_inst.run_master_loop(test_state, outer_its_per_customer=outer_its, inner_its=exp_copy.alns_inner_its)
                    time_ended_seconds = time.time()
                    duration_ms = (time_ended_seconds - time_started_seconds) * 1000

                    result_row = {}

                    record = alns_inst.alns_record

                    if exp_copy.track_operators:
                        alns_records[test_state.random_seed] = deepcopy(record)

                    post_repair_record = record[record['nonservice_penalty'] == 0]

                    result_row['obj_avg'] = post_repair_record['total_objective'].mean()
                    result_row['obj_min'] = alns_inst.best_obj

                    result_row['counter'] = alns_inst.counter
                    result_row['operator_counter'] = alns_inst.operator_counter
                    result_row['use_localsearch'] = exp_copy.use_localsearch
                    result_row['initial_tour_method'] = exp_copy.initial_tour_method

                    result_row['tour_seed'] = test_state.random_seed

                    result_row['network_generator'] = self.network_generator.name
                    result_row['objective_function'] = self.objective_function.name

                    result_row['cust_number'] = test_state.inst.cust_number

                    result_row['algorithm'] = self.agent.algorithm_name
                    result_row['agent_seed'] = model_seed
                    result_row['instance_name'] = exp_copy.instance_name
                    result_row['duration_ms'] = duration_ms
                    result_row['hyps_id'] = self.best_alns_hyperparams_id
                    result_row["alns_outer_multiplier"] = alns_outer_multiplier

                    local_results.append(result_row)

            if agent_instance.is_trainable:
                agent_instance.finalize()

            if self.experiment_conditions.track_operators:
                model_prefix = self.storage.file_paths.construct_model_identifier_prefix(self.agent.algorithm_name,
                                                                                         self.objective_function.name,
                                                                                         self.network_generator.name,
                                                                                         model_seed,
                                                                                         self.best_alns_hyperparams_id)

                self.storage.write_alns_records(model_prefix, alns_records)

        self.storage.write_results("alns", local_results, self.task_id)

    def setup_hybrid_agent(self, exp_copy, ol, model_seed, env, train_graphs, validation_graphs, test_graphs):
        base_options = {}
        base_options["max_nodes"] = find_max_nodes(train_graphs + validation_graphs + test_graphs)
        base_options['random_seed'] = model_seed
        base_options["restore_model"] = True


        base_options['num_mdp_timesteps'] = exp_copy.budget * 2
        log_filename = str(self.storage.file_paths.construct_log_filepath())
        base_options["file_paths"] = self.storage.file_paths
        base_options["log_progress"] = True
        base_options["log_filename"] = log_filename
        base_options.update((self.additional_opts or {}))

        first_sub_agent = DQNAgent(ol, model_seed, env=env)
        first_model_prefix = self.storage.file_paths.construct_model_identifier_prefix(first_sub_agent.algorithm_name,
                                                                                        self.objective_function.name,
                                                                                        self.network_generator.name,
                                                                                        model_seed,
                                                                                        self.best_opselec_hyperparams_id)
        first_model_opts = deepcopy(base_options)
        first_model_opts["model_identifier_prefix"] = first_model_prefix
        first_sub_agent.setup(first_model_opts, self.best_opselec_hyperparams)

        second_sub_agent = VictorRouletteWheelMDPAgent(ol, model_seed, env=env)
        second_model_prefix = self.storage.file_paths.construct_model_identifier_prefix(second_sub_agent.algorithm_name,
                                                                                       self.objective_function.name,
                                                                                       self.network_generator.name,
                                                                                       model_seed,
                                                                                       self.best_hybrid_hyperparams_id)
        second_model_opts = deepcopy(base_options)
        second_model_opts["model_identifier_prefix"] = second_model_prefix
        second_sub_agent.setup(second_model_opts, self.best_hybrid_hyperparams)

        hybrid_agent = HybridAgent(ol, model_seed, env=env)
        hybrid_agent.setup(base_options, {})
        hybrid_agent.pass_sub_agents([first_sub_agent, second_sub_agent])
        return hybrid_agent



def main():
    parser = argparse.ArgumentParser(description="Run a given task.")
    parser.add_argument("--experiment_part", required=True, type=str,
                        help="Whether to setup hyperparameter optimisation, evaluation, ALNS tuning, or ALNS evaluation tasks.",
                        choices=["hyperopt", "eval", "tune", "alns"])

    parser.add_argument("--parent_dir", type=str, help="Root path for storing experiment data.")
    parser.add_argument("--experiment_id", required=True, help="experiment id to use")

    parser.add_argument("--task_id", type=str, required=True, help="Task id to run. Must have already been generated.")
    parser.set_defaults(parent_dir="/experiment_data")

    args = parser.parse_args()

    file_paths = FilePaths(args.parent_dir, args.experiment_id, setup_directories=False)
    task_storage_dir = getattr(file_paths, f"{args.experiment_part}_tasks_dir")
    task_file = task_storage_dir / FilePaths.construct_task_filename(args.experiment_part, args.task_id)
    with open(task_file, 'rb') as fh:
        task = dill.load(fh)

    task.run()



if __name__ == "__main__":
    main()
































































