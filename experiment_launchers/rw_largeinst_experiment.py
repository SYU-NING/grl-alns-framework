import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


from alns.rl.hybrid_agent import HybridAgent
from alns.rl.dqn.dqn_agent import ProbabilisticDQNAgent, DQNAgent
from environment.objective_functions import TotalObjective
from experiment_storage.storage import EvaluationStorage

from alns.rl.pytorch_agent import PyTorchAgent
from alns.rl.rw_mdp import VictorRouletteWheelMDPAgent



import numpy as np

from operators.destroy_operators import GreedyDestroy, RelatedDestroy, RandomDestroy
from operators.repair_operators import GreedyRepair, Regret2Repair


from alns.op_selector import RouletteWheelOperatorSelector, RandomOperatorSelector
from alns.classical_alns import ClassicalALNS
from alns.rl.mdp_agent import RandomMDPAgent, MDPAgent
from environment.operator_env import OperatorEnv
from experiment_storage.file_paths import FilePaths
from operators.operator_library import OperatorLibrary
from state.state_generator import SolomonStateGenerator, GehringHombergerStateGenerator


def find_num_custs(instance_name):
    cust_num_key = instance_name.split("_")[1]
    return int(cust_num_key) * 100


def run_experiments(args):
    all_results = []
    seeds = [args.seed]

    train_inst = "O101200"
    train_suffix = "diffinitrand"

    initial_tour_method = "random"
    # initial_tour_method = "clark_wright"
    # initial_tour_method = "hybrid"

    original_budget = 10

    # agent_classes = [VictorRouletteWheelMDPAgent, RandomOperatorSelector, RouletteWheelOperatorSelector]
    agent_classes = [HybridAgent]

    problem_variant = args.problem_variant

    train_exp_id = f"{train_inst}_{train_suffix}{problem_variant}"

    which_destroy = OperatorLibrary.get_compatible_destroy_operators(problem_variant)
    which_repair = OperatorLibrary.get_compatible_repair_operators(problem_variant)

    gen = GehringHombergerStateGenerator()
    instance_name = args.instance_name

    cust_number = [find_num_custs(instance_name)]
    num_tours = args.num_tours

    data_dir = os.getenv("ATES_EXPERIMENT_DATA_DIR")
    # data_dir = '/Users/vdarvariu/experiment_data/ates'

    fp_in = FilePaths(data_dir, train_exp_id, setup_directories=True)
    storage = EvaluationStorage(fp_in)

    for agent_class in agent_classes:
        print(f"executing for {agent_class.algorithm_name}")
        for seed in seeds:
            print(f"executing for seed {seed}")

            fixed_scale_pc = args.fixed_scale_pc
            ol = OperatorLibrary(fixed_scale_pc=fixed_scale_pc, random_seed=seed, which_destroy=which_destroy, which_repair=which_repair)
            env = OperatorEnv(problem_variant, ol, seed, original_budget)

            if issubclass(agent_class, MDPAgent):
                agent = agent_class(ol, seed, env=env)
            else:
                agent = agent_class(ol, random_seed=seed)

            hyps_search_spaces = storage.get_experiment_details(train_exp_id)["parameter_search_spaces"]


            options = {
                 "max_nodes": find_num_custs(instance_name) + 1,
                 "log_progress": True,
                 "log_filename": str(fp_in.construct_log_filepath()),
                 "log_tf_summaries": True,
                 "random_seed": seed,
                 "file_paths": fp_in,
                 "restore_model": True,
            }

            if agent_class == HybridAgent:
                optimal_hyperparams = storage.retrieve_optimal_hyperparams(train_exp_id, False, "hyperopt", fp_in=fp_in)
                print(optimal_hyperparams)
                standard_dqn_setting = (SolomonStateGenerator.name, TotalObjective.name, DQNAgent.algorithm_name)
                best_opselec_hyperparams, best_opselec_hyperparams_id = optimal_hyperparams[standard_dqn_setting]

                standard_vrw_setting = (SolomonStateGenerator.name, TotalObjective.name, VictorRouletteWheelMDPAgent.algorithm_name)
                best_hybrid_hyperparams, best_hybrid_hyperparams_id = optimal_hyperparams[standard_vrw_setting]

                options['num_mdp_timesteps'] = original_budget * 2

                first_sub_agent = DQNAgent(ol, seed, env=env)
                first_model_prefix = fp_in.construct_model_identifier_prefix(first_sub_agent.algorithm_name,
                                                                                               TotalObjective.name,
                                                                                               SolomonStateGenerator.name,
                                                                                               seed,
                                                                                               best_opselec_hyperparams_id)
                first_model_opts = deepcopy(options)
                first_model_opts["model_identifier_prefix"] = first_model_prefix
                first_sub_agent.setup(first_model_opts, best_opselec_hyperparams)

                second_sub_agent = VictorRouletteWheelMDPAgent(ol, seed, env=env)
                second_model_prefix = fp_in.construct_model_identifier_prefix(second_sub_agent.algorithm_name,
                                                                                                TotalObjective.name,
                                                                                                SolomonStateGenerator.name,
                                                                                                seed,
                                                                                                best_hybrid_hyperparams_id)
                second_model_opts = deepcopy(options)
                second_model_opts["model_identifier_prefix"] = second_model_prefix
                second_sub_agent.setup(second_model_opts, best_hybrid_hyperparams)

                agent = HybridAgent(ol, seed, env=env)
                agent.setup(options, {})
                agent.pass_sub_agents([first_sub_agent, second_sub_agent])

                hyps_id = best_opselec_hyperparams_id


            elif agent_class == VictorRouletteWheelMDPAgent:
                alg_name = VictorRouletteWheelMDPAgent.algorithm_name
                hyperparams, hyps_id = agent.get_default_hyperparameters(), 0
                model_prefix = fp_in.construct_model_identifier_prefix(alg_name,
                                                                    TotalObjective.name,
                                                                    SolomonStateGenerator.name,
                                                                    seed,
                                                                    hyps_id)
                options["model_identifier_prefix"] = model_prefix
                options['restore_model'] = True
                agent.setup(options, hyperparams)
            elif agent_class == RouletteWheelOperatorSelector:
                optimal_hyperparams_tune = storage.retrieve_optimal_hyperparams(train_exp_id, False, "tune", fp_in=fp_in)
                alg_name = RouletteWheelOperatorSelector.algorithm_name
                hyperparams, hyps_id = optimal_hyperparams_tune[SolomonStateGenerator.name, TotalObjective.name, alg_name]
                agent.setup({}, hyperparams)
            else:
                hyperparams ,hyps_id = {}, 0
            print(f"using hyps with id {hyps_id}")

            # run classical ALNS:
            #
            for i in range(0, num_tours):
                test_state = gen.generate_many(problem_variant, [i], cust_number, instance_name=instance_name, initial_tour_method=initial_tour_method)[0]
                print(f"executing tour {i+1}/{num_tours}.")

                alns_inst = ClassicalALNS(agent, random_seed=seed, use_localsearch=False)

                time_started_seconds = time.time()
                alns_inst.run_master_loop(test_state, args.alns_outer_its, args.alns_inner_its)
                time_ended_seconds = time.time()
                duration_ms = (time_ended_seconds - time_started_seconds) * 1000

                result_row = {}

                record = alns_inst.alns_record
                post_repair_record = record[record['nonservice_penalty'] == 0]

                result_row['obj_avg'] = post_repair_record['total_objective'].mean()
                result_row['obj_min'] = alns_inst.best_obj

                result_row['counter'] = alns_inst.counter
                result_row['operator_counter'] = alns_inst.operator_counter
                result_row['use_localsearch'] = False
                result_row['initial_tour_method'] = initial_tour_method

                result_row['tour_seed'] = test_state.random_seed

                result_row['network_generator'] = GehringHombergerStateGenerator.name
                result_row['objective_function'] = TotalObjective.name

                result_row['cust_number'] = test_state.inst.cust_number

                result_row['algorithm'] = agent.algorithm_name
                result_row['agent_seed'] = seed
                result_row['instance_name'] = instance_name
                result_row['duration_ms'] = duration_ms
                result_row['hyps_id'] = hyps_id

                all_results.append(result_row)

    fp_out = FilePaths(data_dir, f"{GehringHombergerStateGenerator.name}_experiments", setup_directories=True)
    out_file = fp_out.alns_results_dir / f"{HybridAgent.algorithm_name}results_{args.problem_variant}_{args.instance_name}_{args.seed}.json"

    with open(out_file, "w") as fh:
        json.dump(all_results, fh, indent=4, sort_keys=True)

    print(f"wrote results file to {fp_out.alns_results_dir}.")

def main():
    parser = argparse.ArgumentParser(description="Run large-scale instance experiments.")
    parser.add_argument("--problem_variant", type=str, help="Problem variant to apply framework to.",
                        choices=["TSP", "CVRP", "VRPTW", "HVRP", "LRP"])

    parser.add_argument("--instance_name", type=str, required=True, help="Instance name")
    parser.add_argument("--fixed_scale_pc", type=float, help="Percentage of customer nodes to use as scale for the operators.", default=10.)

    parser.add_argument("--alns_outer_its", type=int, help="Number of ALNS outer iterations.", default=10)
    parser.add_argument("--alns_inner_its", type=int, help="Number of ALNS inner iterations.", default=5)

    parser.add_argument("--num_tours", type=int, required=False, help="Number of tours", default=128)
    parser.add_argument("--seed", type=int, required=True, help="Random seed")

    args = parser.parse_args()
    run_experiments(args)


if __name__ == '__main__':
    main()

