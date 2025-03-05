import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from experiment_storage.file_paths import FilePaths
from experiment_storage.storage import EvaluationStorage



from itertools import product

import argparse

def main():
    parser = argparse.ArgumentParser(description="Plain utility to check best validation losses so far.")
    parser.add_argument("--experiment_id", required=True, help="experiment id to use")
    parser.add_argument("--parent_dir", type=str, help="Root path for storing experiment data.")
    args = parser.parse_args()

    experiment_id = args.experiment_id
    fp = FilePaths(args.parent_dir, experiment_id, setup_directories=False)

    storage = EvaluationStorage(fp)
    experiment_details = storage.get_experiment_details(experiment_id)

    agent_names = list(experiment_details['agents'])
    experiment_conditions = experiment_details['experiment_conditions']
    objective_functions = experiment_details['objective_functions']
    network_generators = experiment_details['network_generators']
    model_seeds = experiment_conditions['experiment_params']['model_seeds']

    for objective_function in objective_functions:
        for network_generator in network_generators:
            print(f"=================")
            print(f"{network_generator},{objective_function}")
            print(f"=================")
            for agent_name in agent_names:
                if agent_name in experiment_conditions['hyperparam_grids'][objective_function]:
                    print(f"=================")
                    print(f"<<{agent_name}>>")
                    print(f"=================")
                    agent_grid = experiment_conditions['hyperparam_grids'][objective_function][agent_name]
                    num_hyperparam_combs = len(list(product(*agent_grid.values())))

                    for comb in range(num_hyperparam_combs):
                        try:
                            df = storage.fetch_eval_curves(agent_name, comb, fp, objective_function, network_generator, model_seeds, False, nrows_to_skip=0)
                        except ValueError:
                            continue

                        if len(df) > 0:
                            num_started = 0
                            num_total = len(model_seeds)

                            out_str_started = ""

                            best_so_far = []

                            for seed in model_seeds:
                                df_subset = df[(df['model_seed'] == seed) &
                                               (df['network_generator'] == network_generator) &
                                               (df['objective_function'] == objective_function)]

                                if len(df_subset) > 0:
                                    best_perf = df_subset['perf'].max()
                                    training_step = df_subset['timestep'].max()
                                    out_str_started += f"{best_perf:.3f} [{training_step}; {seed}],  "
                                    num_started +=1
                                    best_so_far.append(best_perf)

                            if len(out_str_started) > 0:
                                print(f"{comb}: <<{sum(best_so_far) / len(best_so_far):.4f}>> avg. training started: {num_started / num_total * 100:.2f}%. {out_str_started}")

if __name__ == "__main__":
    main()
