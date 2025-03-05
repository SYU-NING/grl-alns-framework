import json
import dill
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_storage.file_paths import FilePaths


class EvaluationStorage:
    EXPERIMENT_DETAILS_FILENAME = "experiment_details.json"

    def __init__(self, file_paths):
        self.file_paths = file_paths

    def get_hyperparameter_optimisation_data(self,
                                     experiment_id,
                                     train_individually,
                                     task_type,
                                     fp_in=None):

        latest_experiment = self.get_experiment_details(experiment_id)
        file_paths = latest_experiment["file_paths"]
        experiment_conditions = latest_experiment["experiment_conditions"]

        hyperopt_data = []

        network_generators = latest_experiment["network_generators"]
        objective_functions = latest_experiment["objective_functions"]
        agent_names = latest_experiment["agents"]
        param_spaces = latest_experiment["parameter_search_spaces"]

        for objective_function in objective_functions:
            for agent_name in agent_names:
                if agent_name in param_spaces[objective_function]:
                    agent_grid = param_spaces[objective_function][agent_name]
                    search_space_keys = list(agent_grid.keys())

                    for network_generator in network_generators:
                        for hyperparams_id in search_space_keys:

                            if fp_in is not None:
                                hyperopt_dir = fp_in.hyperopt_results_dir if task_type == "hyperopt" else fp_in.tune_results_dir
                            else:
                                hyperopt_dir = file_paths['hyperopt_results_dir'] if task_type == "hyperopt" else file_paths['tune_results_dir']

                            finished_seeds = []
                            for f in Path(hyperopt_dir).glob(f"*{agent_name}-{objective_function}-{network_generator}-*-{hyperparams_id}_best.csv"):
                                if not f.is_dir():
                                    seed = int(f.name.split("-")[3])
                                    finished_seeds.append(seed)

                            for seed in finished_seeds:
                                graph_id = None

                                setting = (network_generator, objective_function, agent_name, graph_id)

                                model_prefix = FilePaths.construct_model_identifier_prefix(agent_name,
                                                                                       objective_function,
                                                                                       network_generator,
                                                                                       seed,
                                                                                       hyperparams_id,
                                                                                       graph_id=graph_id)
                                hyperopt_result_filename = FilePaths.construct_best_validation_file_name(model_prefix)



                                hyperopt_result_path = Path(hyperopt_dir, hyperopt_result_filename)
                                if hyperopt_result_path.exists():
                                    with hyperopt_result_path.open('r') as f:
                                        avg_eval_perf = float(f.readline())
                                        hyperopt_data_row = {"network_generator": network_generator,
                                                             "objective_function": objective_function,
                                                             "agent_name": agent_name,
                                                             "hyperparams_id": hyperparams_id,
                                                             "avg_perf": avg_eval_perf,
                                                             "graph_id": graph_id}

                                        hyperopt_data.append(hyperopt_data_row)

        return param_spaces, pd.DataFrame(hyperopt_data)

    def retrieve_optimal_hyperparams(self,
                                     experiment_id,
                                     train_individually,
                                     task_type="hyperopt",
                                     fp_in=None):

        avg_perfs_df, param_spaces = self.get_grouped_hyp_data(experiment_id, train_individually, task_type=task_type, fp_in=fp_in)
        gb_cols = list(set(avg_perfs_df.columns) - {"avg_perf", "hyperparams_id"})
        avg_perfs_max = avg_perfs_df.loc[avg_perfs_df.groupby(gb_cols)["avg_perf"].idxmax()].reset_index(
            drop=True)

        optimal_hyperparams = {}

        for row in avg_perfs_max.itertuples():
            if not train_individually:
                setting = row.network_generator, row.objective_function, row.agent_name
            else:
                setting = row.network_generator, row.objective_function, row.agent_name, row.graph_id
            optimal_id = row.hyperparams_id
            optimal_hyperparams[setting] = param_spaces[row.objective_function][row.agent_name][optimal_id], optimal_id

        return optimal_hyperparams

    def get_grouped_hyp_data(self, experiment_id, train_individually, task_type, fp_in=None):
        param_spaces, df = self.get_hyperparameter_optimisation_data(experiment_id, train_individually, task_type, fp_in=fp_in)
        # print(df)
        if not train_individually:
            if 'graph_id' in df.columns:
                df = df.drop(columns='graph_id')
        avg_perfs_df = df.groupby(list(set(df.columns) - {"avg_perf"})).mean().reset_index()
        return avg_perfs_df, param_spaces

    def get_evaluation_data(self, experiment_id, task_type="eval"):
        results_dir = getattr(FilePaths(self.file_paths.parent_dir, experiment_id, setup_directories=False), f"{task_type}_results_dir")
        all_results_rows = []
        for results_file in results_dir.iterdir():
            with open(results_file, "rb") as fh:
                try: 
                    result_rows = json.load(fh)
                    all_results_rows.extend(result_rows)
                except:
                    # print("can't load json file only")
                    continue
                

        return all_results_rows

    def insert_experiment_details(self,
                                    experiment_conditions,
                                    started_str,
                                    started_millis,
                                    parameter_search_spaces):
        all_experiment_details = {}
        all_experiment_details['experiment_id'] = self.file_paths.experiment_id
        all_experiment_details['started_datetime'] = started_str
        all_experiment_details['started_millis'] = started_millis
        all_experiment_details['file_paths'] = {k: str(v) for k, v in dict(vars(self.file_paths)).items()}

        conds = dict(vars(deepcopy(experiment_conditions)))
        del conds["objective_functions"]
        del conds["network_generators"]
        del conds["all_agents"]
        del conds["which_destroy"]
        del conds["which_repair"]
        del conds["problem_variant"]

        all_experiment_details['experiment_conditions'] = conds
        all_experiment_details['problem_variant'] = experiment_conditions.problem_variant

        all_experiment_details['agents'] = [agent.algorithm_name for agent in experiment_conditions.all_agents]
        all_experiment_details['objective_functions'] = [obj.name for obj in experiment_conditions.objective_functions]
        all_experiment_details['network_generators'] = [network_generator.name for network_generator in experiment_conditions.network_generators]

        all_experiment_details['which_destroy'] = [op.name for op in experiment_conditions.which_destroy]
        all_experiment_details['which_repair'] = [op.name for op in experiment_conditions.which_repair]

        all_experiment_details['parameter_search_spaces'] = parameter_search_spaces

        import pprint
        pprint.pprint(all_experiment_details)

        with open(self.file_paths.models_dir / self.EXPERIMENT_DETAILS_FILENAME, "w") as fh:
            json.dump(all_experiment_details, fh, indent=4, sort_keys=True)

        return all_experiment_details

    def get_experiment_details(self, experiment_id):
        exp_models_dir = FilePaths(self.file_paths.parent_dir, experiment_id, setup_directories=False).models_dir
        with open(exp_models_dir / self.EXPERIMENT_DETAILS_FILENAME, "rb") as fh:
            exp_details_dict = json.load(fh)
        return exp_details_dict

    def fetch_all_eval_curves(self, agent_name, hyperparams_id, file_paths, objective_functions, network_generators, model_seeds, train_individually, nrows_to_skip=0):
        all_dfs = []
        for obj_fun_name in objective_functions:
            for net_gen_name in network_generators:
                all_dfs.append(self.fetch_eval_curves(agent_name, hyperparams_id, file_paths, obj_fun_name, net_gen_name, model_seeds, train_individually, nrows_to_skip))
        return pd.concat(all_dfs)

    def fetch_eval_curves(self, agent_name, hyperparams_id, file_paths, objective_function, network_generator, model_seeds, train_individually, nrows_to_skip):
        eval_histories_dir = file_paths.eval_histories_dir
        if len(list(eval_histories_dir.iterdir())) == 0:
            return pd.DataFrame()

        data_dfs = []

        for seed in model_seeds:
            g_id = None
            model_identifier_prefix = file_paths.construct_model_identifier_prefix(agent_name, objective_function, network_generator, seed, hyperparams_id, graph_id=g_id)
            filename = file_paths.construct_history_file_name(model_identifier_prefix)
            data_file = eval_histories_dir / filename
            if data_file.exists():
                eval_df = pd.read_csv(data_file, sep=",", header=None, names=['timestep', 'perf'], usecols=[0,1], skiprows=nrows_to_skip)

                model_seed_col = [seed] * len(eval_df)

                eval_df['model_seed'] = model_seed_col
                eval_df['objective_function'] = [objective_function] * len(eval_df)
                eval_df['network_generator'] = [network_generator] * len(eval_df)
                if g_id is not None:
                    eval_df['graph_id'] = [g_id] * len(eval_df)

                data_dfs.append(eval_df)
        all_data_df = pd.concat(data_dfs)
        return all_data_df

    def store_tasks(self, tasks, task_type):
        task_storage_dir = getattr(self.file_paths, f"{task_type}_tasks_dir")
        count_file = self.file_paths.models_dir / f"{task_type}_tasks.count"

        count_file_out = open(count_file, 'w')
        count_file_out.write(f'{len(tasks)}\n')
        count_file_out.close()

        for task in tasks:
            out_file = task_storage_dir / FilePaths.construct_task_filename(task_type, task.task_id)
            with open(out_file, 'wb') as fh:
                dill.dump(task, fh)

    def write_hyperopt_results(self, model_identifier_prefix, perf, task_type="hyperopt"):
        results_dir = self.file_paths.hyperopt_results_dir if task_type == "hyperopt" else self.file_paths.tune_results_dir

        hyperopt_result_file = f"{results_dir.absolute()}/" + \
                               self.file_paths.construct_best_validation_file_name(model_identifier_prefix)
        hyperopt_result_out = open(hyperopt_result_file, 'w')
        hyperopt_result_out.write('%.6f\n' % (perf))
        hyperopt_result_out.close()

    def write_results(self, task_type, local_results, task_id):
        results_storage_dir = getattr(self.file_paths, f"{task_type}_results_dir")

        out_file = results_storage_dir / f"results_{task_id}.json"

        with open(out_file, "w") as fh:
            json.dump(local_results, fh, indent=4, sort_keys=True)

    def write_alns_records(self, prefix, alns_record):
        results_storage_dir = self.file_paths.alns_results_dir
        fname = f"{prefix}-records.pkl"
        out_file = results_storage_dir / fname

        with open(out_file, 'wb') as fh:
            dill.dump(alns_record, fh)

    def read_alns_records(self, prefix):
        results_storage_dir = self.file_paths.alns_results_dir
        fname = f"{prefix}-records.pkl"
        in_file = results_storage_dir / fname

        with open(in_file, 'rb') as fh:
            record = dill.load(fh)

        return record
