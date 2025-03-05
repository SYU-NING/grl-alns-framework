from pathlib import Path
from copy import copy
import subprocess
import os



class FilePaths:
    DATE_FORMAT = "%Y-%m-%d-%H-%M-%S"
    MODELS_DIR_NAME = 'models'
    CHECKPOINTS_DIR_NAME = 'checkpoints'
    SUMMARIES_DIR_NAME = 'summaries'
    EVAL_HISTORIES_DIR_NAME = 'eval_histories'

    HYPEROPT_RESULTS_DIR_NAME = 'hyperopt_results'
    EVAL_RESULTS_DIR_NAME = 'eval_results'
    TUNE_RESULTS_DIR_NAME = 'tune_results'
    ALNS_RESULTS_DIR_NAME = 'alns_results'

    HYPEROPT_TASKS_DIR_NAME = 'tasks_hyperopt'
    EVAL_TASKS_DIR_NAME = 'tasks_eval'

    TUNE_TASKS_DIR_NAME = 'tasks_tune'
    ALNS_TASKS_DIR_NAME = 'tasks_alns'

    OPERATORS_TRACKING_DIR_NAME = 'op_tracking'

    FIGURES_DIR_NAME = 'figures'
    LOGS_DIR_NAME = 'logs'

    DEFAULT_MODEL_PREFIX = 'default'

    def __init__(self, parent_dir, experiment_id, setup_directories=True):
        self.parent_dir = parent_dir
        self.experiment_id = experiment_id

        self.logs_dir = Path(self.parent_dir) / self.LOGS_DIR_NAME

        self.experiment_dir = self.get_dir_for_experiment_id(experiment_id)


        self.models_dir = self.experiment_dir / self.MODELS_DIR_NAME
        self.checkpoints_dir = self.models_dir / self.CHECKPOINTS_DIR_NAME
        self.summaries_dir = self.models_dir / self.SUMMARIES_DIR_NAME
        self.eval_histories_dir = self.models_dir / self.EVAL_HISTORIES_DIR_NAME

        self.hyperopt_results_dir = self.models_dir / self.HYPEROPT_RESULTS_DIR_NAME
        self.eval_results_dir = self.models_dir / self.EVAL_RESULTS_DIR_NAME
        self.tune_results_dir = self.models_dir / self.TUNE_RESULTS_DIR_NAME
        self.alns_results_dir = self.models_dir / self.ALNS_RESULTS_DIR_NAME

        self.hyperopt_tasks_dir = self.models_dir / self.HYPEROPT_TASKS_DIR_NAME
        self.eval_tasks_dir = self.models_dir / self.EVAL_TASKS_DIR_NAME
        self.tune_tasks_dir = self.models_dir / self.TUNE_TASKS_DIR_NAME
        self.alns_tasks_dir = self.models_dir / self.ALNS_TASKS_DIR_NAME

        self.op_tracking_dir = self.models_dir / self.OPERATORS_TRACKING_DIR_NAME

        self.figures_dir = self.experiment_dir / self.FIGURES_DIR_NAME

        if setup_directories:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            self.experiment_dir.mkdir(parents=True, exist_ok=True)

            self.models_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            self.summaries_dir.mkdir(parents=True, exist_ok=True)
            self.eval_histories_dir.mkdir(parents=True, exist_ok=True)
            self.hyperopt_results_dir.mkdir(parents=True, exist_ok=True)
            self.eval_results_dir.mkdir(parents=True, exist_ok=True)
            self.tune_results_dir.mkdir(parents=True, exist_ok=True)
            self.alns_results_dir.mkdir(parents=True, exist_ok=True)

            self.hyperopt_tasks_dir.mkdir(parents=True, exist_ok=True)
            self.eval_tasks_dir.mkdir(parents=True, exist_ok=True)
            self.tune_tasks_dir.mkdir(parents=True, exist_ok=True)
            self.alns_tasks_dir.mkdir(parents=True, exist_ok=True)

            self.op_tracking_dir.mkdir(parents=True, exist_ok=True)


            self.figures_dir.mkdir(parents=True, exist_ok=True)

            self.set_group_permissions()


    def set_group_permissions(self):
        try:
            # for dir in [self.logs_dir, self.experiment_dir]:
            for dir in [self.experiment_dir]:
                abspath = str(dir.absolute())
                subprocess.run(["chmod", "-R", "g+rwx", abspath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def get_dir_for_experiment_id(self, experiment_id):
        return Path(self.parent_dir) / f'{experiment_id}'

    def __str__(self):
        asdict = self.__dict__
        target = copy(asdict)
        for param_name, corresp_path in asdict.items():
            target[param_name] = str(corresp_path.absolute())

        return str(target)

    def __repr__(self):
        return self.__str__()

    def construct_log_filepath(self):
        return self.logs_dir / self.construct_log_filename()

    @staticmethod
    def construct_task_filename(task_type, task_id):
        return f"{task_type}_{task_id}.obj"

    @staticmethod
    def construct_log_filename():
        hostname = os.getenv("HOSTNAME", "unknown")
        return f'experiments_{hostname}.log'

    @staticmethod
    def construct_model_identifier_prefix(agent_name, obj_fun_name, network_generator_name, model_seed,  hyperparams_id, graph_id=None):
        model_identifier_prefix = f"{agent_name}-{obj_fun_name}-{network_generator_name}-{(graph_id + '-') if graph_id is not None else ''}" \
                                  f"{model_seed}-{hyperparams_id}"
        return model_identifier_prefix

    @staticmethod
    def construct_history_file_name(model_identifier_prefix):
        return f"{model_identifier_prefix}_history.csv"

    @staticmethod
    def construct_best_validation_file_name(model_identifier_prefix):
        return f"{model_identifier_prefix}_best.csv"

    @staticmethod
    def construct_op_tracking_file_name(model_identifier_prefix):
        return f"{model_identifier_prefix}_tracked_operators.csv"