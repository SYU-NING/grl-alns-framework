import datetime
import traceback
from pathlib import Path

import torch
import numpy as np

from experiment_storage.file_paths import FilePaths
from alns.rl.mdp_agent import MDPAgent
from utils.config_utils import get_logger_instance


class PyTorchAgent(MDPAgent):
    DEFAULT_GRAPHS_PER_SIM = 1
    DEFAULT_BATCH_SIZE = 16

    def __init__(self, operator_library, random_seed, **kwargs):
        super().__init__(operator_library, random_seed, **kwargs)

        self.enable_assertions = True
        self.hist_out = None

        self.validation_change_threshold = 1e-5
        self.best_validation_changed_step = -1
        self.best_validation_loss = float("inf")

        self.pos = 0
        self.step = 0

    def setup_graphs(self, train_g_list, validation_g_list):
        self.train_g_list = train_g_list
        self.validation_g_list = validation_g_list

    def setup_sample_idxes(self, dataset_size):
        self.sample_idxes = list(range(dataset_size))
        np.random.shuffle(self.sample_idxes)

    def advance_pos_and_sample_indices(self):
        if (self.pos + 1) * self.graphs_per_sim > len(self.sample_idxes):
            self.pos = 0
            np.random.shuffle(self.sample_idxes)

        selected_idx = self.sample_idxes[self.pos * self.graphs_per_sim : (self.pos + 1) * self.graphs_per_sim]
        self.pos += 1
        return selected_idx

    def save_model_checkpoints(self):
        model_dir = self.checkpoints_path / self.model_identifier_prefix
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"agent.model"
        torch.save(self.net.state_dict(), model_path)

    def restore_model_from_checkpoint(self):
        model_path = self.checkpoints_path / self.model_identifier_prefix / f"agent.model"
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(checkpoint)

    def check_validation_loss(self, step_number, max_steps,
                              model_tag=None,
                              save_model_if_better=True):
        if (step_number % self.validation_check_interval == 0 or step_number == max_steps):
            validation_loss = self.log_validation_loss(step_number)
            if self.log_progress: self.logger.info(f"{model_tag if model_tag is not None else 'model'} validation loss: {validation_loss: .4f} at step {step_number}.")
            if (self.best_validation_loss - validation_loss) > self.validation_change_threshold:
                if self.log_progress: self.logger.info(f"rejoice! found a better validation loss at step {step_number}.")
                self.best_validation_changed_step = step_number
                self.best_validation_loss = validation_loss
                if save_model_if_better:
                    if self.log_progress: self.logger.info("saving model.")
                    self.save_model_checkpoints()

    def log_validation_loss(self, step):
        performance = self.eval(self.validation_g_list, validation=True)
        validation_loss = -performance

        if self.hist_out is not None:
            self.hist_out.write('%d,%.6f\n' % (step, performance))
            try:
                self.hist_out.flush()
            except BaseException:
                if self.logger is not None:
                    self.logger.warn("caught an exception when trying to flush evaluation history.")
                    self.logger.warn(traceback.format_exc())

        return validation_loss

    def print_model_parameters(self):
        param_list = self.net.parameters()
        for params in param_list:
            print(params.data)

    def check_stopping_condition(self, step_number, max_steps):
        if step_number >= max_steps \
                or (step_number - self.best_validation_changed_step > self.max_validation_consecutive_steps):
            if self.log_progress: self.logger.info(
                "number steps exceeded or validation plateaued for too long, stopping training.")
            if self.log_progress: self.logger.info("restoring best model to use for predictions.")
            self.restore_model_from_checkpoint()

            return True
        return False

    def setup(self, options, hyperparams):
        self.options = options
        self.hyperparams = hyperparams

        if 'log_filename' in options:
            self.log_filename = options['log_filename']
        if 'log_progress' in options:
            self.log_progress = options['log_progress']
        else:
            self.log_progress = False
        if self.log_progress:
            self.logger = get_logger_instance(self.log_filename)
        else:
            self.logger = None


        if 'batch_size' in options:
            self.batch_size = options['batch_size']
        else:
            self.batch_size = self.DEFAULT_BATCH_SIZE

        if 'graphs_per_sim' in options:
            self.graphs_per_sim = options['graphs_per_sim']
        else:
            self.graphs_per_sim = self.DEFAULT_GRAPHS_PER_SIM

        if 'validation_check_interval' in options:
            self.validation_check_interval = options['validation_check_interval']
        else:
            self.validation_check_interval = 25

        if 'max_validation_consecutive_steps' in options:
            self.max_validation_consecutive_steps = options['max_validation_consecutive_steps']
        else:
            self.max_validation_consecutive_steps = 200000

        if 'pytorch_full_print' in options:
            if options['pytorch_full_print']:
                torch.set_printoptions(profile="full")

        if 'enable_assertions' in options:
            self.enable_assertions = options['enable_assertions']

        if 'model_identifier_prefix' in options:
            self.model_identifier_prefix = options['model_identifier_prefix']
        else:
            self.model_identifier_prefix = FilePaths.DEFAULT_MODEL_PREFIX

        if 'restore_model' in options:
            self.restore_model = options['restore_model']
        else:
            self.restore_model = False

        self.file_paths = options['file_paths']
        self.models_path = self.file_paths.models_dir
        self.checkpoints_path = self.file_paths.checkpoints_dir


    def setup_histories_file(self):
        self.eval_histories_path = self.models_path / FilePaths.EVAL_HISTORIES_DIR_NAME
        model_history_filename = self.eval_histories_path / FilePaths.construct_history_file_name(self.model_identifier_prefix)
        model_history_file = Path(model_history_filename)
        if model_history_file.exists():
            model_history_file.unlink()
        self.hist_out = open(model_history_filename, 'a')

    def finalize(self):
        if self.hist_out is not None and not self.hist_out.closed:
            self.hist_out.close()
