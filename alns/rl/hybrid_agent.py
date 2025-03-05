from alns.rl.pytorch_agent import PyTorchAgent


class HybridAgent(PyTorchAgent):
    algorithm_name = "hybrid"

    is_deterministic = False
    is_trainable = False
    requires_hyperopt = False
    requires_tune = False

    def setup(self, options, hyperparams):
        super().setup(options, hyperparams)
        self.num_mdp_timesteps = options['num_mdp_timesteps']


    def pass_sub_agents(self, sub_agents):
        self.sub_agents = sub_agents


    def select_destroy_operator(self, state, **kwargs):
        search_step = kwargs.pop('search_step')
        sub_agent_idx = 0 if search_step < self.num_mdp_timesteps else 1
        return self.sub_agents[sub_agent_idx].select_destroy_operator(state, **kwargs)


    def select_repair_operator(self, state, **kwargs):
        search_step = kwargs.pop('search_step')
        sub_agent_idx = 0 if search_step < self.num_mdp_timesteps else 1
        return self.sub_agents[sub_agent_idx].select_repair_operator(state, **kwargs)

    def after_operator_applied(self, operator, **kwargs):
        pass
