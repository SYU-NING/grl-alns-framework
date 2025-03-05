import sys
from copy import deepcopy

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Linear, ReLU, Sigmoid, Softmax, MSELoss

from torch.nn.init import kaiming_uniform_, xavier_uniform_, xavier_normal_
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv

from alns.rl.dqn.features.instance_processor import InstanceFeatureProcessor
from operators.base_op import Operator


class QNet(nn.Module):
    def __init__(self, feat_processor, hyperparams, all_possible_acts):
        super(QNet, self).__init__()

        self.feat_processor = feat_processor
        self.hyperparams = hyperparams
        self.arch = hyperparams["arch"]

        self.all_possible_acts = all_possible_acts
        self.output_size = len(all_possible_acts)
        self.construct_action_mappings()

        self.hidden_layers = nn.ModuleList()

        if self.arch == "mlp":
            self.build_mlp_layers()
        elif self.arch in ["gat", "gcn", "sage"]:
            self.build_gnn_layers()
        else:
            raise ValueError(f"unknown architecture {self.arch}")

    def build_gnn_layers(self):
        self.num_hidden_layers = self.hyperparams['num_hidden_layers']
        self.node_feat_size, self.edge_feat_size, self.global_feat_size = self.feat_processor.get_feature_sizes()
        self.lf_dim = self.hyperparams['lf_dim']

        if self.arch == "gat":
            num_heads = 1
            self.hidden_layers.append(GATv2Conv(self.node_feat_size, self.lf_dim, heads=num_heads, edge_dim=self.edge_feat_size))
            for i in range(0, self.num_hidden_layers - 1):
                self.hidden_layers.append(GATv2Conv(self.lf_dim, self.lf_dim, heads=1, edge_dim=self.edge_feat_size))
        elif self.arch == "gcn":
            self.hidden_layers.append(GCNConv(self.node_feat_size, self.lf_dim))
            for i in range(0, self.num_hidden_layers - 1):
                self.hidden_layers.append(GCNConv(self.lf_dim, self.lf_dim))
        elif self.arch == "sage":
            self.hidden_layers.append(SAGEConv(self.node_feat_size, self.lf_dim))
            for i in range(0, self.num_hidden_layers - 1):
                self.hidden_layers.append(SAGEConv(self.lf_dim, self.lf_dim))

        self.output_layer = Linear(self.lf_dim + self.global_feat_size, self.output_size)


    def build_mlp_layers(self):
        self.num_hidden_layers = 0
        self.input_size = self.feat_processor.get_mlp_max_input_size()

        layer_in_size = self.input_size
        layer_out_size = self.hyperparams['first_hidden_size']
        layer_min_size = 16

        while True:
            layer = Linear(layer_in_size, layer_out_size)
            self.hidden_layers.append(layer)
            self.num_hidden_layers += 1

            layer_in_size = layer_out_size
            layer_out_size = int(layer_out_size / 2)

            if layer_out_size <= layer_min_size:
                break

        self.output_layer = Linear(layer_in_size, self.output_size)
        self.init_weights()


    def construct_action_mappings(self):
        self.action_to_encoding = {}
        self.encoding_to_action = {}

        for i in range(self.output_size):
            self.action_to_encoding[self.all_possible_acts[i]] = i
            self.encoding_to_action[i] = self.all_possible_acts[i]

    def init_weights(self):
        for l in self.hidden_layers:
            kaiming_uniform_(l.weight)

        kaiming_uniform_(self.output_layer.weight)

    def forward(self, states, for_actions=None, **kwargs):
        repr_list = self.feat_processor.get_states_representations(states, **kwargs)
        if self.arch == "mlp":
            x = torch.vstack(repr_list)
        elif self.arch in ["gat", "gcn", "sage"]:
            # TODO: currently, does not make use of the global features
            # also using a large batch when evaluating, ok on CPU, may be problematic on GPUs
            loader = DataLoader(repr_list, batch_size=len(repr_list))
            all_data = next(iter(loader))
            x, edge_index, edge_attr, global_feats = all_data.x, all_data.edge_index, all_data.edge_attr, all_data.global_feats


        for i, hidden_layer in enumerate(self.hidden_layers):
            if self.arch == "mlp":
                x = hidden_layer(x)
            elif self.arch == "gat":
                x = hidden_layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
            elif self.arch in ["gcn", "sage"]:
                x = hidden_layer(x=x, edge_index=edge_index)

            x = F.relu(x)

        if self.arch == "mlp":
            preds = self.output_layer(x)
        elif self.arch in ["gat", "gcn", "sage"]:
            batch_num_nodes = all_data.state_num_nodes
            preds = torch.zeros(len(repr_list), self.output_size)
            split_embeds = torch.split(x, batch_num_nodes.tolist())
            split_globals = torch.split(global_feats, self.global_feat_size)

            for g_num, g_embed in enumerate(split_embeds):
                graph_rep = torch.sum(g_embed, dim=0)
                graph_globals = split_globals[g_num]
                graph_rep_and_global = torch.hstack((graph_rep, graph_globals))
                pred = self.output_layer(graph_rep_and_global)
                preds[g_num, :] = pred



        if for_actions is None:
            return preds
        else:
            if not isinstance(for_actions[0], Operator):
                act_idxes_list = [self.action_to_encoding[a] for a in for_actions]
            else:
                act_idxes_list = [self.action_to_encoding[a.name] for a in for_actions]

            act_idxes = torch.LongTensor(act_idxes_list).unsqueeze(1)
            qvals = torch.gather(preds, 1, act_idxes)
            return qvals


class NStepQNet(nn.Module):
    def __init__(self, feat_processor, hyperparams, ol, problem_variant):
        super(NStepQNet, self).__init__()
        self.ol = ol
        self.problem_variant = problem_variant

        unique_acts = [ol.unique_ops_destroy,
                       ol.unique_ops_repair]

        list_mod = []
        for i, all_possible_acts in enumerate(unique_acts):
            list_mod.append(QNet(feat_processor, hyperparams, all_possible_acts))

        self.list_mod = nn.ModuleList(list_mod)
        self.num_steps = len(unique_acts)

    def greedy_actions(self, time_t, est_q_values):
        max_idxes = torch.argmax(est_q_values, dim=1)
        which_subnet = self.list_mod[time_t]
        encoding_to_action = which_subnet.encoding_to_action
        greedy_acts = [encoding_to_action[max_idx] for max_idx in max_idxes.tolist()]
        return greedy_acts

    def get_action_by_index(self, time_t, act_index):
        which_subnet = self.list_mod[time_t]
        encoding_to_action = which_subnet.encoding_to_action
        return encoding_to_action[act_index]

    def get_max_qvals(self, time_t, est_q_values):
        max_idxes = torch.argmax(est_q_values, dim=1)
        index_unsq = torch.unsqueeze(max_idxes, 1)
        max_q_vals = torch.gather(est_q_values, 1, index_unsq).squeeze()
        return max_q_vals


    def forward(self, time_t, states, for_actions=None, **kwargs):
        assert time_t >= 0 and time_t < self.num_steps
        return self.list_mod[time_t](states, for_actions=for_actions, **kwargs)
