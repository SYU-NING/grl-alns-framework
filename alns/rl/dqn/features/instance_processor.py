import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data

from utils.general_utils import maxabs_scale


class InstanceFeatureSpec(object):
    def __init__(self, name, feat_type, datatype, feat_dim, should_rescale):
        self.name = name
        self.feat_type = feat_type
        self.datatype = datatype
        self.feat_dim = feat_dim
        self.should_rescale = should_rescale

class InstanceFeatureProcessor(object):
    def __init__(self, hyperparams, max_nodes, problem_variant, operator_library):
        super().__init__()
        self.hyperparams = hyperparams
        self.max_nodes = max_nodes
        self.problem_variant = problem_variant

        self.arch = hyperparams["arch"]

        self.all_feats = self.get_feature_specs()
        self.node_feats = [f for f in self.all_feats if f.feat_type == "node"]
        self.edge_feats = [f for f in self.all_feats if f.feat_type == "edge"]
        self.global_feats = [f for f in self.all_feats if f.feat_type == "global"]
        self.operator_library = operator_library



    def get_feature_specs(self):
        feats = [
            InstanceFeatureSpec("loc_x", "node", "float", 1, True),
            InstanceFeatureSpec("loc_y", "node", "float", 1, True),
            InstanceFeatureSpec("demand", "node", "float", 1, True),
            InstanceFeatureSpec("position_index", "node", "float", 1, False),
        ]

        if self.arch == "mlp":
            feats.extend([InstanceFeatureSpec("pairwise_dist", "node", "float", 1, False)])

        # tour indices redundant for TSP.
        if self.problem_variant != "TSP":
            if self.hyperparams['onehot_tour_idxes']:
                feats.extend([
                    InstanceFeatureSpec("tour_index_oh", "node", "int", self.max_nodes, False),
                ])
            else:
                feats.extend([
                    InstanceFeatureSpec("tour_index", "node", "float", 1, False),
                ])

        if self.arch in ["gat", "gcn", "sage"]:
            feats.extend([
                InstanceFeatureSpec("neighbour_dist", "edge", "float", 1, False)
            ])

        self.inst_size_repeat = self.hyperparams['inst_size_repeat']

        feats.extend([
            InstanceFeatureSpec("inst_size", "global", "float", 1 * self.inst_size_repeat, False)
        ])

        if self.problem_variant == "VRPTW":
            feats.extend([
                InstanceFeatureSpec("tw_start", "node", "float", 1, True),
                InstanceFeatureSpec("tw_end", "node", "float", 1, True),
                InstanceFeatureSpec("service_time", "node", "float", 1, True),

            ])
        return feats

    def get_feature_sizes(self):
        node_feat_size = sum([f.feat_dim for f in self.node_feats])
        edge_feat_size = sum([f.feat_dim for f in self.edge_feats])
        global_feat_size = sum([f.feat_dim for f in self.global_feats])

        return node_feat_size, edge_feat_size, global_feat_size


    def get_mlp_max_input_size(self):
        input_size = 0
        input_size += sum([f.feat_dim for f in self.node_feats]) * self.max_nodes
        input_size += sum([f.feat_dim for f in self.global_feats])

        return input_size


    def get_states_representations(self, state_list, **kwargs):
        repr_list = []
        for i, state in enumerate(state_list):
            state_repr = self.get_state_representation(state)
            repr_list.append(state_repr)

        return repr_list


    def get_state_representation(self, state):
        if not hasattr(state, "rescaled_feats"):
            rescaled_feats = {}

            for feat in self.all_feats:
                if feat.should_rescale:
                    if feat in ["tw_start", "tw_end", "service_time"]:
                        # special rescaling for these...
                        max_tw_end = max(state.tw_end)
                        if "tw_start" not in rescaled_feats:
                            rescaled_feats["tw_start"] = [v / max_tw_end for v in state.tw_start]
                        if "tw_end" not in rescaled_feats:
                            rescaled_feats["tw_end"] = [v / max_tw_end for v in state.tw_end]
                        if "service_time" not in rescaled_feats:
                            rescaled_feats["service_time"] = [v / max_tw_end for v in state.service_time]
                    else:
                        rescaled_feats[feat.name] = maxabs_scale(getattr(state.inst, feat.name))

            state.rescaled_feats = rescaled_feats

        if self.arch == "mlp":
            return self.get_mlp_representation(state)
        elif self.arch in ["gat", "gcn", "sage"]:
            return self.get_gnn_representation(state)
        else:
            raise ValueError(f"unknown architecture {self.arch}")

    def get_gnn_representation(self, state):
        total_num_nodes = state.num_nodes
        node_feat_size, edge_feat_size, global_feat_size = self.get_feature_sizes()

        node_feats_flat = torch.zeros(node_feat_size * total_num_nodes)
        self.populate_node_features(state, node_feats_flat)
        node_feats = node_feats_flat.view(total_num_nodes, node_feat_size)

        edge_list, edge_index = self.extract_edge_index(state)

        edge_attr = self.construct_edge_feats(state, edge_list)
        global_feats = torch.zeros(global_feat_size)
        self.populate_global_feats(state, 0, global_feats)
        data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attr, global_feats=global_feats,
                    state_num_nodes=torch.IntTensor([total_num_nodes]))
        return data

    def extract_edge_index(self, state):
        alns_list_as_edge_list = []
        for tour in state.alns_list:
            alns_list_as_edge_list.append((0, tour[0]))
            for i in range(len(tour) - 1):
                alns_list_as_edge_list.append((tour[i], tour[i+1]))
            alns_list_as_edge_list.append((tour[-1], 0))

        edge_index = torch.stack([torch.LongTensor([e[0] for e in alns_list_as_edge_list]),
                                  torch.LongTensor([e[1] for e in alns_list_as_edge_list])], dim=0)

        return alns_list_as_edge_list, edge_index

    def get_mlp_representation(self, state):
        repr_size = self.get_mlp_max_input_size()
        out_tensor = torch.zeros(repr_size)

        curr_pos = self.populate_node_features(state, out_tensor)
        curr_pos = self.populate_global_feats(state, curr_pos, out_tensor)

        return out_tensor

    def construct_edge_feats(self, state, edge_list):
        total_num_edges = len(edge_list)
        edge_attr = torch.zeros((total_num_edges, self.get_feature_sizes()[1]))
        ef_dim = 0
        for ef in self.edge_feats:
            feat = torch.zeros((total_num_edges, ef.feat_dim))
            if ef.name == "neighbour_dist":
                for i, edge in enumerate(edge_list):
                    dist = state.distance_matrix[(edge[0], edge[1])]
                    if self.hyperparams['use_rounding']:
                        dist = round(dist, 0)

                    feat[i, :] = torch.FloatTensor([dist])

            else:
                raise ValueError(f"edge feature {ef.name} not known.")

            edge_attr[:, ef_dim:ef_dim + ef.feat_dim] = feat
            ef_dim += ef.feat_dim
        return edge_attr

    def populate_global_feats(self, state, starting_pos, out_tensor):
        for gf in self.global_feats:
            # if gf.name == "was_scale_selected":
            #     scale_indicator = (0 if state.selected_scale == -1 else 1)
            #     oh = F.one_hot(torch.LongTensor([scale_indicator]), num_classes=2).repeat(1, self.scale_repeat)
            #     out_tensor[starting_pos: starting_pos + gf.feat_dim] = oh
            if gf.name == "inst_size":
                inst_size_scaling_factor = 10
                feat_val = torch.FloatTensor([state.inst.cust_number / inst_size_scaling_factor] * self.inst_size_repeat)
                out_tensor[starting_pos: starting_pos + gf.feat_dim] = feat_val

            starting_pos += gf.feat_dim
        return starting_pos

    def populate_node_features(self, state, out_tensor):
        curr_pos = 0
        alns_list_with_depot_feature = ([[0]] + state.alns_list)
        for tour_index, tour in enumerate(alns_list_with_depot_feature):
            for position_index, cust in enumerate(tour):
                for nf in self.node_feats:
                    if nf.name in ["loc_x", "loc_y", "demand", "tw_start", "tw_end", "service_time"]:
                        if not nf.should_rescale:
                            vals = getattr(state.inst, nf.name)
                        else:
                            vals = state.rescaled_feats[nf.name]

                        out_tensor[curr_pos] = vals[cust]
                    elif nf.name == "tour_index":
                        out_tensor[curr_pos] = tour_index
                    elif nf.name == "tour_index_oh":
                        feat_val = F.one_hot(torch.LongTensor([tour_index]), num_classes=self.max_nodes)
                        out_tensor[curr_pos: curr_pos + nf.feat_dim] = feat_val

                    elif nf.name == "position_index":
                        out_tensor[curr_pos] = position_index / self.max_nodes
                    elif nf.name == "pairwise_dist":
                        if position_index == 0 or position_index == len(tour) - 1:  # feature 6. neighbour distance
                            dist = state.distance_matrix[(cust, 0)]
                        else:
                            next_neighbour = tour[position_index + 1]
                            dist = state.distance_matrix[(cust, next_neighbour)]

                        if self.hyperparams['use_rounding']:
                            dist = round(dist, 0)

                        out_tensor[curr_pos] = dist

                    curr_pos += nf.feat_dim
        return curr_pos






