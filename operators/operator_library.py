from operators.base_op import Operator
from operators.localsearch_operators import TwoOptLocalSearch
from operators.destroy_operators import *
from operators.repair_operators import *


class OperatorLibrary(object):
    def __init__(self,
                 fixed_scale_pc,
                 random_seed,
                 which_destroy='all',
                 which_repair='all',
                 which_localsearch='2opt_only',
                 **kwargs):
        
        self.fixed_scale_pc = fixed_scale_pc

        # used by QNet
        self.unique_ops_destroy = []
        self.unique_ops_repair = []

        self.destroy_ops = {}
        self.repair_ops = {}
        self.local_search_ops = {}


        # Retrieve all destroy and repair operators' subclasses
        # original: destroy_op_classes = which_destroy if which_destroy != 'all' else self.get_all_supported_operators('D')
        problem_variant = kwargs["problem_variant"] if "problem_variant" in kwargs else None
        destroy_op_classes = which_destroy if which_destroy != 'all' else self.get_compatible_destroy_operators(problem_variant)

        self.unique_ops_destroy = [d_cls.name for d_cls in destroy_op_classes]

        repair_op_classes = which_repair if which_repair != 'all' else self.get_all_supported_operators('R')
        self.unique_ops_repair = [r_cls.name for r_cls in repair_op_classes]

        for d_cls in destroy_op_classes:
            operator_instance = d_cls(self.fixed_scale_pc, random_seed)   # specific destroy operator class
            op_str = str(operator_instance)                 # specific destroy operator string
            self.destroy_ops[op_str] = operator_instance    # create dictionary {str: class} e.g. {'D-Random-2': D-Random-2, 'D-Greedy-2': D-Greedy-2, 'D-Related-2': D-Related-2} for min_scale = max_scale = 2

        for r_cls in repair_op_classes:
            operator_instance = r_cls(self.fixed_scale_pc, random_seed)
            op_str = str(operator_instance)
            self.repair_ops[op_str] = operator_instance


        localsearch_op_classes = None
        if which_localsearch == 'all':
            localsearch_op_classes = self.get_all_supported_operators('L')
        elif which_localsearch == '2opt_only':
            localsearch_op_classes = [TwoOptLocalSearch]

        for l_cls in localsearch_op_classes:
            ls_scale = 0
            operator_instance = l_cls(ls_scale, random_seed)
            op_str = str(operator_instance)
            self.local_search_ops[op_str] = operator_instance


    def get_available_destroy_ops(self):
        av_ops = list(self.destroy_ops.values())
        return av_ops


    def get_available_repair_ops(self):
        av_ops = list(self.repair_ops.values())
        return av_ops

    def get_available_local_search_ops(self):
        return list(self.local_search_ops.values())


    def get_destroy_op(self, op_str):
        return self.destroy_ops[op_str]


    def get_repair_op(self, op_str):
        return self.repair_ops[op_str]


    def get_localsearch_op(self, op_str):
        return self.local_search_ops[op_str]

    @staticmethod
    def scale_from_op_str(op_str):
        return int(op_str.split("-")[-1])

    @staticmethod
    def get_all_supported_operators(operator_type):
        return [cls for cls in Operator.__subclasses__() if cls.name[0] == operator_type]


    @staticmethod
    def create_operator_from_string(operator_string, random_seed):
        scale = int(operator_string[-1].split("-")[-1])
        operator = "-".join(operator_string.split("-")[:-1])

        matching_operators = [cls for cls in Operator.__subclasses__() if cls.name == operator]
        if len(matching_operators) > 1:
            raise ValueError(f"two operators have the same name, which is not allowed.")

        op = matching_operators[0]
        return op(scale, random_seed)


    @staticmethod
    def get_compatible_destroy_operators(variant, include_noop=True):
        # NodeNeighbourhoodDestroy, ZoneDestroy excluded for TSP.
        universal = [NoOpDestroy] if include_noop else []

        facility_neighbourhood = [FacilityCloseDestroy, FacilityOpenDestroy, # random-based
                                  SameFacilityDestroy,                       # greedy-based
                                  FacilitySwapDestroy                        # related-based
                                  ]  
        time_window_neighbourhood = [GreedyTimeDestroy, RelatedTimeDestroy,  # greedy-based
                                     RelatedDurationDestroy                  # related-based
                                     ]                 
        
        distance_vanilla = [RandomDestroy, GreedyDestroy, RelatedDestroy]
        distance_neighbourhood = [RandomRouteDestroy,                        # random-based
                                  RouteDeviateDestroy, GreedyRouteDestroyMax, GreedyRouteDestroyMin,  # greedy-based
                                  NodeNeighbourhoodDestroy, ZoneDestroy, PairDestroy, RouteNeighbourhoodDestroy, ClusterDestroy,
                                  ShawDestroy                                # general related-based
                                  ]
        
        demand_neighbourhood = [CapacityDeviateDestroy, GreedyCapacityDestroyMin, GreedyCapacityDestroyMax, # greedy-based
                                RelatedDemandDestroy
                                ]
        
        history_neighbourhood = [HistoricalPairDestroy, 
                                 HistoricalKnowledgeDestroy, 
                                 ForbiddenRandomDestroy
                                 ]
        

        if variant == "TSP":
            return universal + [RandomDestroy, GreedyDestroy, RelatedDestroy,
                    PairDestroy, RouteNeighbourhoodDestroy, ClusterDestroy,
                    RouteDeviateDestroy, GreedyRouteDestroyMax, GreedyRouteDestroyMin, RandomRouteDestroy,
                    HistoricalPairDestroy, HistoricalKnowledgeDestroy, ForbiddenRandomDestroy]
      
        elif variant == "CVRP":
            return universal + distance_vanilla + distance_neighbourhood + demand_neighbourhood + history_neighbourhood
        
        elif variant == "HVRP":
            return universal + distance_vanilla + distance_neighbourhood + demand_neighbourhood + history_neighbourhood

        elif variant == "LRP":
            return universal + distance_vanilla + distance_neighbourhood + demand_neighbourhood + history_neighbourhood + facility_neighbourhood
        
        elif variant == "VRPTW":
            return universal + distance_vanilla + distance_neighbourhood + demand_neighbourhood + history_neighbourhood + time_window_neighbourhood

    @staticmethod
    def get_compatible_repair_operators(variant):
        universal = [RandomRepair, GreedyRepair, GreedyRepairPerturbation,
                     DeepGreedyRepair, Regret2Repair, Regret3Repair, Regret4Repair]

        return universal