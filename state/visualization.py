import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from copy import deepcopy
import numpy as np 
import pandas as pd

from environment.objective_functions import TotalObjective, compute_cust_time_window_violation


objective_function_display_names = {TotalObjective.name: "Total Objective"}
fig_dpi = 200
font_scale = 1.75


def plot_eval_histories(results_df,
                      figure_save_path,
                      separate_seeds=True):
    sns.set(font_scale=font_scale)
    plt.rcParams["lines.linewidth"] = 1
    plt.rc('font', family='serif')

    # dims = (16.54, 24.81)
    # dims = (16.54, 16.54)

    objs = results_df["objective_function"].unique()

    num_objs = len(objs)

    dims = (8.26 * num_objs, 8.26)

    fig, axes = plt.subplots(1, num_objs, sharex='none', sharey='none', figsize=dims, squeeze=False)

    for i in range(num_objs):
        obj = objs[i]
        filtered_data = results_df[(results_df['objective_function'] == obj)]

        filtered_data = filtered_data.rename(columns={"timestep": "epoch",
                                                      "perf": "Evaluation performance"}).reset_index()

        ax = axes[0][i]
        ax = sns.lineplot(data=filtered_data, x="epoch", y="Evaluation performance",
                          ax=ax, hue=("model_seed" if separate_seeds else None))

        handles, labels = ax.get_legend_handles_labels()

        if i == 0:
            ax.set_ylabel('$\mathbf{G}^{eval}$ performance', size="small")
        else:
            ax.set_ylabel('')

            #ax.legend_.remove()
            #ax.set_xticks(network_sizes)

    pad = 2.5  # in points

    rows = objs

    for ax, row in zip(axes[:, 0], rows):
        ax.annotate(f"{objective_function_display_names[row]}", xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    rotation=90,
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='medium', ha='right', va='center')

    fig.tight_layout()
    # fig.tight_layout(rect=[0,0,1,0.90])
    # fig.subplots_adjust(left=0.15, top=0.95)
    fig.savefig(figure_save_path, bbox_inches='tight', dpi=fig_dpi)

    # plt.show()
    plt.close()
    plt.rcParams["lines.linewidth"] = 1.0


def plot_geographical_location(state):
    ''' Print out all the nodes on a geographical plot '''
    loc_x = state.inst.loc_x
    loc_y = state.inst.loc_y

    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize = (6, 6))
    plt.scatter(loc_x, loc_y, color="green")

    for i in state.depot:
        plt.plot(loc_x[i], loc_y[i], color="black", marker="s")
        plt.annotate("$Depot_{%d}$" %i, (loc_x[i], loc_y[i]))
        
    for i in state.customers:
        plt.plot(loc_x[i], loc_y[i], color="green")
        plt.annotate('$C_{%d}$' %(i), (loc_x[i], loc_y[i]))
        #plt.annotate('$C_{%d}=%d$' %(i,Demand[i]), (loc_x[i], loc_y[i]))

    node_num = len(loc_y)
    
    plt.xlabel("x-coordinate")
    plt.ylabel("ycoordinate")
    plt.title(f"Geographical Location ({node_num} nodes)")
    plt.show()



def plot_visiting_sequence(state, **kwargs):
    ''' Plot the nodes visiting sequence from ALNS list and removed customers '''
    loc_x = state.inst.loc_x
    loc_y = state.inst.loc_y

    #(1) Print out all underlying nodes for facilities and customers:
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize = (6, 6))
    
    for i in state.depot:
        plt.plot(loc_x[i], loc_y[i], color="black", marker="s")
        plt.annotate("$%d$" %i, (loc_x[i], loc_y[i]))   

    for c in state.customers:
        marker_color = "green" if state.node_status[c] else "red"
        plt.plot(loc_x[c], loc_y[c], color=marker_color, marker="o")
        plt.annotate("$%d$" % c, (loc_x[c], loc_y[c]))

    #(2) Find all active arcs from an ALNS list:
    active_arcs = []
    
    alns_list = kwargs['alns_list'] if 'alns_list' in kwargs else deepcopy(state.alns_list)

    for tour in alns_list:
        for i in range(len(tour) - 1):
            active_arcs.append((tour[i], tour[i+1]))
        if state.inst.problem_variant == "LRP":
            active_arcs.append((tour[0], state.customer_depot_allocation[tour[0]]))
            active_arcs.append((tour[-1], state.customer_depot_allocation[tour[-1]]))
        else: 
            active_arcs.append((0, tour[0]))
            active_arcs.append((tour[-1], 0))

    #(3) Plot the active arcs out: 
    for i, j in active_arcs: 
        node1_x = loc_x[i]
        node1_y = loc_y[i]
        node2_x = loc_x[j]
        node2_y = loc_y[j]
        plt.plot([node1_x, node2_x], [node1_y, node2_y], color="darkblue", alpha=0.7)

    node_num = len(loc_y)

    plt.xlabel("x-coordinate")
    plt.ylabel("ycoordinate")
    plt.title(f"Node Visiting Sequence ({node_num} nodes)")
    #print("The visiting sequence: ", state.alns_list)
    #print([arcs for arcs in active_arcs])
    plt.show()



def plot_objective_value_improvement(state, list_of_alns, list_of_method):
    plt.figure(figsize = (8, 5))
    plt.tight_layout()    
    for index, alns in enumerate(list_of_alns):
        axis_x = [s[0] for s in alns.best_obj_record]   # evaluate improvement brought by each destroy-repair iteration
        axis_y = [s[1] for s in alns.best_obj_record]
        plt.plot(axis_x, axis_y, marker='o', markersize=2, label = str(list_of_method[index])) 
        #axis_x = [s[0] for s in alns.best_obj_record_frequency]   # evaluate improvement brought by each operator 
        #axis_y = [s[1] for s in alns.best_obj_record_frequency]
        #plt.plot(axis_x, axis_y, marker='o', markersize=2, label = str(list_of_method[index])) 
    plt.title("Progress curve with "+str(len(state.customers))+ " customer nodes")
    plt.xlabel("iterations (destroy-repair pairs)")
    plt.ylabel("objective value")
    plt.legend() 
   # plt.savefig("Progresscurve_" + str(len(state.customers)) + "customers.png")
    plt.show()


def box_plot_vrw(objective_dicts_total):
    num_of_seeds = len(list(objective_dicts_total.values())[0])
    Matrix = [[value[i] for value in list(objective_dicts_total.values())] for i in range(num_of_seeds)]      
    df = pd.DataFrame(Matrix, columns=list(objective_dicts_total.keys()))
    df.plot(kind='box')
    plt.title("VRW box plot ({} seeds)".format(num_of_seeds))
    print('Average: ', [np.average(list(obj.values())) for obj in list(objective_dicts_total.values())])
    plt.savefig('vrw_box_plot.png')




### additional VRPTW functions:

def count_operator_usage_frequency(my_list):
    freq = {}
    for items in my_list:
        freq[items] = my_list.count(items)
    freq_sorted = dict(sorted(freq.items(), reverse=True, key=lambda item: item[1]))   # max to min
    return freq_sorted

def check_100_customers(alns_list):
        ''' check if a state has 100 customers in its sequence (alns_list); '''
        full_list = []
        for tour in alns_list:
            for cust in tour: 
                if cust not in full_list:
                    full_list.append(cust)
                else: 
                    full_list.append(cust)
                    print("*** customer appears 2 times: ", cust)
        print("* number of customers inside the tour: ", len(full_list))
        if [x for x in [i for i in range(1, 101)] if x not in full_list] != []:
            print("*** customer missing: ",[x for x in [i for i in range(1, 101)] if x not in full_list])

def compute_tw_violation(state):
    objective = {}
    total_travel_cost = 0
    total_exceed_time = 0
    for tour in state.alns_list[0:1]:
        node2_arrival_time = 0
        for arc in range(len(tour)+1):
            node1 = tour[arc-1] if arc != 0 else 0
            node2 = tour[arc] if arc != len(tour) else 0
            travel_cost = state.distance_matrix[(node1, node2)] 
            total_travel_cost += travel_cost
            
            node1_service_duration = state.inst.service_time[node1] if arc != 0 else 0

            print('{}-->{}:'.format(node1, node2))
            print('({})_serve_time = {}, travel = {}'.format(node1, 
                                                             node1_service_duration,
                                                             travel_cost))
            
            node2_arrival_time += node1_service_duration + travel_cost
            print('({})_arrival = {}, ({})_tw-start = {}, ({})_tw-end = {}'.format( node2, 
                                                                                    node2_arrival_time, 
                                                                                    node2, 
                                                                                    state.inst.tw_start[node2], 
                                                                                    node2, 
                                                                                    state.inst.tw_end[node2]))
                    
            total_exceed_time += compute_cust_time_window_violation(state, node2, node2_arrival_time)
            print('violation = ', compute_cust_time_window_violation(state, node2, node2_arrival_time))
            print()