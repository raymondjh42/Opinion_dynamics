import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import numpy as np
import time
import pickle
import random
import matplotlib.pyplot as plt
import networkx as nx
import random
import pandas as pd
import networkx as nx
import model_file as mf
import multiprocessing
import glob

datasets = [
    '../DATA/snap/facebook_combined.pkl',
    '../DATA/socfb-Mich67/socfb-Mich67.pkl',
    '../DATA/socfb-Wellesley22/socfb-Wellesley22.pkl'
]

target_data = 'fb_mich'

# Unpickle weight matrix of the graph
if target_data == 'fb_snap':
    with open(datasets[0], 'rb') as handle:
        weights = pickle.load(handle)
elif target_data == 'fb_mich':
    with open(datasets[1], 'rb') as handle:
        weights = pickle.load(handle)
elif target_data == 'fb_wel':
    with open(datasets[2], 'rb') as handle:
        weights = pickle.load(handle)
else:
    print("Invalid dataset")
    exit()


epsilon = 0.1
delta = 0.5
runs = 15 
n = weights.shape[0]
s = 0.005

state_init = ['random', 'maxDegree'] # , 'maxDegree'
d_edges = [0, 2, 4, 8, 16, 32] #, 2, 4, 8, 12, 16, 24, 32, 48, 64
iterations = [0, 1, 2, 3, 4] # 
cascade = ['cascade', 'noCascade']

simulations_required = set()

for state in state_init:
    for d in d_edges:
        for i in iterations:
            for c in cascade:
                simulations_required.add((state, d, i, c))

simulations_ran = set()
for file in glob.glob("simulation_results/results_"+target_data+"/*.csv"):
    spl = file.split("_")
    simulations_ran.add((spl[6], int(spl[5]), int(spl[7]), spl[8].split(".")[0]))

simulations_to_run = simulations_required - simulations_ran

print(len(simulations_to_run))

# For every tuple in simulations_to_run ensure that there is cascade and noCascade
to_add = set()
for sim in simulations_to_run:
    if (sim[0], sim[1], sim[2], 'cascade') not in simulations_to_run:
        to_add.add((sim[0], sim[1], sim[2], 'cascade'))
    if (sim[0], sim[1], sim[2], 'noCascade') not in simulations_to_run:
        to_add.add((sim[0], sim[1], sim[2], 'noCascade'))
simulations_to_run = simulations_to_run.union(to_add)

# Convert size 4 tuples in simulations_to_run to size 3 tuples by removing the cascade/noCascade
simulations_to_run = list(set([sim[:3] for sim in simulations_to_run]))

print("There are {} simulations to run".format(len(simulations_to_run)))
print(simulations_to_run)

param_grid = simulations_to_run


def run_simulation(*args):
    print("The args are:", args[0])
    state_init, d_edges, iteration = args[0]

    results_df_1 = pd.DataFrame(columns=['round', 'I_state_counts,', 'A_state_counts,', 'S_state_counts,', 'R_state_counts,', 'polarization_internal', 'polarization_external', 'polarization_equilibrium', 'disagreement_internal', 'disagreement_external', 'disagreement_equilibrium'])
    results_df_2 = pd.DataFrame(columns=['round', 'I_state_counts,', 'A_state_counts,', 'S_state_counts,', 'R_state_counts,', 'polarization_internal', 'polarization_external', 'polarization_equilibrium', 'disagreement_internal', 'disagreement_external', 'disagreement_equilibrium'])

    internal_opinions_1 = np.random.rand(n)
    external_opinions_1 = np.copy(internal_opinions_1)
    internal_opinions_2 = np.copy(internal_opinions_1)
    external_opinions_2 = np.copy(external_opinions_1)

    model_1 = mf.Model(weights, internal_opinions_1, external_opinions_1, epsilon, delta, s, include_cascade=True, state_init=state_init)
    model_1.over_lay_graph(2, re_calculate_weights=False)
    simulation_tuple_1 = model_1.run(runs)

    for i in range(runs):
        results_df_1.loc[i] = [i, 
        simulation_tuple_1[0][i],
        simulation_tuple_1[1][i],
        simulation_tuple_1[2][i],
        simulation_tuple_1[3][i],
        simulation_tuple_1[4][i],
        simulation_tuple_1[5][i],
        simulation_tuple_1[6][i],
        simulation_tuple_1[7][i],
        simulation_tuple_1[8][i],
        simulation_tuple_1[9][i]
        ]
    # Save results_df_1 as csv
    results_df_1.to_csv(f'results_{target_data}/results_df_{d_edges}_{state_init}_{iteration}_cascade.csv')
    

    model_2 = mf.Model(weights, internal_opinions_2, external_opinions_2, epsilon, delta, s, include_cascade=False, state_init=state_init)
    model_2.over_lay_graph(d_edges, re_calculate_weights=False)
    simulation_tuple_2 = model_2.run(runs)

    for i in range(runs):
        results_df_2.loc[i] = [i, 
        simulation_tuple_2[0][i],
        simulation_tuple_2[1][i],
        simulation_tuple_2[2][i],
        simulation_tuple_2[3][i],
        simulation_tuple_2[4][i],
        simulation_tuple_2[5][i],
        simulation_tuple_2[6][i],
        simulation_tuple_2[7][i],
        simulation_tuple_2[8][i],
        simulation_tuple_2[9][i]
        ]
    
    # Save results_df_2 as csv
    results_df_2.to_csv(f'results_{target_data}/results_df_{d_edges}_{state_init}_{iteration}_noCascade.csv')

    # Print total time
    print(f"Run {iteration} with d = {d_edges} and init state {state_init}.")


if __name__ == '__main__': 
    #Generate processes equal to the number of cores
    pool = multiprocessing.Pool(4)

    print(pool._processes)

    # Get start time
    start_time_ = time.time()

    #Distribute the parameter sets evenly across the cores
    res  = pool.map(run_simulation, param_grid)

    end_time_ = time.time()
    print(f"Total time taken: {end_time_ - start_time_} seconds")
