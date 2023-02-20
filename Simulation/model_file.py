import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import numpy as np
import random
import networkx as nx
import math
import time

class Model:
    def __init__(self, weights, internal, external, epsilon, delta, s,  include_cascade = True, state_init = 'random', polarizing_campaign = False) -> None:
        self.weights = weights
        self.internal = internal
        self.external = external
        self.epsilon = epsilon
        self.delta = delta
        self.n = weights.shape[0]
        self.include_cascade = include_cascade
        self.polar = -1
        self.states = np.zeros(self.n, dtype=int)
        self.polarizing_campaign = polarizing_campaign
        if state_init == 'max_degree':
            self.init_states_max_degree(s)
        # elif state_init == 'max_influence':
        #     self.init_states_max_influence(s)
        else:
            self.init_states_random(s)

    def init_states_random(self, s):
        num_seeds = math.ceil(s * self.n)
        # Pick 'num_seeds' random nodes to be seeds
        seeds = random.sample(range(self.n), num_seeds)
        print("The seeds are:", seeds)

        # Set the states of the seeds to 2
        for i in seeds:
            self.states[i] = 2
    
    def init_states_max_degree(self, s):
        seeds = math.ceil(s * self.n)
        degrees = np.sum(self.weights, axis=1).flatten()
        degrees = np.squeeze(np.asarray(degrees))
        ind = np.argpartition(degrees, -seeds)[-seeds:]
        # print("The degrees are:", degrees[ind])
        for i in ind:
            self.states[i] = 2
    
    def get_weights(self):
        return self.weights
    
    def get_internal(self):
        return self.internal
    
    def get_external(self):
        return self.external

    def set_weights(self, weights):
        self.weights = weights

    def replace_weights(self):
        for i in range(self.n):
            # Get the neighbors for node i
            neighbors_i = self.weights[i].nonzero()[1]

            for j in neighbors_i:
                if j <= i:
                    continue
                # Get the neighbors for node j
                neighbors_j = self.weights[j].nonzero()[1]

                # Calculate the Jaccard similarity
                intersection = len(set(neighbors_i).intersection(neighbors_j)) + 2
                union = len(set(neighbors_i).union(neighbors_j))
                jaccard = intersection / union

                # Set the edge weight
                self.weights[i, j] = jaccard
                self.weights[j, i] = jaccard
    
    def over_lay_graph(self, d, re_calculate_weights = False):
        if d == 0:
            return
        edges = len(self.weights.nonzero()[0])
        av_deg = 2 * edges / self.n
        print("av_deg", av_deg)

        #generate the d-regular random graph
        G_dreg = nx.random_regular_graph(d=d, n=self.n)

        # Convert the graph to a sparse matrix
        weights = nx.to_scipy_sparse_matrix(G_dreg, format='csc') * 1 / d

        #add the d-regular random graph on top of the original graph
        
        if re_calculate_weights:
            self.weights = self.weights + weights
            self.replace_weights()
        else:
            self.weights = self.weights / 2 + weights / 2

    def opinion_formation_round(self) -> None:
        intermediate_step = self.weights @ self.external + self.internal
        for i in range(self.n):
            intermediate_step[i] = intermediate_step[i] / (self.weights[i].sum() + 1)
        self.external = intermediate_step
    
    
    def information_spread_round(self) -> None:
        new_states = np.copy(self.states)

        for i, state in enumerate(self.states):
            exposed = []
            neighbors = self.weights[i].nonzero()[1]
            for j in neighbors:
                if self.states[j] == 2:
                    exposed.append(self.weights[i, j])

            if state == 0:
                for p_uv in exposed:
                    if p_uv < random.uniform(0, 1):
                        continue
                    else:
                        # Update the internal opinion
                        if self.polarizing_campaign:
                            if self.internal[i] < 0.5:
                                self.internal[i] = max(self.internal[i] - self.epsilon, 0)
                            else:
                                self.internal[i] = min(self.internal[i] + self.epsilon, 1)
                        else:
                            self.internal[i] = min(self.internal[i] + self.epsilon, 1)
                        new_states[i] = 1
                        if self.delta < random.uniform(0, 1):
                            continue
                        else:
                            new_states[i] = 2
                            break
            elif state == 1:
                for p_uv in exposed:
                    if self.delta * p_uv < random.uniform(0, 1):
                        continue
                    else:
                        new_states[i] = 2
                        break
            elif state == 2:
                new_states[i] = 3
        
        self.states = new_states


    def external_equilibrium(self):
        L = csgraph.laplacian(self.weights, normed=False)
        I_plus_L_inv = sp.linalg.inv((sp.identity(self.n) + L).tocsc())
        z_equilibrium = I_plus_L_inv @ self.external
        return z_equilibrium

    def polarization(self, vect):
        average = vect.mean()
        return np.sum((vect - average) ** 2)

    def disagreement(self, vect):
        # Loop over all edges and calculate the disagreement
        disagreement = 0
        for i in range(self.n):
            neighbors = self.weights[i].nonzero()[1]
            for j in neighbors:
                # Increment weighted disagreement
                disagreement += self.weights[i, j] * (vect[i] - vect[j]) ** 2
        return disagreement / 2
    

    def run(self, rounds):
        I_state_counts = []
        A_state_counts = []
        S_state_counts = []
        R_state_counts = []
        polarization_external = []
        polarization_internal = []
        polarization_equilibrium = []
        disagreement_external = []
        disagreement_internal = []
        disagreement_equilibrium = []


        bins = np.bincount(self.states, minlength=4)
        I_state_counts.append(bins[0])
        A_state_counts.append(bins[1])
        S_state_counts.append(bins[2])
        R_state_counts.append(bins[3])
        polarization_external.append(self.polarization(self.external))
        polarization_internal.append(self.polarization(self.internal))
        polarization_equilibrium.append(self.polarization(self.external_equilibrium()))
        disagreement_external.append(self.disagreement(self.external))
        disagreement_internal.append(self.disagreement(self.internal))
        disagreement_equilibrium.append(self.disagreement(self.external_equilibrium()))



        for i in range(rounds):
            start_time = time.time()
            self.opinion_formation_round()
            if self.include_cascade:
                self.information_spread_round()

            bins = np.bincount(self.states, minlength=4)
            I_state_counts.append(bins[0])
            A_state_counts.append(bins[1])
            S_state_counts.append(bins[2])
            R_state_counts.append(bins[3])
            polarization_external.append(self.polarization(self.external))
            polarization_internal.append(self.polarization(self.internal))
            polarization_equilibrium.append(self.polarization(self.external_equilibrium()))
            disagreement_external.append(self.disagreement(self.external))
            disagreement_internal.append(self.disagreement(self.internal))
            disagreement_equilibrium.append(self.disagreement(self.external_equilibrium()))

            end_time = time.time()
            print(f"Round {i} took {end_time - start_time} seconds")

        
        return I_state_counts, A_state_counts, S_state_counts, R_state_counts, polarization_external, polarization_internal, polarization_equilibrium, disagreement_external, disagreement_internal, disagreement_equilibrium


        
    