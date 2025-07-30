from src.constants import *
from src.viz import display_matrix
from src.genetic_algorithm import *
import pandas as pd
from src.validation import *
import numpy as np
from src.utils import alpha_fit
import matplotlib.pyplot as plt
from scipy.stats import skewnorm


# neuron_data = {}

# for condition in ['experimental', 'control']:    
#     minimal_neurons = []
#     for n in range(5):
#         minimal_neuron = n**2
#         minimal_neurons.append(minimal_neuron)
        
#     neuron_data[condition] = minimal_neurons

# experimental_neurons=neuron_data['experimental']
# control_neurons=neuron_data['control']

# print(experimental_neurons)
# print(control_neurons)

# test = define_criteria(TMAX//BIN_SIZE)
# experimental_criteria = pd.DataFrame(test['experimental'], index=CRITERIA_NAMES, columns=range(TMAX//BIN_SIZE))
# control_criteria = pd.DataFrame(test['control'], index=CRITERIA_NAMES, columns=range(TMAX//BIN_SIZE))    

# print("hi")
# print(test)
# print(experimental_criteria)
# print(control_criteria)
# print("bye")



# print("new vs dna_0")
# print(*list(zip(
#     ACTIVE_SYNAPSES,
#     create_dna_string(new_jh_weights, ACTIVE_SYNAPSES),
#     DNA_0_padded_50)), 
#     sep='\n')

# print("old vs dna_0")
# print(*list(zip(
#     ACTIVE_SYNAPSES,
#     create_dna_string(old_jh_weights, ACTIVE_SYNAPSES),
#     DNA_0_padded_50)), 
#     sep='\n')

# print("new vs old")
# print(*list(zip(
#     ACTIVE_SYNAPSES,
#     create_dna_string(new_jh_weights, ACTIVE_SYNAPSES),
#     create_dna_string(old_jh_weights, ACTIVE_SYNAPSES),
#     )), 
#     sep='\n')

# print(load_dna(DNA_0_padded_50))
# # print(load_dna(create_dna_string(old_padded_dna_0, ACTIVE_SYNAPSES_OLD)))
# t=1
# alpha_array = create_alpha_array(250, L=30)
# # alpha = alpha_fit(alpha_array[0:], t, TMAX)
# # print(*alpha[0:10])

# # alpha = alpha_fit(alpha_array[1:], t, TMAX)
# # print(*alpha[0:10])

# alpha = alpha_fit(alpha_array, t, TMAX)
# print(*alpha[0:10])

# boundary = 1000
# sigma = .5
# mut_rate = 0.5
# gen_count = 200
# synapse = ["MSN1"]
# num_genes = 4
# gene_list = [random.uniform(-boundary, boundary) for _ in range(num_genes)]
# print([f"{gene:.2f}" for gene in gene_list])
# for i in range(gen_count):
#     new_gene_list = []
#     for g in gene_list:
#         gene = random.normalvariate(g, g*sigma) if random.random() < mut_rate else g

#         # Bounding DNA
#         if abs(gene) > boundary:
#             gene = boundary

#         # Inhibitory neurons have negative weights
#         if synapse[0] in INHIBITORY_NEURONS:
#             gene = -abs(gene)

#         new_gene_list.append(gene)

#     gene_list = new_gene_list
#     print(i, [f"{gene:.2f}" for gene in gene_list])
pkl_path = "/Users/stevenwendel/Documents/GitHub/bg/data/K_2025-02-24_01-28-22.pkl"

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)
    print(f"Data from {pkl_path}: {data}")
