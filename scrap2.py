from src.constants import *
from src.utils import *
from src.genetic_algorithm import *
import pandas as pd
from src.validation import *

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

print(load_dna(DNA_0_padded_50))
print(load_dna(create_dna_string(old_padded_dna_0, ACTIVE_SYNAPSES_OLD)))