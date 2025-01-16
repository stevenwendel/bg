from src.constants import *
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

test = define_criteria(TMAX//BIN_SIZE)
experimental_criteria = pd.DataFrame(test['experimental'], index=CRITERIA_NAMES, columns=range(TMAX//BIN_SIZE))
control_criteria = pd.DataFrame(test['control'], index=CRITERIA_NAMES, columns=range(TMAX//BIN_SIZE))    

print("hi")
print(test)
print(experimental_criteria)
print(control_criteria)
print("bye")