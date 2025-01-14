from src.constants import *
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

test = define_criteria(TMAX/BIN_SIZE)
print("hi")
print(test)
print("bye")