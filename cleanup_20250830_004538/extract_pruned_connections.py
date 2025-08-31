#!/usr/bin/env python3

import pickle
import numpy as np
from src.constants import ACTIVE_SYNAPSES, NEURON_NAMES

# Load pruned DNA
data = pickle.load(open('greedy_pruned_dna.pkl', 'rb'))
pruned_dna = data['pruned_dna']

print('Pruned DNA shape:', pruned_dna.shape)
print('Non-zero connections:')
print('Index | From -> To | Weight')
print('-' * 35)

connections = []
for i, weight in enumerate(pruned_dna):
    if weight != 0:
        if i < len(ACTIVE_SYNAPSES):
            from_neuron, to_neuron = ACTIVE_SYNAPSES[i]
            print(f'{i:5d} | {from_neuron:7s} -> {to_neuron:7s} | {weight:6d}')
            connections.append((from_neuron, to_neuron, weight))
        else:
            print(f'{i:5d} | INDEX OUT OF RANGE | {weight:6d}')

print(f'\nTotal non-zero connections: {np.count_nonzero(pruned_dna)}')
print(f'Connection list: {connections}')