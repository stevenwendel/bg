#!/usr/bin/env python3
import pickle
import numpy as np

# Load the best DNA
data = pickle.load(open('best_dna.pkl', 'rb'))
print('Data type:', type(data))
print('Data keys:', data.keys() if isinstance(data, dict) else 'Not a dict')

if isinstance(data, dict) and 'dna' in data:
    dna = data['dna']
    print('DNA shape:', dna.shape if hasattr(dna, 'shape') else len(dna))
    print('DNA vector:', dna)
    print('Non-zero weights:', sum(1 for x in dna if x != 0))
    print('Zero weights:', sum(1 for x in dna if x == 0))
    print('Total score:', data.get('total_score', 'Unknown'))
    print('Weight distribution:')
    print('  Min:', min(dna))
    print('  Max:', max(dna))
    print('  Mean:', np.mean(dna))
    print('  Std:', np.std(dna))
else:
    print('DNA vector:', data)
    print('Non-zero weights:', sum(1 for x in data if x != 0))
    print('Zero weights:', sum(1 for x in data if x == 0))