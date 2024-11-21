import pickle
import pandas as pd
import numpy as np
file_path = './data/run_data.pkl'


with open(file_path,'rb') as f:
        try:
            while True:
                item = pickle.load(f)
                print(f'{item[0]=}')
                print(f'{item[1]=}')
                print(f'{item[2]=}')
                print(f'{item[3]=}')
                print(f'{item[4]=}')
                
        except EOFError:
            # End of file reached
            pass

"""            with open('./data/run_data.pkl','ab') as f:
                pickle.dump((generation, curr_dna, dna_score, neuron_data, binned_differences),f)
"""


# load_neurons('data/run_data.pkl')