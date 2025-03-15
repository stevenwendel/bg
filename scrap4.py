# scrap4.py
import pickle

pkl_file = '/Users/stevenwendel/Documents/GitHub/bg/data/J_high_gen_2025-03-13_04-00-56.pkl'

# Load the combined data from the .pkl file
with open(pkl_file, 'rb') as file:
    combined_data = pickle.load(file)

print(combined_data.keys())