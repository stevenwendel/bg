import pandas as pd
import numpy as np
import pickle
from src.constants import *
from src.neuron import *
from src.utils import *
from src.network import *
from src.validation import *
from src.viz import *
from src.genetic_algorithm import *
from multiprocessing import Pool
from functools import partial
import ast

pkl_file = '/Users/stevenwendel/Documents/GitHub/bg/data/K_high_gen_2025-03-15_01-15-23.pkl'

with open(pkl_file, 'rb') as f: 
    ga_results = pickle.load(f)

intial_df = flatten_pkl(ga_results)

def clean_df(initial_df, threshold=700):
    cleaned_df = initial_df[initial_df['dna_score'] >= threshold]
    cleaned_df = cleaned_df.drop_duplicates(subset='dna')
    cleaned_df = cleaned_df.sort_values(by='dna_score', ascending=False)    
    return cleaned_df

cleaned_df = clean_df(intial_df, threshold=650)
cleaned_df = get_unique_representatives(cleaned_df, max_synapses=20)
print(cleaned_df.head())
