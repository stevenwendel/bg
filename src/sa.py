from random import *
import numpy as np
from scipy.optimize import basinhopping

def get_neighbor(dna: list[float], sigma: float =.1) -> list[float]:
    neighbor=[normalvariate(synapse, sigma) for synapse in dna]
    return neighbor
"""
x = [75.0, 205.0, -90.0, -10.0, 65.0, 80.0, 320.0, -50.0, -50.0, -100.0, 60.0, 45.0, 30.0, -15.0, -90.0, -50.0, 85.0, 90.0, 320.0]
print(len(x))
nei = get_neighbor(dna=x,sigma=10) #Do i make sigma temp?
print(nei)
"""

