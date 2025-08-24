import numpy as np

# Time Config
TMAX = 5000
BIN_SIZE = 100
NBINS = TMAX // BIN_SIZE  # Number of time bins for criterion evaluation

a = np.array((0.03, 0.01, 0.03, 0.03, 0.03, 0.01, 0.03, 0.03, 0.03, 0.03, 0.01, 0.03, 0.03, 0.03), dtype=np.float32)
b = np.array((-2.0, -20.0, -2.0, -2.0, -2.0, -20.0, -2.0, -2.0, -2.0, -2.0, -20.0, -2.0, -2.0, -2.0), dtype=np.float32)
vreset = np.array((-50.0, -55.0, -50.0, -50.0, -50.0, -55.0, -50.0, -50.0, -50.0, -50.0, -55.0, -50.0, -50.0, -50.0), dtype=np.float32)
d = np.array((100.0, 150.0, 100.0, 100.0, 100.0, 150.0, 100.0, 100.0, 100.0, 100.0, 150.0, 100.0, 100.0, 100.0), dtype=np.float32)
k = np.array((0.7, 1.0, 0.7, 0.7, 0.7, 1.0, 0.7, 0.7, 0.7, 0.7, 1.0, 0.7, 0.7, 0.7), dtype=np.float32)
vr = np.array((-60.0, -80.0, -60.0, -60.0, -60.0, -80.0, -60.0, -60.0, -60.0, -60.0, -80.0, -60.0, -60.0, -60.0), dtype=np.float32)
vt = np.array((-40.0, -25.0, -40.0, -40.0, -40.0, -25.0, -40.0, -40.0, -40.0, -40.0, -25.0, -40.0, -40.0, -40.0), dtype=np.float32)
vpeak = np.array((35.0, 40.0, 35.0, 35.0, 35.0, 40.0, 35.0, 35.0, 35.0, 35.0, 40.0, 35.0, 35.0, 35.0), dtype=np.float32)
E = np.array((0.0, 70.0, 120.0, 0.0, 0.0, 70.0, 120.0, 100.0, 0.0, 0.0, 70.0, 120.0, 0.0, 0.0), dtype=np.float32)
C = np.array((100.0, 50.0, 100.0, 100.0, 100.0, 50.0, 100.0, 100.0, 100.0, 100.0, 50.0, 100.0, 100.0, 100.0), dtype=np.float32)

# Setup Config
GO_DURATION = 100 # From the Wang paper directly
GO_STRENGTH = 850.
CUE_STRENGTH = 150.


NEURON_NAMES = ["Somat", "MSN1", "SNR1", "VMprep", "ALMprep", "MSN2", "SNR2", "PPN", "THALgo", "ALMinter", "MSN3", "SNR3", "ALMresp",  "VMresp"]
TONICALLY_ACTIVE_NEURONS = ["SNR1", "SNR2", "SNR3", "PPN", "THALgo"]
INHIBITORY_NEURONS = ["SNR1","SNR2", "SNR3", "MSN1", "MSN2", "MSN3", "ALMinter"]
ACTIVE_SYNAPSES = [
    # Connections from Somat to ALM (1x3)
    ["Somat", "ALMprep"], # ["Somat", "ALMinter"], ["Somat", "ALMresp"], 

    # Connections from ALM to Somat (3x1)
    # ["ALMprep", "Somat"], 
    # ["ALMinter", "Somat"], 
    # ["ALMresp", "Somat"],  

    # Connections from Somat to MSN 
    ["Somat", "MSN1"], ["Somat", "MSN2"], ["Somat", "MSN3"], 

    # Connections from MSN to SNR (3x3)
    ["MSN1", "SNR1"], ["MSN1", "SNR2"], ["MSN1", "SNR3"],
    ["MSN2", "SNR1"], ["MSN2", "SNR2"], ["MSN2", "SNR3"],
    ["MSN3", "SNR1"], ["MSN3", "SNR2"], ["MSN3", "SNR3"],

    # Connections from SNR to VM (3x2)
    ["SNR1", "VMprep"], ["SNR1", "VMresp"],
    ["SNR2", "VMprep"], ["SNR2", "VMresp"],
    ["SNR3", "VMprep"], ["SNR3", "VMresp"],

    # Connections from VM to ALM (2x3)
    ["VMprep", "ALMprep"], ["VMprep", "ALMinter"], ["VMprep", "ALMresp"],
    ["VMresp", "ALMprep"], ["VMresp", "ALMinter"], ["VMresp", "ALMresp"],

    # Connections from ALM to MSN (3x3)
    ["ALMprep", "MSN1"], ["ALMprep", "MSN2"], ["ALMprep", "MSN3"],
    ["ALMinter","MSN1"], ["ALMinter","MSN2"], ["ALMinter","MSN3"],
    ["ALMresp", "MSN1"], ["ALMresp", "MSN2"], ["ALMresp", "MSN3"],

    # Connections from ALM to VM (3x2)
    ["ALMprep", "VMprep"], ["ALMprep", "VMresp"], 
    ["ALMresp", "VMprep"], ["ALMresp", "VMresp"],
    
    # Recurrent MSN connections 
                      ["MSN1", "MSN2"], ["MSN1", "MSN3"],
    ["MSN2", "MSN1"],                   ["MSN2", "MSN3"],
    ["MSN3", "MSN1"], ["MSN3", "MSN2"],

    # Recurrent ALM connections
                            ["ALMprep", "ALMinter"], ["ALMprep", "ALMresp"],
    ["ALMinter","ALMprep"],                          ["ALMinter","ALMresp"],
    ["ALMresp", "ALMprep"], ["ALMresp", "ALMinter"],

    # Other key connections
    ["PPN", "THALgo"], 
    ["THALgo", "ALMinter"], ["THALgo", "ALMresp"],
]

EPOCHS = {
    'sample'   : [1000, 2000], 
    'delay'    : [2000, 3000],
    'response' : [3000, 4000] 
    }

CRITERIA_NAMES = [
        "Somat",
        "ALMprep",
        "ALMinter",
        "ALMresp",
        "SNR1",
        "SNR2",
        "SNR3",
        "VMprep",
        "VMresp",
        "PPN"
    ]

CRITERIA = {
        # These are all intervals which should be ON for experimental condition; should be OFF otherwise 
        "experimental" : {
            "Somat": {
                "interval":[EPOCHS['sample'][0], EPOCHS['sample'][1]],
                "io": "on"
            },
            "ALMprep": {
                "interval":[EPOCHS['sample'][0], EPOCHS['delay'][1] + 200], #Based on empirical results (eyeballing)
                "io": "on"
            },
            "ALMinter": {
                "interval":[EPOCHS['response'][0], EPOCHS['response'][0] + 300],
                "io": "on"
            },
            "ALMresp": {
                "interval":[EPOCHS['response'][0], TMAX], #EPOCHS['response'][1]
                "io": "on"
            },
            "SNR1": {
                "interval":[EPOCHS['sample'][0], EPOCHS['sample'][1]],
                "io": "off"
            },
            "SNR2": {
                "interval":[EPOCHS['sample'][0], EPOCHS['delay'][1]],
                "io": "off"
            },
            "SNR3": {   
                "interval":[EPOCHS['response'][0], TMAX],
                "io": "off"
            },
            "VMprep": {
                "interval":[EPOCHS['sample'][0], EPOCHS['delay'][1]],
                "io": "on"
            },
            "VMresp": {
                "interval":[EPOCHS['response'][0], TMAX],
                "io": "on"
            },
            "PPN": {
                "interval":[EPOCHS['response'][0], EPOCHS['response'][0]+250],
                "io": "on"
            } 
        },
      
        "control" : {
            "Somat": {
                "interval":[EPOCHS['sample'][0], EPOCHS['sample'][1]],
                "io": "off"
            },
            "ALMprep": {
                "interval":[EPOCHS['sample'][0], EPOCHS['delay'][1]],
                "io": "off"
            },
            "ALMinter": {
                "interval":[EPOCHS['response'][0], EPOCHS['response'][0] + 300],
                "io": "on"
            },
            "ALMresp": {
                "interval":[EPOCHS['response'][0], EPOCHS['response'][1]], #tMax -250?
                "io": "off"
            },
            "SNR1": {
                "interval":[0,TMAX],
                "io": "on"
            },
            "SNR2": {
                "interval":[0,TMAX],
                "io": "on"
            },
            "SNR3": {   
                "interval":[0,TMAX],
                "io": "on"
            },
            "VMprep": {
                "interval":[EPOCHS['sample'][0], EPOCHS['delay'][1]],
                "io": "off"
            },
            "VMresp": {
                "interval":[EPOCHS['response'][0], TMAX],
                "io": "off"
            },
            "PPN": {
                "interval":[EPOCHS['response'][0], EPOCHS['response'][0]+250],
                "io": "on"
            }
        }
    }

GA_CONFIG = { 
    "single": {
        "NUM_GENERATIONS" : 1,
        "POP_SIZE" : 1,
        "MUT_RATE" : 0.3,
        "MUT_SIGMA" : 0.5,
        "RANK_DEPTH" : 1,
        "ELITE_SIZE" : 1,
        "CROSSOVER_POINT" : None, # Randomly selecting all genes
        "DNA_BOUNDS" : [0,500]
    },
    "small": {
        "NUM_GENERATIONS" : 10,
        "POP_SIZE" : 100,
        "MUT_RATE" : 0.3,
        "MUT_SIGMA" : 0.5,
        "RANK_DEPTH" : 50,
        "ELITE_SIZE" : 5,
        "CROSSOVER_POINT" : None, # Randomly selecting all genes
        "DNA_BOUNDS" : [0,500]
    },
    "medium": {
        "NUM_GENERATIONS" : 20,
        "POP_SIZE" : 100,
        "MUT_RATE" : 0.3,
        "MUT_SIGMA" : 0.5,
        "RANK_DEPTH" : 25,
        "ELITE_SIZE" : 5,
        "CROSSOVER_POINT" : None, # Randomly selecting all genes
        "DNA_BOUNDS" : [0,500],
        "TIME_TAKEN": [3, 4, .3]
    },
    "large":   {
        "NUM_GENERATIONS" : 100,
        "POP_SIZE" : 500,
        "MUT_RATE" : 0.4,
        "MUT_SIGMA" : 0.5,
        "RANK_DEPTH" : 250,
        "ELITE_SIZE" : 3,
        "CROSSOVER_POINT" : None, # Randomly selecting all genes
        "DNA_BOUNDS" : [0,500],
        "TIME_TAKEN" : [60, 73, 5]
    },
    "E":   {
        "NUM_GENERATIONS" : 300,
        "POP_SIZE" : 300,
        "MUT_RATE" : .45,
        "MUT_SIGMA" : .5,
        "RANK_DEPTH" : 150,
        "ELITE_SIZE" : 8,
        "CROSSOVER_POINT" : None,
        "DNA_BOUNDS" : [0,500], 
        "TIME_TAKEN" : [303, 200,]
    },
    "F":   {
        "NUM_GENERATIONS" : 120,
        "POP_SIZE" : 150,
        "MUT_RATE" : 0.5,
        "MUT_SIGMA" : .5,
        "RANK_DEPTH" : 70,
        "ELITE_SIZE" : 3,
        "CROSSOVER_POINT" : None,
        "DNA_BOUNDS" : [0,500], 
        "TIME_TAKEN" : 620# 10.5 hr
    },
    "G":   {
        "NUM_GENERATIONS" : 250,
        "POP_SIZE" : 1000,
        "MUT_RATE" : 0.4,
        "MUT_SIGMA" : .5,
        "RANK_DEPTH" : 500,
        "ELITE_SIZE" : 3,
        "CROSSOVER_POINT" : None,
        "DNA_BOUNDS" : [0,500], 
        "TIME_TAKEN" : []
    },
    "H":   {
        "NUM_GENERATIONS" : 300,
        "POP_SIZE" : 1000,
        "MUT_RATE" : 0.25,
        "MUT_SIGMA" : .30,
        "RANK_DEPTH" : 500,
        "ELITE_SIZE" : 3,
        "CROSSOVER_POINT" : None,
        "DNA_BOUNDS" : [0,500], 
        "TIME_TAKEN" : [.34, .66]
    },
    "J":   {
        "NUM_GENERATIONS" : 400,
        "POP_SIZE" : 4000,
        "MUT_RATE" : 0.25,
        "MUT_SIGMA" : .30,
        "RANK_DEPTH" : 500,
        "ELITE_SIZE" : 3,
        "CROSSOVER_POINT" : None,
        "DNA_BOUNDS" : [0,500], 
        "TIME_TAKEN" : 360
    },
    "K":   {
        "NUM_GENERATIONS" : 300,
        "POP_SIZE" : 10000,
        "MUT_RATE" : 0.40,
        "MUT_SIGMA" : .80,
        "RANK_DEPTH" : 5000,
        "ELITE_SIZE" : 10,
        "CROSSOVER_POINT" : None,
        "DNA_BOUNDS" : [0,500], 
        "TIME_TAKEN" : 360
    },
    "L":   {
        "NUM_GENERATIONS" : 200,
        "POP_SIZE" : 500,
        "MUT_RATE" : 0.25,
        "MUT_SIGMA" : .35,
        "RANK_DEPTH" : 250,
        "ELITE_SIZE" : 3,
        "CROSSOVER_POINT" : None,
        "DNA_BOUNDS" : [0,500], 
        "TIME_TAKEN" : []
    },
}


new_jh_weights = [
    ("Somat", "ALMprep", 40),
    ("Somat", "MSN1", 220),
    ("MSN1", "SNR1", -90),
    ("SNR1", "VMprep", -10),
    ("VMprep", "ALMprep", 70),
    ("ALMprep", "VMprep", 80),
    ("ALMprep", "MSN2", 320),
    ("MSN2", "SNR2", -50),
    ("SNR2", "VMresp", -100),
    ("PPN", "THALgo", 60),
    ("THALgo", "ALMinter", 55),
    ("ALMinter", "ALMprep", -50),
    ("THALgo", "ALMresp", 30),
    ("ALMresp", "MSN3", 320),
    ("MSN3", "SNR3", -90),
    ("SNR3", "VMresp", -50),
    ("VMresp", "ALMresp", 85),
    ("ALMresp", "VMresp", 90)
]

# old_jh_weights = [
#     ("Somat", "ALMprep", 75.),
#     ("Somat", "MSN1", 205.),
#     ("MSN1", "SNR1", -90.),
#     ("SNR1", "VMprep", -10.),
#     ("VMprep", "ALMprep", 65.),
#     ("ALMprep", "VMprep", 80.),
#     ("ALMprep", "MSN2", 320.),
#     ("MSN2", "SNR2", -50.),
#     ("SNR2", "VMresp", -100.),
#     ("PPN", "THALgo", 60.),
#     ("THALgo", "ALMinter", 45.),
#     ("ALMinter", "ALMprep", -15.),
#     ("THALgo", "ALMresp", 30.),
#     ("ALMresp", "MSN3", 320.),
#     ("ALMresp", "VMresp", 90.),
#     ("MSN3", "SNR3", -90.),
#     ("SNR3", "VMresp", -50.),
#     ("VMresp", "ALMresp", 85.),
# ]

# old_padded_dna_0 = [
#     ("Somat", "ALMprep", 75.0),
#     ("Somat", "MSN1", 205.0),
#     ("MSN1", "SNR1", -90.0),
#     ("SNR1", "VMprep", -10.0),
#     ("VMprep", "ALMprep", 65.0),
#     ("ALMprep", "VMprep", 80.0),
#     ("ALMprep", "MSN2", 320.0),
#     ("MSN2", "SNR2", -50.0),
#     ("SNR2", "VMresp", -100.0),
#     ("PPN", "THALgo", 60.0),
#     ("THALgo", "ALMinter", 45.0),
#     ("ALMinter", "ALMprep", -15.0),
#     ("THALgo", "ALMresp", 30.0),
#     ("ALMresp", "MSN3", 320.0),
#     ("ALMresp", "VMresp", 90.0),
#     ("MSN3", "SNR3", -90.0),
#     ("SNR3", "VMresp", -50.0),
#     ("VMresp", "ALMresp", 85.0),
# ]
