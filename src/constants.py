# Time Config
TMAX = 5000
BIN_SIZE = 100

# Setup Config
GO_DURATION = 100 # From the Wang paper directly
GO_STRENGTH = 850.
CUE_STRENGTH = 150.

# Neuron model parameters (Izhikevich)
import numpy as np
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


NEURON_NAMES = ["Somat", "MSN1", "SNR1", "VMprep", "ALMprep", "MSN2", "SNR2", "PPN", "THALgo", "ALMinter", "MSN3", "SNR3", "ALMresp",  "VMresp"]
TONICALLY_ACTIVE_NEURONS = ["SNR1", "SNR2", "SNR3", "PPN", "THALgo"]
INHIBITORY_NEURONS = ["SNR1","SNR2", "SNR3", "MSN1", "MSN2", "MSN3", "ALMinter"]
ACTIVE_SYNAPSES = [
    # Connections from Somat to ALM (1x3)
    ["Somat", "ALMprep"], ["Somat", "ALMinter"], ["Somat", "ALMresp"], 

    # Connections from ALM to Somat (3x1)
    ["ALMprep", "Somat"], 
    ["ALMinter", "Somat"], 
    ["ALMresp", "Somat"],  

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
    'sample'   : [1000, 2000], #should there be a [0,1000] epoch?
    'delay'    : [2000, 3000],
    'response' : [3000, 4000] #should this be up to 5000?
    }

CRITERIA_NAMES = [
        "Somat",
        "ALMprep",
        "ALMinter",
        "ALMresp",
        "SNR1",
        "SNR2",
        "SNR3", #added this 1/16/2025
        "VMprep",
        "VMresp",
        "PPN"
    ]

# Critical connections that should never be zero - prevents GA from bypassing criteria neurons
CRITICAL_CONNECTIONS = {
    # Somat input pathway - needed for stimulus detection
    ("Somat", "MSN1"): {"min_abs": 10, "reason": "Somat->MSN1 needed for stimulus response"},
    ("Somat", "MSN2"): {"min_abs": 10, "reason": "Somat->MSN2 needed for stimulus response"}, 
    ("Somat", "MSN3"): {"min_abs": 10, "reason": "Somat->MSN3 needed for stimulus response"},
    
    # ALM pathway - critical for preparation and response
    ("VMprep", "ALMprep"): {"min_abs": 15, "reason": "VMprep->ALMprep drives preparation activity"},
    ("ALMprep", "ALMinter"): {"min_abs": 10, "reason": "ALMprep->ALMinter enables response inhibition"},
    ("THALgo", "ALMresp"): {"min_abs": 15, "reason": "THALgo->ALMresp drives response activity"},
    ("ALMresp", "VMresp"): {"min_abs": 20, "reason": "ALMresp->VMresp completes response loop"},
    
    # SNR suppression pathways - essential for basal ganglia function
    ("MSN1", "SNR1"): {"min_abs": 20, "reason": "MSN1->SNR1 inhibition for stimulus response"},
    ("MSN2", "SNR2"): {"min_abs": 20, "reason": "MSN2->SNR2 inhibition for preparation"},
    ("MSN3", "SNR3"): {"min_abs": 30, "reason": "MSN3->SNR3 inhibition critical for response period"},
    
    # SNR output pathways - needed for proper disinhibition
    ("SNR1", "VMprep"): {"min_abs": 5, "reason": "SNR1->VMprep disinhibition"},
    ("SNR1", "VMresp"): {"min_abs": 5, "reason": "SNR1->VMresp disinhibition"},
    ("SNR2", "VMprep"): {"min_abs": 5, "reason": "SNR2->VMprep disinhibition"}, 
    ("SNR2", "VMresp"): {"min_abs": 5, "reason": "SNR2->VMresp disinhibition"},
    ("SNR3", "VMprep"): {"min_abs": 15, "reason": "SNR3->VMprep disinhibition for preparation"},
    ("SNR3", "VMresp"): {"min_abs": 25, "reason": "SNR3->VMresp disinhibition for response"},
    
    # VM output pathways
    ("VMresp", "ALMresp"): {"min_abs": 20, "reason": "VMresp->ALMresp drives response activity"},
    
    # PPN pathway - needed for movement initiation
    ("PPN", "THALgo"): {"min_abs": 10, "reason": "PPN->THALgo drives movement initiation"},
}

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

GA_CONFIG = { # I should store these configurations in the pkl file itself as a metadata field in the dictionary
    "small": {
        "NUM_GENERATIONS" : 10,
        "POP_SIZE" : 50,
        "MUT_RATE" : 0.3,
        "MUT_SIGMA" : 0.5,
        "RANK_DEPTH" : 25,
        "ELITE_SIZE" : 1,
        "CROSSOVER_POINT" : None,
        "DNA_BOUNDS" : [0,500]
    },
    "E":   {
        "NUM_GENERATIONS" : 300,
        "POP_SIZE" : 200,
        "MUT_RATE" : .5,
        "MUT_SIGMA" : .5,
        "RANK_DEPTH" : None,
        "ELITE_SIZE" : 5,
        "CROSSOVER_POINT" : None,
        "DNA_BOUNDS" : [0,500], 
        "TIME_TAKEN" : None
    },
}
