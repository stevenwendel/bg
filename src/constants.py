
# Time Config
TMAX = 5000
BIN_SIZE = 100

# Setup Config
GO_DURATION = 100 # From the Wang paper directly
GO_STRENGTH = 850.
CUE_STRENGTH = 145.

DNA_0 = [75., 205., -90., -10., 65., 80., 320., -50., -100., 60., 45., 30., -15., -90., -50., 85., 90., 320.,
         0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]

NEURON_NAMES = ["Somat", "MSN1", "SNR1", "VMprep", "ALMprep", "MSN2", "SNR2", "PPN", "THALgo", "ALMinter", "MSN3", "SNR3", "ALMresp",  "VMresp"]

ACTIVE_SYNAPSES = [
    ["Somat", "ALMprep"], ["Somat", "MSN1"], ["MSN1", "SNR1"], ["SNR1", "VMprep"],
    ["VMprep", "ALMprep"], ["ALMprep", "VMprep"], ["ALMprep", "MSN2"], ["MSN2", "SNR2"], #["SNR2", "VMprep"] REMOVED 1/17/25
    ["SNR2", "VMresp"], ["PPN", "THALgo"], ["THALgo", "ALMinter"], ["THALgo", "ALMresp"], 
    ["ALMinter", "ALMprep"], ["MSN3", "SNR3"], ["SNR3", "VMresp"], ["VMresp", "ALMresp"], 
    ["ALMresp", "VMresp"], ["ALMresp", "MSN3"],
    
    # These new connections were added 1/17/25
    # "The only impossible connections I think are MSN->cortical, SNR->cortical"

    # Somat to remaining ALM
    ["Somat", "ALMinter"], ["Somat", "ALMresp"],
    
    # All MSN to all SNR (excluding existing)
    ["MSN1", "SNR3"], ["MSN1", "SNR2"],
    ["MSN2", "SNR1"], ["MSN2", "SNR3"],
    ["MSN3", "SNR1"], ["MSN3", "SNR2"],
    
    # All SNR to all VM (excluding existing)
    ["SNR1", "VMresp"],
    ["SNR2", "VMprep"],
    ["SNR3", "VMprep"],
    
    # All ALM to all MSN (excluding existing)
    ["ALMinter", "MSN1"], ["ALMinter", "MSN2"], ["ALMinter", "MSN3"],
    ["ALMresp", "MSN1"], ["ALMresp", "MSN2"],
    
    # All VM to all ALM (excluding existing)
    ["VMprep", "ALMinter"], ["VMprep", "ALMresp"],
    ["VMresp", "ALMprep"], ["VMresp", "ALMinter"],

    # Recurrent MSN connections
    ["MSN1", "MSN2"], ["MSN1", "MSN3"],
    ["MSN2", "MSN3"], ["MSN2", "MSN1"],
    ["MSN3", "MSN1"], ["MSN3", "MSN2"],

    # Recurrent ALM connections
    ["ALMprep", "ALMinter"], ["ALMprep", "ALMresp"],
    ["ALMinter", "ALMresp"], ["ALMinter", "ALMprep"],
    ["ALMresp", "ALMinter"], ["ALMresp", "ALMprep"]
    
    ]

EPOCHS = {
    'sample'   : [1000, 2000], #should there be a [0,1000] epoch?
    'delay'    : [2000, 3000],
    'response' : [3000, 4000] #should this be up to 5000?
    }


go_signal_duration = 100 # / currently, a bunch of parameters are hard-coded in for create_experiment


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

CRITERIA = {
        # These are all intervals which should be ON for experimental condition; should be OFF otherwise 
        "experimental" : {
            "Somat": {
                "interval":[EPOCHS['sample'][0], EPOCHS['sample'][1]],
                "io": "on"
            },
            "ALMprep": {
                "interval":[EPOCHS['sample'][0], EPOCHS['delay'][1]],
                "io": "on"
            },
            "ALMinter": {
                "interval":[EPOCHS['response'][0], EPOCHS['response'][0] + 300],
                "io": "on"
            },
            "ALMresp": {
                "interval":[EPOCHS['response'][0], TMAX-250], #EPOCHS['response'][1]
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
                "interval":[EPOCHS['response'][0], TMAX-250],
                "io": "off"
            },
            "VMprep": {
                "interval":[EPOCHS['sample'][0], EPOCHS['delay'][1]],
                "io": "on"
            },
            "VMresp": {
                "interval":[EPOCHS['response'][0], TMAX-250],
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
                "interval":[EPOCHS['response'][0], TMAX-250],
                "io": "off"
            },
            "PPN": {
                "interval":[EPOCHS['response'][0], EPOCHS['response'][0]+250],
                "io": "on"
            }
        }
    }

GA_CONFIG = {
    "large":   {
        "NUM_GENERATIONS" : 10,
        "POP_SIZE" : 500,
        "MUT_RATE" : 0.15,
        "MUT_SIGMA" : 0.3,
        "RANK_DEPTH" : 250,
        "ELITE_SIZE" : 10,
        "CROSSOVER_POINT" : None, # Randomly selecting all genes
        "DNA_BOUNDS" : [0,400]
    },
    "small": {
        "NUM_GENERATIONS" : 3,
        "POP_SIZE" : 10,
        "MUT_RATE" : 0.15,
        "MUT_SIGMA" : 0.3,
        "RANK_DEPTH" : 5,
        "ELITE_SIZE" : 1,
        "CROSSOVER_POINT" : None, # Randomly selecting all genes
        "DNA_BOUNDS" : [0,400]
    },
    "xlarge":   {
        "NUM_GENERATIONS" : 30,
        "POP_SIZE" : 1000,
        "MUT_RATE" : 0.25,
        "MUT_SIGMA" : 0.4,
        "RANK_DEPTH" : 500,
        "ELITE_SIZE" : 20,
        "CROSSOVER_POINT" : None, # Randomly selecting all genes
        "DNA_BOUNDS" : [0,400]
    },
    "xlarge_highMutation":   {
        "NUM_GENERATIONS" : 75,
        "POP_SIZE" : 2000,
        "MUT_RATE" : 0.35,
        "MUT_SIGMA" : 0.3,
        "RANK_DEPTH" : 400,
        "ELITE_SIZE" : 20,
        "CROSSOVER_POINT" : None, # Randomly selecting all genes
        "DNA_BOUNDS" : [0,400]
    },
        "xxlarge_highMutation":   {
        "NUM_GENERATIONS" : 300,
        "POP_SIZE" : 4000,
        "MUT_RATE" : 0.35,
        "MUT_SIGMA" : 0.5,
        "RANK_DEPTH" : 1000,
        "ELITE_SIZE" : 20,
        "CROSSOVER_POINT" : None, # Randomly selecting all genes
        "DNA_BOUNDS" : [0,400]
    }
}
