# I should make these all caps; will need to learn how to change all varibales across all files though...

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
    }
}

# Time Config
TMAX = 5000
BIN_SIZE = 250

# Setup Config
GO_DURATION = 400 # I should take all the constants from create_experiment and put them here
GO_STRENGTH = 850.
CUE_STRENGTH = 145.

DNA_0 = [75., 205., -90., -10., 65., 80., 320., -50., -50., -100., 60., 45., 30., -15., -90., -50., 85., 90., 320.]

NEURON_NAMES = ["Somat", "MSN1", "SNR1", "VMprep", "ALMprep", "MSN2", "SNR2", "PPN", "THALgo", "ALMinter", "MSN3", "SNR3", "ALMresp",  "VMresp"]

ACTIVE_SYNAPSES = [
    ["Somat", "ALMprep"], ["Somat", "MSN1"], ["MSN1", "SNR1"], ["SNR1", "VMprep"],
    ["VMprep", "ALMprep"], ["ALMprep", "VMprep"], ["ALMprep", "MSN2"], ["MSN2", "SNR2"],
    ["SNR2", "VMprep"], ["SNR2", "VMresp"], ["PPN", "THALgo"], ["THALgo", "ALMinter"],
    ["THALgo", "ALMresp"], ["ALMinter", "ALMprep"], ["MSN3", "SNR3"], ["SNR3", "VMresp"],
    ["VMresp", "ALMresp"], ["ALMresp", "VMresp"], ["ALMresp", "MSN3"]
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
                "interval":[EPOCHS['response'][0], EPOCHS['response'][1]], #tMax -250?
                "io": "on"
            },
            "SNR1": {
                "interval":[EPOCHS['sample'][0], EPOCHS['delay'][1]],
                "io": "off"
            },
            "SNR2": {   
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
        #I don't think I need this
        "control" : {
            "Somat": {
                "interval":[EPOCHS['sample'][0], EPOCHS['sample'][1]],
                "io": "off"
            },
                    "ALMprep": {
            "interval":[EPOCHS['sample'][0], EPOCHS['delay'][1]],
            "io": "on"
            },
            "ALMinter": {
                "interval":[EPOCHS['response'][0], EPOCHS['response'][0] + 300],
                "io": "off"
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