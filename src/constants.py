# I should make these all caps; will need to learn how to change all varibales across all files though...

neuron_names = ["Somat", "MSN1", "SNR1", "VMprep", "ALMprep", "MSN2", "SNR2", "PPN", "THALgo", "ALMinter", "MSN3", "SNR3", "ALMresp",  "VMresp"]

active_synapses = [
    ["Somat", "ALMprep"], ["Somat", "MSN1"], ["MSN1", "SNR1"], ["SNR1", "VMprep"],
    ["VMprep", "ALMprep"], ["ALMprep", "VMprep"], ["ALMprep", "MSN2"], ["MSN2", "SNR2"],
    ["SNR2", "VMprep"], ["SNR2", "VMresp"], ["PPN", "THALgo"], ["THALgo", "ALMinter"],
    ["THALgo", "ALMresp"], ["ALMinter", "ALMprep"], ["MSN3", "SNR3"], ["SNR3", "VMresp"],
    ["VMresp", "ALMresp"], ["ALMresp", "VMresp"], ["ALMresp", "MSN3"]
]

dna_0 = [75., 205., -90., -10., 65., 80., 320., -50., -50., -100., 60., 45., 30., -15., -90., -50., 85., 90., 320.]

# my_free_weights_names = ["_".join(synapse) for synapse in active_synapses] # Do I need this?

epochs = {
    'sample'   : [1000, 2000], #should there be a [0,1000] epoch?
    'delay'    : [2000, 3000],
    'response' : [3000, 4000] #should this be up to 5000?
    }


bin_size = 250

tMax = 5000
dt = 1

go_epoch_length = 400 # I should take all the constants from create_experiment and put them here
go_signal_duration = 100 # / currently, a bunch of parameters are hard-coded in for create_experiment




criteria_names = [
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


criteria_times = {
"experimental_criterion" : {
    "Somat": [epochs['sample'][0], epochs['sample'][1]],
    "ALMprep": [epochs['sample'][0], epochs['delay'][1]],
    "ALMinter": [epochs['response'][0], epochs['response'][0] + 300],
    "ALMresp": [epochs['delay'][0], epochs['delay'][1]], #tMax -250?
    "SNR1": [epochs['sample'][0], epochs['delay'][1]],
    "SNR2": [epochs['response'][0], tMax-250],
    "VMprep": [epochs['sample'][0], epochs['sample'][1]],
    "VMresp": [epochs['response'][0], tMax-250],
    "PPN": [epochs['response'][0], epochs['response'][0]+250]
}, "control_criterion" : {
    "ALMinter": [epochs['response'][0], epochs['response'][0] + 300],
    "SNR1": [0,tMax],
    "SNR2": [0,tMax], 
    "PPN": [epochs['response'][0], epochs['response'][0]+250],

    

}

    

}
