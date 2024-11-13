all_nodes = ["Somat", "MSN1", "SNR1", "VMprep", "ALMprep", "MSN2", "SNR2", "PPN", "THALgo", "ALMinter", "MSN3", "SNR3", "ALMresp",  "VMresp"]

active_synapses = [
    ["Somat", "ALMprep"], ["Somat", "MSN1"], ["MSN1", "SNR1"], ["SNR1", "VMprep"],
    ["VMprep", "ALMprep"], ["ALMprep", "VMprep"], ["ALMprep", "MSN2"], ["MSN2", "SNR2"],
    ["SNR2", "VMprep"], ["SNR2", "VMresp"], ["PPN", "THALgo"], ["THALgo", "ALMinter"],
    ["THALgo", "ALMresp"], ["ALMinter", "ALMprep"], ["MSN3", "SNR3"], ["SNR3", "VMresp"],
    ["VMresp", "ALMresp"], ["ALMresp", "VMresp"], ["ALMresp", "MSN3"]
]

dna_0 = [75., 205., -90., -10., 65., 80., 320., -50., -50., -100., 60., 45., 30., -15., -90., -50., 85., 90., 320.]

# my_free_weights_names = ["_".join(synapse) for synapse in active_synapses] # Do I need this?

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