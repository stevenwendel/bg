import pandas as pd
import matplotlib.pyplot as plt

# Data for MUT_RATE Analysis
mut_rate_data = {
    'parameters': [
        0.126840, 0.142720, 0.146794, 0.218285, 0.225320, 0.235774, 0.264828, 0.265797, 0.267420, 0.277313,
        0.327827, 0.336172, 0.359398, 0.382517, 0.391999, 0.397024, 0.404528, 0.405968, 0.407272, 0.417759,
        0.419805, 0.433515, 0.434110, 0.437249, 0.437369, 0.439157, 0.454667, 0.457255, 0.458477, 0.463504,
        0.468712, 0.489435, 0.499144, 0.525223, 0.541755, 0.554669, 0.561940, 0.576401, 0.588139, 0.598126,
        0.604750, 0.634791, 0.651505, 0.683411, 0.710903, 0.736900, 0.742107, 0.747732, 0.753830, 0.800000
    ],
    'mean': [
        646, 648, 639, 643, 649, 652, 655, 644, 648, 663,
        650, 655, 647, 657, 657, 658, 658, 660, 663, 665,
        656, 656, 663, 661, 659, 663, 664, 653, 661, 654,
        655, 654, 656, 661, 649, 647, 651, 650, 655, 665,
        654, 651, 651, 662, 649, 647, 657, 657, 657, 644
    ],
    'max': [
        646, 648, 639, 643, 649, 652, 655, 644, 648, 663,
        650, 655, 647, 657, 657, 658, 658, 660, 663, 665,
        656, 656, 663, 661, 659, 663, 664, 653, 661, 654,
        655, 654, 656, 661, 649, 647, 651, 650, 655, 665,
        654, 651, 651, 662, 649, 647, 657, 657, 657, 644
    ]
}

# Data for MUT_SIGMA Analysis
mut_sigma_data = {
    'parameters': [
        0.100000, 0.125791, 0.138486, 0.154384, 0.160951, 0.161579, 0.175451, 0.176761, 0.178146, 0.201629,
        0.324681, 0.363312, 0.371933, 0.386119, 0.397649, 0.415385, 0.422231, 0.429780, 0.504644, 0.548440,
        0.552729, 0.575117, 0.576469, 0.578611, 0.578790, 0.578883, 0.582910, 0.587379, 0.621199, 0.634682,
        0.637319, 0.662701, 0.666554, 0.711577, 0.715971, 0.723516, 0.793586, 0.800000
    ],
    'mean': [
        644.0, 648.0, 647.0, 650.0, 651.0, 643.0, 649.0, 655.0, 644.0, 655.0,
        650.0, 663.0, 656.0, 657.0, 646.0, 663.0, 649.0, 652.0, 657.0, 651.0,
        657.0, 653.0, 654.0, 664.0, 662.0, 665.0, 647.0, 661.0, 649.0, 651.0,
        656.0, 648.0, 639.0, 647.0, 655.0, 654.0, 657.0, 659.230769
    ],
    'max': [
        644, 648, 647, 650, 651, 643, 649, 655, 644, 655,
        650, 663, 656, 657, 646, 663, 649, 652, 657, 651,
        657, 653, 654, 664, 662, 665, 647, 661, 649, 651,
        656, 648, 639, 647, 655, 654, 657, 665
    ]
}

# Data for ELITE_SIZE Analysis
elite_size_data = {
    'parameters': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 18, 19, 20],
    'mean': [
        657.0, 657.0, 652.5, 649.0, 658.5, 647.333333, 650.0, 664.0, 652.666667, 655.5,
        656.0, 655.0, 653.0, 654.0, 648.0, 660.0, 652.5, 644.0, 658.75
    ],
    'max': [
        662, 657, 658, 650, 661, 657, 657, 665, 655, 657, 664, 655, 663, 656, 649, 660, 661, 644, 665
    ]
}

# Data for POP_SIZE Analysis
pop_size_data = {
    'parameters': [50, 51, 52, 80, 113, 135, 159, 161, 162, 186, 237, 281, 297, 303, 330, 376, 393, 396, 411, 431, 466, 500],
    'mean': [
        656.5, 654.0, 663.5, 651.0, 651.0, 655.0, 653.0, 664.0, 657.0, 657.0, 647.0, 649.0, 647.0,
        647.0, 655.0, 648.0, 650.0, 652.0, 651.0, 643.0, 650.0, 647.333333
    ],
    'max': [
        665, 654, 665, 651, 651, 655, 653, 664, 657, 657, 647, 649, 647, 647, 655, 648, 650, 652, 651, 643, 650, 654
    ]
}

# Convert to DataFrames
df_mut_rate = pd.DataFrame(mut_rate_data)
df_mut_sigma = pd.DataFrame(mut_sigma_data)
df_elite_size = pd.DataFrame(elite_size_data)
df_pop_size = pd.DataFrame(pop_size_data)

# Plot MUT_RATE Analysis
plt.figure()
plt.plot(df_mut_rate['parameters'], df_mut_rate['mean'], label='Mean')
plt.plot(df_mut_rate['parameters'], df_mut_rate['max'], label='Max')
plt.title('MUT_RATE Analysis')
plt.xlabel('MUT_RATE')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()

# Plot MUT_SIGMA Analysis
plt.figure()
plt.plot(df_mut_sigma['parameters'], df_mut_sigma['mean'], label='Mean')
plt.plot(df_mut_sigma['parameters'], df_mut_sigma['max'], label='Max')
plt.title('MUT_SIGMA Analysis')
plt.xlabel('MUT_SIGMA')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()

# Plot ELITE_SIZE Analysis
plt.figure()
plt.plot(df_elite_size['parameters'], df_elite_size['mean'], label='Mean')
plt.plot(df_elite_size['parameters'], df_elite_size['max'], label='Max')
plt.title('ELITE_SIZE Analysis')
plt.xlabel('ELITE_SIZE')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()

# Plot POP_SIZE Analysis
plt.figure()
plt.plot(df_pop_size['parameters'], df_pop_size['mean'], label='Mean')
plt.plot(df_pop_size['parameters'], df_pop_size['max'], label='Max')
plt.title('POP_SIZE Analysis')
plt.xlabel('POP_SIZE')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()
