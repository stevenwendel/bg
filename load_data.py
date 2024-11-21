import pickle
import pandas as pd
from ydata_profiling import ProfileReport

file_path1 = './data/run_data.pkl'
file_path2 = './data/dfs.pkl'

# def pkl_to_dataframe(file_path: str) -> pd.DataFrame:
#     data = []

#     with open(file_path, 'rb') as file:
#         try:
#             while True:
#                 # Load each dictionary from the pickle file
#                 item = pickle.load(file)
#                 data.append(item)
#         except EOFError:
#             # End of file reached
#             pass

#     # Convert the list of dictionaries to a DataFrame
#     df = pd.DataFrame(data, columns=["generation", "curr_dna","dna_score", "neuron_data", "binned_differences"])
#     return df

# # Example usage

# df = pkl_to_dataframe(file_path)
# print(df)

# # Generate the profile report
# profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)

# # Save the report to an HTML file
# profile.to_file("your_report.html")

df = pd.read_pickle(file_path2)
print(df)
