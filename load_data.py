import pickle
import pandas as pd

file_path1 = './data/run_data.pkl'
file_path2 = '/Users/stevenwendel/Documents/GitHub/bg/data/2024-12-01_23-59-24.pkl'

def pkl_to_dataframe(file_path: str) -> pd.DataFrame:
    data = []
    with open(file_path, 'rb') as file:
        try:
            while True:
                item = pickle.load(file)
                data.append(item)
        except EOFError:
            pass
    
    df = pd.DataFrame(data)
    return df

# Load and analyze file_path2
df = pkl_to_dataframe(file_path2)
print(f"DataFrame shape: {df.shape}")
print("\nAll entries:")
print(df)