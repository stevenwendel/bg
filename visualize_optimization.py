import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the analysis file
def parse_analysis_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Extract top combinations
    top_combinations = []
    for line in lines[4:14]:  # Lines containing top 10 combinations
        if line.strip():
            try:
                # Find the last number in the line (the score)
                parts = line.split()
                score = int(parts[-1])
                
                # Find the dictionary part
                dict_start = line.find('{')
                dict_end = line.rfind('}') + 1
                if dict_start != -1 and dict_end != -1:
                    params_str = line[dict_start:dict_end]
                    params = eval(params_str)
                    top_combinations.append({**params, 'best_score': score})
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing line: {line}")
                continue
    
    return pd.DataFrame(top_combinations)

# Create visualizations
def create_visualizations(df):
    # Set style
    sns.set_theme()
    
    # Remove DNA_BOUNDS column for plotting and correlation
    if 'DNA_BOUNDS' in df.columns:
        df = df.drop(columns=['DNA_BOUNDS'])
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Parameter vs Score Scatter Plots
    params = ['MUT_RATE', 'MUT_SIGMA', 'ELITE_SIZE', 'POP_SIZE', 'NUM_GENERATIONS']
    
    for i, param in enumerate(params, 1):
        plt.subplot(2, 3, i)
        sns.scatterplot(data=df, x=param, y='best_score')
        plt.title(f'{param} vs Best Score')
        plt.xlabel(param)
        plt.ylabel('Best Score')
    
    # 2. Correlation Heatmap
    plt.subplot(2, 3, 6)
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Parameter Correlations')
    
    plt.tight_layout()
    plt.savefig('optimization_analysis.png')
    plt.close()
    
    # Create individual parameter analysis plots
    for param in params:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x=param, y='best_score')
        plt.title(f'{param} Distribution vs Best Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{param}_analysis.png')
        plt.close()

if __name__ == "__main__":
    # Read and parse the analysis file
    df = parse_analysis_file('/Users/stevenwendel/Documents/GitHub/bg/data/bayesian_opt_2025-06-06_08-37-58/analysis.txt')
    
    # Create visualizations
    create_visualizations(df) 