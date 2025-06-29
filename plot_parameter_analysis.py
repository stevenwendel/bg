import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Load the trial results
filepath = "/Users/stevenwendel/Documents/GitHub/bg/results/bayes_20250620_220309/trial_results.pkl"

with open(filepath, "rb") as f:
    trial_results = pickle.load(f)

# Convert to DataFrame for easier analysis
data = []
for trial in trial_results:
    cfg = trial['cfg']
    data.append({
        'POP_SIZE': cfg['POP_SIZE'],
        'NUM_GENERATIONS': cfg['NUM_GENERATIONS'],
        'MUT_RATE': cfg['MUT_RATE'],
        'MUT_SIGMA': cfg['MUT_SIGMA'],
        'ELITE_SIZE': cfg['ELITE_SIZE'],
        'RANK_DEPTH': cfg['RANK_DEPTH'],
        'score': trial['score'],
        'dna': trial['dna']  # This is currently None
    })

df = pd.DataFrame(data)

# Check if DNA data is available
dna_available = df['dna'].notna().any()
print(f"DNA data available: {dna_available}")

# Create subplots for each parameter
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Genetic Algorithm Parameter Analysis', fontsize=16, fontweight='bold')

# Parameters to plot
params = ['POP_SIZE', 'NUM_GENERATIONS', 'MUT_RATE', 'MUT_SIGMA', 'ELITE_SIZE', 'RANK_DEPTH']
titles = ['Population Size', 'Number of Generations', 'Mutation Rate', 'Mutation Sigma', 'Elite Size', 'Rank Depth']

# Flatten axes for easier iteration
axes_flat = axes.flatten()

for i, (param, title) in enumerate(zip(params, titles)):
    ax = axes_flat[i]
    
    # Scatter plot
    ax.scatter(df[param], df['score'], alpha=0.6, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
    
    # Add trend line
    z = np.polyfit(df[param], df['score'], 1)
    p = np.poly1d(z)
    ax.plot(df[param], p(df[param]), "r--", alpha=0.8, linewidth=2)
    
    # Calculate correlation
    correlation = df[param].corr(df['score'])
    
    # Add statistics
    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
            transform=ax.transAxes, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_xlabel(title)
    ax.set_ylabel('Score')
    ax.set_title(f'{title} vs Score')
    ax.grid(True, alpha=0.3)
    
    # Add some statistics
    ax.text(0.05, 0.85, f'Mean: {df[param].mean():.1f}\nStd: {df[param].std():.1f}', 
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))

plt.tight_layout()
plt.savefig('parameter_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n=== PARAMETER ANALYSIS SUMMARY ===")
print(f"Number of trials: {len(df)}")
print(f"Score range: {df['score'].min()} - {df['score'].max()}")
print(f"Mean score: {df['score'].mean():.2f} ± {df['score'].std():.2f}")

print("\n=== CORRELATIONS WITH SCORE ===")
for param in params:
    corr = df[param].corr(df['score'])
    print(f"{param:15s}: {corr:6.3f}")

print("\n=== BEST PERFORMING TRIALS ===")
top_5 = df.nlargest(5, 'score')
print(top_5[['score'] + params].to_string(index=False))

# Create a correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(label='Correlation Coefficient')

# Add correlation values as text
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Parameter Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Create box plots for each parameter grouped by score performance
plt.figure(figsize=(15, 10))

# Create score bins for grouping
df['score_bin'] = pd.cut(df['score'], bins=3, labels=['Low', 'Medium', 'High'])

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Parameter Distribution by Score Performance', fontsize=16, fontweight='bold')

for i, (param, title) in enumerate(zip(params, titles)):
    ax = axes_flat[i]
    
    # Box plot
    df.boxplot(column=param, by='score_bin', ax=ax)
    ax.set_title(f'{title} by Score Performance')
    ax.set_xlabel('Score Performance')
    ax.set_ylabel(title)
    
    # Remove the automatic title that pandas adds
    ax.set_title(f'{title} by Score Performance')

plt.tight_layout()
plt.savefig('parameter_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# DNA Analysis Section
print("\n" + "="*60)
print("DNA ANALYSIS")
print("="*60)

if not dna_available:
    print("⚠️  WARNING: DNA data is not available in the current results!")
    print("   All DNA values are None in the trial results.")
    print("\n📋 To capture DNA data in future runs, you need to modify your code.")
    print("   See the file 'capture_dna_guide.py' for implementation details.")
    
    # Create a summary of what we know about the best configurations
    print("\n🏆 BEST CONFIGURATIONS (without DNA):")
    print("="*40)
    
    # Group by similar scores to find patterns
    high_scoring = df[df['score'] >= 670]
    print(f"High-scoring trials (≥670): {len(high_scoring)}")
    
    if len(high_scoring) > 0:
        print("\nParameter ranges for high-scoring trials:")
        for param in params:
            min_val = high_scoring[param].min()
            max_val = high_scoring[param].max()
            mean_val = high_scoring[param].mean()
            print(f"  {param:15s}: {min_val:6.1f} - {max_val:6.1f} (mean: {mean_val:6.1f})")
    
    # Find the absolute best configuration
    best_trial = df.loc[df['score'].idxmax()]
    print(f"\n🥇 ABSOLUTE BEST TRIAL (Score: {best_trial['score']}):")
    for param in params:
        print(f"  {param:15s}: {best_trial[param]:6.1f}")

print("\n=== ANALYSIS COMPLETE ===")
print("Generated plots:")
print("- parameter_analysis.png: Scatter plots with trend lines")
print("- correlation_matrix.png: Correlation heatmap")
print("- parameter_distributions.png: Box plots by score performance") 