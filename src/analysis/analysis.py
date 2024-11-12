import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.manifold import TSNE
import umap
import plotly.express as px
import plotly.graph_objects as go

def plot_pca(dna_vectors, score_vectors, dna_0=None):
    """
    Plots a 3D PCA of DNA vectors colored by their scores.
    
    Parameters:
        dna_vectors (list of list of float): DNA vectors.
        score_vectors (list of float): Corresponding scores.
        dna_0 (list of float, optional): Specific DNA vector to highlight.
    """
    scaler = StandardScaler()
    dna_scaled = scaler.fit_transform(dna_vectors)
    
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(dna_scaled)
    
    fig = px.scatter_3d(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        z=pca_result[:, 2],
        color=score_vectors,
        labels={'color': 'Score'},
        title='3D PCA of DNA Vectors'
    )
    
    if dna_0:
        dna_0_scaled = scaler.transform([dna_0])
        dna_0_pca = pca.transform(dna_0_scaled)
        fig.add_trace(go.Scatter3d(
            x=[dna_0_pca[0, 0]],
            y=[dna_0_pca[0, 1]],
            z=[dna_0_pca[0, 2]],
            mode='markers',
            marker=dict(size=10, color='red', symbol='diamond'),
            name='Original DNA (dna_0)'
        ))
    
    fig.update_traces(marker=dict(size=5))
    fig.show()

def plot_tsne(dna_vectors, score_vectors):
    """
    Plots a 2D t-SNE of DNA vectors colored by their scores.
    
    Parameters:
        dna_vectors (list of list of float): DNA vectors.
        score_vectors (list of float): Corresponding scores.
    """
    tsne = TSNE(n_components=2, perplexity=50, learning_rate=200, random_state=42)
    tsne_result = tsne.fit_transform(dna_vectors)
    
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=score_vectors, cmap='viridis')
    plt.colorbar(scatter, label='Fitness')
    plt.title('t-SNE of DNA Vectors')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

def plot_umap(dna_vectors, score_vectors):
    """
    Plots a 2D UMAP of DNA vectors colored by their scores.
    
    Parameters:
        dna_vectors (list of list of float): DNA vectors.
        score_vectors (list of float): Corresponding scores.
    """
    reducer = umap.UMAP(random_state=42)
    umap_result = reducer.fit_transform(dna_vectors)
    
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1], c=score_vectors, cmap='viridis')
    plt.colorbar(scatter, label='Score')
    plt.title('UMAP of Vectors')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.show()

def compute_mahalanobis(dna_array, dna_0):
    """
    Computes the Mahalanobis distance and p-value of dna_0 from the DNA array.
    
    Parameters:
        dna_array (np.ndarray): Array of DNA vectors.
        dna_0 (list of float): Specific DNA vector to compare.
    
    Returns:
        tuple: (mahal_dist, p_value)
    """
    mean_vector = np.mean(dna_array, axis=0)
    cov_matrix = np.cov(dna_array, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    
    mahal_dist = mahalanobis(dna_0, mean_vector, inv_cov_matrix)
    df = len(mean_vector)
    p_value = 1 - chi2.cdf(mahal_dist**2, df)
    
    print(f"Mahalanobis distance: {mahal_dist}")
    print(f"P-value: {p_value}")
    return mahal_dist, p_value

def display_stats(top_df, my_free_weights_names, best_dna, dna_0):
    """
    Displays statistical summaries of the top DNA samples.
    
    Parameters:
        top_df (pd.DataFrame): DataFrame containing top DNA samples.
        my_free_weights_names (list of str): Names of the free weights.
        best_dna (list of float): DNA of the best individual.
        dna_0 (list of float): Original DNA.
    """
    dna_df = pd.DataFrame(top_df['DNA'].tolist(), columns=my_free_weights_names)
    mean_df = dna_df.mean()
    median_df = dna_df.median()
    skew_df = dna_df.skew(axis=0)
    std_dev_df = dna_df.std()
    
    stats_df = pd.DataFrame({
        'µ': mean_df,
        'σ': std_dev_df,
        'Skew': skew_df,
        'Best': best_dna,
        'JH': dna_0,
        'JH-µ': np.array(dna_0) - np.array(mean_df),
        'JH-Best': np.array(dna_0) - np.array(best_dna)
    })
    display(stats_df)