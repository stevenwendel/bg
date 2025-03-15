import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

def plot_neurons(neurons, sq_wave, go_wave, t_max):
    fig, axs = plt.subplots(len(neurons), 1, figsize=(6, 3 * len(neurons)))
    for i, neu in enumerate(neurons):
        axs[i].plot(range(t_max), neu.hist_V, label="V")
        axs[i].plot(range(t_max), sq_wave, label="SqWave", alpha=0.8, color="red", linestyle="dotted")
        axs[i].plot(range(t_max), go_wave / 5, label="GoWave", alpha=0.8, color="red", linestyle="dotted")
        axs[i].set_title(f"{neu.name} dynamics")
        axs[i].set_xlabel("ms")
        axs[i].set_ylabel("mV")
        axs[i].grid(True)
        axs[i].legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_pca(dna_vectors, score_vectors):
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
    
    fig.update_traces(marker=dict(size=5))
    fig.show()

def plot_tsne(dna_vectors, score_vectors):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=50, learning_rate=200)
    tsne_result = tsne.fit_transform(dna_vectors)
    
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=score_vectors, cmap='viridis')
    plt.colorbar(label='Fitness')
    plt.title('t-SNE of Vectors')
    plt.show()

def plot_umap(dna_vectors, score_vectors):
    import umap
    reducer = umap.UMAP()
    umap_result = reducer.fit_transform(dna_vectors)
    
    plt.scatter(umap_result[:, 0], umap_result[:, 1], c=score_vectors, cmap='viridis')
    plt.colorbar(label='Score')
    plt.title('UMAP of Vectors')
    plt.show()

def plot_mahalanobis(dna_array, dna_0, path=0):
    mean_vector = np.mean(dna_array, axis=0)
    cov_matrix = np.cov(dna_array, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    
    mahal_dist = mahalanobis(dna_0, mean_vector, inv_cov_matrix)
    df = len(mean_vector)
    p_value = 1 - chi2.cdf(mahal_dist**2, df)
    
    print(f"Mahalanobis distance: {mahal_dist}")
    print(f"P-value: {p_value}")
    
def display_stats(top_df, my_free_weights_names, best_dna, dna_0):
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
        'JH-µ': dna_0 - mean_df,
        'JH-Best': np.array(dna_0) - np.array(best_dna)
    })
    display(stats_df)