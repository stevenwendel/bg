import unittest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.analysis.analysis import plot_pca, plot_tsne, plot_umap, compute_mahalanobis, display_stats

class TestAnalysis(unittest.TestCase):
    def setUp(self):
        # Sample DNA vectors and scores
        self.dna_vectors = [
            [10, 20, 30],
            [20, 30, 40],
            [30, 40, 50]
        ]
        self.score_vectors = [100, 200, 300]
        self.dna_0 = [15, 25, 35]
        self.top_df = pd.DataFrame({
            'DNA': self.dna_vectors,
            'Score': self.score_vectors
        })
        self.my_free_weights_names = ['A_B', 'B_C', 'C_A']
        self.best_dna = [30, 40, 50]
    
    def test_compute_mahalanobis(self):
        dna_array = np.array(self.dna_vectors)
        mahal_dist, p_value = compute_mahalanobis(dna_array, self.dna_0)
        self.assertIsInstance(mahal_dist, float)
        self.assertIsInstance(p_value, float)
    
    def test_display_stats(self):
        # To test display functions, we ensure they run without error
        try:
            display_stats(self.top_df, self.my_free_weights_names, self.best_dna, self.dna_0)
        except Exception as e:
            self.fail(f"display_stats raised an exception {e}")
    
    def test_plot_functions(self):
        # These functions primarily produce plots, so we'll ensure they run without error
        try:
            plot_pca(self.dna_vectors, self.score_vectors, dna_0=self.dna_0)
            plot_tsne(self.dna_vectors, self.score_vectors)
            plot_umap(self.dna_vectors, self.score_vectors)
        except Exception as e:
            self.fail(f"Plotting function raised an exception {e}")

if __name__ == '__main__':
    unittest.main()