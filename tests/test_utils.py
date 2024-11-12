import unittest
import numpy as np
from src.utils import load_dna, create_alpha_array, alpha_fit

class TestUtils(unittest.TestCase):
    def test_load_dna(self):
        free_weights = [["A", "B"], ["B", "C"], ["C", "A"]]
        dna = [10, 20, 30]
        expected_matrix = np.array([
            [0, 10, 0],
            [0, 0, 20],
            [30, 0, 0]
        ])
        loaded_matrix = load_dna(free_weights, dna)
        np.testing.assert_array_equal(loaded_matrix, expected_matrix)
    
    def test_create_alpha_array(self):
        length = 5
        L = 2
        expected_alphas = np.array([
            (1 / L) * np.exp((L - 1) / L),
            (2 / L) * np.exp((L - 2) / L),
            (3 / L) * np.exp((L - 3) / L),
            (4 / L) * np.exp((L - 4) / L),
            (5 / L) * np.exp((L - 5) / L)
        ])
        alphas = create_alpha_array(length, L)
        np.testing.assert_almost_equal(alphas, expected_alphas)
    
    def test_alpha_fit_full_fit(self):
        alp_arr = np.array([1, 2, 3])
        time = 1
        time_max = 5
        expected = np.array([0, 1, 2, 3, 0])
        result = alpha_fit(alp_arr, time, time_max)
        np.testing.assert_array_equal(result, expected)
    
    def test_alpha_fit_truncated_fit(self):
        alp_arr = np.array([1, 2, 3])
        time = 3
        time_max = 5
        expected = np.array([0, 0, 0, 1, 2])
        result = alpha_fit(alp_arr, time, time_max)
        np.testing.assert_array_equal(result, expected)
    
    def test_alpha_fit_partial_fit(self):
        alp_arr = np.array([1, 2, 3, 4])
        time = 2
        time_max = 5
        expected = np.array([0, 0, 1, 2, 3])
        result = alpha_fit(alp_arr, time, time_max)
        np.testing.assert_array_equal(result, expected)

if __name__ == '__main__':
    unittest.main()