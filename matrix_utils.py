import math
import numpy as np

def matrix_to_triu(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices_from(matrix, k=1)]

def triu_to_matrix(triu: np.ndarray) -> np.ndarray:
    matrix_size = int((math.sqrt(1 + 8 *len(triu)) + 1) / 2)
    matrix = np.zeros((matrix_size, matrix_size), dtype=triu.dtype)
    matrix[np.triu_indices_from(matrix, k=1)] = triu
    return matrix + matrix.T

def save_matrix_as_triu(matrix: np.ndarray, filename: str):
    with open(filename, 'wb') as f:
        np.save(f, matrix_to_triu(matrix))
    
def load_matrix_from_triu(filename: str) -> np.ndarray:
    return triu_to_matrix(np.load(filename))
