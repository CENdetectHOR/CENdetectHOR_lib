import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall


def matrix_sparsity(matrix):
    return 1.0 - np.count_nonzero(matrix) / matrix.size

def matrix_closure(adjacency_matrix, sparse_matrix=False):
    if sparse_matrix:
        return 1*np.isfinite(floyd_warshall(csgraph=np.ascontiguousarray(adjacency_matrix), directed=False))
    matrix_closure = adjacency_matrix
    while True:
        new_matrix_closure = matrix_closure | matrix_closure @ matrix_closure
        if np.array_equal(new_matrix_closure, matrix_closure):
            # break
            return matrix_closure
        matrix_closure = new_matrix_closure

def triu_closure(adjacency_triu, sparse_matrix=False):
    return np.triu(matrix_closure(adjacency_triu + adjacency_triu.T, sparse_matrix=sparse_matrix),1)