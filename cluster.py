from matrix_closure import triu_closure, matrix_sparsity
import numpy as np
import editdistance

def build_distance_matrix(seq_blocks):
    seq_triu = np.array([
            (0 if j <= i else editdistance.eval(seq_i, seq_blocks[j]))
            for i, seq_i in enumerate(seq_blocks)
            for j in range(len(seq_blocks))
    ])
    seq_triu.shape = (len(seq_blocks), len(seq_blocks))
    return seq_triu + seq_triu.T

def matrix_prod(a, b):
    # return np.einsum('ij,jk->ikj', a, b)
    res = a[:,None] * b.T
    return res

def matrix_min(matrix, mask):
    return np.array([np.min(matrix, where=cr, initial=np.max(matrix), axis=1) for cr in mask])
    # return (np.min(np.tile(matrix[:,:,None],len(mask)), where=mask.T, initial=np.max(matrix), axis=1)).T

def min_distance(distance_matrix):
    return min(distance_matrix[np.triu_indices(distance_matrix.shape[0],1)])

def merge_clusters(distance_matrix, clusters_expansion = None, max_distance = None, sparsity_threshold = 0.5):
    if max_distance is None:
        max_distance = min_distance(distance_matrix)
    print(f'merge_clusters with distance as {distance_matrix.shape}, clusters as {() if clusters_expansion is None else clusters_expansion.shape}, and max distance {max_distance}')
    adjancency_triu = np.triu(distance_matrix <= max_distance,1)
    print(f'adjacency triu sparsity is {matrix_sparsity(adjancency_triu)}')
    adjancency_triu_closure = triu_closure(adjancency_triu, sparse_matrix=matrix_sparsity(adjancency_triu) > sparsity_threshold)
    indexes_to_suppress = np.sum(adjancency_triu_closure, axis=0)
    np.fill_diagonal(adjancency_triu_closure, 1)
    new_clusters_expansion = np.delete(adjancency_triu_closure, indexes_to_suppress > 0, axis=0)

    new_rows_distance_matrix = matrix_min(distance_matrix, new_clusters_expansion > 0)
    new_distance_matrix = matrix_min(new_rows_distance_matrix, new_clusters_expansion > 0)

    updated_clusters_expansion = new_clusters_expansion if clusters_expansion is None else new_clusters_expansion @ clusters_expansion
    return new_distance_matrix, updated_clusters_expansion, new_clusters_expansion, max_distance

def distance_values(matrix):
    return matrix[np.triu_indices(matrix.shape[0], k = 1)]

def get_seq_as_txt(seq):
    return "".join([(chr(65 + symbol) if symbol < 26 else (chr(71 + symbol) if symbol < 52 else '*')) for symbol in seq])

