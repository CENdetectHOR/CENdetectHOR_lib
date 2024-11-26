from Bio.SeqRecord import SeqRecord
import editdistance
import numpy as np


def build_string_distance_matrix(strings: list[str]) -> np.ndarray:
    seq_triu = np.array([
            (0 if j <= i else editdistance.eval(seq_i, strings[j]))
            for i, seq_i in enumerate(strings)
            for j in range(len(strings))
    ])
    seq_triu.shape = (len(strings), len(strings))
    return seq_triu + seq_triu.T


def build_seqs_distance_matrix(seqs: list[SeqRecord]) -> np.ndarray:
    return build_string_distance_matrix([str(seq.seq) for seq in seqs])


def min_distance(distance_matrix: np.ndarray) -> any:
    return min(distance_matrix[np.triu_indices(distance_matrix.shape[0],1)])


def distance_values(matrix: np.ndarray):
    return matrix[np.triu_indices(matrix.shape[0], k = 1)]