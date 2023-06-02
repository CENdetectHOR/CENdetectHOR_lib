import numpy as np
import editdistance
from collections import defaultdict

def build_distance_matrix(seq_blocks):
    seq_dists = np.array([
        editdistance.eval(seq_block_1, seq_block_2) for seq_block_1 in seq_blocks for seq_block_2 in seq_blocks])
    seq_dists.shape = (len(seq_blocks), len(seq_blocks))
    return seq_dists

def matrix_closure(adjacency_matrix):
    matrix_closure = adjacency_matrix
    while True:
        new_matrix_closure = matrix_closure | np.matmul(matrix_closure, adjacency_matrix)
        if np.array_equal(new_matrix_closure, matrix_closure):
            break
        matrix_closure = new_matrix_closure
    return matrix_closure

def connected_components(adjacency_matrix, min_size=2):
    reachability_matrix = matrix_closure(adjacency_matrix)
    components = {}
    for curr_node in range(adjacency_matrix.shape[0]):
        existing_cluster_found = False
        for prev_node in range(curr_node):
            if reachability_matrix[curr_node][prev_node] == 1:
                components[prev_node].append(curr_node)
                existing_cluster_found = True
                break
        if not existing_cluster_found:
            components[curr_node] = [curr_node]
    return [component for component in components.values() if len(component) >= min_size]

def distance_values(matrix):
    return matrix[np.triu_indices(matrix.shape[0], k = 1)]

def normalize_loop(loop_seq):
    def invert_pos(pos):
        len(loop_seq) - pos if pos > 0 else 0
    options = []
    for start_index in range(len(loop_seq)):
        options.append(loop_seq[start_index:] + loop_seq[:start_index] + [start_index])
    options.sort()
    return options[0][:-1],invert_pos(options[0][-1])

def find_loops(seq, max_loop_size = 20, min_loops = 4):
    loops_found = {} #defaultdict(list)
    curr_loops = {loop_size:0 for loop_size in range(1, max_loop_size + 1)}

    def last_of_size(curr_position, loop_size):
        if curr_loops[loop_size] >= min_loops * loop_size:
            loop_start = curr_position - curr_loops[loop_size]
            loop_length = curr_loops[loop_size]
            loop_laps = loop_length // loop_size
            loop_items = seq[loop_start:loop_start + loop_size]
            normal_loop, in_loop_start_position = normalize_loop(loop_items)
            normal_loop_str = str(normal_loop)
            if normal_loop_str not in loops_found:
                loops_found[normal_loop_str] = (
                    normal_loop,
                    [(loop_start, loop_length, loop_laps, in_loop_start_position)]
                )
            else:
                loops_found[normal_loop_str][1].append((
                    loop_start, loop_length, loop_laps, in_loop_start_position
                ))

        # curr_loops[loop_size] = 0
        
    for curr_position, curr_symbol in enumerate(seq):
        for loop_size in range(1, min(max_loop_size, curr_position) + 1):
            if seq[curr_position - loop_size] == curr_symbol:
                curr_loops[loop_size] += 1
            else:
                last_of_size(curr_position, loop_size)
                curr_loops[loop_size] = 0
    
    for loop_size in range(1, max_loop_size + 1):
        last_of_size(len(seq), loop_size)

    return loops_found.values()

def clusterings_with_hors(distance_matrix, max_num_clusters = None):
    if max_num_clusters is None:
        max_num_clusters = distance_matrix.shape[0] / 4
    last_num_clusters = max_num_clusters + 1
    clusterings = []
    for max_distance in range(distance_matrix.max()):
        clusters = connected_components(distance_matrix <= max_distance, min_size=1)
        if len(clusters) < last_num_clusters:
            seq_to_cluster = {
                seq_index:cluster_index
                for cluster_index, cluster in enumerate(clusters)
                for seq_index in cluster
            }
            seq_as_clusters = [seq_to_cluster[seq_pos] for seq_pos in range(distance_matrix.shape[0])]
            loops = find_loops(seq_as_clusters)
            seq_as_clusters_txt = "".join([chr(65 + cluster_index) for cluster_index in seq_as_clusters])
            clusterings.append((
                len(clusters), clusters, seq_to_cluster,
                seq_as_clusters, loops, seq_as_clusters_txt
            ))
            last_num_clusters = len(clusters)
    return clusterings

