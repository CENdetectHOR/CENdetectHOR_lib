from typing import List
import numpy as np
import editdistance
from collections import defaultdict
from math import log2, ceil
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall


def build_distance_matrix(seq_blocks):
    seq_dists = np.array([
        editdistance.eval(seq_block_1, seq_block_2) for seq_block_1 in seq_blocks for seq_block_2 in seq_blocks])
    seq_dists.shape = (len(seq_blocks), len(seq_blocks))
    return seq_dists

# def matrix_closure(adjacency_matrix, base_matrix_closure=None, check_interval=1):
#     matrix_closure = adjacency_matrix
#     if base_matrix_closure is not None:
#         matrix_closure = matrix_closure | np.matmul(matrix_closure, base_matrix_closure)
#         if np.array_equal(matrix_closure, base_matrix_closure):
#             return matrix_closure
#     for remaining_steps in range(ceil(log2(adjacency_matrix.shape[0])),0,-1):
#         new_matrix_closure = matrix_closure | np.matmul(matrix_closure, matrix_closure)
#         if remaining_steps > 1 and (remaining_steps % check_interval == 0) and np.array_equal(new_matrix_closure, matrix_closure):
#             break
#         matrix_closure = new_matrix_closure
#     return matrix_closure

def matrix_closure(adjacency_matrix, base_matrix_closure=None, check_interval=1, sparse_matrix=False):
    if sparse_matrix:
        # if base_matrix_closure is not None:
        #     matrix_closure = csr_matrix(matrix_closure)
        #     base_matrix_closure = csr_matrix(base_matrix_closure)
        #     matrix_closure = matrix_closure | matrix_closure @ base_matrix_closure
        return 1*np.isfinite(floyd_warshall(csgraph=adjacency_matrix, directed=False))
    matrix_closure = adjacency_matrix
    if base_matrix_closure is not None:
        matrix_closure = matrix_closure | matrix_closure @ base_matrix_closure
        if np.array_equal(matrix_closure, base_matrix_closure):
            return matrix_closure
    # for remaining_steps in range(ceil(log2(adjacency_matrix.shape[0])),0,-1):
    #     new_matrix_closure = matrix_closure | np.matmul(matrix_closure, matrix_closure)
    #     if remaining_steps > 1 and (remaining_steps % check_interval == 0) and np.array_equal(new_matrix_closure, matrix_closure):
    #         break
    #     matrix_closure = new_matrix_closure
    while True:
        new_matrix_closure = matrix_closure | matrix_closure @ matrix_closure
        if np.array_equal(new_matrix_closure, matrix_closure):
            break
        matrix_closure = new_matrix_closure
    return matrix_closure

def connected_components(adjacency_matrix, min_size=2, return_matrix=False, base_matrix_closure=None, closure_check_interval=1, sparse_matrix=False):
    reachability_matrix = matrix_closure(adjacency_matrix, base_matrix_closure=base_matrix_closure, check_interval=closure_check_interval, sparse_matrix=sparse_matrix)
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
    result = [component for component in components.values() if len(component) >= min_size]
    if return_matrix:
        return result, reachability_matrix
    else:
        return result

def distance_values(matrix):
    return matrix[np.triu_indices(matrix.shape[0], k = 1)]

def get_seq_as_txt(seq):
    return "".join([(chr(65 + symbol) if symbol < 26 else (chr(71 + symbol) if symbol < 52 else '*')) for symbol in seq])

def normalize_loop(loop_seq):
    def invert_pos(pos):
        return len(loop_seq) - pos if pos > 0 else 0
    options = []
    for start_index in range(len(loop_seq)):
        options.append(loop_seq[start_index:] + loop_seq[:start_index] + [start_index])
    options.sort()
    return (options[0][:-1],invert_pos(options[0][-1]))

def denormalize_loop(loop_seq, in_loop_start):
    return loop_seq[in_loop_start:] + loop_seq[:in_loop_start]

class Loop:
    loop_seq: List[int]

    def __init__(self, loop_seq):
        self.loop_seq = loop_seq

    def __str__(self):
        return get_seq_as_txt(self.loop_seq)

class LoopSpanInSeq:
    def __init__(self, span_start, span_length, num_of_laps, in_loop_start):
        self.span_start = span_start
        self.span_length = span_length
        self.num_of_laps = num_of_laps
        self.in_loop_start = in_loop_start

    def __str__(self):
        return (
            f'[{self.span_start}:{self.span_start + self.span_length}]' +
            (f'#{self.in_loop_start}' if self.in_loop_start != 0 else '')
        )

class LoopInSeq:
    loop: Loop
    spans_in_seq: List[LoopSpanInSeq]

    def __init__(self, loop, spans_in_seq = []):
        self.loop = loop
        self.spans_in_seq = spans_in_seq

    def add_span(self, span_in_seq):
        self.spans_in_seq.append(span_in_seq)

    def __str__(self):
        return (
            f'{self.loop}' +
            (
                f' in {",".join([str(span) for span in self.spans_in_seq])}'
                    if len(self.spans_in_seq) > 0 else ''
            )
        )

def find_loops(seq, min_loop_size = 2, max_loop_size = 20, min_loops = 4):
    loops_found = {} #defaultdict(list)
    curr_loops = {loop_size:0 for loop_size in range(1, max_loop_size + 1)}

    def last_of_size(curr_position, loop_size):
        if curr_loops[loop_size] >= min_loops * loop_size:
            loop_start = curr_position - curr_loops[loop_size] - loop_size
            loop_length = curr_loops[loop_size] + loop_size
            loop_laps = loop_length // loop_size
            loop_items = seq[loop_start:loop_start + loop_size]
            (normal_loop, in_loop_start_position) = normalize_loop(loop_items)
            normal_loop_str = str(normal_loop)
            loop_span = LoopSpanInSeq(loop_start, loop_length, loop_laps, in_loop_start_position)
            if normal_loop_str not in loops_found:
                loops_found[normal_loop_str] = LoopInSeq(
                    Loop(normal_loop),
                    [loop_span]
                )
            else:
                loops_found[normal_loop_str].add_span(loop_span)
        
    for curr_position, curr_symbol in enumerate(seq):
        max_loop_length_closed = 0
        for loop_size in range(1, min(max_loop_size, curr_position) + 1):
            if seq[curr_position - loop_size] == curr_symbol:
                curr_loops[loop_size] += 1
            else:
                if curr_loops[loop_size] > max_loop_length_closed:
                    last_of_size(curr_position, loop_size)
                    max_loop_length_closed = curr_loops[loop_size]
                curr_loops[loop_size] = 0
    
    max_loop_length_closed = 0
    for loop_size in range(1, max_loop_size + 1):
        if curr_loops[loop_size] > max_loop_length_closed:
            last_of_size(len(seq), loop_size)
            max_loop_length_closed = curr_loops[loop_size]

    loops = list(loops_found.values())

    if min_loop_size > 1:
        loops = [loop for loop in loops if len(loop.loop.loop_seq) >= min_loop_size]

    for loop in loops:
        spans = loop.spans_in_seq
        if all([span.in_loop_start == spans[0].in_loop_start for span in spans]):
            loop.loop.loop_seq = denormalize_loop(loop.loop.loop_seq, spans[0].in_loop_start)
            for span in spans:
                span.in_loop_start = 0

    return loops

class ClusteredSeq:
    def __init__(self, clusters, loops = []):
        self.clusters = clusters
        self.loops = []
        self.loops.extend(loops)
        self.seq_to_cluster = {
            seq_index:cluster_index
            for cluster_index, cluster in enumerate(clusters)
            for seq_index in cluster
        }
        self.seq_as_clusters = [
            self.seq_to_cluster[seq_pos] for seq_pos in sorted(self.seq_to_cluster.keys())
        ]

    def add_loop(self, loop):
        self.loops.append(loop)

    def add_loops(self, loops):
        self.loops.extend(loops)

    def __str__(self):
        return (
            f'Num clusters: {len(self.clusters)}, ' +
            f'Seq: {get_seq_as_txt(self.seq_as_clusters)}, ' +
            f'Loops: {[str(loop) for loop in self.loops]}'
        )
    
    def to_dict(self):
        return {
            "num_clusters": len(self.clusters),
            "seq": get_seq_as_txt(self.seq_as_clusters),
            "loops": [{
                "loop_seq": get_seq_as_txt(l.loop.loop_seq),
                "spans": [{
                    "span_start": span_inseq.span_start,
                    "span_length": span_inseq.span_length,
                    "num_of_laps": span_inseq.num_of_laps,
                    "in_loop_start": span_inseq.in_loop_start
                } for span_inseq in l.spans_in_seq]
            } for l in self.loops]
        }        

def matrix_sparsity(matrix):
    return 1.0 - np.count_nonzero(matrix) / matrix.size


def clusterings_with_hors(
        distance_matrix, min_distance=0, max_num_clusters=None, order_clusters=True,
        require_loop=True, min_len_loop=2,
        closure_check_interval=1, closure_sparsity_threshold = 0.5):
    if max_num_clusters is None:
        max_num_clusters = distance_matrix.shape[0] / 4
    last_num_clusters = max_num_clusters + 1
    clusterings = []
    curr_closure_matrix = None
    sparse_matrix = True
    for max_distance in range(min_distance, distance_matrix.max()):
        print(f'Checking clusters for max distance {max_distance}...')
        adjancecy_matrix = distance_matrix <= max_distance
        if sparse_matrix and matrix_sparsity(adjancecy_matrix) < closure_sparsity_threshold:
            sparse_matrix = False
        clusters,curr_closure_matrix = connected_components(
            adjancecy_matrix, min_size=1, return_matrix=True,
            base_matrix_closure=curr_closure_matrix,
            closure_check_interval=closure_check_interval,
            sparse_matrix=sparse_matrix)
        if order_clusters:
            clusters.sort(key=len, reverse=True)
        print(f'Found {len(clusters)} clusters')
        if len(clusters) < last_num_clusters:
            clustered_seq = ClusteredSeq(clusters)
            print(f'Looking for loops in {str(clustered_seq)}...')
            loops = find_loops(clustered_seq.seq_as_clusters, min_loop_size=min_len_loop)
            clustered_seq.add_loops(loops)
            if len(loops) > 0 or not require_loop:
                clusterings.append(clustered_seq)
            last_num_clusters = len(clusters)
            # print(f'Loops: {loops}')
            if last_num_clusters <= 1:
                break
    return clusterings

