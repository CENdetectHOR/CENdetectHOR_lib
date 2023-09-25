from cluster import get_seq_as_txt, merge_clusters
from loops import find_loops
import numpy as np

class ClusteredSeq:
    def __init__(self, clusters_expansion, loops = []):
        self.clusters_expansion = clusters_expansion
        self.loops = []
        self.loops.extend(loops)
        self.seq_as_clusters = list(np.arange(len(clusters_expansion)) @ clusters_expansion)

    def add_loop(self, loop):
        self.loops.append(loop)

    def add_loops(self, loops):
        self.loops.extend(loops)

    def __str__(self):
        return (
            f'Num clusters: {len(self.clusters_expansion)}, ' +
            f'Seq: {get_seq_as_txt(self.seq_as_clusters)}, ' +
            f'Loops: {[str(loop) for loop in self.loops]}'
        )
    
    def to_dict(self):
        return {
            "num_clusters": len(self.clusters_expansion),
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

def coverage_includes(coverage_a, coverage_b):
    return all([
        any([
            span_a.span_start <= span_b.span_start and span_a.span_start + span_a.span_length >= span_b.span_start + span_b.span_length
            for span_a in coverage_a
        ]) for span_b in coverage_b
    ])

def clusterings_with_hors(
        distance_matrix, min_distance=0,
        min_num_clusters=2, max_num_clusters=None, order_clusters=True,
        require_loop=True, min_len_loop=2, max_len_loop=30, min_loop_reps=3,
        require_increasing_loop_coverage=True,
        closure_sparsity_threshold = 0.5):
    print(f'Start of clusterings_with_hors')
    if max_num_clusters is None:
        max_num_clusters = distance_matrix.shape[0] / 4
    # last_num_clusters = max_num_clusters + 1
    clusterings = []
    curr_distance_matrix = distance_matrix
    curr_clusters_expansion = None
    last_loops_coverage = []

    while curr_clusters_expansion is None or len(curr_clusters_expansion) >= min_num_clusters:
        curr_distance_matrix, curr_clusters_expansion = merge_clusters(
            curr_distance_matrix,
            clusters_expansion=curr_clusters_expansion,
            sparsity_threshold=closure_sparsity_threshold)
        if len(curr_clusters_expansion) <= max_num_clusters:
            if order_clusters:
                clusters_size = np.sum(curr_clusters_expansion, axis=1)
                ordered_clusters_expansion = curr_clusters_expansion[clusters_size.argsort()[::-1]]
                clustered_seq = ClusteredSeq(ordered_clusters_expansion)
            else:
                clustered_seq = ClusteredSeq(curr_clusters_expansion)
            print(f'Looking for loops in {str(clustered_seq)}...')
            loops = find_loops(clustered_seq.seq_as_clusters, min_loop_size=min_len_loop, max_loop_size=max_len_loop, min_loops=min_loop_reps)
            print(f'Loops found: {len(loops)}')
            clustered_seq.add_loops(loops)


            if len(loops) > 0 or not require_loop:
                loops_coverage = [span_in_seq for loop in loops for span_in_seq in loop.spans_in_seq]
                if not require_increasing_loop_coverage or not coverage_includes(last_loops_coverage, loops_coverage):
                    clusterings.append(clustered_seq)
                    last_loops_coverage = loops_coverage

    return clusterings

