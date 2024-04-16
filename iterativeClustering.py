from collections.abc import Iterable
from cluster import build_string_distance_matrix, get_seq_as_txt, merge_clusters, min_distance
from featureUtils import feature_to_seq, label_to_phyloxml_sequence, location_to_feature, order_by_indices, order_matrix_by_indeces, sorted_locations_indeces
from hor import HORInSeq, hor_tree_to_phylogeny, loops_to_HORs, name_hor_tree
from loops import LoopInSeq, LoopSpanInSeq, find_loops
import numpy as np
from Bio.Phylo.PhyloXML import Phyloxml, Clade, Other
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord

from treeFromClusters import merge_clades, new_leaves, new_phylogeny, features_to_leaves


class ClusteredSeq:
    """
    A list of contiguous monomers, possibly with gaps in it, classified
    in a set of clusters (families).
    Information about found loops may be represented.
    Loops are repeated patterns in the list of monomers labelled with the
    clusters they belong to.

    Attributes
    ----------
    clades: list[Clade]
        Clades in the phylogenetic tree corresponding to each of the clusters;
        must be of length m, where m is the number of clusters.
    clusters_expansion : np.ndarray
        Boolean m x n matrix representing the cluster assignments,
        where m is the number of clusters and n is the number of monomers.
    loops : list[LoopInSeq]
        List of found loops (expressed in terms of local clusters).
    hors: list[]
        List of found HORs (loops expressed in terms of clades).
    seq_split_indeces: list[int]
        List of the indeces at which there are gaps in the list of monomers.
    seqs_as_clusters: list[list[int]]
        Each list of contiguous monomers (obtained from the original
        list split by the gaps), represented as the list of corresponding
        cluster numbers.

    """

    def __init__(
        self,
        clusters_expansion: np.ndarray,
        loops: list[LoopInSeq] = [],
        clades: list[Clade] = None,
        gap_indeces: list[int] = [],
        seq_locations: Iterable[SimpleLocation] = None
    ):
        self.clades = clades
        self.clusters_expansion = clusters_expansion
        self.loops = []
        self.add_loops(loops)
        whole_seq_as_clusters = list(
            np.arange(len(clusters_expansion)) @ clusters_expansion
        )
        seq_split_indeces = [0] + gap_indeces + [len(whole_seq_as_clusters)]
        self.seqs_as_clusters = [
            whole_seq_as_clusters[seq_split_indeces[i]:seq_split_indeces[i+1]]
            for i in range(len(gap_indeces) + 1)
        ]
        self.seq_locations = seq_locations
        # self.seq_as_clusters = list(np.arange(len(clusters_expansion)) @ clusters_expansion)

    def add_loops(
        self,
        loops: Iterable[LoopInSeq]
    ):
        self.loops.extend(loops)
        if self.clades is not None:
            self.hors = loops_to_HORs(
                self.loops, self.clades, seq_locations=self.seq_locations
            )

    def __str__(self):
        return (
            f'Num clusters: {len(self.clusters_expansion)}, ' +
            f'Seqs: {[get_seq_as_txt(seq_as_clusters) for seq_as_clusters in self.seqs_as_clusters]}, ' +
            f'Loops: {[str(loop) for loop in self.loops]}'
        )

    def to_dict(self):
        return {
            "num_clusters": len(self.clusters_expansion),
            "seqs": [
                get_seq_as_txt(seq_as_clusters)
                for seq_as_clusters in self.seqs_as_clusters
            ],
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


def matrix_sparsity(matrix: np.ndarray) -> float:
    return 1.0 - np.count_nonzero(matrix) / matrix.size

# Compares two collections of spans in a sequence, called "coverages"
# Each coverage implicitly represents the set of items of the sequence included in some span
# Returns true iff the set of items associated with coverage_a contains the one associated with coverage_b


def coverage_includes(
    coverage_a: Iterable[LoopSpanInSeq],
    coverage_b: Iterable[LoopSpanInSeq]
) -> bool:
    return all([
        any([
            span_a.span_start <= span_b.span_start and span_a.span_start +
            span_a.span_length >= span_b.span_start + span_b.span_length
            for span_a in coverage_a
        ]) for span_b in coverage_b
    ])


def coverage_diff(
    coverage_a: Iterable[LoopSpanInSeq],
    loops_b: Iterable[LoopInSeq]
) -> list[LoopInSeq]:
    return [
        loop_b
        for loop_b in loops_b
        if any([
            all([
                span_a.span_start > span_b.span_start or span_a.span_start +
                span_a.span_length < span_b.span_start + span_b.span_length
                for span_a in coverage_a
            ])
            for span_b in loop_b.spans_in_seq
        ])
    ]


def clusterings_with_hors(
    monomer_seqs: list[SeqRecord] = None,
    monomers_as_features: list[SeqFeature] = None,
    monomer_locations: list[SimpleLocation] = None,
    references: dict[SeqRecord] = None,
    sorted: bool = False,
    sorted_by_positive_strand_location: bool = False,
    gap_indeces: list[int] = [],
    max_allowed_bases_gap_in_hor: int = 10,
    distance_matrix: np.ndarray = None,
    seq_labels_prefix: str = '',
    starting_distance: int = 0,
    min_num_clusters: int = 2, max_num_clusters: int = None,
    order_clusters: bool = False,
    require_loop: bool = True,
    min_len_loop: int = 1, max_len_loop: int = 30,
    min_loop_reps: int = 3,
    require_increasing_loop_coverage: bool = True,
    require_relevant_loop_coverage: bool = True,
    incremental_loops: bool = False,
    closure_sparsity_threshold: float = 0.97,
    build_tree: bool = True
) -> tuple[Phyloxml, HORInSeq, list[list[ClusteredSeq]]]:
    """Given a set of related DNA/RNA sequences, occurring in contiguous
    blocks and called monomers, this function looks for higher order
    repeats, i.e. repeats of sequence of families of sequences. 

    The function has been designed with the purpose of analysing the
    structure of repetitive patterns in centromeres, but there is no
    reason it cannot be used in other contexts.

    The monomers are subsequences of larger reference genetic sequences.
    The reference sequences are given through the references parameter
    or left implicit. The positions of the monomers in the reference
    sequences are defined using monomer_locations or monomers_as_features.
    If reference sequences are not explicitly given, the actual monomer
    sequences must also be given, using monomer_seqs.

    The main steps performed by the function are the following ones
    (some of them may be skipped if the corresponding information is
    already given in input):
    - extraction of the monomer sequences from the references;
    - computation of the matrix of Levenshtein distances between each
    pair of monomer sequences;
    - construction of a phylogenetic tree through hierarchical clustering,
    specifically single-linkage clustering;
    - for each cut of the tree, from leaves to root, discovery of HORs
    by looking how the clades at that cut repeat in the reference sequences;
    - linking of the HORs to form a rooted tree;
    - prune the HOR tree of duplications;
    - output both the phylogenetic tree of monomers and the tree of HORs.

    Parameters
    ----------
    monomer_seqs : list[SeqRecord], optional
        The monomer sequences, as an array of SeqRecord objects;
        required if 'references' is not given.
    monomers_as_features: list[SeqFeature], optional
        The monomers as subsequences of rereference sequences, using
        SeqRecord objects; if neither 'monomers_as_features',
        'monomer_locations', nor 'gap_indeces' are given the monomers
        are considered all contiguous.
    monomer_locations: list[SimpleLocation], optional
        The monomers as subsequences of rereference sequences, using
        SimpleLocation objects; used only if 'monomers_as_features' is
        not given; if neither 'monomers_as_features', 'monomer_locations',
        nor 'gap_indeces' are given the monomers are considered all
        contiguous.
    references: dict[SeqRecord], optional
        Dictionary with the reference sequences; the keys must correspond
        with the values for ref in monomers_as_features or monomer_locations;
        required if 'monomer_seqs' is not given.
    sorted: bool, default=False
        True if the monomers are sorted in a way that contiguous ones on
        the same strand are continguous in the given list.
    sorted_by_positive_strand_location: bool, default=False
        True if the monomers are sorted by location on the reference
        (positive) strand.
    gap_indeces: list[int], optional
        List of the indeces of the gaps over monomer_seqs; used only if
        neither 'monomers_as_features' nor 'monomer_locations' are given;
        if neither 'monomers_as_features', 'monomer_locations', nor
        'gap_indeces' are given the monomers are considered all contiguous.
    max_allowed_bases_gap_in_hor: int, default=10
        Maximum accepted gap (in number of bases) between monomers
        considered contiguous for HOR discovery purposes.
    distance_matrix: np.ndarray, optional
        Matrix of pairwise monomer distances, if already computed;
        it must be of shape (n,n), where n is the number of monomers;
        if not given it is computed by scratch.
    seq_labels_prefix: str, optional
        Optional string prefix to use when generating labels, in the case
        neither 'monomers_as_features' nor 'monomer_locations' are given.
    starting_distance: int, optional
        If given, the clustering starts at this distance, rather than
        from zero.
    min_num_clusters: int, default=2
        Deprecated parameter, do not use
    max_num_clusters: int, optional
        If given, the search for HORs starts only after the number of
        clusters is smaller than this number.
    order_clusters: bool, default=False
        If true, clusters at each level are ordered from largest to smallest.
    require_loop: bool, default=True
        Deprecated parameter, do not use
    min_len_loop: int, default=1
        Deprecated parameter, do not use
    max_len_loop: int, default=30,
        Maximum loop length of recognized HORs.
    min_loop_reps: int = 3,
        Minimum number of repetitions required for identifying as an HOR.
    require_increasing_loop_coverage: bool, default=True
        Deprecated parameter, do not use
    require_relevant_loop_coverage: bool, default=True
        Deprecated parameter, do not use
    incremental_loops: bool, default=False
        Deprecated parameter, do not use
    closure_sparsity_threshold: float, default=0.97
        Threshold used to decide the algorithm to be used for closure on
        the adjancency matrix, based on the sparsity of that matrix.
    build_tree: bool, default=True
        Deprecated parameter, do not use

    Returns
    -------
    tuple[Phyloxml, HORInSeq, list[list[ClusteredSeq]]]
        A triple with:
            - a Phyloxml object with two phylogenies, one for the monomers,
              one for the HORs;
            - the root HOR as an HORInSeq object;
            - all the found clusters and HORs, level by level.
    """
    print(f'Start of clusterings_with_hors')

    if monomers_as_features is not None:
        monomer_locations = [
            feature.location for feature in monomers_as_features]

    if monomer_locations is not None:

        if not sorted:
            print(f'Reorder')
            if sorted_by_positive_strand_location:
                print(f'Reorder negative strand')
                indexed_locations = list(enumerate(monomer_locations))
                positive_location_indexes = [
                    indexed_location[0] for indexed_location in indexed_locations
                    if indexed_location[1].strand is None or indexed_location[1].strand == 1
                ]
                negative_location_indexes = list(reversed([
                    indexed_location[0] for indexed_location in indexed_locations
                    if indexed_location[1].strand is not None and indexed_location[1].strand == -1
                ]))
                reordered_indeces = positive_location_indexes + negative_location_indexes
            else:
                print(f'Reorder all')
                reordered_indeces = sorted_locations_indeces(monomer_locations)
            print(f'Indexes built, now reordering lists...')
            monomer_locations = order_by_indices(
                monomer_locations, reordered_indeces)
            monomers_as_features = order_by_indices(
                monomers_as_features, reordered_indeces)
            monomer_seqs = order_by_indices(monomer_seqs, reordered_indeces)
            print(f'Lists reordered, now reordering matrix...')
            distance_matrix = order_matrix_by_indeces(
                distance_matrix, reordered_indeces)
            print(f'Matrix reordered')

        if monomers_as_features is None:
            monomers_as_features = [
                location_to_feature(location)
                for location in monomer_locations
            ]

        if monomer_seqs is None:
            monomer_seqs = [
                feature_to_seq(feature, references)
                for feature in monomers_as_features
            ]

        gap_indeces = [
            monomer_index + 1
            for monomer_index in range(len(monomer_locations) - 1)
            if monomer_locations[monomer_index + 1].ref != monomer_locations[monomer_index].ref
            or monomer_locations[monomer_index + 1].strand != monomer_locations[monomer_index].strand
            or monomer_locations[monomer_index + 1].start - monomer_locations[monomer_index].end > max_allowed_bases_gap_in_hor
        ]

    if distance_matrix is None:
        plain_seqs = [str(seq.seq) for seq in monomer_seqs]
        print(f'Computing distances...')
        distance_matrix = build_string_distance_matrix(plain_seqs)
        print(f'Distance matrix computed!')

    if max_num_clusters is None:
        max_num_clusters = distance_matrix.shape[0] / 4

    if require_relevant_loop_coverage:
        require_increasing_loop_coverage = True

    if build_tree:
        if monomers_as_features is not None:
            curr_clades = features_to_leaves(monomers_as_features)
        else:
            seq_labels = [seq_labels_prefix +
                          str(i) for i in range(distance_matrix.shape[0])]
            curr_clades = new_leaves(seq_labels)
    else:
        curr_clades = None

    clusterings = []
    curr_distance_matrix = distance_matrix
    curr_clusters_expansion = None
    curr_clusters_max_internal_distance = 0
    last_loops_coverage = []
    while curr_clusters_expansion is None or len(curr_clusters_expansion) >= min_num_clusters:
        curr_distance_matrix, curr_clusters_expansion, merged_clusters_expansion, merged_clusters_distance = merge_clusters(
            curr_distance_matrix,
            clusters_expansion=curr_clusters_expansion,
            sparsity_threshold=closure_sparsity_threshold,
            max_distance=max(starting_distance, min_distance(curr_distance_matrix)))
        if build_tree:
            curr_clades = merge_clades(
                clades=curr_clades, new_clusters_matrix=merged_clusters_expansion,
                branch_length=(merged_clusters_distance-curr_clusters_max_internal_distance)/2)
        if len(curr_clusters_expansion) <= max_num_clusters:
            if order_clusters:
                clusters_size = np.sum(curr_clusters_expansion, axis=1)
                ordered_clusters_expansion = curr_clusters_expansion[
                    clusters_size.argsort()[::-1]
                ]
                clustered_seq = ClusteredSeq(
                    ordered_clusters_expansion,
                    clades=curr_clades,
                    gap_indeces=gap_indeces,
                    seq_locations=monomer_locations
                )
            else:
                clustered_seq = ClusteredSeq(
                    curr_clusters_expansion,
                    clades=curr_clades,
                    gap_indeces=gap_indeces,
                    seq_locations=monomer_locations
                )
            print(f'Looking for loops in {str(clustered_seq)}...')
            loops = find_loops(
                clustered_seq.seqs_as_clusters,
                min_loop_size=min_len_loop, max_loop_size=max_len_loop,
                min_loops=min_loop_reps
            )
            print(f'Loops found: {len(loops)}')
            if incremental_loops:
                clustered_seq.add_loops(
                    coverage_diff(last_loops_coverage, loops)
                )
            else:
                clustered_seq.add_loops(loops)

            if len(loops) > 0 or not require_loop:
                loops_coverage = [
                    span_in_seq for loop in loops for span_in_seq in loop.spans_in_seq]
                if not require_increasing_loop_coverage or not coverage_includes(last_loops_coverage, loops_coverage):
                    clusterings.append(clustered_seq)
                    last_loops_coverage = loops_coverage

        curr_clusters_max_internal_distance = merged_clusters_distance

    if require_relevant_loop_coverage:
        new_clusterings_reversed = [clusterings[-1]]
        curr_preserved_loops = clusterings[-1].loops
        curr_hor_tree_nodes = clusterings[-1].hors
        for clustering_level in reversed(range(len(clusterings) - 1)):
            new_preserved_loops = []
            new_hor_tree_nodes = []
            new_loops = []
            clustered_seq = clusterings[clustering_level]
            for existing_loop_index in range(len(curr_preserved_loops)):
                existing_loop = curr_preserved_loops[existing_loop_index]
                existing_hor = curr_hor_tree_nodes[existing_loop_index]
                corresponding_candidate_loops = []
                corresponding_candidate_hors = []
                specificity_increased = False
                for loop_index in range(len(clustered_seq.loops)):
                    loop = clustered_seq.loops[loop_index]
                    hor = clustered_seq.hors[loop_index]
                    if coverage_includes(existing_loop.spans_in_seq, loop.spans_in_seq):
                        corresponding_candidate_loops.append(loop)
                        corresponding_candidate_hors.append(hor)
                        if (len(loop.loop.loop_seq) > len(existing_loop.loop.loop_seq) or
                                len(set(loop.loop.loop_seq)) > len(set(existing_loop.loop.loop_seq))):
                            specificity_increased = True
                if len(corresponding_candidate_loops) > 1 or specificity_increased:
                    new_preserved_loops.extend(corresponding_candidate_loops)
                    new_hor_tree_nodes.extend(corresponding_candidate_hors)
                    new_loops.extend(corresponding_candidate_loops)
                    existing_hor.sub_hors = corresponding_candidate_hors
                    for sub_hor in corresponding_candidate_hors:
                        sub_hor.super_hor = existing_hor
                        sub_hor.sub_hors = []
                else:
                    new_preserved_loops.append(existing_loop)
                    new_hor_tree_nodes.append(existing_hor)
            clustered_seq.loops = new_loops
            if len(new_loops) > 0 or not require_loop:
                new_clusterings_reversed.append(clustered_seq)
            curr_preserved_loops = new_preserved_loops
            curr_hor_tree_nodes = new_hor_tree_nodes

        hor_tree_roots = new_clusterings_reversed[0].hors

        clusterings = list(reversed(new_clusterings_reversed))

    if build_tree:
        seq_tree_root = curr_clades[0]
        hor_tree_root = hor_tree_roots[0]
        name_hor_tree(hor_tree_root)
        reference_seqs_element = (
            Other(
                tag='reference-sequences',
                children=[
                    label_to_phyloxml_sequence(ref_id)
                    for ref_id in references.keys()
                ]
            )
            if references is not None else None
        )
        return (
            Phyloxml(
                phylogenies=[
                    new_phylogeny(seq_tree_root),
                    hor_tree_to_phylogeny(hor_tree_root)
                ],
                attributes={'xsd': 'http://www.w3.org/2001/XMLSchema'},
                other=reference_seqs_element
            ),
            hor_tree_root, clusterings
        )
    else:
        return clusterings


def bfs_merge_rec(hor_tree_level: list[HORInSeq]):
    if not any([hor_in_level.sub_hors for hor_in_level in hor_tree_level]):
        return [hor_tree_level]
    sub_hors = [sub_hor for hor_in_level in hor_tree_level for sub_hor in (
        hor_in_level.sub_hors or [hor_in_level])]
    return [sub_hors] + bfs_merge_rec(sub_hors)


def bfs_merge(hor_tree_root: HORInSeq):
    return bfs_merge_rec([hor_tree_root])
