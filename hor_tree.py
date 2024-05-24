from Bio.Phylo.BaseTree import Tree, Clade
from Bio.SeqFeature import SeqFeature
from featureUtils import SeqFeaturesByContiguity
from hor import HORInSeq, loop_to_HOR, name_hor_tree
from loops import find_loops, loop_to_spans
from phylogeny_to_levels import extract_features_from_leaves, phylogeny_to_levels

def levels_to_hor_tree(
    labelled_items_by_level: list[list[int]],
    clades_by_level: list[list[Clade]],
    seq_features: list[SeqFeature],
    gap_indices: list[int],
    min_loop_size: int = 1,
    max_loop_size: int = 30,
    min_loops: int = 3,
    allowed_mismatch_rate: float = 0.0,
    allow_hor_overlap: bool = False
) -> HORInSeq:
    seq_locations = [seq_feature.location for seq_feature in seq_features]
        
    def level_to_hors(
        spans_to_search: list[tuple[int,int]],
        level: int,
        curr_loop_diversity: int
    ):
        if level == 0:
            return []
        loops_in_seq = find_loops(
            whole_seq=labelled_items_by_level[level],
            seq_spans=spans_to_search,
            min_loop_size=min_loop_size,
            max_loop_size=max_loop_size,
            min_loops=min_loops,
            allowed_mismatch_rate=allowed_mismatch_rate,
            allow_overlap=allow_hor_overlap
        )
        if (len(loops_in_seq) == 0):
            return []
        
        if (
            len(loops_in_seq) == 1 and
            len(set(loops_in_seq[0].loop.loop_seq)) <= curr_loop_diversity
        ):
            return level_to_hors(
                spans_to_search=loop_to_spans(loops_in_seq[0]),
                level=level - 1,
                curr_loop_diversity=curr_loop_diversity
            )
        
        hors_in_seq = []
        for loop_in_seq in loops_in_seq:
            hor_in_seq = loop_to_HOR(
                loop_in_seq,
                clades=clades_by_level[level],
                seq_locations=seq_locations
            )
            hor_in_seq.sub_hors = level_to_hors(
                spans_to_search=loop_to_spans(loop_in_seq),
                level=level - 1,
                curr_loop_diversity=len(set(loop_in_seq.loop.loop_seq))
            )
            hors_in_seq.append(hor_in_seq)
            
        return hors_in_seq
    
    split_limits = [0] + gap_indices + [len(seq_locations)]
    num_splits = len(gap_indices) + 1
    hor_tree_root = level_to_hors(
        spans_to_search=[
            (split_limits[split_index], split_limits[split_index + 1])
            for split_index in range(num_splits)
        ],
        level=len(labelled_items_by_level) - 1,
        curr_loop_diversity=0
    )[0]
    name_hor_tree(hor_tree_root)
    return hor_tree_root

def phylogeny_to_hor_tree(
    phylogeny: Tree,
    min_loop_size: int = 1,
    max_loop_size: int = 30,
    min_loops: int = 3,
    allowed_mismatch_rate: float = 0.0,
    allow_hor_overlap: bool = False
) -> HORInSeq:
    sfbc = SeqFeaturesByContiguity(
            seq_features=extract_features_from_leaves(phylogeny)
    )
    levels_res = phylogeny_to_levels(
        phylogeny=phylogeny,
        item_position_to_leaf_index=sfbc.reordered_indices
    )
    return levels_to_hor_tree(
        labelled_items_by_level=levels_res.labelled_items_by_level,
        clades_by_level=levels_res.clades_by_level,
        seq_features=sfbc.sorted_seq_features,
        gap_indices=sfbc.gap_indices,
        min_loop_size=min_loop_size,
        max_loop_size=max_loop_size,
        min_loops=min_loops,
        allowed_mismatch_rate=allowed_mismatch_rate,
        allow_hor_overlap=allow_hor_overlap
    )
    
