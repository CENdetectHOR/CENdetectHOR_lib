from dataclasses import dataclass
import json
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from treeFromClusters import feature_to_leave, new_clade, new_phylogeny
from Bio.SeqFeature import SeqFeature
from Bio.Phylo.PhyloXML import Clade, Phylogeny, Sequence, Phyloxml

class SimplePhylogeny:
    num_leaves: int
    children: list[list[int]]
    root_clade_index: int
    num_clades: int
    
    def __init__(
        self,
        num_leaves: int,
        children: list[list[int]]
    ):
        self.num_leaves = num_leaves
        self.children = children
        self.num_clades = num_leaves + len(children)
        self.root_clade_index = self.num_clades - 1
    
    def get_clade_children(self, clade_index: int) -> list[int]:
        return (
            [] if clade_index < self.num_leaves
            else self.children[clade_index - self.num_leaves]
        )
    
    # def set_clade_children(self, clade_index: int, children: list[int]):
    #     return (
    #         [] if clade_index < self.num_leaves
    #         else self.children[clade_index - self.num_leaves]
    #     )
    
    def get_clade_height(self, clade_index: int) -> int:
        return (
            0 if clade_index < self.num_leaves
            else 1 + max([
                self.get_clade_height(subclade_index)
                for subclade_index in self.get_clade_children(clade_index)
            ])
        )

class SimplePhylogenyWithDistances(SimplePhylogeny):
    max_distances: list[int]
    
    def __init__(
        self,
        num_leaves: int,
        children: list[list[int]],
        max_distances: list[int]
    ):
        super().__init__(num_leaves=num_leaves, children=children)
        self.max_distances = max_distances
    
    def get_clade_distance(self, clade_index: int):
        return (
            0 if clade_index < self.num_leaves
            else self.max_distances[clade_index - self.num_leaves]
        )
    
class SimplePhylogenyWithBranchLengths(SimplePhylogeny):
    branch_lengths: list[int] = None
    
    def __init__(
        self,
        num_leaves: int,
        children: list[list[int]],
        branch_lengths: list[int]
    ):
        super().__init__(num_leaves=num_leaves, children=children)
        self.branch_lengths = branch_lengths
    
    def get_branch_length(self, clade_index: int):
        return self.branch_lengths[clade_index]
    
def save_phylogeny(
    phylogeny: SimplePhylogenyWithBranchLengths,
    filename: str
):
    with open(filename + '.json', 'w') as f:
        json.dump({
            'num_leaves': phylogeny.num_leaves,
            'children': phylogeny.children,
            'branch_lengths': phylogeny.branch_lengths
        }, f)
        
def load_phylogeny(
    filename: str
) -> SimplePhylogenyWithBranchLengths:
    with open(filename + '.json') as f:
        dict = json.load(f)
        return SimplePhylogenyWithBranchLengths(
            num_leaves=dict['num_leaves'],
            children=dict['children'],
            branch_lengths=dict['branch_lengths']
        )
        
def compact_phylogeny(
    input_phylogeny: SimplePhylogenyWithDistances
) -> SimplePhylogenyWithBranchLengths:
    def clades_to_descendants_at_distance(clade_indeces: list[int], distance: int) -> list[Clade]:
        return [
            descendant_clade
            for child_clade_index in clade_indeces
            for descendant_clade in clade_to_descendants_at_distance(child_clade_index, distance)
        ]

    def clade_to_descendants_at_distance(clade_index: int, distance: int) -> list[Clade]:
        if input_phylogeny.get_clade_distance(clade_index) < distance:
            return [clade_index]
        descendant_clades_at_distance = clades_to_descendants_at_distance(
            input_phylogeny.get_clade_children(clade_index), distance
        )
        if len(descendant_clades_at_distance) == 0:
            return [clade_index]
        return descendant_clades_at_distance
    
    new_internal_clades_distances = []
    new_internal_clades_children = []
    
    def compact_clade(clade_index: int) -> int:
        subclades = input_phylogeny.get_clade_children(clade_index)
        max_distance = input_phylogeny.get_clade_distance(clade_index)
        if len(subclades) == 0:
            return clade_index
        descendants = clades_to_descendants_at_distance(
            subclades,
            max_distance
        )
        new_internal_clades_children.append([compact_clade(clade) for clade in descendants])
        new_internal_clades_distances.append(max_distance)
        new_clade_index = input_phylogeny.num_leaves + len(new_internal_clades_distances) - 1
        return new_clade_index 

    new_root_clade_index = compact_clade(input_phylogeny.root_clade_index)
    
    return SimplePhylogenyWithDistances(
        num_leaves=input_phylogeny.num_leaves,
        children=new_internal_clades_children,
        max_distances=new_internal_clades_distances
    )
    
def distances_to_branch_lengths(
    input_phylogeny: SimplePhylogenyWithDistances
) -> SimplePhylogenyWithBranchLengths:
    branch_lengths = [0] * (input_phylogeny.num_clades)
    def set_clade_branch_length(clade_index: int):
        for subclade_index in input_phylogeny.get_clade_children(clade_index):
            branch_lengths[subclade_index] = (
                input_phylogeny.get_clade_distance(clade_index) -
                input_phylogeny.get_clade_distance(subclade_index)
            ) / 2
            set_clade_branch_length(subclade_index)
    
    set_clade_branch_length(input_phylogeny.root_clade_index)
    
    return SimplePhylogenyWithBranchLengths(
        num_leaves=input_phylogeny.num_leaves,
        children=input_phylogeny.children,
        branch_lengths=branch_lengths
    )
    
def hierarchical_clustering(
    dist_matrix: np.ndarray,
    sort: bool = True
) -> SimplePhylogeny:
    clustering = AgglomerativeClustering(
        metric='precomputed',
        compute_full_tree=True,
        linkage='single',
        compute_distances = True,
        n_clusters=1)

    clustering.fit(dist_matrix)
    
    phylogeny = distances_to_branch_lengths(
        compact_phylogeny(
            SimplePhylogenyWithDistances(
                num_leaves=int(dist_matrix.shape[0]),
                children=[[int(subclade) for subclade in subclades] for subclades in clustering.children_],
                max_distances=clustering.distances_
            )
        )
    )
    
    if sort:
        sort_by_leaf_indexes(phylogeny)
        
    return phylogeny


def build_phylogeny(
    simple_phylogeny: SimplePhylogenyWithBranchLengths,
    features: list[SeqFeature]
) -> Phylogeny:
    
    def build_clade(clade_index: int):
        subclade_indices = simple_phylogeny.get_clade_children(clade_index)
        branch_length = simple_phylogeny.get_branch_length(clade_index)
        if len(subclade_indices) == 0:
            return feature_to_leave(features[clade_index], branch_length=branch_length)
        subclades = [build_clade(subclade_index) for subclade_index in subclade_indices]
        return Clade(clades=subclades, branch_length=branch_length)
        
    root_clade = build_clade(simple_phylogeny.root_clade_index)
    philogeny = new_phylogeny(root_clade)

    return philogeny

@dataclass
class CladeSortResult:
    subclade_index: int
    min_index: int    

def sort_by_leaf_indexes(phylogeny: SimplePhylogeny):
    
    def sort_clade_by_leaf_indices(clade_index: int):
        if len(phylogeny.get_clade_children(clade_index)) == 0:
            return clade_index
        subclade_results = [
            CladeSortResult(
                subclade_index,
                sort_clade_by_leaf_indices(subclade_index)
            )
            for subclade_index in phylogeny.get_clade_children(clade_index)
        ]
        subclade_results.sort(
            key=lambda c: c.min_index
        )
        phylogeny.children[clade_index - phylogeny.num_leaves] = [
            subclade_result.subclade_index
            for subclade_result in subclade_results
        ]
        return subclade_results[0].min_index
    
    sort_clade_by_leaf_indices(phylogeny.root_clade_index)
        
def get_clades_by_level(
    phylogeny: SimplePhylogeny
) -> tuple[list[list[int]],list[list[int]]]:
    
    clades_by_level = []
    children_num_by_level = []

    def set_by_height(clade_height: int, clade_index: int, children_num: int):
        if clade_height >= len(clades_by_level):
            missing_levels = clade_height - len(clades_by_level) + 1
            clades_by_level.extend([[]] * missing_levels)
            children_num_by_level.extend([[]] * missing_levels)
        clades_by_level[clade_height].append(clade_index)
        children_num_by_level[clade_height].append(children_num)

    def extract_clades_by_height(clade_index: int):
        subclade_indices = phylogeny.get_clade_children(clade_index)
        subclade_heights = [
            phylogeny.get_clade_height(subclade_index)
            for subclade_index in subclade_indices
        ]
        clade_height = (
            0 if len(subclade_indices) == 0
            else 1 + max(subclade_heights)
        )
        for subclade_position, subclade_height in enumerate(subclade_heights):
            subclade_index = subclade_indices[subclade_position]
            extract_clades_by_height(subclade_index)
            for height in range(subclade_height + 1, clade_height):
                set_by_height(clade_height=height, clade_index=subclade_indices[subclade_position], children_num=1)            
        set_by_height(clade_height=clade_height, clade_index=clade_index, children_num=len(subclade_indices))

    extract_clades_by_height(phylogeny.root_clade_index)
    return clades_by_level, children_num_by_level

def matrix_sparsity(matrix: np.ndarray) -> float:
    return 1.0 - np.count_nonzero(matrix) / matrix.size

@dataclass
class ClusteringMatricesResult:
    clustering_matrices: list[np.ndarray]
    clades_by_level: list[list[int]]

def get_clustering_matrices(phylogeny: SimplePhylogeny) -> ClusteringMatricesResult:
    clades_by_level, children_num_by_height = get_clades_by_level(phylogeny)
    print(f"Computing clustering matrices for {len(clades_by_level)} levels...")

    def get_clustering_matrix_for_level(level: int) -> np.ndarray:
        print(f"Computing clustering matrix for level {level}...")
        subclade_start_indexes = np.array(
            [0] +
            [
                children_num
                for children_num in children_num_by_height[level]]
        ).cumsum()
        return np.array([
            [
                subclade_index >= subclade_start_indexes[clade_index] and
                subclade_index < subclade_start_indexes[clade_index + 1]
                for subclade_index in range(subclade_start_indexes[-1])
            ]
            for clade_index in range(len(children_num_by_height[level]))
        ])        
        
    leaf_confusion_matrix = np.array([
        [
            original_leaf_index == clades_by_level[0][leaf_index]
            for original_leaf_index in range(phylogeny.num_leaves)
        ]
        for leaf_index in range(phylogeny.num_leaves)
    ])

    return ClusteringMatricesResult(
        clustering_matrices = [leaf_confusion_matrix] + [
            get_clustering_matrix_for_level(level)
            for level in range(1, len(children_num_by_height))
        ],
        clades_by_level = clades_by_level
    )
    
@dataclass
class MatricesResult:
    clustering_matrices: list[np.ndarray]
    expansion_matrices: list[np.ndarray]
    clades_by_level: list[list[int]]

def get_matrices(phylogeny: SimplePhylogeny) -> MatricesResult:
    res = get_clustering_matrices(phylogeny)
    clustering_matrices = res.clustering_matrices
    expansion_matrices = [clustering_matrices[0]]
    num_levels = len(clustering_matrices)
    print(f"Computing expansion matrices for {num_levels} levels...")
    for level in range(1, num_levels):
        print(f"Computing expansion matrix for level {level}...")
        expansion_matrices.append(clustering_matrices[level] @ expansion_matrices[level - 1])
    return MatricesResult(
        clustering_matrices=clustering_matrices,
        expansion_matrices=expansion_matrices,
        clades_by_level=res.clades_by_level
    )
    
def save_matrix(
    matrix: np.ndarray,
    filename: str,
    closure_sparsity_threshold: float = 0.97
):
    if matrix_sparsity(matrix) > closure_sparsity_threshold:
        save_npz(filename, )

def save_clustering_matrices(
    matrices_result: ClusteringMatricesResult,
    base_dir: str = '',
    clustering_matrices_template: str = 'clustering_{level}',
    clades_by_level_template: str = 'clades_{level}',
    closure_sparsity_threshold: float = 0.97
):
    clustering_matrices_template = base_dir + clustering_matrices_template
    clades_by_level_template = base_dir + clades_by_level_template
    num_levels = len(matrices_result.clustering_matrices)
    for level in range(num_levels):
        with open(clustering_matrices_template.format(level=level) + '.npy', 'wb') as f:
            np.save(f, matrices_result.clustering_matrices[level])
        with open(clades_by_level_template.format(level=level) + '.csv', 'w') as f:
            f.write('\n'.join([str(clade) for clade in matrices_result.clades_by_level[level]]))



def save_matrices(
    matrices_result: MatricesResult,
    base_dir: str = '',
    clustering_matrices_template: str = 'clustering_{level}',
    expansion_matrices_template: str = 'expansion_{level}',
    clades_by_level_template: str = 'clades_{level}',
    closure_sparsity_threshold: float = 0.97
):
    clustering_matrices_template = base_dir + clustering_matrices_template
    expansion_matrices_template = base_dir + expansion_matrices_template
    clades_by_level_template = base_dir + clades_by_level_template
    num_levels = len(matrices_result.clustering_matrices)
    for level in range(num_levels):
        with open(clustering_matrices_template.format(level=level) + '.npy', 'wb') as f:
            np.save(f, matrices_result.clustering_matrices[level])
        with open(expansion_matrices_template.format(level=level) + '.npy', 'wb') as f:
            np.save(f, matrices_result.expansion_matrices[level])
        with open(clades_by_level_template.format(level=level) + '.csv', 'w') as f:
            f.write('\n'.join([str(clade) for clade in matrices_result.clades_by_level[level]]))

def load_matrices(
    base_dir: str = '',
    clustering_matrices_template: str = 'clustering_{level}',
    expansion_matrices_template: str = 'expansion_{level}',
    clades_by_level_template: str = 'clades_{level}'
) -> MatricesResult:
    clustering_matrices_template = base_dir + clustering_matrices_template
    expansion_matrices_template = base_dir + expansion_matrices_template
    clades_by_level_template = base_dir + clades_by_level_template
    
    clustering_matrices = []
    expansion_matrices = []
    clades_by_level = []
    
    level = 0
    try:
        while(True):
            clustering_matrices.append(
                np.load(
                    clustering_matrices_template.format(level=level) + '.npy'
                )
            )
            expansion_matrices.append(
                np.load(
                    expansion_matrices_template.format(level=level) + '.npy'
                )
            )
            with open(
                clades_by_level_template.format(level=level) + '.csv'
            ) as f:
                clades_by_level.append(
                    [int(line.rstrip()) for line in f]
                )
            level = level + 1
    except IOError:
        return MatricesResult(
            clustering_matrices=clustering_matrices,
            expansion_matrices=expansion_matrices,
            clades_by_level=clades_by_level
        )

