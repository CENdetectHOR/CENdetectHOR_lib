from dataclasses import dataclass
import numpy as np
from clustering_to_phylogeny import SimplePhylogeny, get_clades_by_level
from matrix_utils import build_sparse_matrix

def matrix_sparsity(matrix: np.ndarray) -> float:
    return 1.0 - np.count_nonzero(matrix) / matrix.size

@dataclass
class ClusteringMatricesResult:
    clustering_matrices: list[np.ndarray]
    clades_by_level: list[list[int]]

def get_clustering_matrices(
    phylogeny: SimplePhylogeny,
    sparse_matrix_type,
    sparsity_threshold = 0.97,
) -> ClusteringMatricesResult:
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
        num_cols = subclade_start_indexes[-1]
        num_rows = len(children_num_by_height[level])
        if sparsity_threshold != 1 and num_rows >= 1 / (1 - sparsity_threshold):
            sparse_matrix = build_sparse_matrix((num_rows, num_cols), sparse_matrix_type=sparse_matrix_type)
            for clade_index in range(num_rows):
                for subclade_index in range(subclade_start_indexes[clade_index], subclade_start_indexes[clade_index + 1]):
                    sparse_matrix[clade_index][subclade_index] = True
            return sparse_matrix
        return np.array([
            [
                subclade_index >= subclade_start_indexes[clade_index] and
                subclade_index < subclade_start_indexes[clade_index + 1]
                for subclade_index in range(num_cols)
            ]
            for clade_index in range(num_rows)
        ])        
        
    if sparsity_threshold != 1 and phylogeny.num_leaves >= 1 / (1 - sparsity_threshold):
        leaf_confusion_matrix = build_sparse_matrix((phylogeny.num_leaves, phylogeny.num_leaves), sparse_matrix_type=sparse_matrix_type)
        for leaf_index in range(phylogeny.num_leaves):
            leaf_confusion_matrix[leaf_index][clades_by_level[0][leaf_index]] = True
    else:
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
    sparsity_threshold: float = 0.97
):
    if matrix_sparsity(matrix) > sparsity_threshold:
        save_npz(filename, )

def save_clustering_matrices(
    matrices_result: ClusteringMatricesResult,
    base_dir: str = '',
    clustering_matrices_template: str = 'clustering_{level}',
    clades_by_level_template: str = 'clades_{level}',
    sparsity_threshold: float = 0.97
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

