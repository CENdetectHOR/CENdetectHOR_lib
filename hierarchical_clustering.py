import numpy as np
from sklearn.cluster import AgglomerativeClustering
from treeFromClusters import features_to_leaves, new_clade, new_phylogeny
from Bio.SeqFeature import SeqFeature
from Bio.Phylo.PhyloXML import Clade, Phylogeny, Sequence, Phyloxml

def clades_to_descendants_at_distance(clades: list[Clade], distance: int) -> list[Clade]:
    return [
        descendant_clade
        for child_clade in clades
        for descendant_clade in clade_to_descendants_at_distance(child_clade, distance)
    ]

def clade_to_descendants_at_distance(clade: Clade, distance: int) -> list[Clade]:
    if clade.branch_length < distance:
        return []
    # if clade.branch_length > distance:
    #     raise Exception("Unexpected distance growth going down the tree")
    descendant_clades_at_distance = clades_to_descendants_at_distance(clade.clades, distance)
    if len(descendant_clades_at_distance) == 0:
        return [clade]
    return descendant_clades_at_distance

def compact_clade(clade: Clade) -> Clade:
    if len(clade.clades) == 0:
        return clade
    children_distance = clade.clades[0].branch_length
    # if len(set([subclade.branch_length for subclade in clade.clades])) != 1:
    #     raise Exception("Unexpected presence of multiple distances in children nodes")
    new_subclades = clades_to_descendants_at_distance(clade.clades, children_distance)
    # if len(set([subclade.branch_length for subclade in new_subclades])) != 1:
    #     raise Exception(f"Unexpected creation of multiple distances in children nodes while looking for distance {children_distance}: {set([subclade.branch_length for subclade in new_subclades])}")
    return Clade(
        clades=[compact_clade(clade) for clade in new_subclades],
        branch_length=clade.branch_length
    )

def compact_phylogeny(phylogeny: Phylogeny) -> Phylogeny:
    return Phylogeny(
        name=phylogeny.name,
        root=compact_clade(phylogeny.root)
    )
    
def hierarchical_clustering(
    features: list[SeqFeature],
    dist_matrix: np.ndarray
) -> Phylogeny:
    clustering = AgglomerativeClustering(
        metric='precomputed',
        compute_full_tree=True,
        linkage='single',
        compute_distances = True,
        n_clusters=1)

    clustering.fit(dist_matrix)
    tree_nodes = features_to_leaves(features)
    
    for node_index, node_children in enumerate(clustering.children_):
        tree_nodes.append(new_clade(
            label=None,
            branch_length=clustering.distances_[node_index],
            clades=[tree_nodes[child_index] for child_index in node_children]
        ))
        
    binary_philogeny = new_phylogeny(tree_nodes[-1])
    multichild_phylogeny = compact_phylogeny(binary_philogeny)
    return fix_phylogeny_branch_length(multichild_phylogeny)

def fix_clade_branch_length(clade: Clade) -> Clade:
    max_children_distance = 0 if len(clade.clades) == 0 else max([subclade.branch_length for subclade in clade.clades])
    return Clade(
        clades=[fix_clade_branch_length(subclade) for subclade in clade.clades],
        branch_length= (clade.branch_length - max_children_distance) / 2
    )
    
def fix_phylogeny_branch_length(phylogeny: Phylogeny) -> Phylogeny:
    return Phylogeny(
        name=phylogeny.name,
        root=Clade(clades=[
            fix_clade_branch_length(subclade)
            for subclade in phylogeny.root.clades
        ])
    )