from dataclasses import dataclass
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from treeFromClusters import features_to_leaves, new_clade, new_phylogeny
from Bio.SeqFeature import SeqFeature
from Bio.Phylo.PhyloXML import Clade, Phylogeny, Sequence, Phyloxml

def clades_equal(
    clade_a: Clade, clade_b: Clade,
    check_branch_lengths: bool = True
) -> bool:
    if check_branch_lengths and clade_a.branch_length != clade_b.branch_length:
        return False
    if len(clade_a.clades) != len(clade_b.clades):
        return False
    if len(clade_a.clades) == 0:
        return clade_a.name == clade_b.name
    return all([
        clades_equal(clade_a.clades[subclade_index],clade_b.clades[subclade_index])
        for subclade_index in range(len(clade_a))
    ])      

def phylogenies_equal(
    phylogeny_a: Phylogeny, phylogeny_b: Phylogeny,
    check_branch_lengths: bool = True
) -> bool:
    return clades_equal(
        phylogeny_a.root, phylogeny_b.root,
        check_branch_lengths=check_branch_lengths
    )