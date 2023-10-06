from Bio.Phylo.BaseTree import Clade, Tree

def new_clade(label=None, branch_length=None, clades=None):
    if clades is not None:
        if branch_length is not None:
            for clade in clades:
                clade.branch_length += branch_length
        if len(clades) == 1:
            return clades[0]
    return Clade(name=label, clades=clades, branch_length=0)

def merge_clades(clades, new_clusters_matrix, branch_length=None):
    return [
        new_clade(clades=[clade for index,clade in enumerate(clades) if new_cluster_row[index] > 0], branch_length=branch_length)
        for new_cluster_row in new_clusters_matrix
    ]

def new_leaves(leaf_labels):
    return [new_clade(label=leaf_label) for leaf_label in leaf_labels]

def new_tree(clade):
    return Tree(root=clade)
