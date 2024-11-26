from dataclasses import dataclass
from Bio.Phylo.BaseTree import Clade, Tree

@dataclass
class CladeSortResult:
    sorted_clade: Clade
    min_label: str    

def sort_clade_by_leaf_names(clade: Clade) -> CladeSortResult:
    if len(clade.clades) == 0:
        return CladeSortResult(sorted_clade=clade, min_label=clade.name)
    sorted_subclade_results = [
        sort_clade_by_leaf_names(subclade) for subclade in clade.clades
    ]
    sorted_subclade_results.sort(
        key=lambda c: c.min_label
    )
    clade.clades = [
        clade_sort_result.sorted_clade
        for clade_sort_result in sorted_subclade_results
    ]
    return CladeSortResult(
        sorted_clade=clade,
        min_label=sorted_subclade_results[0].min_label
    )
    
def sort_phylogeny_by_leaf_names(phylogeny: Tree):
    sort_clade_by_leaf_names(phylogeny.root)