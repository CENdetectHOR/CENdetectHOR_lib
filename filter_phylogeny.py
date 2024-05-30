from typing import Callable
from Bio.Phylo import PhyloXML
from Bio.Phylo.PhyloXML import Sequence, Phylogeny
from Bio.Phylo.BaseTree import Tree, Clade
from copy import deepcopy, copy


def seq_to_chrom(seq: Sequence) -> str:
    return seq.location.split(':')[0]

def sequences_from_chromosome(chromosome_label: str) -> Callable[[Sequence], bool]:
    return lambda seq : seq_to_chrom(seq) == chromosome_label

            
def filter_phylogeny(
    phylogeny: Tree,
    select_fun: Callable[[Clade], bool],
    clade_merges: dict[str, str] = None
) -> Tree:

    def filter_clade(clade: Clade) -> Clade:
        if len(clade.clades) == 0:
            if select_fun(clade):
                return clade
            else:
                return None
        new_subclades = [
            filter_clade(subclade)
            for subclade in clade.clades
        ]
        new_subclades = [
            subclade
            for subclade in new_subclades
            if subclade is not None
        ]
        if len(new_subclades) == 0:
            return None
        if len(new_subclades) == 1:
            new_subclades[0].branch_length += clade.branch_length
            if clade.name is not None:
                if (
                    new_subclades[0].name is not None and
                    len(new_subclades[0].clades) > 0
                ):
                    print(f"Clade label clash collapsing parent {clade.name} and child {new_subclades[0].name}")
                    if clade_merges is not None:
                        clade_merges[new_subclades[0].name] = clade.name
                new_subclades[0].name = clade.name
            return new_subclades[0]
        new_clade = copy(clade)
        new_clade.clades = new_subclades
        return new_clade
    
    new_root = filter_clade(phylogeny.root)
    if new_root is None:
        return None
    
    new_phylogeny = copy(phylogeny)
    new_phylogeny.root = new_root
    return new_phylogeny
            
# def filter_phylogeny(
#     phylogeny: Tree,
#     select_fun: Callable[[Clade], bool]
# ) -> Tree:
#     phylogeny = deepcopy(phylogeny)
#     for leaf in phylogeny.get_terminals():
#         if not select_fun(leaf):
#             phylogeny.prune(leaf)
#     return phylogeny
            
def filter_phylogeny_by_chromosome(
    phylogeny: Tree,
    chromosome_label: str,
    clade_merges: dict[str, str] = None
) -> Tree:
    return filter_phylogeny(
        phylogeny=phylogeny,
        select_fun=lambda leaf: any([
            seq_to_chrom(seq) == chromosome_label
            for seq in leaf.sequences
        ]),
        clade_merges=clade_merges
    )
    
def split_phylogeny_by_chromosome(
    phylogeny: Tree
) -> dict[str, tuple[Tree, dict[str, str]]]:
    chromosome_labels = set([
        seq_to_chrom(seq)
        for leaf in phylogeny.get_terminals()
        for seq in leaf.sequences
    ])
    # phylogeny_by_chromosome : dict[str, tuple[Tree, dict[str, str]]]= {}
    # for chromosome_label in chromosome_labels:
    #     clade_merges = {}
    #     filtered_phylogeny = filter_phylogeny_by_chromosome(
    #         phylogeny=phylogeny,
    #         chromosome_label=chromosome_label,
    #         clade_merges=clade_merges
    #     )
    #     phylogeny_by_chromosome[chromosome_label] = [filtered_phylogeny, clade_merges]
        
    # return phylogeny_by_chromosome
    return {
        chromosome_label: filter_phylogeny_by_chromosome(
            phylogeny=phylogeny,
            chromosome_label=chromosome_label
        ) for chromosome_label in chromosome_labels
    }

def filter_hor_clade(
    hor_clade: PhyloXML.Clade,
    select_fun: Callable[[Sequence], bool]
) -> PhyloXML.Clade:
    filtered_seqs = [seq for seq in hor_clade.sequences if select_fun(seq)]
    if len(filtered_seqs) == 0:
        return None
    filtered_subhors = filter(
        lambda s: s is not None,
        [
            filter_hor_clade(hor_clade=subhor, select_fun=select_fun)
            for subhor in hor_clade.clades
        ]
    )
    return PhyloXML.Clade(
        name=hor_clade.name,
        sequences=filtered_seqs,
        clades=filtered_subhors
    )
            
def filter_hor_tree(
    hor_tree: Phylogeny,
    select_fun: Callable[[Sequence], bool]
) -> Phylogeny:
    return Phylogeny(
        name=hor_tree.name,
        root=filter_hor_clade(
            hor_clade=hor_tree.root,
            select_fun=select_fun
        )
    )
            
def filter_hor_tree_by_chromosome(
    hor_tree: Phylogeny,
    chromosome_label: str
) -> Phylogeny:
    return filter_hor_tree(
        hor_tree=hor_tree,
        select_fun=lambda seq: seq_to_chrom(seq) == chromosome_label
    )
    
def split_hor_tree_by_chromosome(
    hor_tree: Phylogeny
) -> dict[str, Phylogeny]:
    chromosome_labels = set([
        seq_to_chrom(seq)
        for seq in hor_tree.root.sequences
    ])
    return {
        chromosome_label: filter_hor_tree_by_chromosome(
            hor_tree=hor_tree,
            chromosome_label=chromosome_label
        ) for chromosome_label in chromosome_labels
    }

def split_phyloxml_by_chromosomes(
    phyloxml: PhyloXML.Phyloxml
) -> dict[str, PhyloXML.Phyloxml]:
    phylogeny_lists_by_chromosome : dict[str,list[Phylogeny]] = {}
    
    def add_phylogenies(phylogenies_by_chromosome : dict[str, Phylogeny]):
        for chromosome, phylogeny in phylogenies_by_chromosome.items():
            if chromosome not in phylogeny_lists_by_chromosome:
                phylogeny_lists_by_chromosome[chromosome] = []
            phylogeny_lists_by_chromosome[chromosome].append(phylogeny)
    
    try:
        add_phylogenies(split_phylogeny_by_chromosome(phyloxml['monomers']))
    except KeyError:
        pass
    
    try:
        add_phylogenies(split_hor_tree_by_chromosome(phyloxml['hors']))
    except KeyError:
        pass
    
    return {
        chromosome: PhyloXML.Phyloxml(
            attributes=phyloxml.attributes,
            phylogenies=phylogenies,
            other=phyloxml.other
        )
        for chromosome, phylogenies in phylogeny_lists_by_chromosome.items()
    }