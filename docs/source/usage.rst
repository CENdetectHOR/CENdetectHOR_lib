Usage
=====

.. _installation:

Installation
------------

To use the CENdetectHOR library, first install it using pip:

.. code-block:: console

   (.venv) $ pip install cen_detect_hor

Looking for HORs given a phylogeny of genomic sequences
-------------------------------------------------------

Given a set of situated genomic sequences organised in a putative phylogenetic tree,
the function ``cen_detect_hor.hor_tree.phylogeny_to_hor_tree()`` looks for candidate higher-order-repeats (HORs) and return them in a structured format:

.. autofunction:: cen_detect_hor.hor_tree.phylogeny_to_hor_tree

The ``phylogeny`` parameter must contain the phylogenetic tree of the sequences.
The leaves of the tree must have positional information of the corresponding sequence.
The other parameters are optional and control details of how the search is performed.

For example:

>>> from cen_detect_hor.clustering_to_phylogeny import clustering_to_phylogeny
>>> from Bio.Phylo import PhyloXMLIO
>>> from Bio.Phylo.PhyloXML import Phyloxml
>>> phyloXml = PhyloXMLIO.read('data/chr4_human/monomer_phylogeny.xml')
>>> hor_tree = phylogeny_to_hor_tree(phylogeny)
>>> complete_phyloXml = Phyloxml(phylogenies=[phylogeny, hor_tree.as_phyloxml], attributes=None)
>>> PhyloXMLIO.write(complete_phyloXml, './data/chr4_human/monomer_phylogeny_and_HORs.xml')

Creating a phylogeny from sequences
-----------------------------------

To create a phylogeny out of a set of genomic sequences,
via aggregative hierarchical clustering,
you can use the ``cen_detect_hor.clustering_to_phylogeny.clustering_to_phylogeny()`` function:

.. autofunction:: cen_detect_hor.clustering_to_phylogeny.clustering_to_phylogeny

The ``items_as_seq_features`` parameter must contain the specification of the sequence features to be clustered,
while the ``seq_references`` parameter must contain the actula genomic sequences those features are defined on.
The optional parameter ``dist_matrix`` my contain the matrix of distances between features, if it computed externally (this allows both saving computation time and flexibility on distance calculation methods).
The object returned is of type ``cen_detect_hor.clustering_to_phylogeny.ClusteringToPhylogenyResult``,
which contains in the field ``phylogeny`` the actual generated phylogney
and in the field ``item_position_to_leaf_index`` an information potentially useful for further analysis.

For example:

>>> from cen_detect_hor.clustering_to_phylogeny import clustering_to_phylogeny
>>> from cen_detect_hor.featureUtils import BED_file_to_features
>>> from Bio import SeqIO
>>> from Bio.Phylo.PhyloXML import Phyloxml
>>> from Bio.Phylo import PhyloXMLIO
>>> references = {seq.id : seq for seq in SeqIO.parse("data/chr4_human/HSA.chr4.fasta", "fasta")}
>>> monomers_as_features = BED_file_to_features("data/chr4_human/monomers.bed")
>>> phylo_res = clustering_to_phylogeny(items_as_seq_features=monomers_as_features, seq_references=references)
>>> phyloXml = Phyloxml(phylogenies=[phylo_res.phylogeny], attributes=None)
>>> PhyloXMLIO.write(phyloXml, 'data/chr4_human/monomer_phylogeny.xml')

