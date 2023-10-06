import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from Bio import Phylo

def show_hor(hor_in_seq, tree=None, seq_length=None,
             label='unnamed',
             color_palette=[mcolors.TABLEAU_COLORS[color_id] for color_id in mcolors.TABLEAU_COLORS],
             seq_fg_color='black',
             seq_bg_color='#D3D3D3',
             tree_bg_color='#D3D3D3',
             hor_item_size=(0.3,0.3),
             seq_item_size=(0.01,0.2),
             monomer_size=107):
    
    spans = hor_in_seq.spans_in_seq

    bp_start = spans[0].span_start * monomer_size
    bp_end = (spans[-1].span_start + spans[-1].span_length) * monomer_size

    print(f'HOR: {label}')
    print(f'Coverage (bp): {bp_start}-{bp_end}')

    if seq_length is None:
        seq_length = tree.count_terminals()

    clade_seq = hor_in_seq.hor.clade_seq
    for clade_index,clade in enumerate(set(clade_seq)):
        clade.color = color_palette[clade_index]

    fig = plt.figure(constrained_layout=True, figsize=(7,5))
    fig.suptitle(f'HOR {label} ({bp_start}-{bp_end})')

    gs = fig.add_gridspec(
        3, 2,
        width_ratios=[0.04*len(clade_seq),1-0.04*len(clade_seq)],
        height_ratios=[1,20,1],
        hspace=0.1 
    )
    ax_hor = plt.subplot(gs[0, 0])
    ax_tree = plt.subplot(gs[1, :])
    ax_seq = plt.subplot(gs[2, :])

    ax_hor.set_xlim([0,len(clade_seq)])
    ax_hor.grid(True)
    ax_hor.set_xticks(np.arange(0, len(clade_seq) + 1, 1))
    ax_hor.get_yaxis().set_visible(False)
    for clade_pos,clade in enumerate(clade_seq):
        ax_hor.add_patch(patches.Rectangle((clade_pos,0),1,1,facecolor=clade.color.to_hex()))

    bp_seq_length = seq_length * monomer_size

    ax_seq.set_xlim([0,bp_seq_length])
    ax_seq.get_yaxis().set_visible(False)

    ax_seq.add_patch(patches.Rectangle((0,0),bp_seq_length,1,facecolor=seq_bg_color))
    for span in spans:
        span_bp_start = span.span_start * monomer_size
        span_bp_length = span.span_length * monomer_size
        ax_seq.add_patch(patches.Rectangle((span_bp_start,0),span_bp_length,1,facecolor=seq_fg_color))

    ax_tree.get_yaxis().set_visible(False)
    ax_tree.get_xaxis().set_visible(False)
    # ax_tree.spines['top'].set_visible(False)
    # ax_tree.spines['right'].set_visible(False)
    # ax_tree.spines['bottom'].set_visible(False)
    # ax_tree.spines['left'].set_visible(False)
    ax_tree.get_yaxis().set_ticks([])
    # ax_tree.axis('off')
    if tree is not None:
        tree.root.color = tree_bg_color
        Phylo.draw(tree, axes=ax_tree)

    for clade_index,clade in enumerate(set(hor_in_seq.hor.clade_seq)):
        clade.color = None
    if tree is not None:
        tree.root.color = None

