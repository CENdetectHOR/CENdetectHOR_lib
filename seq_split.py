import numpy as np
from hmmlearn import hmm

def split_by_positions(seq, positions):
    return (
        seq[0:positions[0]],
        [
            seq[position : positions[order + 1]]
            for order,position in enumerate(positions[:-1])
        ],
        seq[positions[-1]:]
    )

def seq_split_by_hmm_cycles(seq, input_hmm: hmm.CategoricalHMM):
    """Split a sequence of symbols according to cycles in a HMM"""

    (log_prob, state_sequence) = input_hmm.decode(seq.reshape(1, -1))
    state_bins = np.bincount(state_sequence)
    main_cycle_states = (state_bins == np.argmax(np.bincount(state_bins))).nonzero()[0]

    for start_cycle_state in state_sequence:
        if start_cycle_state in main_cycle_states:
            break

    start_cycle_positions = [
        pos
        for pos,state in enumerate(state_sequence)
        if state == start_cycle_state
    ]
    
    state_blocks = split_by_positions(state_sequence, start_cycle_positions)
    seq_blocks = split_by_positions(seq, start_cycle_positions)

    return seq_blocks, state_blocks
