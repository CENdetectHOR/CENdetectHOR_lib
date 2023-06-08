from collections import defaultdict
import numpy as np
from sklearn.preprocessing import normalize
from smm import StochasticMooreMachine
from frtt import FunctionalRealtimeTransducer
from simplifyTransducer import simplify_transducer
from reverseTransducer import reverse_transducer

EPSILON = -1

def normalize_trans_simple(trans_row):
    return [
        (prob, trans_row[pos][1])
        for pos,prob in enumerate(
            normalize([
                [prob for prob,new_state in trans_row]
            ], norm='l1')[0]
        )
    ]

def normalize_trans(trans_row):
    if len(trans_row) == 0:
        return []
    return [
        (prob, trans_row[pos][1], trans_row[pos][2])
        for pos,prob in enumerate(
            normalize([
                [prob for prob,new_state,output in trans_row]
            ], norm='l1')[0]
        )
    ]

def smm_to_autoencoder(
        smm: StochasticMooreMachine,
        cutoff_prob=0.0,
        base_prob=None,
        multi_symbol_encoding=True,
        multi_symbol_depth_limit=None) -> tuple[FunctionalRealtimeTransducer,FunctionalRealtimeTransducer]:
    """From a stochastic Moore machine, derive a quasi-optimal encoding
    """
    
    if not base_prob:
        base_prob = cutoff_prob / 2
    go_to_shuffle_prob = base_prob * smm.num_states
    # go_to_shuffle_prob = cutoff_prob

    # Now supporting only binary encoding, to be possibly extended in the future
    encoding_alphabet_size = 2

    deterministic_trans = {}
    num_output_states = smm.num_states

    def get_new_state():
        nonlocal num_output_states
        num_output_states += 1
        return num_output_states - 1

    shuffle_state = None

    def get_shuffle_state():
        nonlocal shuffle_state
        if shuffle_state:
            return shuffle_state
        else:
            shuffle_state = get_new_state()
            return shuffle_state

    def trim_zero_probs(trans_from_state):
        return [
            (prob, new_state)
            for prob, new_state in trans_from_state
            if prob > 0.0
        ]
    
    states_leading_to_shuffle_state = []

    def trim_by_cutoff(state, trans_from_state):
        if cutoff_prob==0.0 or all(prob > cutoff_prob for prob, new_state in trans_from_state):
            return trans_from_state
        else:
            shuffle_state = get_shuffle_state()
            states_leading_to_shuffle_state.append(state)
            return [
                (prob * (1 - go_to_shuffle_prob), new_state)
                for prob, new_state in normalize_trans_simple([
                    # (prob - base_prob, new_state)
                    (prob, new_state)
                    for prob, new_state in trans_from_state
                    if prob > cutoff_prob
                ])
            ] + [(go_to_shuffle_prob, shuffle_state)]
        
    def add_trans_output(trans_from_state):
        return [
            (prob, new_state, [new_state])
            for prob, new_state in trans_from_state
        ]


    stochastic_trans = {
        input_state:
            add_trans_output(sorted(
                trim_by_cutoff(input_state, trim_zero_probs([
                    (prob, new_state) for new_state, prob in enumerate(row)
                ])),
            reverse=True)) if input_state not in smm.final_states else []
        for input_state, row in enumerate(smm.trans_prob)
    }
    stochastic_state_reaching_prob = {i:1.0 for i in range(smm.num_states)}
    stochastic_state_algo_depth = {i:0 for i in range(smm.num_states)}

    if shuffle_state:
        stochastic_trans[shuffle_state] = add_trans_output([
            (1.0 / smm.num_states, go_to_state) for go_to_state in range(smm.num_states)
        ])
        stochastic_state_reaching_prob[shuffle_state] = go_to_shuffle_prob
        stochastic_state_algo_depth[shuffle_state] = 0

    curr_state = 0
    target_prob = 1.0 / encoding_alphabet_size

    while(True):
        if len(stochastic_trans[curr_state]) == 0:
            deterministic_trans[curr_state] = {}
        elif len(stochastic_trans[curr_state]) == 1:
            deterministic_trans[curr_state] = {
                EPSILON: (
                    stochastic_trans[curr_state][0][1],
                    stochastic_trans[curr_state][0][2]
                )
            }
        else:
            
            if (multi_symbol_encoding and
                (not multi_symbol_depth_limit or
                 stochastic_state_algo_depth[curr_state] < multi_symbol_depth_limit) and
                stochastic_trans[curr_state][0][0] > target_prob):
                curr_prob = 1.0
                curr_diff = 1.0
                num_grouped_rows = 0
                state_path = [curr_state]
                output_seq = []
                curr_ahead_state = curr_state
                incremental_prob_seq = []
                prob_seq = []
                while(True):
                    if len(stochastic_trans[curr_ahead_state]) == 0:
                        break
                    prob,next_state,curr_output = stochastic_trans[curr_ahead_state][0]
                    new_diff = abs(curr_prob * prob - target_prob)
                    # print("new_diff : " + str(new_diff))
                    # if (new_diff > curr_diff or
                    #     curr_prob * prob * stochastic_state_reaching_prob[curr_state] <= cutoff_prob):
                    #     break
                    if (new_diff > curr_diff):
                        break
                    curr_prob *= prob
                    incremental_prob_seq.append(curr_prob)
                    prob_seq.append(prob)
                    curr_diff = new_diff
                    state_path.append(next_state)
                    output_seq.append(curr_output)
                    curr_ahead_state = next_state

                state_zero_num = curr_ahead_state
                state_zero_output = [output_symbol for output in output_seq for output_symbol in output]

                # if len(state_path) < 2:
                    # print("Short state_path (" + str(state_path) + ") in state " + str(curr_state) + " with stochastic_trans " + str(stochastic_trans[curr_state]))

                back_prob = 1.0
                next_new_state = None
                for back_path_pos,curr_ahead_state in enumerate(state_path[-2::-1]):
                    output_leading_to = output_seq[-2-back_path_pos] if back_path_pos < len(output_seq) - 1  else []
                    if back_prob == 1.0:

                        if len(stochastic_trans[curr_ahead_state]) == 2:
                            new_state = stochastic_trans[curr_ahead_state][1][1]
                            new_output = output_leading_to + stochastic_trans[curr_ahead_state][1][2]
                        else:
                            new_state = get_new_state()
                            new_output = output_seq[-2-back_path_pos] if back_path_pos < len(output_seq) - 1  else None
                            new_output = output_leading_to
                            stochastic_trans[new_state] = normalize_trans(stochastic_trans[curr_ahead_state][1:])
                            stochastic_state_reaching_prob[new_state] = incremental_prob_seq[-2-back_path_pos] if back_path_pos < len(incremental_prob_seq) - 1 else 1.0
                            stochastic_state_algo_depth[new_state] = stochastic_state_algo_depth[curr_state] + 1
                    else:

                        new_state = get_new_state()
                        new_output = output_leading_to
                        stochastic_trans[new_state] = sorted(normalize_trans([
                            ((prob * (1 - back_prob),next_new_state,next_new_output) if pos == 0 else (prob,next_state,next_output))
                            for pos,(prob, next_state, next_output) in enumerate(stochastic_trans[curr_ahead_state])
                        ]), reverse=True)
                        stochastic_state_reaching_prob[new_state] = (incremental_prob_seq[-2-back_path_pos] if back_path_pos < len(incremental_prob_seq) - 1 else 1.0) * (1 - prob_seq[-1-back_path_pos])
                        stochastic_state_algo_depth[new_state] = stochastic_state_algo_depth[curr_state] + 1

                    next_new_state = new_state
                    next_new_output = new_output
                    back_prob *= stochastic_trans[curr_ahead_state][0][0]
                state_one_num = next_new_state
                state_one_output = next_new_output
            else:
                curr_prob = 0.0
                curr_diff = 1.0
                num_grouped_rows = 0
                for prob,next_state,output in stochastic_trans[curr_state]:
                    new_diff = abs(curr_prob + prob - target_prob)
                    if new_diff >= curr_diff:
                        break
                    curr_prob += prob
                    curr_diff = new_diff
                    num_grouped_rows += 1
                if num_grouped_rows == 1:
                    state_zero_num = stochastic_trans[curr_state][0][1]
                    state_zero_output = stochastic_trans[curr_state][0][2]
                else:
                    state_zero_num = get_new_state()
                    stochastic_trans[state_zero_num] = normalize_trans(stochastic_trans[curr_state][:num_grouped_rows])
                    stochastic_state_reaching_prob[state_zero_num] = stochastic_state_reaching_prob[curr_state] * curr_prob
                    stochastic_state_algo_depth[state_zero_num] = stochastic_state_algo_depth[curr_state]
                    state_zero_output = []
                if num_grouped_rows == len(stochastic_trans[curr_state]) - 1:
                    state_one_num = stochastic_trans[curr_state][num_grouped_rows][1]
                    state_one_output = stochastic_trans[curr_state][num_grouped_rows][2]
                else:
                    state_one_num = get_new_state()
                    stochastic_trans[state_one_num] = normalize_trans(stochastic_trans[curr_state][num_grouped_rows:])
                    stochastic_state_reaching_prob[state_one_num] = stochastic_state_reaching_prob[curr_state] * (1 - curr_prob)
                    stochastic_state_algo_depth[state_one_num] = stochastic_state_algo_depth[curr_state]
                    state_one_output = []

            deterministic_trans[curr_state] = {0:(state_zero_num,state_zero_output), 1:(state_one_num,state_one_output)}

        curr_state += 1
        if (curr_state >= num_output_states):
            break

    decoder = simplify_transducer(
        FunctionalRealtimeTransducer(
            deterministic_trans,
            initial_state = 0,
            input_alphabet = range(encoding_alphabet_size),
            output_alphabet = range(smm.num_states),
            epsilon = EPSILON
        )
    )

    encoder = simplify_transducer(reverse_transducer(decoder))

    return encoder,decoder
