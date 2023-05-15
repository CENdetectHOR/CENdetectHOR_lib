import numpy as np
from sklearn.preprocessing import normalize
from stochasticmoore import StochasticMooreMachine
from frtt import FunctionalRealtimeTransducer

def normalize_trans(trans_row):
    return [
        (prob,trans_row[pos][1])
        for pos,prob in enumerate(
            normalize([
                [prob for prob,new_state in trans_row]
            ], norm='l1')[0]
        )
    ]

def to_decoder(smm: StochasticMooreMachine, cutoff_prob=0.0, base_prob=None, multi_symbol_encoding=True) -> FunctionalRealtimeTransducer:
    """From a stochastic Moore machine, derive a quasi-optimal encoding
    """
    
    if not base_prob:
        base_prob = cutoff_prob / 2
    go_to_shuffle_prob = base_prob * smm.num_states

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

    def trim_by_cutoff(trans_from_state):
        if cutoff_prob==0.0 or all(prob > cutoff_prob for prob, new_state in trans_from_state):
            return trans_from_state
        else:
            shuffle_state = get_shuffle_state()
            if len([
                *[
                    (prob * (1 - go_to_shuffle_prob), new_state)
                    for prob, new_state in normalize_trans([
                        (prob - base_prob, new_state)
                        for prob, new_state in trans_from_state
                        if prob > cutoff_prob
                    ])
                ], (go_to_shuffle_prob, shuffle_state)
            ]) < 2:
                print('Short')
            return [
                *[
                    (prob * (1 - go_to_shuffle_prob), new_state)
                    for prob, new_state in normalize_trans([
                        (prob - base_prob, new_state)
                        for prob, new_state in trans_from_state
                        if prob > cutoff_prob
                    ])
                ], (go_to_shuffle_prob, shuffle_state)
            ]

    stochastic_trans = {
        input_state:sorted(
            trim_by_cutoff([(prob, new_state) for new_state, prob in enumerate(row)]),
            reverse=True)
        for input_state, row in enumerate(smm.trans_prob)
    }
    if shuffle_state:
        stochastic_trans[shuffle_state] = [(1.0 / smm.num_states, go_to_state) for go_to_state in range(smm.num_states)]

    curr_state = 0
    target_prob = 1.0 / encoding_alphabet_size

    while(True):
        print(curr_state)
#        print(stochastic_trans[curr_state])
        if (multi_symbol_encoding and stochastic_trans[curr_state][0][0] > target_prob):
            curr_prob = 1.0
            curr_diff = 1.0
            num_grouped_rows = 0
            state_path = [curr_state]
            curr_ahead_state = curr_state
            while(True):
                prob,next_state = stochastic_trans[curr_ahead_state][0]
                new_diff = abs(curr_prob * prob - target_prob)
                if new_diff >= curr_diff:
                    break
                curr_prob *= prob
                curr_diff = new_diff
                state_path.append(next_state)
                curr_ahead_state = next_state

            state_zero_num = curr_ahead_state
            state_zero_output = state_path[1:]

            back_prob = 1.0
            next_new_state = None
            for curr_ahead_state in state_path[1::-1]:
                if back_prob == 1.0:
                    if len(stochastic_trans[curr_ahead_state]) == 2:
                        new_state = stochastic_trans[curr_ahead_state][1][1]
                    else:
                        new_state = get_new_state()
                        stochastic_trans[new_state] = normalize_trans(stochastic_trans[curr_ahead_state][1:])
                else:
                    new_state = get_new_state()
#                    output_trans[new_state] = normalize_trans(input_trans[curr_state][0] input_trans[curr_state][1:])
                    stochastic_trans[new_state] = normalize_trans([
                        ((prob * (1 - back_prob)),next_new_state if pos == 0 else (prob,next_state))
                        for pos,(prob, next_state) in enumerate(stochastic_trans[curr_ahead_state])])
                next_new_state = new_state
                back_prob *= stochastic_trans[curr_ahead_state][0][0]
            state_one_num = next_new_state
            state_one_output = [] if len(state_path) > 2 else [next_new_state]

        else:
            curr_prob = 0.0
            curr_diff = 1.0
            num_grouped_rows = 0
            for prob,next_state in stochastic_trans[curr_state]:
                new_diff = abs(curr_prob + prob - target_prob)
                if new_diff >= curr_diff:
                    break
                curr_prob += prob
                curr_diff = new_diff
                num_grouped_rows += 1
            if num_grouped_rows == 1:
                state_zero_num = stochastic_trans[curr_state][0][1]
                state_zero_output = [state_zero_num]
            else:
                state_zero_num = get_new_state()
                stochastic_trans[state_zero_num] = normalize_trans(stochastic_trans[curr_state][:num_grouped_rows])
                state_zero_output = []
            if num_grouped_rows == len(stochastic_trans[curr_state]) - 1:
                state_one_num = stochastic_trans[curr_state][num_grouped_rows][1]
                state_one_output = [state_one_num]
            else:
                state_one_num = get_new_state()
                stochastic_trans[state_one_num] = normalize_trans(stochastic_trans[curr_state][num_grouped_rows:])
                state_one_output = []

        deterministic_trans[curr_state] = {0:(state_zero_num,state_zero_output), 1:(state_one_num,state_one_output)}

        curr_state += 1
        if (curr_state >= num_output_states):
            break
    return FunctionalRealtimeTransducer(num_output_states, range(encoding_alphabet_size), range(smm.num_states), deterministic_trans)
