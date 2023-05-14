import numpy as np
from sklearn.preprocessing import normalize
from stochasticmoore import StochasticMooreMachine
from frtt import FunctionalRealtimeTransducer

def normalize_trans(trans_row):
    return [(prob,trans_row[pos][1]) for pos,prob in enumerate(normalize([[prob for prob,new_state in trans_row]], norm='l1')[0])]

def toDecoder(smm: StochasticMooreMachine, cutoff_prob=None) -> FunctionalRealtimeTransducer:
    """From a stochastic Moore machine, derive a quasi-optimal encoding
    """

    input_trans = {input_state:sorted([(prob, new_state) for new_state, prob in enumerate(row)], reverse=True) for input_state, row in enumerate(smm.transitionMatrix)}
    output_trans = {}
    num_states = np.shape(smm.transitionMatrix)[0]
    #deterministic_states = num_states
    curr_state = 0
    target = 0.5

    def get_new_state():
        num_states += 1
        return num_states - 1

    while(True):
        # print("Curr state: " + curr_state)

        if (input_trans[curr_state][0][0] > target):
            curr_prob = 1.0
            curr_diff = 1.0
            num_grouped_rows = 0
            state_path = [curr_state]
            curr_ahead_state = curr_state
            while(True):
                prob,next_state = input_trans[curr_ahead_state][0]
                new_diff = abs(curr_prob * prob - target)
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
            for curr_ahead_state in state_path[:-1:-1]:
                new_state = get_new_state()
                if back_prob == 1.0:
                    if len(input_trans[curr_ahead_state]) == 2:
                        new_state = input_trans[curr_ahead_state][1][1]
                    else:
                        new_state = get_new_state()
                        output_trans[new_state] = normalize_trans(input_trans[curr_ahead_state][1:])
                else:
                    new_state = get_new_state()
#                    output_trans[new_state] = normalize_trans(input_trans[curr_state][0] input_trans[curr_state][1:])
                    output_trans[new_state] = [(prob * (1 - back_prob),next_new_state if pos == 0 else prob,next_state) for pos,(prob, next_state) in enumerate(input_trans[curr_ahead_state])]
                next_new_state = new_state
                back_prob *= input_trans[curr_ahead_state][0][0]
            state_one_num = next_new_state
            state_one_output = []

        else:
            curr_prob = 0.0
            curr_diff = 1.0
            num_grouped_rows = 0
            for prob,next_state in input_trans[curr_state]:
                new_diff = abs(curr_prob + prob - target)
                if new_diff >= curr_diff:
                    break
                curr_prob += prob
                curr_diff = new_diff
                num_grouped_rows += 1
            if num_grouped_rows == 1:
                state_zero_num = input_trans[curr_state][0][1]
                state_zero_output = [state_zero_num]
            else:
                state_zero_num = get_new_state()
                output_trans[state_zero_num] = normalize_trans(input_trans[curr_state][:num_grouped_rows])
                state_zero_output = []
            if num_grouped_rows == len(input_trans[curr_state]) - 1:
                state_one_num = input_trans[curr_state][num_grouped_rows][1]
                state_one_output = [state_one_num]
            else:
                state_one_num = get_new_state()
                output_trans[state_one_num] = normalize_trans(input_trans[curr_state][num_grouped_rows:])
                state_one_output = []

        output_trans[curr_state] = {0:(state_zero_num,state_zero_output), 1:(state_one_num,state_one_output)}

        curr_state += 1
        if (curr_state >= num_states):
            break
    return output_trans
