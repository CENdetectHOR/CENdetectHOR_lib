from frtt import FunctionalRealtimeTransducer
from collections import defaultdict

def commonSubList(list1, list2):
    common = []
    for index,item in enumerate(list1):
        if list2[index] != item:
            break
        common.append(item)
    return common

def reverse_transducer(transducer: FunctionalRealtimeTransducer):
    input_states_to_reverse_states = {}
    reverse_transitions = {}

    num_reverse_states = 0
    def get_new_state():
        nonlocal num_reverse_states
        new_state = num_reverse_states
        reverse_transitions[new_state] = {}
        num_reverse_states += 1
        return new_state

    def possible_outputs(input_state):
        output_symbols_transitions = defaultdict(list)
        for input, (next_state, output) in transducer.transitions[input_state].items():
            if input == transducer.epsilon:
                continue
            if len(output) > 0:
                output_symbols_transitions[output[0]].append((output[1:], next_state, [input]))
            else:
                for next_output_symbol, next_transitions in (possible_outputs(next_state).items()):
                    output_symbols_transitions[next_output_symbol].extend([
                        (remaining_output, next_next_state, [input] + next_input)
                        for remaining_output, next_next_state, next_input in next_transitions])
        return output_symbols_transitions
    
    def convert_from_transitions(transitions):
        if all([len(output) == 0 for (output, _, _) in transitions]):
            return (convert_from_input_state(transitions[0][1]), transitions[0][2])
            
        output_symbols_transitions = defaultdict(list)
        for (output, next_state, input) in transitions:
            if len(output) > 0:
                output_symbols_transitions[output[0]].append((output[1:], next_state, input))
            else:
                for next_output_symbol, next_transitions in possible_outputs(next_state).items():
                    output_symbols_transitions[next_output_symbol].extend([
                        (remaining_output, next_next_state, input + next_input)
                        for remaining_output, next_next_state, next_input in next_transitions])
        common_input = None
        for _, transitions in output_symbols_transitions.items():
            for _, _, input in transitions:
                if common_input is None:
                    common_input = input
                else:
                    common_input = commonSubList(common_input, input)
        new_state = get_new_state()
        for output_symbol, transitions in output_symbols_transitions.items():
            if len(transitions) > 0:
                reverse_transitions[new_state][output_symbol] = convert_from_transitions([
                    (remaining_output, next_state, input[len(common_input):])
                    for remaining_output, next_state, input in transitions
                ])
        
        return (new_state, common_input)
        

    def convert_from_input_state(input_state):
        if input_state in input_states_to_reverse_states:
            return input_states_to_reverse_states[input_state]
        new_state = get_new_state()
        input_states_to_reverse_states[input_state] = new_state
        for output_symbol, transitions in possible_outputs(input_state).items():
            if len(transitions) > 0:
                reverse_transitions[new_state][output_symbol] = convert_from_transitions(transitions)
        return new_state


    convert_from_input_state(transducer.initial_state)
    return FunctionalRealtimeTransducer(
        reverse_transitions,
        input_alphabet=transducer.output_alphabet,
        output_alphabet=transducer.input_alphabet,
        epsilon=transducer.epsilon,
        initial_state=transducer.initial_state)

