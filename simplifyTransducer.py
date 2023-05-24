from frtt import FunctionalRealtimeTransducer
from collections import defaultdict

def simplify_transducer(transcoder: FunctionalRealtimeTransducer):
    transition_map = transcoder.transitions.copy()
    initial_state = transcoder.initial_state

    state_map = defaultdict(list)
    for id_state, trans in transition_map.items():
        state_map[str(trans)].append(id_state)

    back_links = defaultdict(list)
    for id_state, trans in transition_map.items():
        for next_state, _ in trans.values():
            if next_state != id_state:
                back_links[next_state].append(id_state)

    while(True):
        simplifiable_pairs = [
            (trans_str,state_list)
            for trans_str,state_list in state_map.items()
            if len(state_list) > 1
        ]
        if len(simplifiable_pairs) == 0:
            break
        curr_trans_str, curr_state_list = simplifiable_pairs[0]
        remaining_state = curr_state_list[0]
        if initial_state in curr_state_list:
            initial_state = remaining_state
        for removed_state in curr_state_list[1:]:
            transition_map.pop(removed_state)
            for state_leading_to_removed in back_links[removed_state]:
                if state_leading_to_removed in transition_map:
                    old_trans_str = str(transition_map[state_leading_to_removed])
                    transition_map[state_leading_to_removed] = {
                        input: ((remaining_state,output) if next_state == removed_state else (next_state,output))
                        for (input,(next_state,output)) in transition_map[state_leading_to_removed].items()
                    }
                    new_trans_str = str(transition_map[state_leading_to_removed])
                    if old_trans_str in state_map: 
                        state_map.pop(old_trans_str)
                    if (new_trans_str not in state_map or
                        state_leading_to_removed not in state_map[new_trans_str]):
                        state_map[new_trans_str].append(state_leading_to_removed)
            back_links.pop(removed_state)
        state_map[curr_trans_str] = [remaining_state]

    return FunctionalRealtimeTransducer(
        transition_map,
        initial_state = initial_state,
        input_alphabet = transcoder.input_alphabet,
        output_alphabet = transcoder.output_alphabet,
        epsilon = transcoder.epsilon)