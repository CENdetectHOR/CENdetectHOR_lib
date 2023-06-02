# -*- coding: utf-8 -*-

class FunctionalRealtimeTransducer(object):
    """Functional (completely specified deterministic) real-time transducer
    """

    def __init__(self, transitions, input_alphabet = None, output_alphabet = None, epsilon = None, initial_state = None):
        """
        states: Set of state identifiers
        input_alphabet: Alphabet for input
        output_alphabet: Alphabet for output
        transitions: Transition Table
        epsilon: Special symbol used for transition without reading input
        """

        self.states = frozenset(transitions.keys())
        self.initial_state = initial_state if initial_state is not None else min(self.states)
        self.input_alphabet = frozenset(input_alphabet or [
            input_symbol
            for trans in transitions.values()
            for input_symbol in trans.keys()
            if epsilon is None or input_symbol != epsilon
        ])
        self.output_alphabet = frozenset(output_alphabet or [
            output_symbol
            for trans in transitions.values()
            for _,output in trans.values()
            for output_symbol in output
        ])
        self.transitions = transitions
        self.epsilon = epsilon
    
    def epsilon_closure(self, state_and_output=(0,[]), trace=None):
        state,output = state_and_output
        silents_state_seq = []
        while(self.epsilon in self.transitions[state]):
            state,new_output = self.transitions[state][self.epsilon]
            if len(new_output) == 0:
                if state in silents_state_seq:
                    break
                silents_state_seq.append(state)
            else:
                silents_state_seq = []
                output = output + new_output
            state_seq = state_seq + [state]
            if trace is not None:
                trace = trace + [(state,new_output)]
        return state,output,trace

    def transition(self, state_and_output, input, trace=None):
        state,output = state_and_output
        if input not in self.transitions[state]:
            raise Exception("Unexpected input " + str(input) + " in state " + str(state))
        state,new_output = self.transitions[state][input]
        output = output + new_output
        if trace is not None:
            trace = trace + [(state,new_output)]
        state,output,trace = self.epsilon_closure((state,output), trace=trace)
        return state,output,trace

    def transcode(self, input, show_trace=False):
        """Return transducer's output when a given list (or string) is given as input"""
        #temp_list = list(input)
        current_state,output,trace = self.epsilon_closure(trace=[] if show_trace else None)
        for x in input:
            current_state,output,trace = self.transition((current_state,output),x,trace=trace)
        if show_trace:
            return output,trace
        return output
    
    def get_execution(self):
        return FrttExecution(self)

    def __str__(self):
        """"Pretty Print the Transducer"""

        output = "\nFunctional (completely specified deterministic) real-time transducer" + \
                 "\nNum States " + str(len(self.states)) + \
                 "\nInput Alphabet " + str(self.input_alphabet) + \
                 "\nOutput Alphabet " + str(self.output_alphabet) + \
                 "\nTransitions " + str(self.transitions)

        return output
    
class FrttExecution(object):
    def __init__(self, transducer: FunctionalRealtimeTransducer):
        self.transducer = transducer
        self.curr_state = transducer.initial_state
        self.curr_state, self.output_cache, _ = self.transducer.epsilon_closure(
            (self.curr_state, []))

    def write_symbol(self, input_symbol):
        self.curr_state, self.output_cache, _ = self.transducer.transition(
            (self.curr_state, self.output_cache), input_symbol)

    def write(self, input):
        for input_symbol in input:
            self.write_symbol(input_symbol)

    def read(self, input = []):
        self.write(input)
        output = self.output_cache
        self.output_cache = []
        return output
    
"""
transducer = FunctionalRealtimeTransducer(
    4, ['a' , 'b'], ['c', 'd'],
    {
        0 : {
            'a' : (1, ['c','c'])
            'b' : (3, ['d'])

        },
        1: {
            'a': (3, ['c','d','c','d']),
            'b': (1, ['c'])
        },
        2: {
            'a': (0, []),
            'b': (3, ['d','d','d'])
        },
        3: {
            'a': (3, ['d','c','d']),
            'b': (2, ['c','c','c','c','c'])
        }
    }
)
print(transducer)
print(transducer.transcode('abbba'))

"""