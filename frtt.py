# -*- coding: utf-8 -*-

class FunctionalRealtimeTransducer(object):
    """Functional (completely specified deterministic) real-time transducer
    """

    def __init__(self, num_states, input_alphabet, output_alphabet, transitions):
        """
        num_states: Number of states
        input_alphabet: Alphabet for input
        output_alphabet: Alphabet for output
        transitions: Transition Table
        """

        self.num_states = num_states
        self.input_alphabet = input_alphabet
        self.output_alphabet = output_alphabet
        self.transitions = transitions
    
    def transcode(self, input):
        """Return Moore Machine's output when a given list (or string) is given as input"""
        temp_list = list(input)
        output = []
        current_state = self.initial_state
        output.extend(self.output_table[current_state])
        for x in temp_list:
            current_state = self.transitions[current_state][x]
            output.extend(self.output_table[current_state])

        return output

    def __str__(self):
        """"Pretty Print the Transducer"""

        output = "\nFunctional (completely specified deterministic) real-time transducer" + \
                 "\nNum States " + str(self.num_states) + \
                 "\nInput Alphabet " + str(self.input_alphabet) + \
                 "\nOutput Alphabet " + str(self.output_alphabet) + \
                 "\nTransitions " + str(self.transitions)

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