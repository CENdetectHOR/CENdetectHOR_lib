import numpy as np
import numpy.typing as npt
from typing import Callable

class StochasticMooreMachine:
    def __init__(self,
                 trans_prob: npt.ArrayLike,
                 output_alphabet_size: int,
                 output_fun: Callable[[int], int],
                 initial_state = 0,
                 final_states = None):
        self.trans_prob = np.asarray(trans_prob)
        self.num_states = self.trans_prob.shape[0]
        self.output_alphabet_size = output_alphabet_size
        self.output_fun = output_fun
        self.initial_state = initial_state
        self.final_states = final_states if final_states is not None else [self.num_states - 1]
