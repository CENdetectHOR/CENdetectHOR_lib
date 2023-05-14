import numpy as np
import numpy.typing as npt
from typing import Callable

class StochasticMooreMachine:
    def __init__(self, transitionMatrix: npt.ArrayLike, numOutputs: int, outputFunction: Callable[[int], int]):
        self.transitionMatrix = np.asarray(transitionMatrix)
        self.numOutputs = numOutputs
        self.outputFunction = outputFunction
