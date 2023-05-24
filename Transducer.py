import numpy as np
import numpy.typing as npt
from typing import Callable

class Tansducer:
    def __init__(self, transitionTable: npt.ArrayLike, numOutputs: int, outputTable: npt.ArrayLike):
        self.transitionTable = np.asarray(transitionTable)
        self.outputTable = np.asarray(outputTable)
        self.numOutputs = numOutputs
        self.numInputs = transitionTable.shape[0]

    def transcode(input: npt.ArrayLike):
        
