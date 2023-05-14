import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import normalize
from stochasticmoore import StochasticMooreMachine

def to_moore(input_hmm: hmm.CategoricalHMM, exit_prob) -> StochasticMooreMachine:
    """Convert a CategoricalHMM to a probabilistic automaton
    
    Convert a categorical hidden Markov model, represented in hmmlearn format, to a probabilistic
    automata.
    """
    # the alphabet includes starting symbol 0 and termination symbol n
#    alphabet_size = hmModel.n_features + 2
#    trans_prob = np.array([hmModel.transmat_, alphabet_size])

    # one state for each input node/output combination, plus initial and final states
    n_states = input_hmm.n_components * input_hmm.n_features + 2
    firstProd = (
        np.repeat(input_hmm.transmat_, input_hmm.n_features, axis=1)
        * np.reshape(input_hmm.emissionprob_, input_hmm.emissionprob_.size)
        * (1 - exit_prob))
    secondProd = np.repeat(firstProd, input_hmm.n_features, axis=0)
    fromInitialStateTrans = np.repeat(input_hmm.startprob_, input_hmm.n_features) * np.reshape(input_hmm.emissionprob_,input_hmm.emissionprob_.size)
    toInitialStateTrans = np.reshape(np.zeros(n_states - 1),(n_states - 1,1))
    toFinalStateTrans = np.reshape(np.ones(n_states - 1),(n_states - 1,1)) * exit_prob
    fromfinalStateTrans = np.hstack((np.zeros(n_states - 1), np.ones(1)))
    finalTrans = np.vstack((
        np.hstack((
            toInitialStateTrans,
            np.vstack((fromInitialStateTrans, secondProd)),
            toFinalStateTrans)),
        fromfinalStateTrans))

    outputFunction = lambda s: None if s == 0 or s == n_states - 1 else (s - 1) % input_hmm.n_features
    return StochasticMooreMachine(finalTrans, input_hmm.n_features, outputFunction)

