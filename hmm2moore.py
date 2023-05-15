import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import normalize
from stochasticmoore import StochasticMooreMachine

def to_moore(input_hmm: hmm.CategoricalHMM, exit_prob) -> StochasticMooreMachine:
    """Convert a CategoricalHMM to a stochastic Moore machine
    
    Convert a categorical hidden Markov model, represented in hmmlearn format, to a stochastic
    Moore machine, i.e. a machine in which, while the transitions remain stochastic, the output
    is a deterministic function of the state.
    """
    # the alphabet includes starting symbol 0 and termination symbol n
#    alphabet_size = hmModel.n_features + 2
#    trans_prob = np.array([hmModel.transmat_, alphabet_size])

    # one state for each input state/output combination, plus initial and final states
    n_states = input_hmm.n_components * input_hmm.n_features + 2
    hmm_state_to_intermediate_moore_state = (
        np.repeat(input_hmm.transmat_, input_hmm.n_features, axis=1)
        * np.reshape(input_hmm.emissionprob_, input_hmm.emissionprob_.size)
        * (1 - exit_prob))
    intermediate_state_to_intermediate_state = (
        np.repeat(hmm_state_to_intermediate_moore_state, input_hmm.n_features, axis=0))
    initial_state_to_intermediate_state = (
        np.repeat(input_hmm.startprob_, input_hmm.n_features)
        * np.reshape(input_hmm.emissionprob_,input_hmm.emissionprob_.size))
    non_final_state_to_initial_state = np.reshape(np.zeros(n_states - 1),(n_states - 1,1))
    non_final_state_to_intermediate_state = np.vstack((
        initial_state_to_intermediate_state,
        intermediate_state_to_intermediate_state)),
    non_final_state_to_final_state = (
        np.reshape(np.ones(n_states - 1),(n_states - 1,1))
        * exit_prob)
    final_state_to_any_state = np.hstack((np.zeros(n_states - 1), np.ones(1)))
    any_state_to_any_state = np.vstack((
        np.hstack((
            non_final_state_to_initial_state,
            non_final_state_to_intermediate_state,
            non_final_state_to_final_state)),
        final_state_to_any_state))

    output_function = lambda s: None if s == 0 or s == n_states - 1 else (s - 1) % input_hmm.n_features
    return StochasticMooreMachine(any_state_to_any_state, input_hmm.n_features, output_function)

