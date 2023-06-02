import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import normalize
from smm import StochasticMooreMachine

def to_smm(input_hmm: hmm.CategoricalHMM, exit_prob):
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
        intermediate_state_to_intermediate_state))
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

    initial_state = 0
    final_state = input_hmm.n_components * input_hmm.n_features + 1
    output_function = lambda s: None if s == 0 or s == n_states - 1 else (s - 1) % input_hmm.n_features
    
    def state_encoder(hmm_state, hmm_output):
        if (hmm_state < 0 or
            hmm_state >= input_hmm.n_components):
            raise Exception("HMM state " + str(hmm_state) + " does not exist")
        if (hmm_output < 0 or
            hmm_output >= input_hmm.n_features):
            raise Exception("HMM feature " + str(hmm_state) + " does not exist")
        return hmm_state * input_hmm.n_features + 1 + hmm_output
    
    def state_decoder(moore_state):
        if moore_state == initial_state:
            raise Exception("Found initial state when inner state expected")
        if moore_state == final_state:
            raise Exception("Found final state when inner state expected")
        if (moore_state < initial_state or
            moore_state > final_state):
            raise Exception("Moore state " + str(moore_state) + " does not exist")
        hmm_state = (moore_state - 1) // input_hmm.n_features
        hmm_output = (moore_state - 1) - hmm_state *  input_hmm.n_features
        return (hmm_state, hmm_output)
    
    def seq_encoder(hmm_state_seq, hmm_output_seq, from_start=True, until_end=True):
        if len(hmm_state_seq) != len(hmm_output_seq):
            raise Exception("Length of state sequence (" + str(len(hmm_state_seq)) + ") " +
                            "should be equal to output sequence (" + str(len(hmm_output_seq)) + ") ")
        return (([0] if from_start else []) +
                [state_encoder(hmm_state, hmm_output_seq[pos])
                    for pos, hmm_state in enumerate(hmm_state_seq)] +
                ([input_hmm.n_components * input_hmm.n_features + 1] if until_end else []))

    def seq_decoder(moore_state_seq, from_start=True, until_end=True):
        hmm_state_seq = []
        hmm_output_seq = []
        for pos, moore_state in enumerate(moore_state_seq):
            if pos == 0:
                if moore_state == initial_state:
                    continue
                elif from_start:
                    raise Exception("Initial state missing")
            if moore_state == initial_state:
                raise Exception("Initial state in intermediate position")
            if until_end and pos == len(moore_state_seq) - 1 and moore_state != final_state:
                raise Exception("Final state missing")
            if moore_state == final_state:
                break
            hmm_state, hmm_output = state_decoder(moore_state)
            hmm_state_seq.append(hmm_state)
            hmm_output_seq.append(hmm_output)
        return (hmm_state_seq, hmm_output_seq)
    
    return (StochasticMooreMachine(any_state_to_any_state, input_hmm.n_features, output_function),
            seq_encoder, seq_decoder)

