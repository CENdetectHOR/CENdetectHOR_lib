import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from pyhmmer import plan7
from sklearn.preprocessing import normalize

defaultTransProb = np.array([[
    9.7468591e-01, 1.2656685e-02, 1.2656685e-02, 2.3076856e-01,
    7.6923406e-01, 3.3333409e-01, 6.6666341e-01]])

defaultInsertEmission = np.full((1, 4), 0.25)

def toHMM(phmm: plan7.HMM,
          edgeTransProb = defaultTransProb,
          edgeInsertEmission = defaultInsertEmission) -> hmm.CategoricalHMM:
    """Convert a pHMM in HMMER format to a cyclic HMM in hmmlearn
    
    Convert a Profile Hidden Markov Model represented in HMMER Plan7 format
    to a standard categorical HMM.
    Match and inserttion states from pHMM are converted to corresponding nodes of the output HMM.
    Deletion states in pHMM are 'silent' (do not emit) so cannot be copied to the output HMM.
    They are emulated in the HMM by collapsing them as transitions between arbitrary match states, 
    with corresponding probabilities.
    The pHMM is interpreted as cyclic, considering that the final states (M_n, D_n, I_n) connect
    back to the first ones (M_1, D_1). Rather than reusing existing edge transition probabilites
    (either the ones from begin to M_1, D_1 or from M_n, D_n, I_n to end) and edge insert emitions
    (from I_n), novel values are provided as parameters (edgeTransProb, edgeInsertEmission). 
    """
    matchEmissionProb = np.delete(phmm.match_emissions, 0, axis=0)
    insertEmissionProb = np.concatenate((
        np.delete(phmm.insert_emissions, [0, phmm.M], axis=0),
        edgeInsertEmission))
    baseTransProb = np.concatenate((
        np.delete(phmm.transition_probabilities, [0, phmm.M], axis=0),
        edgeTransProb))
    (m2m, m2i, m2d, i2m, i2i, d2m, d2d) = tuple(np.transpose(baseTransProb))
    dSilentClosure = 1 / (1 - np.prod(d2d))
    m2m_new = np.roll(np.diag(m2m), 1, axis=1)
    for i in range(phmm.M):
        for j in range(phmm.M):
            dTransProb = np.prod(d2d[i + 1 : j]) if j > i + 1 else np.prod(d2d[:j]) * np.prod(d2d[i + 1:])
            m2m_new[i,j] += m2d[i] * dTransProb * dSilentClosure * d2m[j-1]
    hmModel = hmm.CategoricalHMM(
        n_components=phmm.M * 2,
        n_features=4,
        init_params='')
    hmModel.transmat_ = np.block([
        [m2m_new, np.diag(m2i)],
        [np.roll(np.diag(i2m), 1, axis=1), np.roll(np.diag(i2i), 1, axis=1)]])
    hmModel.startprob_ = hmModel.transmat_[:phmm.M].mean(0)
    hmModel.emissionprob_ = np.concatenate((matchEmissionProb, insertEmissionProb))
    hmModel.transmat_ = normalize(hmModel.transmat_, axis=1, norm='l1')
    hmModel.emissionprob_ = normalize(hmModel.emissionprob_, axis=1, norm='l1')
    hmModel.startprob_ = np.transpose(normalize(hmModel.startprob_[:,np.newaxis], axis=0, norm='l1'))[0]
    return hmModel

