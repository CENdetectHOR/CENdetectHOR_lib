import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm

def transMatrix(num_states, prob_ahead):
    if (num_states == 1):
        return np.diag(np.ones(num_states))
    aheadMatrix = np.roll(np.diag(np.ones(num_states)),1,axis=1)
    return aheadMatrix * prob_ahead + (1 - aheadMatrix) * ((1 - prob_ahead) / (num_states - 1))

def guidedModel(num_states, trainSeq, random_state=None):
    model = hmm.CategoricalHMM(
        n_components=num_states,
        transmat_prior=transMatrix(num_states, 0.9),
        emissionprob_prior=np.ones(4)*0.25,
        startprob_prior=np.append([1.], np.zeros(num_states - 1)),
        random_state=random_state)
    model.fit(trainSeq)
#    return model
    return trimModel(model)

def unguidedModel(num_states, trainSeq, random_state=None):
    model = hmm.CategoricalHMM(
        n_components=num_states,
        startprob_prior=np.append([1.], np.zeros(num_states - 1)),
        random_state=random_state)
    model.fit(trainSeq)
    return trimModel(model)

def evaluatedModel(num_states, trainSeq, random_state=None):
    model = unguidedModel(num_states, trainSeq, random_state=random_state)
    return (model, model.score(trainSeq))

def modelSet(num_states, trainSeq, num_models=1):
    return sorted(map(
        lambda model: (model.score(trainSeq), model),
        map(
            lambda int_seed: unguidedModel(num_states, trainSeq, random_state=int_seed),
            range(num_models))), key=lambda scoreAndModel: scoreAndModel[0],reverse=True)
    # return list(map(
    #     lambda model: (model.score(trainSeq), model),
    #     map(
    #         lambda int_seed: unguidedModel(num_states, trainSeq, random_state=int_seed),
    #         range(num_models)))) #.sort()

def guidedModelSet(num_states, trainSeq, num_models=1):
    return sorted(map(
        lambda model: (model.score(trainSeq), model),
        map(
            lambda int_seed: guidedModel(num_states, trainSeq, random_state=int_seed),
            range(num_models))), key=lambda scoreAndModel: scoreAndModel[0],reverse=True)

def trimModel(model):
    unreachableStates = np.nonzero(np.sum(model.emissionprob_,axis=1) == 0)
    numUnreachableStates = np.size(unreachableStates)
    if numUnreachableStates == 0:
        return model
    newModel = hmm.CategoricalHMM(model.n_components - numUnreachableStates)
    newModel.startprob_ = np.delete(model.startprob_, unreachableStates)
    newModel.emissionprob_ = np.delete(model.emissionprob_, unreachableStates, axis=0)
    newModel.transmat_ = np.delete(np.delete(model.transmat_, unreachableStates, axis=0), unreachableStates, axis=1)
    return newModel