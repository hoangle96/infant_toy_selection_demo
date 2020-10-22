import pomegranate as pom
import numpy as np
from sklearn.utils import check_random_state

def create_independent_dist(feature, seed):
    unique_val = np.unique(feature)
    init_dict = {}
    random_state = check_random_state(seed)
    init_prob = random_state.rand(len(unique_val),1)
    init_prob = init_prob/init_prob.sum()
    
    for idx, i in enumerate(unique_val):
        init_dict[int(i)] = init_prob[idx].item()
    return pom.DiscreteDistribution(init_dict)

def create_dist_for_states(n_states, feature_list, seed):
    distributions = []
    i = 0
    for _ in range(n_states):
        dist_list = []
        for f in feature_list:
            dist_ = create_independent_dist(f, i)
            i += 1
            dist_list.append(dist_)
        distributions.append(pom.IndependentComponentsDistribution(dist_list))
    return distributions

def init_hmm(n_components, feature_list, seed):
    random_state_trans = check_random_state(seed**seed)
    transitions = random_state_trans.rand(n_components, n_components)
    transitions = transitions/transitions.sum()
    
    random_state_start = check_random_state(seed**2)
    starts = random_state_start.rand(n_components)
    starts = starts/starts.sum()
    distributions = create_dist_for_states(n_components, feature_list, seed)
    model = pom.HiddenMarkovModel.from_matrix(transitions, distributions, starts)
    model.bake()
    
    return model