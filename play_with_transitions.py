import pickle
import numpy as np
from ndr.learn import run_main_search as learn_ndrs
from ndr.learn import print_rule_set
from copy import deepcopy
from settings import AgentConfig as ac

with open('bakingrealistic_demonstrations.pkl', 'rb') as f:
    transitions = pickle.load(f)

max_ee_transitions = ac.max_zpk_explain_examples_transitions['Bakingrealistic']

def get_batch_probs():
    assert False, 'assumed off'

init_rule_sets = None
_rand_state = np.random.RandomState(seed=1)

for action_predicate in transitions:
    if action_predicate.name == 'preheat-oven-with-cake-settings':
        dups = []
        for t in transitions[action_predicate]:
            dups.append(deepcopy(t))
            dups.append(deepcopy(t))
            dups.append(deepcopy(t))
            dups.append(deepcopy(t))
            dups.append(deepcopy(t))
            dups.append(deepcopy(t))
            dups.append(deepcopy(t))

        transitions[action_predicate].extend(dups)
rule_set = {}
for action_predicate in transitions:
    if action_predicate.name == 'preheat-oven-with-cake-settings':
        learned_ndrs = learn_ndrs({action_predicate : transitions[action_predicate]},
            max_timeout=ac.max_zpk_learning_time,
            max_action_batch_size=ac.max_zpk_action_batch_size['Bakingrealistic'],
            get_batch_probs=get_batch_probs,
            init_rule_sets=init_rule_sets,
            rng=_rand_state,
            max_ee_transitions=max_ee_transitions,
        )
        rule_set[action_predicate] = learned_ndrs[action_predicate]

print_rule_set(rule_set)