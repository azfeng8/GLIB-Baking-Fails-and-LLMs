import pddlgym
import gym
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pickle
from ndr.learn import run_main_search as learn_ndrs
from settings import AgentConfig as ac
from ndr.ndrs import NOISE_OUTCOME
from collections import defaultdict
from pddlgym.structs import LiteralConjunction
from pddlgym.parser import Operator
from pddlgym.structs import Predicate, Exists, State
from settings import AgentConfig as ac
from planning_modules.base_planner import Planner, PlannerTimeoutException, \
    NoPlanFoundException
from agent import Agent

# with open('transition_datasets/nondeterministic_learner_for_beateggwhites0_operator.pkl', 'rb') as f:
with open('bakingrealistic_demonstrations.pkl', 'rb') as f:
    dataset = pickle.load(f)

MAX_EE_TRANSITIONS = ac.max_zpk_explain_examples_transitions['Bakingrealistic']

def get_batch_probs():
    assert False, 'assumed off'

def learn_and_print():
    init_rule_sets = None
    _rand_state = np.random.RandomState(seed=1)


    rule_set = {}
    for action_predicate in dataset:
        learned_ndrs = learn_ndrs({action_predicate : dataset[action_predicate]},
            max_timeout=ac.max_zpk_learning_time,
            max_action_batch_size=ac.max_zpk_action_batch_size['Bakingrealistic'],
            get_batch_probs=get_batch_probs,
            init_rule_sets=init_rule_sets,
            rng=_rand_state,
            max_ee_transitions=MAX_EE_TRANSITIONS,
        )
        rule_set[action_predicate] = learned_ndrs[action_predicate]
    operators = []
    for act_pred in rule_set:
        name_suffix = 0
        ndrset = rule_set[act_pred]
        for ndr in ndrset.ndrs:
            op_name = "{}{}".format(ndr.action.predicate.name, name_suffix)
            indices = [i for i, eff in enumerate(ndr.effects) if len(eff) > 0 ]
            effs = ndr.effects
            for idx in indices:
                op_name = "{}{}".format(ndr.action.predicate.name, name_suffix)
                effects = LiteralConjunction(sorted(effs[idx]))
                if len(effects.literals) == 0 or NOISE_OUTCOME in effects.literals:
                    continue
                preconds = LiteralConjunction(sorted(ndr.preconditions) + [ndr.action])
                params = set()
                for lit in preconds.literals + effects.literals:
                    for v in lit.variables:
                        params.add(v)
                params= sorted(params)
                operators.append(Operator(op_name, params, preconds, effects))
                name_suffix += 1

    operators = sorted(operators, key=lambda op: op.name)
    s = ''
    for o in operators:
        s += o.pddl_str()
    print(s)
    return s

results = []
NUM_TRIALS = 1
for _ in range(NUM_TRIALS):
    results.append(learn_and_print())

assert all(results[i] == results[0] for i in range(NUM_TRIALS))