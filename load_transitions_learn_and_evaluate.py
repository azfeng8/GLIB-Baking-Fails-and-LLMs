import pickle
import numpy as np
from ndr.learn import run_main_search as learn_ndrs
from ndr.learn import print_rule_set
from copy import deepcopy
from settings import AgentConfig as ac
from pddlgym.structs import LiteralConjunction
from pddlgym.parser import Operator
import pddlgym
import gym


with open('transitions.pkl', 'rb') as f:
    transitions = pickle.load(f)


# learn NDRs
max_ee_transitions = ac.max_zpk_explain_examples_transitions['Bakingrealistic']

def get_batch_probs():
    assert False, 'assumed off'

init_rule_sets = None
_rand_state = np.random.RandomState(seed=1)


rule_set = {}
for action_predicate in transitions:
    learned_ndrs = learn_ndrs({action_predicate : transitions[action_predicate]},
        max_timeout=ac.max_zpk_learning_time,
        max_action_batch_size=ac.max_zpk_action_batch_size['Bakingrealistic'],
        get_batch_probs=get_batch_probs,
        init_rule_sets=init_rule_sets,
        rng=_rand_state,
        max_ee_transitions=max_ee_transitions,
    )
    rule_set[action_predicate] = learned_ndrs[action_predicate]

# print_rule_set(rule_set)
print("Loaded NDRs")
# NDRs to operators
from ndr.ndrs import NOISE_OUTCOME
ops = []
for act_pred in rule_set:
    ndrset = rule_set[act_pred]
    suffix = 0
    for ndr in ndrset.ndrs:
        op_name = "{}{}".format(ndr.action.predicate.name, suffix)
        probs, effs = ndr.effect_probs, ndr.effects
        max_idx = np.argmax(probs)
        max_effects = LiteralConjunction(sorted(effs[max_idx]))
        preconds = LiteralConjunction(sorted(ndr.preconditions) + [ndr.action])
        params = set()
        for lit in preconds.literals + max_effects.literals:
            for v in lit.variables:
                params.add(v)
        params= sorted(params)
        operator = Operator(op_name, params, preconds, max_effects)
        if len(operator.effects.literals) == 0 or NOISE_OUTCOME in operator.effects.literals:
            continue
        ops.append(operator)
        suffix += 1
with open('ops.pkl', 'wb') as f:
    pickle.dump(ops, f)
    # ops = pickle.load(f)
print("Loaded ops")
# for o in ops:
#     print(o.pddl_str())
# evaluate operators on test tasks.

from settings import AgentConfig as ac

from pddlgym.structs import ground_literal
import sys
import os
import re
import subprocess
import time
from pddlgym.structs import Predicate, Exists, State
from pddlgym.parser import PDDLProblemParser
from settings import AgentConfig as ac

import random
import abc
import os

from planning_modules.base_planner import Planner, PlannerTimeoutException, \
    NoPlanFoundException



BAKING_REALISTIC_TEST_CASES_DESCRIPTIONS = {
    0: "Bake souffle and cake, without damaging pans, putting them on plates.",
    1: "move-baked-good-in-container-to-different-container",
    2: "set-oven-with-souffle-bake-time-and-press-start",
    3: "set-oven-with-cake-bake-time-and-press-start",
    4: "fold-stiff-egg-whites-into-mixture",
    5: "pour-mixture-only",
    6: "use-stand-mixer for cake",
    7: "use-stand-mixer for souffle",
    8: "beat-egg-whites",
    9: "separate-egg-whites",
    10: "transfer-butter-from-pan-or-bowl",
    11: "transfer-egg-from-pan-or-bowl",
    12: "pour-powdery-ingredient-from-container",
    13: "remove-pan-from-oven",
    14: "put-pan-in-oven",
    15: "crack-egg",
    16: "preheat-souffle",
    17: "preheat-cake",
    18: "pour-powdery-ingredient-from-measuring-cup",
    19: "put-butter-in-container-from-measuring-cup",
}
from agent import Agent

domain_name = 'Bakingrealistic'


test_env = pddlgym.make("PDDLEnvBakingrealisticTest-v0")

ac.seed = 10
ac.train_env = pddlgym.make("PDDLEnvBakingrealistic-v0")
ac.planner_timeout = 400
agent = Agent(domain_name, test_env.action_space,
                test_env.observation_space, "GLIB_G1", "LNDR", log_llm_path='',
                planning_module_name=ac.planner_name[domain_name])
        

for o in ops:
    agent._planning_module._learned_operators.add(o)

    agent._planning_module._planning_operators.add(o)

for o in agent._planning_module._learned_operators:
    print(o.pddl_str())

assert len(agent._planning_module._learned_operators) != 0
assert len(agent._planning_module._learned_operators) == len(ops), f"{len(agent._planning_module._learned_operators)} vs. {len(ops)}"

        
num_successes = 0
for i in range(len(test_env.problems))[::-1]:
    test_env.fix_problem_index(i)
    obs, debug_info = test_env.reset()

    
    try:
        policy = agent.get_policy(debug_info["problem_file"], use_learned_ops=True)
    except (NoPlanFoundException,PlannerTimeoutException) as e:
        # Automatic failure
        if isinstance(e, NoPlanFoundException):
            print("No plan found")
        else:
            print("Planner timed out.")
        print("\tTest case {}/{}, FAILED. {} successes so far. {}".format(
        i+1, len(test_env.problems), num_successes, BAKING_REALISTIC_TEST_CASES_DESCRIPTIONS[i]))
        continue

    # Test plan open-loop
    reward = 0.
    for _ in range(28):
        try:
            action = policy(obs)
        except (NoPlanFoundException, PlannerTimeoutException):
            break
        obs, reward, done, _ = test_env.step(action)
        if done:
            break

    # Reward is 1 iff goal is reached
    if reward == 1.:
        num_successes += 1
    else:
        assert reward == 0.

    result_str = "PASSED" if reward == 1. else "FAILED"
    print("\tTest case {}/{}, {}. {} successes so far. {}".format(
    i+1, len(test_env.problems), result_str, num_successes, BAKING_REALISTIC_TEST_CASES_DESCRIPTIONS[i]))