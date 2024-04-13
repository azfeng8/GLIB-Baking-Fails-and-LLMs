"""Unit tests for code sections of the learn() method in the LLMZPKWarmStartOperatorLearningModule.

Also, contains starting implementations that are developed in this file before adding it to the learn() method.
"""

path = '/home/catalan/GLIB-Baking-Fails-and-LLMs/results_openstack/llm_cache'
minecraft_files = [
    '69720a18fa697893bfe1308b000c1148_0_160_1.pkl',
    '69720a18fa697893bfe1308b000c1148_0_169_1.pkl',
    'e26f090f8f380088273db6ae0867935e_0_168_1.pkl',
    'e26f090f8f380088273db6ae0867935e_0_167_1.pkl',
    'e26f090f8f380088273db6ae0867935e_0_166_1.pkl',
    'b7f2f2528d7e3bb51e18f3087df8f087_0_165_1.pkl',
    'b7f2f2528d7e3bb51e18f3087df8f087_0_164_1.pkl',
    '69720a18fa697893bfe1308b000c1148_0_163_1.pkl',
    'b7f2f2528d7e3bb51e18f3087df8f087_0_162_1.pkl',
    '555427674f5f034d06ad7c66d86fcbdc_0_161_1.pkl',
]

baking_files = [
   'da7329cb0a2cd85ce2fe724b67878971_0_140_1.pkl',
   '049ccb3f0ac8b2d9b62c7aca77313f2c_0_149_1.pkl',
   '687a66037dcefb5d228cc6685243119a_0_148_1.pkl',
   '687a66037dcefb5d228cc6685243119a_0_147_1.pkl',
'40e9ec9f8e89428925707746d14b5f89_0_146_1.pkl',
'd82f43e420cd59ebdde6e8ef3ef39913_0_145_1.pkl',
'cc6e762b1266d851216d9643a9408499_0_144_1.pkl',
'1e5b444b68a5789ab01fad71ca56ad3c_0_143_1.pkl',
'cfeb4483d2b1543340b6d22df821353d_0_142_1.pkl',
'813864bf6a50a4c724f7585dc905801f_0_141_1.pkl',
]

import itertools
from copy import deepcopy
from collections import defaultdict
from pprint import pprint
import re
import os, pickle
import shutil
from llm_parsing import LLM_PDDL_Parser, find_closing_paran
import pddlgym
from pddlgym.structs import Literal, LiteralConjunction

env = pddlgym.make('PDDLEnvBaking-v0')
ap = {p.name: p for p in env.action_space.predicates}
op = {p.name: p for p in env.observation_space.predicates}
# Collect the object types in this domain.
types = set()
for p in (env.action_space.predicates + env.observation_space.predicates):
    for t in p.var_types:
        types.add(t)
llm_parser = LLM_PDDL_Parser(ap, op, types)
def ops_equal(op1, op2):
    # Check that the # params are equal, of each type.
    op1_type_to_paramnames:dict[str, list] = defaultdict(list)
    op2_type_to_paramnames:dict[str, list] = defaultdict(list)

    for param in op1.params:
        t = param._str.split(':')[-1]
        op1_type_to_paramnames[t].append(param.split(':')[0])
    
    for param in op2.params:
        t = param._str.split(':')[-1]
        op2_type_to_paramnames[t].append(param.split(':')[0])

    # If the number of types don't match, return False
    if len(op2_type_to_paramnames) != len(op1_type_to_paramnames):
        return False

    # If the number of params of each type don't match, return False
    for t in op1_type_to_paramnames:
        if t not in op2_type_to_paramnames:
            return False
        if len(op2_type_to_paramnames[t]) != len(op1_type_to_paramnames[t]):
            return False
 
    # Get all parameterizations of the op1 params.
        # get all the variable names in a list, and use itertools.permutations(var_names)
    op1_params_list = []
    for param in op1.params:
        op1_params_list.append(param._str.split(':')[0])
    for perm in itertools.permutations(op1_params_list):
        # map from the original variable name list to the permutation
        variables = dict(zip(op1_params_list, perm))
        # Change the preconds and effects of op1 to the new arg names
        # Change the name from op1 param to the corresponding op2 param in preconditions and effects
        preconds = []
        for l in op1.preconds.literals:
            args = []
            for v in l.variables:
                args.append(variables[v.split(':')[0]])
            preconds.append(Literal(l.predicate, args))
        effects = []
        for l in op1.effects.literals:
            args = []
            for v in l.variables:
                args.append(variables[v.split(':')[0]])
            effects.append(Literal(l.predicate, args))

        # Check that the preconditions and effects of the changed op1 are the same as in op2
        if (set(op2.preconds.literals) == set(preconds)) and (set(op2.effects.literals) == set(effects)):
        # If the preconds and effects match, return True
            return True
 
    return False


def _llm_output_to_operators(llm_output):
    """Parse the LLM output."""

    # Split the response into chunks separated by "(:action ", and discard the first chunk with the header and LLM chat.
    operator_matches = list(re.finditer("\(\:action\s", llm_output))
    operators = []
    end = len(llm_output)
    for match in operator_matches[::-1]: # Read matches from end of file to top of file
        start = match.start()
        operator_str = find_closing_paran(llm_output[start:end])
        ops = llm_parser.parse_operators(operator_str)
        if ops is None:
            continue
        for o in ops:
            if o is None: continue
            operators.append(o)
        end = match.start()

    return operators


def add_ops_no_duplicates(ops_to_add, ops):
    """ Adds `ops_to_add` to  `ops`, no duplicate operators, make sure all ops are named according to the scheme `action_pred{int}` starting with int=0
    """
    for op1 in ops_to_add:
        already_in = False
        for op2 in ops:
            if ops_equal(op1, op2):
                already_in = True
        if not already_in:
            ops.append(op1)
    # Rename all the ops
    op_dict = defaultdict(list)
    for op in ops:
        op_dict[[p.predicate for p in op.preconds.literals if p.predicate in env.action_space.predicates][0].name].append(op)
    for action_name in op_dict:
        for i, op in enumerate(op_dict[action_name]):
            op.name = f'{action_name}{i}'
    return ops 


all_ops = []
for file in baking_files:
    with open(os.path.join(path, file), 'rb') as f:
        response = pickle.load(f)[0]
        ops = _llm_output_to_operators(response)
        # for o in ops:
            # if 'craftplank' in o.name:
            #     print(o)
        add_ops_no_duplicates(ops, all_ops)



bakecakes = []
cleanpans = []
bakesouffles = []
for o in all_ops:
    if 'bakecake' in o.name:
        bakecakes.append(o)
    if 'bakesouffle' in o.name:
        bakesouffles.append(o)
    if 'cleanpan' in o.name:
        cleanpans.append(o)

def get_set(op, ops):
    """Get the set of operators with the same preconditions as the operator passed in, agnostic to different parametrizations.

    Args:
        ops (list[Operator]): operators all of the same skill.
    """
    same_preconds_ops = []
    op_counts = {}
    op_params = set()
    for lit in op.preconds.literals:
        op_counts.setdefault(lit.predicate.name, 0)
        op_counts[lit.predicate.name] += 1
        # Get a list of parameter names that exist in the preconditions
        for v in lit.variables:
            v_name = v._str.split(':')[0]
            op_params.add(v_name)
    op_params = list(op_params)
 
    for o in ops:
        # if the number of literals is different, continue
        if len(o.preconds.literals) != len(op.preconds.literals):
            continue
        # if the num predicates are different (count # of each predicate), continue
        counts = {}
        params = set()
        for lit in o.preconds.literals:
            counts.setdefault(lit.predicate.name, 0)
            counts[lit.predicate.name] += 1
            # Get the parameter names that exist in the preconditions
            for v in lit.variables:
                v_name = v._str.split(':')[0]
                params.add(v_name)

        if op_counts != counts:
            continue
            
        # Get a list of parameter names that exist in the preconditions
        params = list(params)
 
        # Get all permutations of them, using each as a new mapping to the list of param names for `op`
        for perm in itertools.permutations(params):
            # create the new set of literals with the new names.
            new_preconds = []
            names_map = dict(zip(perm, op_params))
            for lit in o.preconds.literals:
                args = []
                for v in lit.variables:
                    args.append(names_map[v._str.split(':')[0]])
                new_preconds.append(Literal(lit.predicate, args))
            
            # if the new set matches the set(op.preconds.literals):
            if set(new_preconds) == set(op.preconds.literals):
                # add the operator
                same_preconds_ops.append(o)
                break
    return same_preconds_ops


bakecake0 = bakecakes[0]
bakecakes = bakecakes[1:]
# print(bakecake0)
s_cake = get_set(bakecake0, bakecakes)
s_clean = get_set(cleanpans[0], cleanpans[1:])
s_souffle = get_set(bakesouffles[0], bakesouffles[1:])

# print(cleanpans[0])
# print(bakesouffles[0])

print("------")
for s in (s_cake,s_clean, s_souffle):
    print('>')
    for o in s:
        print(o)
    # Manually verify that the set of operators adheres to the spec of `get_set`
    