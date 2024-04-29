from openai_interface import OpenAI_Model
from tqdm import tqdm
import os
from operator_learning_modules.zpk.zpk_operator_learning import ops_equal
import pddlgym
import itertools
import pickle
import gym
import numpy as np
from llm_parsing import LLM_PDDL_Parser
from pddlgym.structs import LiteralConjunction
from pddlgym.parser import Operator
from collections import defaultdict

# domain_name = "Easygripper"
# domain_name = "Minecraft"
# domain_name = "Blocks"
# domain_name = "Glibdoors"
# domain_name = "Travel"
domain_name = "Baking"
train_env = gym.make("PDDLEnv{}-v0".format(domain_name))
types = set()
ap = {p.name: p for p in train_env.action_space.predicates}
op = {p.name: p for p in train_env.observation_space.predicates}
for p in (train_env.action_space.predicates + train_env.observation_space.predicates):
    for t in p.var_types:
        types.add(t)
llm_parser = LLM_PDDL_Parser(ap, op, types)
llm = OpenAI_Model()

preds = [p for p in train_env.observation_space.predicates]
lines = []
for p in preds:
    s = f"({p.name} " + " ".join(p.pddl_variables()) + ")"
    lines.append(s)
predicates = '\n'.join(lines)



def create_final_operators(operators_and_skills:list[tuple[Operator, str]]) -> list[Operator]:
    """Adds the skill to the operators, and renames and removes duplicate operators."""
    # situate the arguments of the skill within the operator, in all possible ways, adding each one.
    operators = []
    op_names = defaultdict(lambda: 0)
    for operator, skill in operators_and_skills:
        skip_operator = False
        action_pred = [p for p in train_env.action_space.predicates if p.name == skill][0]
        # Variable type to parameter name in the operator
        type_to_op_param_names:dict[str, list[str]] = {}
        type_to_action_param_names = {}
        for v in action_pred.pddl_variables():
            name, var_type = v.split(' - ')
            type_to_op_param_names[var_type] = []
            type_to_action_param_names.setdefault(var_type, [])
            type_to_action_param_names[var_type].append(name)
        for param in operator.params:
            name, v_type = param._str.split(':')
            if v_type in type_to_op_param_names:
                type_to_op_param_names[v_type].append(name)
        # Maintain a dict of type => parameter name maps
        type_to_param_name_maps = defaultdict(list)
        # For each variable type in the action predicate
        for v in action_pred.pddl_variables():
            if len(type_to_op_param_names[var_type]) < len(type_to_action_param_names[var_type]):
                skip_operator = True
                break
 
            # Get all combinations of operator params of that variable type
            name, var_type = v.split(' - ')
            for comb in itertools.combinations(type_to_op_param_names[var_type], len(type_to_action_param_names[var_type])):
            # For each combination
                # Get all permutation of the variables in the combination
                for perm in itertools.permutations(comb):
                # For each permutation
                    # Create a mapping from type_to_action_param_names[v_type] to the permutation of operator param names
                    # Add the assignment from action param names to operator param names for variables of this type
                    type_to_param_name_maps[var_type].append(list(zip(type_to_action_param_names[var_type], perm)))
        if skip_operator:
            continue
 
        # Take itertools.product on the values of the dict
        # For each assignment/permutation,
        for a in itertools.product(*list(type_to_param_name_maps.values())):
            if len(a) < len(type_to_param_name_maps):
                continue
            # Map the action predicate to the operator parameters
            args = []
            # Action name to operator name
            assignment = {}
            for type_list in a:
                assignment.update(dict(type_list) )
            for v in action_pred.pddl_variables():
                name, v_type = v.split(' - ')
                args.append(assignment[name])
            lit = action_pred(*args)
            # Create the operator with the action predicate in the precondition
            preconds = operator.preconds.literals + [lit]
            new_op = Operator(operator.name, operator.params, LiteralConjunction(preconds), operator.effects)
            # don't add duplicates
            equal = False
            for op in operators:
                if ops_equal(op, new_op):
                    equal = True
                    break
            if not equal:
                # ensure operators are of different names (append an int to the end of the names)
                suffix = op_names[new_op.name]
                op_names[new_op.name] += 1
                new_op.name = f'{new_op.name}{suffix}'
                operators.append(new_op)
        
    return operators

def get_op_descriptions(n) -> list[tuple[str, str]]:
    """
    Args:
        n (int): number of descriptions per skill
    Returns:
        List of (conversation, skill_name)
    """
    op_descriptions = []
    for action in tqdm(ap):
        op_description_prompt = f""";;;; You are an agent in the “{domain_name}” environment. Describe what are the preconditions and effects of an operator that uses the skill “{action}”."""
        conv = [{"role":"user", "content":op_description_prompt}]
        for _ in range(n):
            responses, path = llm.sample_completions(conv, 1.0, 0, num_completions=1)
            c = [{"role":"user", "content":op_description_prompt}, {"role":"assistant", "content":responses[0]}]
            op_descriptions.append((c, action))
            conv.append({"role":"assistant", "content":responses[0]})
            conv.append({"role": "user", "content": "Please try again."})
    return op_descriptions
    
def get_op_definitions(convos, temp, n):
    """
    Returns:
        list of (Operator, skill_name)
    """
    ops = []
    for convo, action in tqdm(convos):
        prompt = f""";;;; Using these predicates, translate the operator description into one or more PDDL operators associated with the skill "{action}".\nPredicates:\n{predicates}"""
        convo.append({"role":"user", "content": prompt})
        for _ in range(n):
            responses, path = llm.sample_completions(convo, temp, seed=0, num_completions=1, disable_cache=True)
            for op in llm_parser.parse_operators(responses[0]):
                ops.append(op)
            convo.append({"role":"assistant", "content":responses[0]})
            convo.append({"role": "user", "content": "Please try again."})
    return ops
    

if __name__ == '__main__':
    temp = 1
    n=4

    dir = f'skill_conditioned_2stage_temp{temp}'
    os.makedirs(dir, exist_ok=True)
    os.makedirs(os.path.join(dir, domain_name), exist_ok=True)

    # d = get_op_descriptions(n)
    with open(f'skill_conditioned_2stage_temp0/{domain_name}/op_descriptions.pkl', 'rb') as f:
        # pickle.dump(d, f)
        d = pickle.load(f)

    definitions = get_op_definitions(d, temp, n)
    with open(f'skill_conditioned_2stage_temp{temp}/{domain_name}/op_definitions.pkl', 'wb') as f:
        pickle.dump(definitions, f)
        # definitions = pickle.load(f)
        
    ops_no_duplicates = []
    for o in definitions:
        is_dup = False
        for op in ops_no_duplicates:
            if ops_equal(o, op):
                is_dup = True
                break
        if not is_dup:
            ops_no_duplicates.append(o)
            
    print(len(ops_no_duplicates))
    with open(f'skill_conditioned_2stage_temp{temp}/{domain_name}/ops.pkl', 'wb') as f:
        pickle.dump(ops_no_duplicates, f)

