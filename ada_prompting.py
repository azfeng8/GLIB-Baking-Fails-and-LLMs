"""This file uses the ADA prompting scheme to generate operators ready for consumption by the WarmStart module."""

from openai_interface import OpenAI_Model
import pddlgym
import gym
import numpy as np
from llm_parsing import LLM_PDDL_Parser

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


GOAL_TRANSLATION_PROMPT = """;;;; Translate the goal into natural language.

Q: (and (movie-rewound)
    (counter-at-zero)
    (have-chips)
    (have-dip)
    (have-pop)
    (have-cheese)
    (have-crackers)
    )

A: The movie is rewound, the counter is at zero, and the agent has chips, dip, pop, cheese, and crackers.

Q: (and (person-at person0 f5-5f))

A: person0 is at location f5-5f.

Q:  (and (ontable shot15)
    (dispenses dispenser1 ingredient1)
    (contains shot1 cocktail5)
    )

A: The shot15 glass is on the table, the dispenser dispenser1 is set up to dispense ingredient1, and the shot glass shot1 contains cocktail5.

Q: (and  (at bear-0 loc-4-2)  (holding pawn-1) )

A: bear-0 is at location loc-4-2, and the agent is holding pawn-1.
"""

TASK_DECOMPOSITION_PROMPT = """;;;; Given natural language goals, predict a sequence of PDDL actions.

Q: 

Domain: Casino
Goal: The agent has secured prize-1-1 and prize-1-2 from the first prize group, prize-2-1 and prize-2-2 from the second prize group, and prize-3-1 and prize-3-2 from the third prize group.
Objects: loc-3-3:location,loc-5-0:location,loc-1-6:location,loc-7-8:location,loc-3-1:location,loc-3-9:location,loc-7-6:location,loc-2-0:location,loc-0-0:location,loc-5-6:location,prize-2-2:prize2,loc-7-2:location,loc-1-1:location,loc-6-1:location,loc-6-5:location,loc-0-2:location,loc-7-3:location,loc-3-0:location,loc-5-5:location,loc-6-8:location,loc-4-3:location,loc-3-7:location,loc-4-0:location,prize-1-1:prize1,loc-2-2:location,loc-2-1:location,loc-2-5:location,loc-0-8:location,loc-5-9:location,loc-1-8:location,loc-7-1:location,prize-3-1:prize3,loc-2-4:location,loc-2-8:location,loc-1-9:location,loc-6-9:location,loc-5-2:location,loc-0-6:location,prize-2-1:prize2,loc-5-8:location,loc-2-3:location,loc-4-6:location,loc-5-3:location,loc-2-9:location,loc-3-2:location,loc-4-8:location,loc-4-2:location,loc-6-6:location,loc-0-7:location,loc-0-1:location,prize-1-2:prize1,loc-6-0:location,loc-4-1:location,loc-4-7:location,loc-6-2:location,loc-1-4:location,loc-2-6:location,loc-3-4:location,loc-6-7:location,loc-0-4:location,loc-0-3:location,loc-3-6:location,loc-3-8:location,loc-2-7:location,loc-7-0:location,loc-7-5:location,loc-1-2:location,loc-7-9:location,prize-3-2:prize3,loc-5-7:location,loc-1-3:location,loc-4-9:location,loc-4-5:location,loc-1-7:location,loc-6-4:location,loc-7-4:location,loc-1-5:location,loc-0-9:location,loc-5-1:location,loc-0-5:location,loc-6-3:location,loc-5-4:location,loc-4-4:location,loc-7-7:location,loc-1-0:location,loc-3-5:location
State: iscasino(loc-5-4),at(loc-0-0)

A:

(moveto loc-0-0 loc-5-4),(getprize3 prize-3-2 loc-5-4),(getprize3 prize-3-1 loc-5-4),(getprize2 prize-2-2 loc-5-4),(getprize2 prize-2-1 loc-5-4),(getprize1 prize-1-2 loc-5-4),(getprize1 prize-1-1 loc-5-4)

Q:

Domain: Elevator
Goal: Service has been provided to person p0.
Objects: f1:floor,p0:passenger,f0:floor
State: lift-at(f0),origin(p0,f1),above(f0,f1),destin(p0,f0)

A:

(up f0 f1),(board f1 p0),(down f1 f0),(depart f0 p0)
"""
OPERATOR_DEFINITION_PROMPT = \
f""";;;; You are a software engineer who will be writing planning operators in the PDDL planning language. These operators are based on the following PDDL domain definition.

### The predicates in Sokoban are:

(move-dir ?v0 - location ?v1 - location ?v2 - direction)
	(is-nongoal ?v0 - location)
	(clear ?v0 - location)
	(is-stone ?v0 - thing)
	(at ?v0 - thing ?v1 - location)
	(is-player ?v0 - thing)
	(at-goal ?v0 - thing)
	(move ?v0 - direction)
	(is-goal ?v0 - location)

Q: Propose a PDDL operator called "move".

A: (:action move
		:parameters (?p - thing ?from - location ?to - location ?dir - direction)
		:precondition (and (move ?dir)
			(is-player ?p)
			(at ?p ?from)
			(clear ?to)
			(move-dir ?from ?to ?dir))
		:effect (and
			(not (at ?p ?from))
			(not (clear ?to))
			(at ?p ?to)
			(clear ?from))
	)

Q: Propose an operator called "push-to-goal".
A: 	(:action push-to-goal
		:parameters (?p - thing ?s - thing ?ppos - location ?from - location ?to - location ?dir - direction)
		:precondition (and (move ?dir)
			(is-player ?p)
			(is-stone ?s)
			(at ?p ?ppos)
			(at ?s ?from)
			(clear ?to)
			(move-dir ?ppos ?from ?dir)
			(move-dir ?from ?to ?dir)
			(is-goal ?to))
		:effect (and
			(not (at ?p ?ppos))
			(not (at ?s ?from))
			(not (clear ?to))
			(at ?p ?from)
			(at ?s ?to)
			(clear ?ppos)
			(at-goal ?s))
	)
	
Q: Propose an operator called "push-to-nongoal".
A: 	(:action push-to-nongoal
		:parameters (?p - thing ?s - thing ?ppos - location ?from - location ?to - location ?dir - direction)
		:precondition (and (move ?dir)
			(is-player ?p)
			(is-stone ?s)
			(at ?p ?ppos)
			(at ?s ?from)
			(clear ?to)
			(move-dir ?ppos ?from ?dir)
			(move-dir ?from ?to ?dir)
			(is-nongoal ?to))
		:effect (and
			(not (at ?p ?ppos))
			(not (at ?s ?from))
			(not (clear ?to))
			(at ?p ?from)
			(at ?s ?to)
			(clear ?ppos)
			(not (at-goal ?s)))
	)
"""

SKILL_ASSOCIATION_PROMPT = \
""";;;; Given the list of skills and a PDDL operator, pick the skill that is needed to execute the PDDL operator.

Q:

Domain: Sokoban
Skills: move(?v0 - direction), throw(?v0 - ball), walk(?v3 - loc), pick(?v1 - object)
Operator: 
(:action push-to-goal
		:parameters (?p - thing ?s - thing ?ppos - location ?from - location ?to - location ?dir - direction)
		:precondition (and (move ?dir)
			(is-player ?p)
			(is-stone ?s)
			(at ?p ?ppos)
			(at ?s ?from)
			(clear ?to)
			(move-dir ?ppos ?from ?dir)
			(move-dir ?from ?to ?dir)
			(is-goal ?to))
		:effect (and
			(not (at ?p ?ppos))
			(not (at ?s ?from))
			(not (clear ?to))
			(at ?p ?from)
			(at ?s ?to)
			(clear ?ppos)
			(at-goal ?s))
	)

A: move

Q:

Domain: Spanner
Skills: (walk ?v0 - location ?v1 - location ?v2 - man),(pickup_spanner ?v0 - location ?v1 - spanner ?v2 - man),(rotate ?v0 - location ?v1 - spanner ?v2 - man ?v3 - nut)
Operator:
(:action tighten_nut 
        :parameters (?l - location ?s - spanner ?m - man ?n - nut)
        :precondition (and (at ?m ?l) 
		      	   (at ?n ?l)
			   (carrying ?m ?s)
			   (useable ?s)
			   (loose ?n))
        :effect (and (not (loose ?n))(not (useable ?s)) (tightened ?n)))
)

A: rotate
"""

MAX_NUM_TASKS = 5
def get_goals_and_init_states(seed, n) -> list[tuple[str, str]]:
    """Returns the natural language goals and state.

    Returns:
        list[tuple[str, str]]: list of (natural language goal, state)
    """
    rets = []
    num_problems = len(train_env.problems)
    # for idx in np.random.choice(range(num_problems), size=min(MAX_NUM_TASKS, num_problems), replace=False):
    for idx in np.random.choice(range(num_problems), size=np.ceil(.1 * num_problems), replace=False):
        train_env.fix_problem_index(idx)
        init_state, info = train_env.reset()
        objects_str = "Objects: " + ','.join([o._str for o in init_state.objects])
        lits = []
        for lit in init_state.literals:
            lits.append(str(lit.predicate) + "(" + ",".join(lit.pddl_variables()) + ")")
        state_str = "State: " + ",".join(lits)
        goal = init_state.goal.pddl_str()
        #query LLM few-shot prompt, temp 1.0, sequentially for n responses.
        prompt = \
f"""{GOAL_TRANSLATION_PROMPT}
Q: {goal}
A:
"""
        conv = [{"role": "user", "content": prompt}]
        responses, path = llm.sample_completions(conv, 1.0, seed, 1)
        response = responses[0]
        rets.append((f'Goal: {response}', objects_str + '\n' +  state_str))
        for _ in range(max(0, n - 1)):
            conv.append({"role": "assistant", "content": response})
            conv.append({"role": "user", "content": "Please give another translation."})
            responses, path = llm.sample_completions(conv, 1.0, seed, 1)
            response = responses[0]
            rets.append((f'Goal: {response}', objects_str + '\n' +  state_str))
    return rets

def get_task_decompositions(domain_name, goals_and_states:list[tuple[str,str]], n, seed):
    """Returns n task decompositions per goal/state pair.

    Args:
        goals_and_states (list[tuple[str,str]])
    Returns:
        list[str]
    """
    task_decomps = []
    for goal, state in goals_and_states:
        prompt = \
f"""{TASK_DECOMPOSITION_PROMPT}

Q:

Domain: {domain_name}
{goal}
{state}

A:
"""
        responses, path = llm.sample_completions([{"role":"user", "content":prompt}], 1.0, seed, n)
        task_decomps.append(responses)
    return task_decomps

from tqdm import tqdm
def get_operator_definitions(task_decompositions, seed, n) -> list[str]:
    names = set()
    for task_decomp_str in task_decompositions:
        for literal in task_decomp_str.split(','):
            name = literal.split(' ')[0][1:]
            names.add(name)
    operators = []
    preds = [p for p in train_env.observation_space.predicates]
    lines = []
    for p in preds:
        s = f"({p.name} " + " ".join(p.pddl_variables()) + ")"
        lines.append(s)
    predicates = '\n'.join(lines)

    for op_name in tqdm(names):
        prompt = \
f"""{OPERATOR_DEFINITION_PROMPT}

### The predicates in {domain_name} are:

{predicates}

Q: Propose an operator called {op_name}.

A:
"""
        conv = [{"role":"user", "content":prompt}]
        for _ in range(n):
            responses, path = llm.sample_completions(conv, 1.0, seed, 1)
            operators.append(responses[0])
            conv.append({"role":"assistant", "content":responses[0]})
            conv.append({"role": "user", "content": "Please try again."})
    return operators

from pddlgym.parser import Operator
def associate_operators_with_skills(operator_proposals, domain_name, seed, n) -> list[tuple[Operator, str]]:
    """Returns tuples of the parsed operator + the name of the skill associated with it."""
    skills_list = ','.join([p.pddl_str() for p in train_env.action_space.predicates])
    operators_and_skills = []
    for proposal in operator_proposals:
        # Parse the operator into pddlgym
        ops = llm_parser.parse_operators(proposal)
        if ops is None: # Parsing failed.
            continue
        # Get the pddl string
        operator = ops[0]
        op_str = operator.pddl_str()

        prompt = \
f"""{SKILL_ASSOCIATION_PROMPT}

Q:

Domain: {domain_name}
Skills: {skills_list}
Operator:
{op_str}

A:
"""
        responses, path = llm.sample_completions([{"role":"user", "content":prompt}], 0, seed, n)
        for response in responses:
            operators_and_skills.append((operator, response))
    return operators_and_skills
    
import itertools
from collections import defaultdict
from operator_learning_modules.zpk.zpk_operator_learning import ops_equal

def create_final_operators(operators_and_skills:list[tuple[Operator, str]]) -> list[Operator]:
    """Adds the skill to the operators, and renames and removes duplicate operators."""
    # situate the arguments of the skill within the operator, in all possible ways, adding each one.
    operators = []
    op_names = defaultdict(lambda: 0)
    for operator, skill in operators_and_skills:
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
            # Get all combinations of operator params of that variable type
            name, var_type = v.split(' - ')
            for comb in itertools.combinations(type_to_op_param_names[var_type], len(type_to_action_param_names[var_type])):
            # For each combination
                # Get all permutation of the variables in the combination
                for perm in itertools.permutations(comb):
                # For each permutation
                    # Create a mapping from type_to_action_param_names[v_type] to the permutation
                    # add the map to the maintained dict
                    type_to_param_name_maps[var_type].append(list(zip(type_to_action_param_names[var_type], perm)))
        # Take itertools.product on the values of the dict
        # For each assignment/permutation,
        for assignment in itertools.product(*list(type_to_param_name_maps.values())):
            # Map the action predicate to the operator parameters
            args = []
            # Action name to operator name
            assignment = dict(assignment)
            for v in action_pred.pddl_variables():
                name, v_type = v.split(' - ')
                args.append(assignment[name])
            lit = action_pred(*args)
            # Create the operator with the action predicate in the precondition
            preconds = op.preconds.literals + [lit]
            new_op = Operator(f"{op.name}{suffix}", op.params, preconds, op.effects)
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
        raise Exception
        
    return operators


import pickle
import os
#TODO: save each set of responses in ada_init_operators/{domain_name}/operators.pkl, add to the repo.
# rets = get_goals_and_init_states(0, 4)
# with open('goal_translations.pkl', 'rb') as f:
#     # pickle.dump(rets, f)
#     rets = pickle.load(f)

# for r in rets:
#     print(r[0] + r[1])
#     print('>')

# print('----')
# task_decomps = get_task_decompositions(domain_name, rets, 4, 0)
# with open('task_decomps.pkl', 'rb') as f:
#     # pickle.dump(task_decomps, f)
#     task_decomps = pickle.load(f)

# for t in task_decomps:
#     print("+++")
#     for a in t:
#         print(a)
#         print('-')
# t = []
# for a in task_decomps:
#     t.extend(a)
# ops = get_operator_definitions(t, 0, 3)
with open('ada_init_operators/Baking/operator_proposals.pkl', 'rb') as f:
    # pickle.dump(ops, f)
    op_proposals = pickle.load(f)
# print(len(ops))
ops = []
for o in op_proposals:
    ops.extend(llm_parser.parse_operators(o))
skill_list = []
for o in ops:
    skills = [p.name for p in train_env.action_space.predicates]
    prompt = ""
    for i,s in enumerate(skills):
        prompt += f'[{i}] {s}\n'
    skill_list.append(skills[int(input(prompt))])
# operators_and_skills = associate_operators_with_skills(ops, domain_name, 0, 3)

# with open('ada_init_operators/Baking/ops_and_skills.pkl', 'rb') as f:
#     # pickle.dump(operators_and_skills, f)
#     operators_and_skills = pickle.load(f)

# print(len(operators_and_skills))
create_final_operators(list(zip(ops, skill_list)))