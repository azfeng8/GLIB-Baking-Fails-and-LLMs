"""Operator search for LLM iterative method.

Debugging notes:

- If change transition_coverage or consistency_score, likely need to change the other as well.
"""

import pickle
import numpy as np
from typing import Iterable
from collections import defaultdict
from pddlgym.parser import Operator
from pddlgym.structs import Anti, Type, LiteralConjunction, Literal,TypedEntity
import os

def LEAP_operator_search(operators:list[Operator], dataset:list[list[tuple]], iter) -> list[Operator]:
    #TODO: [performance] the heavy computation in improve_coverage only needs to be done once.
    #TODO: [performance] cache the dataset partitions and best score / thing that earned the best score. For actions with only 1 operator, automatically take it.
    """Hill-climbing search over operator sets for best score.

    Using the following search operators, start search from an empty set:
        ImproveCoverage search operator
        ReduceComplexity search operator
    
    Inspired by: https://openreview.net/pdf?id=_gZLyRGGuo
    """
    with open(f'/home/catalan/temp/iter_{iter}/all_operators.pkl', 'wb') as f:
        pickle.dump(operators, f)

    op_idxes = set()
    j_last = np.inf
    lambda_val = 0.0001
    j_curr = LEAP_score([operators[i] for i in op_idxes], dataset, lambda_val)
    print("\nStarting search\n")

    while j_curr < j_last:
        j_last = j_curr
        op_set_prime = improve_coverage(op_idxes, dataset, operators)
        j = LEAP_score([operators[i] for i in op_set_prime], dataset, lambda_val)
        if j < j_curr:
            op_idxes = op_set_prime
            j_curr = j
        for op_i in op_idxes:
            op_set_prime = reduce_complexity(op_idxes, op_i)
            j = LEAP_score([operators[i] for i in op_set_prime], dataset, lambda_val)
            if j < j_curr:
                op_idxes = op_set_prime
                j_curr = j
    res =  [operators[i] for i in op_idxes]

    return res

def improve_coverage(op_idxes, dataset, operators) -> list[Operator]:
    """Finds a transition not covered yet in the dataset and adds an operator to better cover it.

    Use notion of "best consistent operator" by counting difference in add and delete effects.

    Prunes out null data operators.

    """
    # Get the set of transitions not covered.
    _, uncovered_transitions, covered_transitions, *_ = LEAP_coverage(dataset, [operators[i] for i in op_idxes])
    if len(uncovered_transitions) == 0:
        # Can't improve coverage more.
        return op_idxes

    ### Find the operator, that if added, covers the most uncovered transitions.
    # For each transition.
        # Pick the best consistent operator to cover it
    # Add the operator with most transitions assigned to the operator set.
    op_i_to_transitions = defaultdict(list)
    for index in np.random.permutation(range(len(uncovered_transitions))):
        uncovered_transition = uncovered_transitions[index]

        best_op_i = None
        best_score = np.inf
        for i, op in enumerate(operators):
            if i in op_idxes: continue

            s = transition_score(op, uncovered_transition, False)
            if s < np.inf and s == best_score:
                # Tied score, so increment the op being overwritten
                best_op_i.append(i)
            elif s < best_score:
                best_op_i = [i]
                best_score = s

        if best_op_i is not None:
            for o in best_op_i:
                op_i_to_transitions[o].append(uncovered_transition)
        else:
            # The transitions should probably always be covered by some operator.
            # with open("/home/catalan/temp/ops.pkl", 'wb') as f:
            #     pickle.dump([operators[i] for i in op_idxes], f)
            # with open("/home/catalan/temp/all_ops.pkl", 'wb') as f:
            #     pickle.dump(operators, f)
            # with open('/home/catalan/temp/uncovered_transition.pkl', 'wb') as f:
            #     pickle.dump(uncovered_transition, f)
            print("Got uncovered transition for action:", uncovered_transition[1])

    if len(op_i_to_transitions) == 0:
        return op_idxes
    
    op_i = max(op_i_to_transitions, key=lambda x: len(op_i_to_transitions[x]))
    ops = op_idxes | set([op_i])

    covered_transitions.extend(op_i_to_transitions[op_i])
    ops = prune(ops, covered_transitions, operators)

    return ops

def transition_score(op:Operator, transition:tuple, coverage=False) -> float:
    """Score how much the operator covers the transition.

    Score has same meaning on two different scales for the operator search algorithm.

    Given the ground operator effects E,
        the observed transition effects e,
        ground operator precondition positive literals P+ and negative literals P-,
        the positive state literals S+ and negative state literals S-,
        the assignment A of variables to objects,

    1. Coverage score: fractional measure, bounded [0,1]. 1 is perfectly cover, 0 is not covering at all.

        if skill isn't associated with the transition:
            score = 0

        otherwise, maximize over all possible groundings of the preconditions and effects:
            score = max[0,  (|E & e| - |E - (E & e)| ) / |e| - ( |P+ \ S+| + |P- \ S-| ) / |S| )]

    2. Consistency score: Integer measure, bounds [0, inf], 0 is perfectly cover, inf is operator is not applicable to the transition.

            Minimize over all groundings of the precondition and groundings of the effects,
                score =  |E \ e| + |e \ E| + max(0, |P+ \ S+| + |P- \ S-| - |P- & S-| - |P+ & S+|)
    
    Args:
        coverage (bool): If true, return 1. Coverage score. Else, return 2. Consistency score.
    """
    ### For all valid: Ground operator precondition in state.objects

    state, action, eff = transition
    lifted_precond = set()
    action_matches = False
    for lit in op.preconds.literals:
        if lit.predicate == action.predicate:
            action_matches = True
            continue
        lifted_precond.add(lit)
    if not action_matches:
        if coverage:
            return 0 
        else:
            return np.inf

    add_eff, delete_eff = get_add_delete_eff(eff)

    if coverage:
        score = 0
    else:
        score = np.inf

    for precond, assignment in _ground_literals(lifted_precond, state.objects):

        # Handle negative preconditions
        consistency_precond_score = 0
        coverage_precond_score = 0
        positive_preconds = set()
        for p in precond:
            if p.predicate.is_negative:
                if p.predicate.positive(*p.variables) in state.literals:
                    consistency_precond_score += 1
                    coverage_precond_score += 1
                else:
                    consistency_precond_score -= 1
            else:
                positive_preconds.add(p)


        consistency_precond_score += len(positive_preconds - state.literals) - len(positive_preconds & state.literals)
        consistency_precond_score = max(0, consistency_precond_score)
        coverage_precond_score += len(positive_preconds - state.literals)

        ground_effects_assign = _ground_literals(op.effects.literals, state.objects, assignment)
        # if len(ground_effects_assign) > 1: print("Warning: computed multiple effects possible")

        for ground_effects, _ in ground_effects_assign:
            # calculate score
            if coverage:
                correct = len(ground_effects & eff) 
                hallucinated = len(ground_effects) - len(ground_effects & eff)
                s =  max(0, (correct - hallucinated) / len(eff)) - coverage_precond_score / len(state.literals)
                if s > score:
                    score = s
            else:
                pred_add_eff, pred_delete_eff= get_add_delete_eff(ground_effects)

                s = len(add_eff - pred_add_eff) + len(pred_add_eff - add_eff)  + len(pred_delete_eff - delete_eff) + len(delete_eff - pred_delete_eff) + consistency_precond_score
                if s < score:
                    score = s
    return score

def get_add_delete_eff(effects:Iterable[Literal]) -> tuple[set[Literal], set[Literal]]:
    """_summary_

    Args:
        effects (Iterable[Literal]): _description_

    Returns:
        tuple[set[Literal], set[Literal]]: add_effects, delete_effects
    """

    # NOTE: What is the difference between .anti and .negative for literals
    delete_eff = set()
    add_eff = set()
    for e in effects:
        if e.is_anti or e.is_negative:
            delete_eff.add(e)
        else:
            add_eff.add(e)
    return add_eff, delete_eff



def prune(op_idxes:set[int], covered_transitions:list[tuple], operators):
    """Prune out null data operators.

    If several operators cover a transition, assign the transition to the operator that 'best' describes it.
    Get rid of operators that don't have any transition assigned to it.

    """
    op_to_num_transition = defaultdict(lambda: 0)
    for t in covered_transitions:
        best_score = np.inf
        best_ops = []
        for i in op_idxes:
            op = operators[i]
            s = transition_score(op, t, False)
            if s < best_score:
                best_score = s
                best_ops = [i]
            elif s == best_score:
                best_ops.append(i)

        best_op = np.random.choice(best_ops)
        op_to_num_transition[best_op] += 1
    
    ops = set()
    for op in op_to_num_transition:
        if op_to_num_transition[op] != 0:
            ops.add(op)
    return ops
    

def reduce_complexity(op_idxes, op_i):
    """Delete a single operator.

    """
    o = op_idxes - set([op_i])
    return o               

def LEAP_coverage(dataset, operators:Iterable[Operator]) -> tuple:
    """
    coverage(D,Ω): fraction of transitions the operator set explains

    Args:
    
        dataset [ [ transition ] ]: Dataset of episodes of transitions
        operators [ Operator ] : operator set

    Returns:
    
        float: coverage score
        uncovered_transitions: list of uncovered transitions 
        covered_transitions:  list of covered transitions 
        uncovered_transition_indices: indices (episode index, transition index) into the dataset

    """
    num_covered_transitions = 0
    total_transitions = 0
    uncovered_transitions = []
    uncovered_transition_indices = []
    covered_transitions = []
    for i,episode in enumerate(dataset):
        total_transitions += len(episode)
        for j,transition in enumerate(episode):
            best_score = 0
            for op in operators:
                # When using LNDR, better to have a close-but-not-perfect operator effects than no operator at all.
                c = transition_score(op, transition, True)
                if c > best_score:
                    best_score = c
            
            if best_score > 0:
                num_covered_transitions += best_score
                covered_transitions.append(transition)
            else:
                uncovered_transitions.append(transition)
                uncovered_transition_indices.append((i,j))
    return num_covered_transitions / total_transitions, uncovered_transitions, covered_transitions, uncovered_transition_indices


def LEAP_score(operators:list[Operator], dataset:list[list[tuple]], lambda_val):
    """Score a set of operators according to:
    
    J(Ω) ≜ (1 - coverage(D, Ω)) + λcomplexity(Ω)

    lower is better. J > 0 lower bound.

    D: dataset of transitions, separated by episode
    Ω: operator set
    coverage(D,Ω): fraction of transitions the operator set explains
    complexity(Ω): number of operators
    lambda: set to not decrease complexity at expense of lower coverage

    Inspired by: https://openreview.net/pdf?id=_gZLyRGGuo
    """
    coverage, *_ = LEAP_coverage(dataset, operators)
    J = 1 - coverage + lambda_val * len(operators)
    return J


def _ground_literals(lifted_literals:Iterable[Literal], objects:frozenset[TypedEntity], partial_assignment={}) -> list[tuple[set[Literal], dict]]:
    """Get all possible groundings of lifted literals with the variable-to-object assignments.

    Args:
        lifted_precond (set[Literal]): _description_
        objects (frozenset[TypedEntity]): _description_

    Returns:
        list[set[Literal]]: list of grounded preconditions
    """
    # create a map from var name to type
    var_to_type = {}
    for lit in lifted_literals:
        for v_name, v_type in zip(lit.pddl_variables_typed(), lit.predicate.var_types):
            var_to_type[v_name] = v_type

    # create a map from type to object
    type_to_object = defaultdict(list)
    for o in objects:
        name, v_type = o._str.split(":")
        type_to_object[v_type].append(o)

    full_assignments = []

    # All the variables to be assigned. Variables at the same indices as `partial_assign` are assigned to those objects in `partial_assign`
    variables = []
    # Create the initial partial assignment. Objects at the same indices as `variables` are assigned to those variables in `variables`
    partial_assign = []
    for v in var_to_type:
        if v in partial_assignment:
            partial_assign.append(partial_assignment[v])
            variables.append(v)
    for v in var_to_type:
        if v not in variables:
            variables.append(v)

    def walk_tree(index:int, current_permutation:list[TypedEntity]) -> list[dict]:
        """Returns maps from variable name to object

        Args:
            index: index into `variables`, to assign an object to
            current_permutation: growing list of the assigned objects, in order of `variables`
        """
        if len(current_permutation) == len(variables):
            full_assignments.append(current_permutation)
            return
        for object in type_to_object[var_to_type[variables[index]]]:
            if object in current_permutation[:index]:
                continue
            walk_tree(index + 1, current_permutation + [object])


    index = max(len(partial_assign), 0)
    walk_tree(index, partial_assign)
    ground_preconds = []
    for assignment in full_assignments:
        a = dict(zip(variables, assignment))
        precond = set()
        for lit in lifted_literals:
            lit_args = []
            for v_name, v_type in zip(lit.pddl_variables_typed(), lit.predicate.var_types):
                obj = a[v_name]
                obj_type = obj._str.split(":")[-1]
                assert v_type == obj_type , f"{v_type} vs {obj_type}"
                lit_args.append(obj)
            l = lit.predicate(*lit_args)
            precond.add(l)
        ground_preconds.append((precond, a))
    return ground_preconds
        
### Debugging
if __name__ == "__main__":
    # with open('/home/catalan/temp/dataset.pkl', 'rb')  as f:
    #     stuff = pickle.load(f)
    # with open('/home/catalan/temp/ops.pkl', 'rb') as f:
    #     ops = pickle.load(f)
    # with open('/home/catalan/temp/all_ops.pkl', 'rb') as f:
    #     all_ops = pickle.load(f)
    # with open('/home/catalan/temp/uncovered_transition.pkl', 'rb') as f:
    #     uncovered_transition = pickle.load(f)

    # print(consistency_score(o, uncovered_transition))

    iter = 800
    with open(f'/home/catalan/temp/iter_{iter}/dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    with open(f'/home/catalan/temp/iter_{iter}/all_operators.pkl', 'rb') as f:
        ops = pickle.load(f)

    for op in ops:
        print('\n',op)
    print(len(LEAP_operator_search(ops, dataset, -1)))