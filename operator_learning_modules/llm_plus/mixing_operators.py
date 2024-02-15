from settings import AgentConfig as ac
from itertools import combinations
import math
import numpy as np
from copy import deepcopy
from pddlgym.parser import Operator
from pddlgym.structs import Anti, TypedEntity, Type, LiteralConjunction, Literal


def rename_lits_in_operator(operator_llm:Operator, operator_learner:Operator) -> Operator:
    """Rename the lits in the learner's operator to match the LLM's operator.

    After mapping the action predicate arguments to be the same,
    arbitrarily map the objects of same types, by which comes first in the parameters list.

    Example:
        LLM operator params: [x - pan, y - pan]
        Learner operator params: [z - pan, w - pan]

        z will be mapped to x because it is the first unassigned variable name of the same type.
        w will be mapped to y.
    
    Args:
        operator_llm : Operator
        operator_learner : Operator

    Returns:
        Operator: the revised operator from the learner that shares parameter names with the LLM operator.
    """
    # Map variable names by type from LLM to Learner
    llm_arg_names:list[str] = []
    llm_var_types:list[str] = []
    for param in operator_llm.params:
        var_name, v_type = param._str.split(':')
        llm_arg_names.append(var_name)
        llm_var_types.append(v_type)


    # Rename the learner's operators. Map from old variable name to new.

    names_mapping = {}

    remaining_params_to_map = deepcopy(operator_learner.params)

    # First map the action arguments.
    for lit in operator_llm.preconds.literals:
        if lit.predicate in ac.train_env.action_space.predicates:
            llm_action_lit = lit
    for lit in operator_learner.preconds.literals:
        if lit.predicate in ac.train_env.action_space.predicates:
            learner_action_lit = lit

    for v_learner, v_llm in zip(learner_action_lit.variables, llm_action_lit.variables):
        learner_var_name, learner_v_type = v_learner._str.split(':')
        llm_var_name, llm_v_type = v_llm._str.split(':')
        names_mapping[learner_var_name] = llm_var_name
        llm_var_types[llm_arg_names.index(llm_var_name)] = None
        remaining_params_to_map.remove(v_learner)

    # Then, arbitrarily map the names of the rest of the parameters.
    next_var_name = f"?x{len(operator_llm.params)}"

    for param in remaining_params_to_map:
        var_name, v_type = param._str.split(':')
        if v_type in llm_var_types:
            i = llm_var_types.index(v_type)
            # Mark the variable as taken
            llm_var_types[i] = None
            names_mapping[var_name] = llm_arg_names[i]
        else:
            names_mapping[var_name] = next_var_name
            next_var_name = "?x" + str(int(next_var_name.lstrip('?x')) + 1)
    
    ## Rename the Learner's variables
    # Rename the precondition
    precond_lits = []
    for lit in operator_learner.preconds.literals:
        args = []
        for v in lit.variables:
            v_name, v_type = v._str.split(':')
            args.append(TypedEntity(names_mapping[v_name], Type(v_type)))
        precond_lits.append(lit.predicate(*args))
    precond = LiteralConjunction(precond_lits)
            
    # Rename the effects
    effect_lits = []
    for lit in operator_learner.effects.literals:
        args = []
        for v in lit.variables:
            v_name, v_type = v._str.split(':')
            args.append(TypedEntity(names_mapping[v_name], Type(v_type)))
        effect_lits.append(lit.predicate(*args))
    effect = LiteralConjunction(effect_lits)
            
    # Recreate the params
    params = set()
    for l in precond.literals + effect.literals:
        for v in l.variables:
            params.add(v)
    return Operator(operator_learner.name, params, precond, effect)


def mix_two_operators(llm_operator:Operator, operator_learner:Operator) -> list[Operator]:
    """Mix two operators: an interpolation between the 2 operators' preconditions and effects.
    
    All operators returned are influenced by the LLM operator.

    Args:
        llm_operator (Operator): _description_
        operator_learner (Operator): _description_

    Returns:
        list[Operator]: _description_
    """
    # Maximum number of literals
    MAX_LITS = ac.LLM_iterative_max_lits
    learner_operator = rename_lits_in_operator(llm_operator, operator_learner)
    px = set(llm_operator.preconds.literals)
    py = set(learner_operator.preconds.literals)
    ex = set(llm_operator.effects.literals)
    ey = set(learner_operator.effects.literals)
    px_m_py = set([(p, 'remove', 'precondition') for p in (px - py)])
    py_m_px = set([(p, 'add', 'precondition') for p in (py - px)])
    ex_m_ey = set([(e, 'remove', 'effect') for e in (ex - ey)])
    ey_m_ex = set([(e, 'add', 'effect') for e in (ey - ex)])
    n = len(px_m_py) + len(py_m_px) + len(ex_m_ey) + len(ey_m_ex)

    operators = set([llm_operator])

    # Interpolate using method 2
    if n > MAX_LITS:

        arr = [(num, math.comb(n, num)) for num in range(1, n)]
        n_samples = 2**MAX_LITS
        num_ops_list = []
        interval = (2**n - 2) / n_samples
        idx = 0
        num, count = arr[idx]
        # print("arr", arr)
        # print("n samples", n_samples)
        # print("interval", interval)
        for _ in range(1, n_samples + 1):
            r = interval
            if r < count:
                count = count - r
                num_ops_list.append(num)
            else:
                while r >= count:
                    r -= count
                    idx += 1
                    if idx == len(arr):
                        assert r == 0
                        break
                    num, count = arr[idx]
                if r > 0:
                    count -= r
                num_ops_list.append(num)
        # Sample from preconditions and effects sets
        for number_ops in num_ops_list:
            pe = list(px_m_py | py_m_px | ex_m_ey | ey_m_ex)
            operator = llm_operator
            while number_ops > 0:
                i = np.random.choice(len(pe))
                lit, act, p_or_e = pe.pop(i)
                if act == 'add':
                    operator, number_ops = append_literal(operator, lit, p_or_e, number_ops) 
                elif act == 'remove':
                    operator, number_ops = remove_literal(operator, lit, p_or_e, number_ops)
            operators.add(operator)
        return operators

    else:
        pe = px_m_py | py_m_px | ex_m_ey | ey_m_ex
        # Generate the powerset b/c 2**n can be handled.
        for x in nonempty_powerset(pe):
            operator = llm_operator
            for y in x:
                lit, act, p_or_e = y
                if act == 'add':
                    operator, _ = append_literal(operator, lit, p_or_e, -1) 
                elif act == 'remove':
                    operator, _ = remove_literal(operator, lit, p_or_e, -1)
            operators.add(operator)
        return operators

def nonempty_powerset(s):
    n = len(s)
    for r in range(1, n+1):
        for combo in combinations(s, r):
            yield combo

def append_literal(operator:Operator, literal:Literal, precond_or_eff:str, number_ops) -> Operator:
    """Helper for mix_two_operators. Adds a literal to the operator in the effect or precondition.

    Args:
        operator (Operator) 
        literal
        precond_or_eff (str): equal to 'precondition' or 'effect'

    Returns:
        Operator: the edited Operator.
    """
    assert isinstance(operator.preconds, LiteralConjunction)
    assert isinstance(operator.effects, LiteralConjunction)

    effects = deepcopy(operator.effects)
    preconds = deepcopy(operator.preconds)
    if precond_or_eff == 'precondition':
        if literal.negative in preconds.literals and not literal.is_negative:
            preconds.literals.remove(literal.negative)
            number_ops -= 1
        elif literal.positive in preconds.literals and literal.is_negative:
            preconds.literals.remove(literal.positive)
            number_ops -= 1
        preconds.literals.append(literal)
        number_ops -= 1
    elif precond_or_eff == 'effect':
        if literal.inverted_anti in effects.literals:
            effects.literals.remove(literal.inverted_anti)
            number_ops -= 1
        effects.literals.append(literal)
        number_ops -= 1
    else:
        raise ValueError(f"{precond_or_eff} expected to be `precondition` or `effect`.")
    params = set()
    for lit in preconds.literals + effects.literals:
        for v in lit.variables:
            params.add(v)
    return Operator(operator.name, params, preconds, effects), number_ops
        
def remove_literal(operator:Operator, literal, precond_or_eff, number_ops) -> Operator:
    """Helper for mix_two_operators. Adds a literal to the operator in the effect or precondition.

    Args:
        operator (Operator) 
        literal  : 
        precond_or_eff (str): equal to 'precondition' or 'effect'

    Returns:
        Operator: the edited Operator.
    """
    assert isinstance(operator.effects, LiteralConjunction)
    assert isinstance(operator.preconds, LiteralConjunction)

    effects = deepcopy(operator.effects)
    preconds = deepcopy(operator.preconds)
    if precond_or_eff == 'precondition':
        if literal in preconds.literals: # literal could be already removed by `append_literal()` by adding its negation
            preconds.literals.remove(literal)
            number_ops -= 1
    elif precond_or_eff == 'effect':
        if literal in effects.literals:# literal could be already removed by `append_literal()` by adding its negation
            effects.literals.remove(literal)
            number_ops -= 1
    else:
        raise ValueError(f"{precond_or_eff} expected to be `precondition` or `effect`.")
    params = set()
    for lit in preconds.literals + effects.literals:
        for v in lit.variables:
            params.add(v)
    return Operator(operator.name, params, preconds, effects), number_ops

