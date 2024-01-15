"""
Strategy: LLM proposes operators based on training data, and score all the learned operators + LLM's operators every once in a while.

Currently, triggered every 20 learning cycles.

Problem: data consists only of successfully executed actions, and LLM only sees this data to improve on the seen operators, not helping explore.
"""
import numpy as np
import pickle
import re
from typing import Iterable
from collections import defaultdict
from operator_learning_modules import ZPKOperatorLearningModule
from settings import AgentConfig as ac
from openai_interface import OpenAI_Model
from pddlgym.parser import Operator
from pddlgym.structs import Anti, Type, LiteralConjunction, Literal,TypedEntity
from abc import abstractmethod
import pdb


def _LEAP_operator_search(operators:list[Operator], dataset:list[list[tuple]]) -> list[Operator]:
    """Hill-climbing search over operator sets for best score.

    Using the following search operators, start search from an empty set:
        ImproveCoverage search operator
        ReduceComplexity search operator
    
    Inspired by: https://openreview.net/pdf?id=_gZLyRGGuo
    """
    def improve_coverage(op_idxes, dataset) -> list[Operator]:
        """Finds a transition not covered yet in the dataset and adds an operator to better cover it.

        Use notion of "best consistent operator" by counting difference in add and delete effects.

        Prunes out null data operators.

        """
        # Get the set of transitions not covered.
        coverage, uncovered_transitions, covered_transitions = LEAP_coverage(dataset, [operators[i] for i in op_idxes])
        uncovered_transitions = set(uncovered_transitions)

        ### Find the operator, that if added, covers the most uncovered transitions.
        # For each transition.
            # Pick the best consistent operator to cover it
        # Add the operator with most transitions assigned to the operator set.
        op_to_transitions = defaultdict(list)
        for uncovered_transition in uncovered_transitions:

            best_op = None
            best_score = np.inf
            for i, op in enumerate(operators):
                if i in op_idxes: continue

                s = consistency_score(op, uncovered_transition)
                if s == 0 and best_score == 0:
                    # Tied score, so increment the op being overwritten
                    op_to_transitions[best_op].append(uncovered_transition)
                    best_op = i
                    best_score = s
                elif s < best_score:
                    best_op = i
                    best_score = s

            if best_op is not None:
                op_to_transitions[best_op].append(uncovered_transition)
        
        op_i = max(op_to_transitions, key=lambda x: len(op_to_transitions[x]))
        op_idxes.add(op_i)

        covered_transitions.extend(op_to_transitions[op_i])
        op_idxes = prune(op_idxes, covered_transitions)

        return op_idxes

    def consistency_score(op:Operator, transition:tuple):
        """Integer measure of how much the operator covers the transition.

        Given the ground operator add effects E+ and delete effects E-, and
        the observed transition add effects e+ and delete effects e-:
        
        score = |E+ \ e+| +
                |e+ \ E+| +
                |E− \ e-| +
                |e- \ E−|
        
        0 means the operator covers the transition perfectly.
        np.inf means that the operator does not cover the transition at all.
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
        if not action_matches: return np.inf
        
        score = np.inf
        for precond, assignment in _ground_literals(lifted_precond, state.objects):
            if precond.issubset(state.literals):
                ground_effects_assign = _ground_literals(op.effects.literals, state.objects, assignment)
                assert len(ground_effects_assign) == 1
                ground_effects, _ = ground_effects_assign[0]

                # calculate score
                pred_delete_eff = set()
                pred_add_eff = set()
                for e in ground_effects:
                    if e.is_anti:
                        pred_delete_eff.add(e)
                    else:
                        pred_add_eff.add(e)
                delete_eff = set()
                add_eff = set()
                for e in eff:
                    if e.is_anti:
                        delete_eff.add(e)
                    else:
                        add_eff.add(e)
                        
                if ((add_eff & pred_add_eff) == set()) and ((delete_eff & pred_delete_eff) == set()):
                    s = np.inf
                else:
                    s = len(add_eff - pred_add_eff) + len(pred_add_eff - add_eff)  + len(pred_delete_eff - delete_eff) + len(delete_eff - pred_delete_eff)
                if s < score:
                    score = s
            #TODO: if precondition doesn't exactly match, but effects do, then return score > 0
        return score
    
    def prune(op_idxes:set[int], covered_transitions:list[tuple]):
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
                s = consistency_score(op, t)
                if s < best_score:
                    best_score = s
                    best_ops = [i]
                elif s == best_score:
                    best_ops.append(i)

            best_op = np.random(best_ops)
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
    op_idxes = set()
    j_last = np.inf
    lambda_val = 0.1
    j_curr = LEAP_score([operators[i] for i in op_idxes], dataset, lambda_val)
    # print(j_curr)
    while j_curr < j_last:
        op_set_prime = improve_coverage(op_idxes, dataset)
        j = LEAP_score([operators[i] for i in op_idxes], dataset, lambda_val)
        if j < j_curr:
            op_idxes = op_set_prime
            j_curr = j
        for op_i in op_idxes:
            op_idxes = reduce_complexity(op_idxes, op_i)
            j = LEAP_score([operators[i] for i in op_idxes], dataset, lambda_val)
            if j < j_curr:
                op_idxes = op_set_prime
                j_curr = j
                break
    return [operators[i] for i in op_idxes]


def transition_coverage(op:Operator, transition:tuple) -> bool:
    """Returns the amount the operator covers the transition.
    
    1 if covers exactly.
    0 if doesn't cover at all.

    max(0, ( len(pred_effects AND pred_effects) - len(pred_effects - pred_effects AND effects) ) / len(effects) ) otherwise.

    Args:
        op (Operator) 
        transition (tuple): (state, action, effects)
    """
    state, action, eff = transition
    # if satisfy precond
    lifted_precond = set()
    action_matches = False
    for lit in op.preconds.literals:
        if lit.predicate == action.predicate:
            action_matches = True
            continue
        lifted_precond.add(lit)
    if not action_matches: return 0
    
    # for all possible groundings
    score = 0
    for precond, assignment in _ground_literals(lifted_precond, state.objects):
        if precond.issubset(state.literals):
            # compute operator effects
            ground_effects_assign = _ground_literals(op.effects.literals, state.objects, assignment)
            assert len(ground_effects_assign) == 1
            ground_effects, _ = ground_effects_assign[0]
            # if operator effects match transition
            if ground_effects == eff:
                return 1
            correct = len(ground_effects & eff) 
            hallucinated = len(ground_effects) - len(ground_effects & eff)
            s =  max(0, (correct - hallucinated) / len(eff))
            if s > score:
                score = s
    #TODO: if precondition doesn't exactly match, but the effects match, should give score > 0
    return score 
                
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

    """
    num_covered_transitions = 0
    total_transitions = 0
    uncovered_transitions = []
    covered_transitions = []
    for episode in dataset:
        total_transitions += len(episode)
        for transition in episode:
            for op in operators:
                # When using LNDR, better to have a close-but-not-perfect operator effects than no operator at all.
                c = transition_coverage(op, transition)
                if c > 0:
                    num_covered_transitions += c
                    covered_transitions.append(transition)
                else:
                    uncovered_transitions.append(transition)
            
    return num_covered_transitions / total_transitions, uncovered_transitions, covered_transitions


def LEAP_score(operators:list[Operator], dataset:list[list[tuple]], lambda_val):
    """Score a set of operators according to:
    
    J(Ω) ≜ (1 - coverage(D, Ω)) + λcomplexity(Ω)

    lower is better. J > 0 lower bound.

    D: dataset of transitions, separated by episode
    Ω: operator set
    coverage(D,Ω): fraction of transitions the operator set explains
    complexity(Ω): number of operators
    lambda: set to not decrease complexity at expense of lower coverage
        #TODO: experiment with different lambda

    Inspired by: https://openreview.net/pdf?id=_gZLyRGGuo
    """
    coverage, _ = LEAP_coverage(dataset, operators)
    J = 1- coverage + lambda_val * len(operators)
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

    def dfs(assignment, type_to_object, var_to_type) -> list[dict]:
        assignments = []
        for v in var_to_type:
            possible_objects = type_to_object[var_to_type[v]]
            if len(possible_objects) == 0:
                # No assignment possible.
                return []
            elif len(possible_objects) == 1:
                assignment[v] = possible_objects[0] 
            else:
                for o in possible_objects:
                    assignment[v] = o
                    # Search with this object mapped
                    type_to_object[var_to_type[v]].remove(o)
                    v2t = var_to_type.copy()
                    del v2t[v]
                    assignments.extend(dfs(assignment, type_to_object, v2t))
                    type_to_object[var_to_type[v]].append(o)
                    v2t[v] = var_to_type[v]
        # At leaf node
        if len(assignments) == 0:
            return [assignment]
        # Passing assignments back up
        else:
            return assignments

    # maps of var to object
    ground_preconds = []
    for assignment in dfs(partial_assignment, type_to_object, var_to_type):
        precond = set()
        for lit in lifted_literals:
            lit_args = []
            for v_name in lit.pddl_variables_typed():
                lit_args.append(assignment[v_name])
            precond.add(lit.predicate(*lit_args))
        ground_preconds.append((precond, assignment))
    return ground_preconds
        

class BaseLLMIterativeOperatorLearningModule:
    """_summary_
    """
    def __init__(self, learned_operators, domain_name, llm):
        self._llm:OpenAI_Model = llm
        self._llm_learn_interval = ac.LLM_learn_interval[domain_name]
        self._learn_iter = 0
        self._learned_operators = learned_operators
        self._observation_predicates = {p.name: p for p in ac.train_env.observation_space.predicates}
        self._action_predicates = {p.name: p for p in ac.train_env.action_space.predicates}

        # List of of (s,a,effects) in the current episode
        # self._trajectory = []

        # Dataset of transitions
        self._trajectories = []
        
    def observe(self, state, action, effects, start_episode=False, **kwargs):
        """Observe a transition.

        Args:
            state (_type_): _description_
            action (_type_): _description_
            effects (_type_): _description_
            start_episode (bool, optional): _description_. Defaults to False.
        """

        self.learner.observe(state, action, effects)

        # TODO: When episode ends, discard trajectories of length 0.
        #TODO: store all the episodes
        if start_episode:
            # self._trajectory = []
            self._trajectories.append([])

        # exclude no-ops
        if len(effects) != 0:
            # self._trajectory.append((state, action, effects))
            self._trajectories[-1].append((state, action, effects))
        # else:
        #     print('noop')


    def learn(self):
        """_summary_

        Returns:
            bool: if operators were updated
        """
        is_updated = self.learner.learn()
        self._learned_operators = self.learner._learned_operators
        if self._learn_iter % self._llm_learn_interval != 0:
            self._learn_iter += 1
            return is_updated

        # sample random episode. Use all the actions in the episode.
        traj = self._sample_trajectory()

        #TODO: if no trajectories yet, log fail message.
        
        # LLM proposes new operator for each of the actions in the trajectory.
        ops = self._propose_operators(traj)

        # score and filter the PDDL operators
        ops.extend(self.learner._learned_operators)
        ops = self._score_and_filter(ops)

        # update learner operators
        is_updated =  self._update_operator_rep(ops)

        self._learn_iter += 1
        return is_updated
        
            
    def _sample_trajectory(self):
        """Returns the current trajectory for the episode.

        Later can cap the number of actions in trajectory and sample from a random episode. 
        """
        assert len(self._trajectories[-1]) > 0

        #TODO: use the most recent episode if possible. Otherwise sample from some past episode
        traj = self._trajectories[-1]
        init =  traj[0]
        init_state, _, _ = init

        # Try to get longest trajectory with different goal state than initial state
        for i,t in enumerate(traj[::-1]):
            end_state, _, _ = t
            if init_state.literals != end_state.literals:
                return traj[:-i]
        
        return traj

    def _propose_operators(self, transitions):
        """
        Args:
            transitions (list): list of (s,a,effects) transitions in the trajectory
        """
        #TODO: change the few-shots to examples from a different domain.
        prompt_few_shot = """;;;; Translate the following state into natural language.

Q:
isflour(flour-0:ingredient), panisclean(pan-0:pan), isflour(flour-1:ingredient), hypothetical(new-1:ingredient), isegg(egg-1:ingredient), hypothetical(new-0:ingredient), isegg(egg-0:ingredient)

A:
There is flour (flour-0) as an ingredient, and the pan (pan-0) is clean. Additionally, there is another type of flour (flour-1) as an ingredient, and a hypothetical new ingredient (new-1). Furthermore, there is an egg (egg-1) as an ingredient, and another hypothetical new ingredient (new-0). Finally, there is an egg (egg-0) as another ingredient.

Q:
isflour(flour-0:ingredient), paninoven(pan-0:pan), panisclean(pan-0:pan), isflour(flour-1:ingredient), ovenisfull(oven-0:oven), inoven(pan-0:pan,oven-0:oven), hypothetical(new-1:ingredient), isegg(egg-1:ingredient), hypothetical(new-0:ingredient), isegg(egg-0:ingredient)

A:
There is flour (flour-0) as an ingredient, and a pan (pan-0) is in the oven (oven-0) that is also clean. Additionally, there is another type of flour (flour-1) as an ingredient. The oven (oven-0) is full, and the pan (pan-0) is inside the oven. Moreover, there is a hypothetical new ingredient (new-1). An egg (egg-1) is also present as an ingredient. Furthermore, there is another hypothetical new ingredient (new-0), and there is an egg (egg-0) as another ingredient.
"""
        # translate the start state into natural language.
        init_state, _, _ = transitions[0]
        prompt_start = prompt_few_shot + f"\nQ:\n" + " ".join([str(l) for l in init_state.literals]) + "\nA:\n"
        #TODO: May vary the temperature on this one
        # init_state_description = self._llm.sample_completions([{"role": "user", "content": prompt_start}], 0, ac.seed, 1)[0]
        #TODO: Uncomment. hardcode example to avoid spurious queries during dev
        init_state_description = "'The pan (pan-0) is clean. There is flour (flour-0) and an egg (egg-0) as ingredients. Additionally, there is a hypothetical new ingredient (new-0).'"

        # translate the end state into natural language.
        goal_state, _, effects = transitions[-1]
        goal_lits = set()
        goal_lits |= goal_state.literals
        for e in effects:
            if e.is_anti:
                goal_lits.remove(Anti(e))
            else:
                goal_lits.add(e)
        prompt_goal = prompt_few_shot + f"\nQ:\n" + " ".join([str(l) for l in goal_lits]) + "\nA:\n"
        #TODO: May vary the temperature
        #TODO: Uncomment. hardcode example to avoid spurious queries during dev
        # goal_state_desription = self._llm.sample_completions([{"role": "user", "content": prompt_goal}], 0, ac.seed, 1)[0]
        goal_state_desription = "The pan (pan-0) is clean and the oven (oven-0) is full. There is a hypothetical new ingredient (new-0). The pan (pan-0) is in the oven (oven-0). There is flour (flour-0) as an ingredient and an egg (egg-0) as another ingredient."

        # create the task decomposition 
        task_decomp = []
        task_decomp_str = ""
        for _, a, _ in transitions:
            task_decomp.append(f"{a.predicate}")
            task_decomp_str += f"{a.predicate}(" + ",".join(a.pddl_variables()) + ") "

        op_convos = []
        for action in task_decomp:
            prompt_operator = f"""
            ;;;;  Given actions from initial state to goal state describe an operator called "{action}" in natural language.

            Actions:
            {task_decomp_str}
            
            Initial State:
            {init_state_description}
            
            Goal State:
            {goal_state_desription}
            """
            #TODO: May vary the temperature on this one
            #TODO: Uncomment. hardcode example to avoid spurious queries during dev
            # response = self._llm.sample_completions([{"role": "user", "content": prompt_operator}], 0, ac.seed, 1)[0]
            response = 'The "putpaninoven" operator in natural language can be described as follows:\n\nThis operator represents the action of placing a pan into an oven. In this specific context, the pan is identified as \'pan-0\' and the oven as \'oven-0\'. The initial state before this action is performed is that the pan is clean and outside the oven. There are also ingredients present, namely flour, an egg, and a hypothetical new ingredient. \n\nWhen the "putpaninoven" action is performed, the pan is moved from its initial location and placed inside the oven. The state of the pan remains clean and the oven is now considered full. The ingredients remain the same, with the flour, egg, and hypothetical new ingredient still present. The goal state is achieved when the pan is successfully placed inside the oven.'
            op_convos.append([{"role": "user", "content": prompt_operator}, {"role": "assistant", "content": response}])

        # Get predicates
        env = ac.train_env
        preds = [p for p in env.action_space.predicates] + [p for p in env.observation_space.predicates]
        lines = []
        for p in preds:
            s = f"({p.name} " + " ".join(p.pddl_variables()) + ")"
            lines.append(s)
        predicates = '\n'.join(lines)
        operators = []
        for conversation in op_convos:
            prompt = f"""Given these predicates, translate the description into a PDDL operator:
            Predicates:
            {predicates}

            Use the format:
            
            (:action 
                :parameters ()
                :precondition (
                )
                :effect (
                )
            
            """
            conversation.append({"role": "user", "content": prompt})
            #TODO: Uncomment. hardcode example to avoid spurious queries during dev
            # response = self._llm.sample_completions(conversation, 0, ac.seed, 1)[0]
            response = """Sure, here is the PDDL operator for the action "putpaninoven":

```pddl
(:action putpaninoven
    :parameters (?v0 - pan ?v1 - oven)
    :precondition (and 
                    (panisclean ?v0)
                    (not (ovenisfull ?v1))
                  )
    :effect (and 
              (paninoven ?v0)
              (ovenisfull ?v1)
              (not (panisclean ?v0))
            )
)
```

This operator represents the action of putting a clean pan in an oven that is not full. After the action, the pan is in the oven, the oven is full, and the pan is no longer clean."""
            operators.append(self._parse_operator(response))

        with open("/home/catalan/temp/ops.pkl", 'wb') as f:
            pickle.dump(operators, f)

        return operators

    def _parse_operator(self, llm_response:str) -> Operator:
        # Find the PDDL operator in the response.
        match = re.search("\(\:action", llm_response)
        # Count parantheses: look for the closing to "(:action" to get the operator string.
        open_parans = 0
        close = 0
        i = match.end()
        operator_str = None
        for c in llm_response[match.end():]:
            if c == "(":
                open_parans += 1
            elif c == ")":
                close += 1
            if close > open_parans:
                operator_str = llm_response[match.start():i]
                break
            i+=1

        if operator_str is None: raise Exception(f"Parsing error: {llm_response}")
        # Extract operator name.
        match = re.search("\(\:action\s\w+", operator_str)
        op_name = operator_str[match.start() + len("(:action "):match.end()]

        # Extract parameters.
            # NOTE: Assume parameters are written on one line.
        match = re.search("\:parameters[^\)]*\)", operator_str)
        param_str = operator_str[match.start() + len(":parameters ("): match.end()].rstrip(')')
        param_names:list[str] = []
        param_types:list[str] = []
        for s in param_str.split('?'):
            if s == "": continue
            name, var_type = s.split(' - ')
            name = name.strip()
            var_type = var_type.strip()
            param_names.append(name)
            param_types.append(var_type)

        # Extract preconditions.
        match = re.search(":precondition([\s\S]*?):effect", operator_str)
        precond_str = operator_str[match.start() + len(":precondition (") : match.end() - len(":effect")]
        literals = self._get_literals(precond_str, param_names)

        # NOTE: Prompting the action multiple times will result in different operators.
        action_pred = self._action_predicates[op_name]
        args = []
        for v_type in action_pred.var_types:
            v_name = param_names[param_types.index(str(v_type))]
            args.append(Type(v_name))
        action = action_pred(*args)
        preconds = LiteralConjunction(literals + [action])

        # Extract effects.
        effect_str_match = re.search(":effect([\s\S]*?)\s\)", operator_str)
        effect_str = operator_str[effect_str_match.start():effect_str_match.end()]
        eliterals = self._get_literals(effect_str, param_names)
        effects = LiteralConjunction(eliterals)

        # Make parameters
        params = set()
        for literal in effects.literals + preconds.literals:
            for v in literal.variables:
                params.add(v)

        #TODO: need thorough checking of names, types. Importantly, the lifted predicate argument names should be consistent in parameters, preconditions, and effects.

        return Operator(op_name, params, preconds, effects)


    def _score_and_filter(self, ops:list[Operator]) -> list[Operator]:

        ops = self._renumber_operators(ops)
        ops = _LEAP_operator_search(ops, self._trajectories)
        return ops
    
    def _renumber_operators(self, ops:list[Operator]) -> list[Operator]:
        """Rename the operators so names are all different.
        """
        # NOTE: Assume initially, operator names are action names or have digits at the end of them.
        # Strip the trailing digits.
        for op in ops:
            op.name = op.name.rstrip('0123456789')
        # Renumber them.
        unique_names = defaultdict(lambda: 0)
        for op in ops:
            i = unique_names[op.name]
            unique_names[op.name] += 1
            op.name = f"{op.name}{i}"
        return ops
    
    def _get_literals(self, precond_or_eff_str, param_names) -> list[Literal]:
        """Helper for _parse_operator.

        Args:
            precond_or_eff_str (_type_): _description_
            param_names (_type_): _description_

        Raises:
            Exception: _description_

        Returns:
            list[Literal]: _description_
        """
        literals = []
        for predicate_str_match in re.finditer("\([\w]+[\s\?\w]*\)", precond_or_eff_str):
            predicate_str = predicate_str_match.group(0)
            if "and" in predicate_str or "not" in predicate_str: continue
            pred_name = re.search("[\w]+", predicate_str).group(0)

            args = []
            for arg_name in re.findall("\?[\w\d]+", predicate_str):
                arg_name = arg_name[1:] # remove the "?" in front
                if arg_name not in param_names:
                    raise Exception(f"Argument for {pred_name} in not in parameters: {arg_name}")
                # arg_type = param_types[param_names.index(arg_name)]
                args.append(Type(f"{arg_name}"))
            literals.append(Literal(self._observation_predicates[pred_name], args))
        return literals



    @abstractmethod
    def update_operator_rep(self):
        raise NotImplementedError("Override me!")
        
class LLMZPKIterativeOperatorLearningModule(BaseLLMIterativeOperatorLearningModule):
    def __init__(self, learned_operators, domain_name, llm):
        self.learner = ZPKOperatorLearningModule(learned_operators, domain_name)

        super().__init__(learned_operators, domain_name, llm)

    def _update_operator_rep(self, ops:list[Operator]) -> bool:
        """Update the PDDL operator representation, such as NDRs or FOLDTs.

        Args:
            ops (list[Operator])

        Returns:
            bool: if the operators have changed
        """
        self._ndrs = self.learner._ndrs
        # update NDRs and learned_operators
        raise NotImplementedError("TODO")
        return True