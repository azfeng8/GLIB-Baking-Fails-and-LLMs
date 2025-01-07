"""Goal-literal babbling with grounded novelty. Outputs single-literal goals and
also actions.
"""

import numpy as np
from settings import AgentConfig as ac
from curiosity_modules.goal_babbling import GoalBabblingCuriosityModule
from pddlgym import structs
from pddlgym.parser import Operator
from pddlgym.structs import Anti, Type, LiteralConjunction, Literal,TypedEntity
import itertools
from typing import Iterable
import logging

class GLIBG1CuriosityModule(GoalBabblingCuriosityModule):
    _ignore_statics = True

    def _initialize(self):
        self._num_steps = 0
        self._name = "glibg1"
        self._static_preds = self._compute_static_preds()
        self._visited_state_action_pairs = set()
        self._last_action = None
        # Keep track of the number of times that we follow a plan
        self.line_stats = []
        self.llm_line_stats = []

    def reset_episode(self, state, ops=None):
        """Recompute the set of ground literals to sample from.

        Args:
            state (pddlgym.structs.State): Starting state.
            ops (set[Operator], optional): New ops from LLM-iterative method, if they exist. Defaults to None.
        """
        self._sampling_iterator = self._yield_goal_action_pairs(state)
        self._visited_state_action_pairs = set() # Reset novelty, just as in original implementation
        self._start_state = state
        self._last_state = set()
        self._plan = []

    def _get_action(self, state, goal):
        in_plan, operator_name, action = super()._get_action(state, goal)
        return in_plan, operator_name, action

    def learning_callback(self):
        super().learning_callback()
        self._static_preds = self._compute_static_preds()

    def _yield_goal_action_pairs(self, state):
        """Generate all grounded (goal, action) pairs in a uniformly random order."""
        goals = sorted([p for p in self._observation_space.all_ground_literals(state) if p.predicate.name not in ('different', 'name-less-than')])
        actions = sorted(self._action_space.all_ground_literals(state))

        items = [np.arange(len(goals))] + [np.arange(len(actions))]
        gen = itertools.product(*items)
        num_in_gen = len(goals) * len(actions)
        generated_items = {} # Map from generation index to the item generated at that index
        next_idx_to_generate = 0
        for index in self._rand_state.permutation(num_in_gen):
            if index in generated_items:
                goal, action = generated_items[index]
                del generated_items[index]
                yield (goal, action)
            else:
                while next_idx_to_generate < index:
                    goal_i, action_i = next(gen)
                    generated_items[next_idx_to_generate] = (goals[goal_i], actions[action_i])
                    next_idx_to_generate += 1
                goal_i, action_i = next(gen)
                next_idx_to_generate += 1
                yield (goals[goal_i], actions[action_i])

    def _sample_goal(self, state):
        """
        Returns:
            goal
            from_llm (bool): False, the goal is not from the LLM
        
        """
        try:
            goal, action = next(self._sampling_iterator)
            while (goal, action) in self._visited_state_action_pairs:
                goal, action = next(self._sampling_iterator)
            self._last_action = action
            return goal, False
        except StopIteration:
            return None, False

    def _goal_is_valid(self, goal):
        return not (goal is None)

    def _finish_plan(self, plan):
        self._last_state = None
        if len(plan) == 0:
            self.line_stats.append('EMPTY PLAN - babbled')
        action = self._last_action
        self._last_action = None
        return plan + [action]

    def observe(self, state, action, effects):
        for lit in state:  # update novelty
            self._visited_state_action_pairs.add(((lit, action)))

def ground_literals(lifted_literals:Iterable[Literal], objects:frozenset[TypedEntity], partial_assignment={}) -> list[tuple[set[Literal], dict]]:
    """Get all possible groundings of lifted literals with the variable-to-object assignments.

    Args:
        lifted_precond (set[Literal]): _description_
        objects (frozenset[TypedEntity]): _description_

    Returns:
        [ (set[Literal], dict) ]: list of (sets of grounded literals, assignment)
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

