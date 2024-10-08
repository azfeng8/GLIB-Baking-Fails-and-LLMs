"""Goal-literal babbling with grounded novelty. Outputs single-literal goals and
also actions.
"""

import numpy as np
from settings import AgentConfig as ac
from curiosity_modules.goal_babbling import GoalBabblingCuriosityModule
from pddlgym import structs
from operator_learning_modules.llm_plus.operator_search import ground_literals
from itertools import combinations
import logging

class GLIBG1CuriosityModule(GoalBabblingCuriosityModule):
    _ignore_statics = True

    def _initialize(self):
        self._num_steps = 0
        self._rand_state = np.random.RandomState(seed=ac.seed)
        self._name = "glibg1"
        self._static_preds = self._compute_static_preds()
        self._visited_state_action_pairs = set()
        # Keep track of the number of times that we follow a plan
        self.line_stats = []
        self.llm_line_stats = []

    def reset_episode(self, state, ops=None):
        """Recompute the set of ground literals to sample from.

        Args:
            state (pddlgym.structs.State): Starting state.
            ops (set[Operator], optional): New ops from LLM-iterative method, if they exist. Defaults to None.
        """
        self._start_state = state
        self._visited_state_action_pairs = None
        self._last_state = set()
        self._plan = []

    def _get_action(self, state, goal):
        if self._visited_state_action_pairs is None:
            self._visited_state_action_pairs = set()
            self._start_state = state
        in_plan, operator_name, action = super()._get_action(state, goal)
        for lit in state:  # update novelty
            self._visited_state_action_pairs.add(((lit, action)))
        return in_plan, operator_name, action

    def learning_callback(self):
        super().learning_callback()
        self._static_preds = self._compute_static_preds()
        self._visited_state_action_pairs = None

    def _sample_grounding(self, predicate, objects):
        """Sample a grounded predicate given the list of objects (pddlgym.TypedEntity)."""
        args = []
        for i in np.random.permutation(np.arange(len(objects))):
            o = objects[i]
            next_idx = len(args)
            if len(predicate.var_types) < next_idx + 1:
                break
            object_name, object_type = o._str.split(':')
            if object_type == predicate.var_types[next_idx]:
                args.append(o)
        if len(args) == len(predicate.var_types):
            ground_pred = predicate(*args)
            return ground_pred
        else:
            return None
    def _sample_goal(self, state):
        """
        Returns:
            goal
            from_llm (bool): False, the goal is not from the LLM
        """
        if not self._visited_state_action_pairs:
            return None, False
        goal_act = None
        while goal_act is None or goal_act in self._visited_state_action_pairs:
            # Sample a random goal
            if self._domain_name.lower() == 'bakingrealistic':
                goal_pred = np.random.permutation([p for p in self._observation_space.predicates if p.name not in ('different', 'name-less-than')])[0]
            else:
                goal_pred = np.random.permutation(self._observation_space.predicates)[0]
            ground_goal = self._sample_grounding(goal_pred, list(self._start_state.objects))
            if ground_goal is None or ground_goal in self._static_preds:
                goal_act = None
                continue
            act_pred = np.random.permutation(self._action_space.predicates)[0]
            ground_action = self._sample_grounding(act_pred, list(self._start_state.objects))
            if ground_action is None:
                goal_act = None
                continue
            goal_act = (ground_goal, ground_action)
        goal, act = goal_act
        return goal, False

    def _goal_is_valid(self, goal):
        return not (goal is None)

    def _finish_plan(self, plan):
        self._last_state = None
        if len(plan) == 0:
            self.line_stats.append('EMPTY PLAN - babbled')
        return plan

    def observe(self, state, action, effects):
        pass

# class GLIBG2CuriosityModule(GoalBabblingCuriosityModule):
#     #TODO: mutex goals
#     _ignore_statics = True
#     _ignore_mutex = True

#     def _initialize(self):
#         self._num_steps = 0
#         self._rand_state = np.random.RandomState(seed=ac.seed)
#         self._name = "glibg2"
#         self._static_preds = self._compute_static_preds()
#         # Keep track of the number of times that we follow a plan
#         self.line_stats = []
#         self.llm_line_stats = []

#     def reset_episode(self, state, ops=None):
#         """Recompute the set of ground literals to sample from.

#         Args:
#             state (pddlgym.structs.State): Starting state.
#             ops (set[Operator], optional): New ops from LLM-iterative method, if they exist. Defaults to None.
#         """
#         self._recompute_unseen_lits_acts(state)
#         self._last_state = set()
#         self._plan = []
#         # if self._ignore_mutex:
#         #     mutex_pairs = self._compute_ground_mutex_pairs(state)
#         #     self._unseen_lits_acts = list(filter(
#         #             lambda ga: frozenset(ga[0]) not in mutex_pairs,
#         #             self._unseen_lits_acts))   


#     def _compute_ground_mutex_pairs(self, state):
#         raise NotImplementedError

#     def _recompute_unseen_lits_acts(self, state):
#         self._unseen_lits_acts = set()
#         obs_ground_lits = list(self._observation_space.all_ground_literals(state))
#         action_ground_lits = self._action_space.all_ground_literals(state)
#         for i, lit in enumerate(obs_ground_lits):
#             if self._ignore_statics and \
#             lit.predicate in self._static_preds:  # ignore static goals
#                 continue
#             for lit2 in obs_ground_lits[i:]:
#                 if self._ignore_statics and \
#                 lit2.predicate in self._static_preds:  # ignore static goals
#                     continue
#                 if lit == lit2: continue
#                 for act in action_ground_lits:
#                         self._unseen_lits_acts.add(((lit, lit2), act, False))
#         self._unseen_lits_acts = sorted(self._unseen_lits_acts)

#     def _get_action(self, state):
#         if self._unseen_lits_acts is None:
#             self._recompute_unseen_lits_acts(state)
#         action = super()._get_action(state)
#         state_lits = list(state)
#         for i,lit in enumerate(state_lits):  # update novelty
#             for lit2 in state_lits[i:]:
#                 if lit == lit2: continue
#                 if ((lit, lit2), action, False) in self._unseen_lits_acts:
#                     self._unseen_lits_acts.remove(((lit, lit2), action, False))
#                 if ((lit2, lit), action, False) in self._unseen_lits_acts:
#                     self._unseen_lits_acts.remove(((lit2, lit), action, False))
#         return action

#     def learning_callback(self):
#         super().learning_callback()
#         self._static_preds = self._compute_static_preds()
#         self._unseen_lits_acts = None

#     def _sample_goal(self, state):
#         """
#         Returns:
#             goal
#             from_llm (bool): False, the goal is not from the LLM
#         """
#         if not self._unseen_lits_acts:
#             return None, False
#         goal, act, _ = self._unseen_lits_acts[self._rand_state.choice(
#             len(self._unseen_lits_acts))]
#         goal = structs.LiteralConjunction(goal)
#         self._last_sampled_action = act
#         return goal, False

#     def _goal_is_valid(self, goal):
#         return not (goal is None)

#     def _finish_plan(self, plan):
#         self._last_state = None
#         if len(plan) == 0:
#             self.line_stats.append('EMPTY PLAN - babbled')
#         return plan + [self._last_sampled_action]

#TODO: update the goal sampling as in GLIB_G1
# class GLIBG1LLMCuriosityModule(GoalBabblingCuriosityModule):
#     _ignore_statics = True

#     def _initialize(self):
#         self._llm_goal_actions = []
#         self._rand_state = np.random.RandomState(seed=ac.seed)
#         self._static_preds = self._compute_static_preds()
#         self.line_stats = []
#         self.llm_line_stats = []
#         self._num_steps = 0
#         self._name = "llm-glibg1"

#     def _sample_goal(self, state):
#         """
#             Returns:
#                 goal
#                 from_llm (bool): if the goal was from the LLM
#         """
#         if not self._unseen_lits_acts:
#             return None, False
#         goal, act, from_llm = self._unseen_lits_acts.pop(0)
#         self._last_sampled_action = act
#         return self._structify_goal(goal), from_llm

#     def learning_callback(self):
#         super().learning_callback()
#         self._static_preds = self._compute_static_preds()
#         self._unseen_lits_acts = None

#     def reset_episode(self, state):
#         self._recompute_unseen_lits_acts(state)
#         self._objects = state.objects
#         self._last_state = set()
#         self._plan = []

#     def _goal_is_valid(self, goal):
#         return not (goal is None)

#     def _finish_plan(self, plan):
#         self._last_state = None
#         return plan + [self._last_sampled_action]


#     def _get_action(self, state):
#         """Goal-babble and attempt to sample a goal to get an action, or fallback to random. Update the novelty measure.

#         Args:
#             state (_type_): _description_

#         Returns:
#             Literal: grounded action
#         """
#         if self._unseen_lits_acts is None:
#             self._recompute_unseen_lits_acts(state)
#         action = super()._get_action(state)

#         # update novelty
#         for lit in state:
#             if ((lit,), action, False) in self._unseen_lits_acts:
#                 self._unseen_lits_acts.remove(((lit,), action, False))

#         removes = set()
#         for goal,act,from_llm in self._unseen_lits_acts:
#             all_in_state = True
#             for g in goal:
#                 if g.predicate.is_negative:
#                     if g.predicate.positive(*g.variables) in state:
#                         all_in_state = False
#                 elif g not in state:
#                     all_in_state = False
#             if all_in_state:
#                 removes.add((goal,act,from_llm))
#         for r in removes:
#             self._unseen_lits_acts.remove(r)
                    
#         return action

#     def _recompute_unseen_lits_acts(self, state):

#         self._objects = state.objects

#         self._unseen_lits_acts = set()
#         for lit in self._observation_space.all_ground_literals(state):
#             if self._ignore_statics and \
#                lit.predicate in self._static_preds:  # ignore static goals
#                 continue
#             for act in self._action_space.all_ground_literals(state):
#                 self._unseen_lits_acts.add(((lit,), act, False))

#         self._recompute_llm_goal_actions()
#         for goal, action in self._llm_goal_actions:
#             self._unseen_lits_acts.add((goal, action, True))

#         self._unseen_lits_acts = sorted(self._unseen_lits_acts, key=self.priority_goal_action)


#     def learn(self, itr):
#         """Update the goal action priorities from the LLM.

#         Args:
#             itr (int): iteration #
#         """
#         if itr % ac.LLM_learn_interval[self._domain_name] == 0:
#             self._recompute_llm_goal_actions()
#             for goal,action in self._llm_goal_actions:
#                 self._unseen_lits_acts.append((goal, action, True))

#             self._unseen_lits_acts = sorted(self._unseen_lits_acts, key=self.priority_goal_action)

#     def _recompute_llm_goal_actions(self):
#         """Add the learner's operators preconditions to the LLM proposed preconditions with all combinations.
#         """
#         self._llm_goal_actions = []
#         for op in self._llm_learned_ops:
#             learner_op = self._llm_learned_ops[op]
#             learned_lits_combinations = []
#             if learner_op is not None:
#                 op_lits = [l.predicate for l in op.preconds.literals]
#                 learned_lits = []
#                 for lit in learner_op.preconds.literals:
#                     if lit.predicate not in op_lits:
#                         learned_lits.append(lit)
#                 for i in range(1, len(learned_lits)+1):
#                     for lits in combinations(learned_lits, i):
#                         learned_lits_combinations.append(lits)

#             learned_lits_combinations.append(tuple())
#             for addition in learned_lits_combinations:
#                 literals = [l for l in addition] + op.preconds.literals
#                 for preconds,_ in ground_literals(literals, self._objects):
#                     action = [p for p in preconds
#                         if p.predicate in self._action_space.predicates][0]
#                     goal = tuple(sorted(set(preconds) - {action}))
#                     if len(goal) == 0:
#                         continue
#                     self._llm_goal_actions.append((goal, action))

#         # print("\n\nUpdated LLM Goal/Actions\n")
#         # print(self._llm_goal_actions)

#     def _structify_goal(self, goal):
#         """Create LiteralConjunction struct for a goal."""
#         if len(goal) == 1:
#             return goal[0]
#         else:
#             return structs.LiteralConjunction(goal)

#     def priority_goal_action(self, goal_action):
#         tiebreak = self._rand_state.uniform()
#         if goal_action in self._llm_goal_actions:
#             #TODO: consider return (-1, len(goal_action[0]), tiebreak)
#             return (-1, tiebreak)
#         return (len(goal_action[0]), tiebreak)

