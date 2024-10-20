"""Goal-literal babbling with grounded novelty. Outputs single-literal goals and
also actions.
"""

import numpy as np
from settings import AgentConfig as ac
from curiosity_modules.goal_babbling import GoalBabblingCuriosityModule
from pddlgym import structs
from operator_learning_modules.llm_plus.operator_search import ground_literals
import itertools
import logging

class GLIBG1CuriosityModule(GoalBabblingCuriosityModule):
    _ignore_statics = True

    def _initialize(self):
        self._num_steps = 0
        self._rand_state = np.random.RandomState(seed=ac.seed)
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

