#TODO: if LLM has no learned operators, then pick some operator preconditions to try first (operators with action that has most no-ops should be prioritized).

"""Goal-literal babbling with grounded novelty. Outputs single-literal goals and
also actions.
"""

import numpy as np
from settings import AgentConfig as ac
from curiosity_modules.goal_babbling import GoalBabblingCuriosityModule
from pddlgym import structs
from operator_learning_modules.llm_plus.operator_search import ground_literals
from itertools import combinations

class GLIBG1CuriosityModule(GoalBabblingCuriosityModule):
    _ignore_statics = True

    def _initialize(self):
        self._num_steps = 0
        self._rand_state = np.random.RandomState(seed=ac.seed)
        self._name = "glibg1"
        self._static_preds = self._compute_static_preds()
        # Keep track of the number of times that we follow a plan
        self.line_stats = []

    def reset_episode(self, state, ops=None):
        """Recompute the set of ground literals to sample from.

        Args:
            state (pddlgym.structs.State): Starting state.
            ops (set[Operator], optional): New ops from LLM-iterative method, if they exist. Defaults to None.
        """
        self._recompute_unseen_lits_acts(state)
        self._last_state = set()
        self._plan = []

    def _recompute_unseen_lits_acts(self, state):
        self._unseen_lits_acts = set()
        for lit in self._observation_space.all_ground_literals(state):
            if self._ignore_statics and \
               lit.predicate in self._static_preds:  # ignore static goals
                continue
            for act in self._action_space.all_ground_literals(state):
                self._unseen_lits_acts.add((lit, act))
        self._unseen_lits_acts = sorted(self._unseen_lits_acts)

    def _get_action(self, state):
        if self._unseen_lits_acts is None:
            self._recompute_unseen_lits_acts(state)
        action = super()._get_action(state)
        for lit in state:  # update novelty
            if (lit, action) in self._unseen_lits_acts:
                self._unseen_lits_acts.remove((lit, action))
        return action

    def learning_callback(self):
        super().learning_callback()
        self._static_preds = self._compute_static_preds()
        self._unseen_lits_acts = None

    def _sample_goal(self, state):
        if not self._unseen_lits_acts:
            return None
        goal, act = self._unseen_lits_acts[self._rand_state.choice(
            len(self._unseen_lits_acts))]
        self._last_sampled_action = act
        return goal

    def _goal_is_valid(self, goal):
        return not (goal is None)

    def _finish_plan(self, plan):
        self._last_state = None
        return plan + [self._last_sampled_action]

class GLIBG1LLMCuriosityModule(GoalBabblingCuriosityModule):
    _ignore_statics = True

    def _initialize(self):
        self._llm_goal_actions = []
        self._rand_state = np.random.RandomState(seed=ac.seed)
        self._static_preds = self._compute_static_preds()
        self.line_stats = []
        self._num_steps = 0
        self._name = "llm-glibg1"

    def _sample_goal(self, state):
        if not self._unseen_lits_acts:
            return None
        goal, act = self._unseen_lits_acts.pop(0)
        self._last_sampled_action = act
        return self._structify_goal(goal)

    def learning_callback(self):
        super().learning_callback()
        self._static_preds = self._compute_static_preds()
        self._unseen_lits_acts = None

    def reset_episode(self, state):
        self._recompute_unseen_lits_acts(state)
        self._objects = state.objects
        self._last_state = set()
        self._plan = []

    def _goal_is_valid(self, goal):
        return not (goal is None)

    def _finish_plan(self, plan):
        self._last_state = None
        return plan + [self._last_sampled_action]


    def _get_action(self, state):
        """Goal-babble and attempt to sample a goal to get an action, or fallback to random. Update the novelty measure.

        Args:
            state (_type_): _description_

        Returns:
            Literal: grounded action
        """
        if self._unseen_lits_acts is None:
            self._recompute_unseen_lits_acts(state)
        action = super()._get_action(state)

        # update novelty
        for lit in state:
            if ((lit,), action) in self._unseen_lits_acts:
                self._unseen_lits_acts.remove(((lit,), action))

        removes = set()
        for goal,act in self._unseen_lits_acts:
            all_in_state = True
            for g in goal:
                if g.predicate.is_negative:
                    if g.predicate.positive(*g.variables) in state:
                        all_in_state = False
                elif g not in state:
                    all_in_state = False
            if all_in_state:
                removes.add((goal,act))
        for r in removes:
            self._unseen_lits_acts.remove(r)
                    
        return action

    def _recompute_unseen_lits_acts(self, state):

        self._objects = state.objects

        self._unseen_lits_acts = set()
        for lit in self._observation_space.all_ground_literals(state):
            if self._ignore_statics and \
               lit.predicate in self._static_preds:  # ignore static goals
                continue
            for act in self._action_space.all_ground_literals(state):
                self._unseen_lits_acts.add(((lit,), act))

        self._recompute_llm_goal_actions()
        for goal_action in self._llm_goal_actions:
            self._unseen_lits_acts.add(goal_action)

        self._unseen_lits_acts = sorted(self._unseen_lits_acts, key=self.priority_goal_action)


    def learn(self, itr):
        """Update the goal action priorities from the LLM.

        Args:
            itr (int): iteration #
        """
        j = (itr - ac.LLM_start_interval[self._domain_name])
        if (j>=0) and (j % ac.LLM_learn_interval[self._domain_name] == 0):
            self._recompute_llm_goal_actions()
            for goal_action in self._llm_goal_actions:
                self._unseen_lits_acts.append(goal_action)

            self._unseen_lits_acts = sorted(self._unseen_lits_acts, key=self.priority_goal_action)

    def _recompute_llm_goal_actions(self):
        """Add the learner's operators preconditions to the LLM proposed preconditions with all combinations.
        """
        self._llm_goal_actions = []
        for op in self._llm_precondition_goal_ops:
            for preconds,_ in ground_literals(op.preconds.literals, self._objects):
                action = [p for p in preconds
                    if p.predicate in self._action_space.predicates][0]
                goal = tuple(sorted(set(preconds) - {action}))
                self._llm_goal_actions.append((goal, action))


        # for op in self._llm_precondition_goal_ops:
        #     learner_op = self._llm_precondition_goal_ops[op]
        #     if learner_op is not None:
        #         learned_lits_combinations = self.mix_lifted_preconditions(op, learner_op)
        #     else:
        #         learned_lits_combinations = [op.preconds.literals]
        #     for literals in learned_lits_combinations:
        #         for preconds,_ in ground_literals(literals, self._objects):
        #             action = [p for p in preconds
        #                 if p.predicate in self._action_space.predicates][0]
        #             goal = tuple(sorted(set(preconds) - {action}))
        #             self._llm_goal_actions.append((goal, action))

        # print("\n\nUpdated LLM Goal/Actions\n")
        # print(self._llm_goal_actions)

    def _structify_goal(self, goal):
        """Create LiteralConjunction struct for a goal."""
        if len(goal) == 1:
            return goal[0]
        else:
            return structs.LiteralConjunction(goal)

    def priority_goal_action(self, goal_action):
        tiebreak = self._rand_state.uniform()
        if goal_action in self._llm_goal_actions:
            #TODO: consider return (-1, len(goal_action[0]), tiebreak)
            return (-1, tiebreak)
        return (len(goal_action[0]), tiebreak)

