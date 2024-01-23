# TODO: experiment 6, see why success rate is so low.
#TODO: experiment 6, iteration 300: shouldn't the next goal (combined with learner's preconditions) also be sampled?
#TODO: Since target preconditions, should the score be revised to value preconditions more?

#TODO: if LLM has no learned operators, then pick some operator preconditions to try first (operators with action that has most no-ops should be prioritized).
#TODO: implement LLM precondition targets for GLIB-L2
#TODO: run experiment
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
        
    def reset_episode(self, state):
        self._recompute_unseen_lits_acts(state)
        self._objects = state.objects
        self._last_state = set()
        self._plan = []

    def priority_goal_action(self, goal_action):
        tiebreak = self._rand_state.uniform()
        if goal_action in self._llm_goal_actions:
            return (-1, tiebreak)
        return (len(goal_action[0]), tiebreak)

    def _goal_is_valid(self, goal):
        return not (goal is None)

    def _finish_plan(self, plan):
        self._last_state = None
        return plan + [self._last_sampled_action]


    def _get_action(self, state):
        if self._unseen_lits_acts is None:
            self._recompute_unseen_lits_acts(state)
        action = super()._get_action(state)

        # update novelty
        for lit in state:
            if ((lit,), action) in self._unseen_lits_acts:
                self._unseen_lits_acts.remove((lit, action))

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
        if itr % ac.LLM_learn_interval[self._domain_name] == 0:
            self._llm_goal_actions = []
            self._recompute_llm_goal_actions()
            for goal_action in self._llm_goal_actions:
                self._unseen_lits_acts.append(goal_action)

            self._unseen_lits_acts = sorted(self._unseen_lits_acts, key=self.priority_goal_action)

    def _recompute_llm_goal_actions(self):
        """Add the learner's operators preconditions to the LLM proposed preconditions with all combinations.
        """
        self._llm_goal_actions = []
        for op in self._llm_learned_ops:
            learner_op = self._llm_learned_ops[op]
            learned_lits_combinations = []
            if learner_op is not None:
                op_lits = [l.predicate for l in op.preconds.literals]
                learned_lits = []
                for lit in learner_op.preconds.literals:
                    if lit.predicate not in op_lits:
                        learned_lits.append(lit)
                for i in range(1, len(learned_lits)+1):
                    for lits in combinations(learned_lits, i):
                        learned_lits_combinations.append(lits)

            learned_lits_combinations.append(tuple())
            for addition in learned_lits_combinations:
                literals = [l for l in addition] + op.preconds.literals
                for preconds,_ in ground_literals(literals, self._objects):
                    action = [p for p in preconds
                        if p.predicate in self._action_space.predicates][0]
                    goal = tuple(sorted(set(preconds) - {action}))
                    self._llm_goal_actions.append((goal, action))

        # print("\n\nUpdated LLM Goal/Actions\n")
        # print(self._llm_goal_actions)

    def _structify_goal(self, goal):
        """Create Exists struct for a goal."""
        body = structs.LiteralConjunction(goal)
        return body