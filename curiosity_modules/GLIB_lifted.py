from curiosity_modules.goal_babbling import GoalBabblingCuriosityModule 
from settings import AgentConfig as ac
from pddlgym import structs
from pddlgym.inference import find_satisfying_assignments

from collections import defaultdict
import copy
import pickle
import itertools
import numpy as np
import logging
import time

GLIB_L_LOGGER = logging.getLogger("GLIB_Lifted")


class GLIBLCuriosityModule(GoalBabblingCuriosityModule):

    _k = None # Must be set by subclasses
    _ignore_statics = True
    _ignore_mutex = True

    ### Initialization ###

    def _initialize(self):
        super()._initialize()
        self.llm_line_stats = []
        self._rand_state = np.random.RandomState(seed=ac.seed)
        self._name = "newiw"
        self._episode_start_state = None
        self._seen_state_actions = set()

        if self._domain_name.lower() == 'bakingrealistic':
            self._sampling_iterator = self._yield_goal_action(self._action_space.predicates, [p for p in self._observation_space.predicates if p.name not in ('different', 'name-less-than')], self._k)
        else:
            self._sampling_iterator = self._yield_goal_action(self._action_space.predicates, self._observation_space.predicates, self._k)

    @classmethod
    def _use_goal_preds(cls, goal_preds):
        return True

    def _yield_goal_action(self, action_predicates, observation_predicates, max_num_lits):
        """Returns a generator that samples a pair of (goal_literal, action) uniformly at random and yields
        it, until all lifted goals have been yielded. Useful for generating goals to try to plan towards for the
        purpose of exploration.

        Parameters
        ----------
        action_predicates : { Predicate }
        observation_predicates : { Predicate }
        max_num_lits : max_num_lits
        Returns
        -------
        unseen_goal_actions : (tuple(Literal), Literal)
            Tuples of (goal literals, action, from_LLM). Note that the
            final element (from_LLM) will always be false in this case.
        """
        # When all of the lifted goals are visited, there needs to be a way to stop this loop.
        for num_lits in np.random.permutation(np.arange(1, max_num_lits+1)):
            goal_preds_generator = generator_goal_literals(observation_predicates, num_lits)
            for goal_preds in goal_preds_generator:
                for action_pred_index in np.random.permutation(np.arange(len(action_predicates))):
                    action_pred = action_predicates[action_pred_index]
                    if not self._use_goal_preds(goal_preds):
                        continue
                    goal_action_preds = list(goal_preds) + [action_pred]
                    # Find all possible variable assortments for the goal action predicates
                    # We're going to first create unique placeholders for the slots in the predicates
                    ph_to_pred_slot = {}
                    goal_action_lits_with_phs = []
                    # Create the placeholders
                    for i, pred in enumerate(goal_action_preds):
                        ph_for_lit = []
                        for j, var_type in enumerate(pred.var_types):
                            ph = var_type('ph{}_{}'.format(i, j))
                            ph_to_pred_slot[ph] = (i, j)
                            ph_for_lit.append(ph)
                        ph_lit = pred(*ph_for_lit)
                        goal_action_lits_with_phs.append(ph_lit)
                    phs = sorted(ph_to_pred_slot.keys())
                    # Consider a random substitution of placeholders to variables.
                    tuple_of_phs = tuple(self._iter_vars_from_phs(phs))
                    vs = tuple_of_phs[np.random.choice(len(tuple_of_phs))]
                    goal_vs = {v.name for i, v in enumerate(vs) if
                            ph_to_pred_slot[phs[i]][0] != len(goal_action_preds)-1} # len(goal_action_preds) - 1 indexes the action, so exclude those variables.
                    action_vs = {v.name for i, v in enumerate(vs) if
                                ph_to_pred_slot[phs[i]][0] == len(goal_action_preds)-1}
                    # If there are different variables in the action literal than the goal literals, 
                    # then the variables in the action literal that don't exist in the goal literals
                    # must take on a greater or equal index ?x# than the maximum index variable in the goal literals.
                    if goal_vs and action_vs-goal_vs and min(action_vs-goal_vs) < max(goal_vs):
                        continue
                    goal_action_lits = [copy.deepcopy(lit) for lit in goal_action_lits_with_phs]
                    # Perform substitution
                    for k, v in enumerate(vs):
                        ph = phs[k]
                        (i, j) = ph_to_pred_slot[ph]
                        goal_action_lits[i].update_variable(j, v)
                    # Goal lits cannot have repeated vars
                    goal_action_valid = True
                    for lit in goal_action_lits:
                        if len(set(lit.variables)) != len(lit.variables):
                            goal_action_valid = False
                            break
                    # Finish the goal and add it
                    if goal_action_valid:
                        goal = tuple([l for l in goal_action_lits if l.predicate != action_pred])
                        action = [l for l in goal_action_lits if l.predicate == action_pred][0]
                        yield goal, action, False


    ### Reset ###

    def _iw_reset(self):
        if self._ignore_statics:  # ignore static goals
            start = time.time()
            self._goal_static_preds = self._compute_static_preds()
            logging.info(f"Static preds compute took {time.time() - start} s")
        if self._ignore_mutex:  # ignore mutex goals
            start = time.time()
            self._goal_mutex_pairs = self._compute_lifted_mutex_literals(self._episode_start_state)
            logging.info(f"Goal mutex pairs compute took {time.time() - start} s")
        # Forget the goal-action that was going to be taken at the end of the plan in progress
        if self._domain_name.lower() == 'bakingrealistic':
            start = time.time()
            self._sampling_iterator = self._yield_goal_action(self._action_space.predicates, [p for p in self._observation_space.predicates if p.name not in ('different', 'name-less-than')], self._k)
            logging.info(f'Creating goal sampler generator took {time.time() - start} s')
        else:
            self._sampling_iterator = self._yield_goal_action(self._action_space.predicates, self._observation_space.predicates, self._k)
        self._current_goal_action = None

    def _get_goal_action_priority(self, goal_action):
        return (len(goal_action[0]), self._rand_state.uniform())

    def reset_episode(self, state):
        super().reset_episode(state)
        self._episode_start_state = state
        self._iw_reset()

    def learning_callback(self):
        super().learning_callback()
        self._iw_reset()

    def _get_fallback_action(self, state):
        GLIB_L_LOGGER.debug("Get fallback action, setting self._current_goal_action to None")
        self._current_goal_action = None
        return super()._get_fallback_action(state)

    ### Get an action ###

    def _get_action(self, state):
        # First check whether we just finished a plan and now must take the final action
        if (not (self._current_goal_action is None)) and (len(self._plan) == 0):
            action = self._get_ground_action_to_execute(state)
            # GLIB_L_LOGGER.debug("*** Finished the plan")
            if action != None:
                # GLIB_L_LOGGER.debug("*** Finished the plan, now executing the action")
                # Execute the action
                self.line_stats.append('FINISHED PLAN - babbled')
                return False, None, action
        # Either continue executing a plan or make a new one (or fall back to random)
        return super()._get_action(state)

    def _get_ground_action_to_execute(self, state):
        lifted_goal, lifted_action = self._current_goal_action
        # Forget this goal-action because we're about to execute it
        # GLIB_L_LOGGER.debug("Setting None in _get_ground_action_to_execute")
        self._current_goal_action = None
        # Sample a grounding for the action conditioned on the lifted goal and state
        action = self._sample_action_from_goal(lifted_goal, lifted_action, state, self._rand_state)
        # If the action is None, that means that the plan was wrong somehow.
        return action

    @staticmethod
    def _sample_action_from_goal(lifted_goal, lifted_action, state, rand_state):
        """Sample a grounding for the action conditioned on the lifted goal and state"""
        # Try to find a grounding of the lifted goal in the state
        all_assignments = find_satisfying_assignments(state.literals, lifted_goal,
            allow_redundant_variables=False)
        # If none exist, return action None
        if len(all_assignments) == 0:
            return None
        assignments = all_assignments[0]
        # Sample an action conditioned on the assignments.
        # Find possible groundings for each object by type.
        types_to_objs = defaultdict(set)
        for lit in state.literals:
            for obj in lit.variables:
                types_to_objs[obj.var_type].add(obj)
        # Sample a grounding for all the unbound variables.
        grounding = []
        for v in lifted_action.variables:
            if v in assignments:
                # Variable is ground, so go with ground variable.
                grounding.append(assignments[v])
            else:
                # Sample a grounding. Make sure it's not an assigned value.
                choices = set(types_to_objs[v.var_type])
                choices -= set(assignments.values())
                choices -= set(grounding)
                # There's no way to bind the variables of the action.
                if len(choices) == 0:
                    return None
                choice = sorted(choices)[rand_state.choice(len(choices))]
                grounding.append(choice)
        assert len(grounding) == len(set(grounding))
        return lifted_action.predicate(*grounding)

    def _sample_goal(self, state):
        """Produce a new goal to try to plan towards
        
        Returns:
            goal
            from_llm (bool): whether the goal is from the LLM"""
        goal_action_val = None
        # Sample all of the lifted (goal, action) pairs in random order, until they run out.
        while goal_action_val is None:
            try:
                goal_action_val = next(self._sampling_iterator)
            except StopIteration:
                # Sampled all of the lifted goals, so stop sampling.
                self._current_goal_action = None 
                return None, False
            goal, lifted_action, from_llm = goal_action_val
            # Ignore static
            if self._ignore_statics and any(lit.predicate in self._goal_static_preds for lit in goal):
                continue
            # Ignore mutex
            if self._ignore_mutex and goal in self._goal_mutex_pairs:
                continue
            for state, action in self._seen_state_actions:
                goal_assignments = find_satisfying_assignments(state.literals, tuple([l for l in goal]), allow_redundant_variables=False)
                action_assignment = find_satisfying_assignments([action], [lifted_action], allow_redundant_variables=False)
                if len(goal_assignments) != 0 and len(action_assignment) != 0:
                    goal_action_val = None
                    break
                
        self._current_goal_action = (goal, lifted_action)
        return self._structify_goal(goal), from_llm

    def _finish_plan(self, plan):
        # If the plan is empty, then we want to immediately take the action.
        if len(plan) == 0:
            action = self._get_ground_action_to_execute(self._last_state)
            # print("Goal is satisfied in the current state; taking action now:", action)
            if action is None:
                # There was no way to bind the lifted action. Fallback
                action = self._get_fallback_action(self._last_state)
            else:
                self.line_stats.append('EMPTY PLAN - babbled')
            return [action]
        # Otherwise, we'll take the last action once we finish the plan
        # print("Setting a plan:", plan)
        return plan

    def _goal_is_valid(self, goal):
        return not (goal is None)

    def _plan_is_good(self):
        return True

    @staticmethod
    def _structify_goal(goal):
        """Create Exists struct for a goal."""
        variables = sorted({ v for lit in goal for v in lit.variables })
        body = structs.LiteralConjunction(goal)
        return structs.Exists(variables, body)

    ### Update caches based on new observation ###

    def observe(self, state, action, _effects):
        self._seen_state_actions.add((state, action))

class GLIBL1CuriosityModule(GLIBLCuriosityModule):
    _k = 1
class GLIBL2CuriosityModule(GLIBLCuriosityModule):
    _k = 2

def generator_goal_literals(obs_predicates: list, num_lits: int):
    """Yields the cartesian product of predicates in a random order."""
    n = len(obs_predicates)
    for random_indices in itertools.product(*[np.random.permutation(np.arange(n)) for _ in range(num_lits)]):
        yield [obs_predicates[i] for i in random_indices]
# class LLMGLIBL2CuriosityModule(GLIBL2CuriosityModule):

#     def _initialize(self):
#         self.llm_line_stats = []
#         super()._initialize()

#     ### Update goals with LLM proposed operator preconditions
    
#     def learn(self, itr):
#         """Set self._llm_goal_actions with the LLM and learner operators."""
#         j = (itr - ac.LLM_start_interval[self._domain_name])
#         if (j>=0) and (j % ac.LLM_learn_interval[self._domain_name] == 0):
#             self._recompute_llm_goal_actions()
#             self._unseen_goal_actions.update(self._llm_goal_actions)
#             self._untried_episode_goal_actions.extend(self._llm_goal_actions)
#             self._untried_episode_goal_actions = sorted(self._untried_episode_goal_actions, key=self._get_goal_action_priority)
#             if self._ignore_statics:  # ignore static goals
#                 static_preds = self._compute_static_preds()
#                 self._untried_episode_goal_actions = list(filter(
#                     lambda ga: any(lit.predicate not in static_preds for lit in ga[0]),
#                     self._untried_episode_goal_actions))
#             if self._ignore_mutex:  # ignore mutex goals
#                 mutex_pairs = self._compute_lifted_mutex_literals(self._episode_start_state)
#                 self._untried_episode_goal_actions = list(filter(
#                     lambda ga: frozenset(ga[0]) not in mutex_pairs,
#                     self._untried_episode_goal_actions))   

#     def _get_goal_action_priority(self, goal_action):
#         tiebreak = self._rand_state.uniform()
#         if goal_action in self._llm_goal_actions:
#             return (-1, len(goal_action[0]), tiebreak)
#         return (1, len(goal_action[0]), tiebreak)
    
#     def _recompute_llm_goal_actions(self):
#         """Get the (goal, action) tuples from the LLM proposed operators.
#         """
#         self._llm_goal_actions = []
#         for o in self._llm_learned_ops:
#             if self._llm_learned_ops[o] is not None:
#                 combined_preconds = self.mix_lifted_preconditions(o, self._llm_learned_ops[o])

#                 for precond in combined_preconds:
#                     action = [p for p in precond
#                                     if p.predicate in self._action_space.predicates][0]
#                     precond.remove(action) 
#                     self._llm_goal_actions.append((tuple(precond), action))
#             else:
#                 action = [p for p in o.preconds.literals
#                         if p.predicate in self._action_space.predicates][0]
#                 goal = tuple(sorted(set(o.preconds.literals) - {action}))
#                 self._llm_goal_actions.append((goal, action, True))
