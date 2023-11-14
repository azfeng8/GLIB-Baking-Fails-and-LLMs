"""Utility to run using lifted sequences of goal,action pairs.

To use, provide the file following the format displayed in sample.txt:

goal_pred_0,?x1,?x2,..,goal_pred_1,arg0,arg1,...action_pred,arg0,arg1...
goal_pred_0,?x0,?x1,..,goal_pred_1,arg0,arg1,...action_pred,arg0,arg1...
...

Each line has 1+ goal predicates and args followed by 1 action predicate and args (k can be varied amongst lines).

Initially created for 2 reasons:
    1. to try sequences of goals (curriculum) to babble instead of random sampling a goal from a nontrivial goal set (non-static and non-mutex).
    2. to reproduce successful (100% success rate) sequences of goals on Baking domain, with a saver utility TBD.

"""

from curiosity_modules.goal_babbling import GoalBabblingCuriosityModule
from settings import AgentConfig as ac
from pddlgym import structs
from pddlgym.inference import find_satisfying_assignments

from collections import defaultdict
import numpy as np


class GLIBLSCuriosityModule(GoalBabblingCuriosityModule):

    _ignore_statics = True

    ### Initialization ###

    def _initialize(self):
        super()._initialize()
        self._rand_state = np.random.RandomState(seed=ac.seed)
        self._name = "newiw"
        self._episode_start_state = None
        self.observation_preds = {o.name: o for o in self._observation_space.predicates}
        self.action_preds = {o.name: o for o in self._action_space.predicates}


        self.random_actions = False
        self.random_goals = False
        self.directed_goals = False
        self.init_counts = 0
        # Index into the goal,action from the file
        self.index = 0
        self.goal_actions = self._parse_file_into_goal_action_preds(self._replay_fname)
        if len(self.goal_actions) == 0:
            raise Exception("No goal,action lines could be parsed")

    def _parse_file_into_goal_action_preds(self,fname):
        """Parse sequence of goal,action pairs.

        Args:
            fname (str): path to file

        Lines in the file look like:
        
        goal_pred_0,?x1,?x2,..,goal_pred_1,arg0,arg1,...action_pred,arg0,arg1...
        goal_pred_0,?x0,?x1,..,goal_pred_1,arg0,arg1,...action_pred,arg0,arg1...
        """
        fh = open(fname, 'r')
        goal_action_tuples = []
        goal_counts_per_line = []
        for line_no,line in enumerate(fh.readlines()):
            if "Starting..." in line: continue
            # All lines should have the same number of goals
            num_goals = 0
            found_action = False
            current_pred = None
            current_args = []
            # 2 elts, ((goal0(), goal1(), goal2(), ...),action())
            preds_with_args = [[]]
            for token in line.split(','):
                token = token.strip()
                if token in self.observation_preds:
                    num_goals += 1
                    # Not the first goal pred
                    if current_pred is not None:
                        preds_with_args[0].append(current_pred(*current_args))
                        current_args = []    
                    current_pred = self.observation_preds[token]
                elif token in self.action_preds:
                    goal_counts_per_line.append(num_goals)
                    found_action = True
                    preds_with_args[0].append(current_pred(*current_args))
                    current_args = []    
                    # Make the goal preds a nested tuple
                    preds_with_args[0] = tuple(preds_with_args[0])
                    current_pred = self.action_preds[token]
                else:
                    i = len(current_args)
                    var = current_pred.var_types[i](token)
                    current_args.append(var)
            if not found_action:
                raise Exception("Parsing error: action predicate not found at end of goal preds/args list")
            # `current_pred` is an action pred
            preds_with_args.append(current_pred(*current_args))
            goal_action_tuples.append(tuple(preds_with_args))
            # # (optional) make sure all lines have the same number of goal literals, error checking the parsing 
            # diff = 1 - ((np.array(goal_counts_per_line) / goal_counts_per_line[0]) == np.ones(len(goal_counts_per_line)))
            # if np.any(diff):
            #     raise Exception("Parsing error: number of goals different on lines 1,{}".format("".join([str(a) for a in (1 + np.argwhere(diff).flatten()).tolist()])))
        return goal_action_tuples

### Reset ###

    def _iw_reset(self):
        """Called by agent.py through Agent.learn() whenever operators changed.

        Do x rounds of random actions, and then run the sequence of goals from a file.
        """
        # Do one episode of random actions, then trade between one episode of random goals, one episode of directed goals
        if self.init_counts < 10:
            self.random_actions = True
            self.init_counts += 1
        elif self.random_actions:
            self.random_actions = False
            self.directed_goals = True
            if self._ignore_statics:  # ignore static goals
                static_preds = self._compute_static_preds()
                self.goal_actions = list(filter(
                    lambda ga: any(lit.predicate not in static_preds for lit in ga[0]),
                    self.goal_actions
                ))
            # With arbitrary k, don't filter out mutex goals
            
        # Forget the goal-action that was going to be taken at the end of the plan in progress
        self._current_goal_action = None

    def reset_episode(self, state):
        super().reset_episode(state)
        self._episode_start_state = state
        self._iw_reset()

    def learning_callback(self):
        super().learning_callback()
        self._iw_reset()

    def _get_fallback_action(self, state):
        self._current_goal_action = None
        return super()._get_fallback_action(state)

    ### Get an action ###

    def _get_action(self, state):
        # First check whether we just finished a plan and now must take the final action
        if (not (self._current_goal_action is None)) and (len(self._plan) == 0):
            action = self._get_ground_action_to_execute(state)
            if action != None:
                print("*** Finished the plan, now executing the action", action)
                # Execute the action
                self.line_stats.append(1)
                return action
        # Either continue executing a plan or make a new one (or fall back to random)
        return super()._get_action(state)

    def _get_ground_action_to_execute(self, state):
        lifted_goal, lifted_action = self._current_goal_action
        # Forget this goal-action because we're about to execute it
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
        """Produce a new goal to try to plan towards"""
        if self.random_actions:
            return None, None, None
        elif self.directed_goals:
            if self.index == len(self.goal_actions):
                # print("Starting to read from file")
                self.index = 0
            goal, action  = self.goal_actions[self.index]
            print("sampled goal, action:", goal, action, self.index)
            self._current_goal_action = (goal, action)
            self.index += 1
            return self._structify_goal(goal), goal, action
        else:
            raise Exception("Unexpected goal sampling strategy selected for this episode.")


    def _finish_plan(self, plan):
        # If the plan is empty, then we want to immediately take the action.
        if len(plan) == 0:
            action = self._get_ground_action_to_execute(self._last_state)
            print("Goal is satisfied in the current state; taking action now:", action)
            if action is None:
                # There was no way to bind the lifted action. Fallback
                action = self._get_fallback_action(self._last_state)
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

    def observe(self, state, action, _effects):
        pass