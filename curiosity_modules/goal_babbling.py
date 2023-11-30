"""Curiosity module that samples previously unachieved goals and plans to
achieve them with the current operators.
"""

import json
import os
from planning_modules.base_planner import PlannerTimeoutException, NoPlanFoundException
from settings import AgentConfig as ac
from curiosity_modules import BaseCuriosityModule


def stringify_grounded_action(action):
    """Turn the PDDLGym struct into a parseable string.

    Args:
        action (PDDLGym.Literal): PDDLGym literal
    Returns:
        str: parseable PDDLGym string
    """
    return f"{action.predicate.name}," + ",".join(action.pddl_variables())


def stringify_lifted_goal(goal_tuple):
    """Puts the goal into a parseable string.

    Args:
        goal_tuple (tuple): tuple of lifted goals, e.g. (inoven(?x1:pan,?x0:oven),)

    Returns:
        str: parseable string representation of a goal

        AND conjunction of goal(?arg0,?arg1), semicolon-separated

        goal_pred_0,?x0,?x1;goal_pred_1,?x1;goal_pred_2...
    """
    line = ""
    for g in goal_tuple:
        line += f"{g.predicate.name}," + ",".join(g.pddl_variables()) + ";"
    line = line[:-1]
    return line


class GoalBabblingCuriosityModule(BaseCuriosityModule):
    """Curiosity module that samples a completely random literal and plans to
    achieve it with the current operators.
    """

    def _initialize(self):
        self._num_steps = 0
        self._name = "goalbabbling"

    def reset_episode(self, state):
        self._last_state = set()
        # self._current_goal_action = None
        self._plan = []

    def _sample_goal(self, state):
        return self._observation_space.sample_literal(state)

    def _goal_is_valid(self, goal):
        return True

    def _finish_plan(self, plan):
        return plan

    def get_action(self, state, iter_path):
        """Execute plans open loop until stuck, then replan

        Args:
            state (PDDLGym.Literal??)
            iter (int): training iteration #, used for logging
        Returns:
            action (PDDLGym.Literal??)
        """
        action, following_plan, ground_action = self._get_action(state, iter_path)
        self._num_steps += 1
        return action, following_plan, ground_action

    def _get_action(self, state, iter_path=None):
        """Babble goals # sampling tries until found a plan to that goal; if no plans found, then fallback to random action.

        If the empty plan is found, then attempt to ground the babbled action and execute it. Otherwise, use a random action.

        Args:
            state (PDDLGym.Literal): literal
            iter_path (str, optional): If not None, do logging. Defaults to None.

        Returns:
            PDDLGym.Literal: action to take
            bool: True if the action is on a plan, False if fallback to random action
            bool: True if grounded action is found on this call, False if fallback to random action
        """
        last_state = self._last_state
        self._last_state = state

        # Continue executing plan?
        if self._plan and (last_state != state):
            self._save_iteration_explorer_info(
                iter_path, [], self._plan[0], option="following_plan"
            )
            return self._plan.pop(0), True, False

        babbled = []

        # Try to sample a goal for which we can find a plan
        sampling_attempts = planning_attempts = 0
        while (
            sampling_attempts < ac.max_sampling_tries
            and planning_attempts < ac.max_planning_tries
        ):
            goal, goal_lifted, action_lifted = self._sample_goal(state)

            if iter_path:
                if goal is not None:
                    babbled.append(
                        self._create_parseable_babble(goal_lifted, action_lifted)
                    )
                else:
                    babbled.append(None)

            sampling_attempts += 1

            if not self._goal_is_valid(goal):
                continue

            # Create a pddl problem file with the goal and current state
            problem_fname = self._create_problem_pddl(state, goal, prefix=self._name)

            # Get a plan
            try:
                self._plan = self._planning_module.get_plan(
                    problem_fname, use_cache=False
                )
            except NoPlanFoundException:
                os.remove(problem_fname)
                continue
            except PlannerTimeoutException:
                os.remove(problem_fname)
                break
            os.remove(problem_fname)
            planning_attempts += 1

            if len(self._plan) == 0:
                # Do the action babbled, or random if not able to ground it in the current state
                self._plan, not_random = self._finish_plan(self._plan)
                if not_random:
                    self._save_iteration_explorer_info(
                        iter_path,
                        babbled,
                        self._plan[0],
                        "ground_action",
                        goal_lifted,
                        action_lifted,
                        self._plan,
                    )
                    return self._plan.pop(0), False, True
                else:
                    self._save_iteration_explorer_info(
                        iter_path,
                        babbled,
                        self._plan[0],
                        "random_action",
                        goal_lifted,
                        action_lifted,
                        self._plan,
                    )
                    return self._plan.pop(0), False, False
            else:
                # Follow the plan found
                self._save_iteration_explorer_info(
                    iter_path,
                    babbled,
                    self._plan[0],
                    "following_new_plan",
                    goal_lifted,
                    action_lifted,
                    self._plan,
                )
                return self._plan.pop(0), True, False

        # No plan found within budget; take a random action
        action = self._get_fallback_action(state)
        self._save_iteration_explorer_info(iter_path, babbled, action, "no_plan_found")
        return action, False, False

    def _get_fallback_action(self, state):
        return self._action_space.sample(state)

    def _create_parseable_babble(self, goal_tuple, action):
        """Utility to create a parseable string of a babbled goal, action sequence.

        Example line format:

        goal_pred_0,?x1,?x2,..,goal_pred_1,?x3,?x5,...action_pred,arg0,arg1...

        Args:
            goal_tuple (tuple): goal is a tuple of PDDLGym Literals (observation predicates)
            action (PDDLGym.Literal): action is a PDDLGym Literal (action predicate)
        """
        line = ""
        for g in goal_tuple:
            line += f"{g.predicate.name}," + ",".join(g.pddl_variables()) + ","
        line += f"{action.predicate.name}," + ",".join(action.pddl_variables())
        return line

    def _save_iteration_explorer_info(
        self,
        iter_path,
        babbled_goal_actions,
        action,
        option,
        lifted_goal=None,
        lifted_action=None,
        plan=None,
    ):
        """Format the explorer.json (iteration-level explorer information) and save it.

        Args:
            iter_path (str or None): if None, don't do logging. Otherwise, save "explorer.json" to here.
            babbled_goal_actions (_type_): _description_
            action (PDDLGym.Literal): grounded action to take at this iteration
            option (str): one of ['following_new_plan', 'following_plan', 'ground_action', 'random_action']
            lifted_goal (PDDLGym.Literal): lifted goal where the plan was found, or None if random action
            plan (list[PDDLGym.Literal]): sequence of actions to take, or None if random action
        """
        if iter_path:
            action_str = stringify_grounded_action(action)

            if option == "following_new_plan":
                goal_str = stringify_lifted_goal(lifted_goal)
                lifted_action = stringify_grounded_action(lifted_action)
                plan_strings = [stringify_grounded_action(act) for act in plan]
                explorer_dict = {
                    "babbled": babbled_goal_actions,
                    "action": action_str,
                    "plan_found": {
                        "goal": goal_str,
                        "action": lifted_action,
                        "plan": plan_strings,
                    },
                }

            elif option == "following_plan":
                explorer_dict = {
                    "babbled": [],
                    "action": action_str,
                    "following_plan": True,
                }

            elif option == "random_action":
                explorer_dict = {
                    "babbled": babbled_goal_actions,
                    "action": action_str,
                    "empty_plan_so_random_action": True,
                }

            elif option == "ground_action":
                goal_str = stringify_lifted_goal(lifted_goal)
                lifted_action = stringify_grounded_action(lifted_action)
                plan_strings = [stringify_grounded_action(act) for act in plan]
                explorer_dict = {
                    "babbled": babbled_goal_actions,
                    "action": action_str,
                    "empty_plan_so_grounded_action": {
                        "goal": goal_str,
                        "action": lifted_action,
                        "plan": plan_strings,
                    },
                }

            elif option == "no_plan_found":
                explorer_dict = {
                    "babbled": babbled_goal_actions,
                    "action": action_str,
                    "found_no_plans_so_random_action": True,
                }

            elif option == "babbled_action":
                explorer_dict = {
                    "babbled": babbled_goal_actions,
                    "action": action_str,
                    "action_after_plan": True
                }
            else:
                raise Exception("Logging option not found")
            fname = os.path.join(iter_path, "explorer.json")
            with open(fname, "w") as f:
                json.dump(explorer_dict, f, indent=4)
