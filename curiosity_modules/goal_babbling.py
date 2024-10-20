"""Curiosity module that samples previously unachieved goals and plans to
achieve them with the current operators.
"""

import numpy as np
import logging
import os
from planning_modules.base_planner import PlannerTimeoutException, \
    NoPlanFoundException
from settings import AgentConfig as ac
from curiosity_modules import BaseCuriosityModule
from pddlgym.parser import Operator
from pddlgym.structs import Type, TypedEntity, Literal, LiteralConjunction
from itertools import combinations
from pddlgym.parser import PDDLDomainParser
from copy import deepcopy

GOAL_BABBLING_LOGGER = logging.getLogger("GOAL_BABBLING")

class GoalBabblingCuriosityModule(BaseCuriosityModule):
    """Curiosity module that samples a completely random literal and plans to
    achieve it with the current operators.
    """
    def _initialize(self):
        self._num_steps = 0
        self._name = "goalbabbling"
        self._llm_goal_actions = []
        # Keep track of the number of times that we follow a plan
        self.line_stats = []
        # Keep track of the number of times that we follow a plan using the LLM
        self.llm_line_stats = []

    def reset_episode(self, state):
        self._last_state = set()
        self._plan = []

    def _sample_goal(self, state):
        return self._observation_space.sample_literal(state)

    def _goal_is_valid(self, goal):
        return True

    def _finish_plan(self, plan):
        return plan

    def get_action(self, state, goal=None):
        """Execute plans open loop until stuck, then replan
        
        Returns:
            in_plan (bool): True if the action is in a plan
            op_name (Optional[str]): name of the operator executed in the plan, or None if action is not in a plan
            action (Literal): action literal taken
            """
        if goal is not None:
            logging.info(f"Getting plan to subgoal {goal}")
        in_plan, op_name, action = self._get_action(state, goal)
        self._num_steps += 1
        return in_plan, op_name, action

    def _get_action(self, state, goal):
        """
        Returns:
            in_plan (bool): True if the action is in a plan
            op_name (Optional[str]): name of the operator executed in the plan, or None if action is not in a plan
            action (Literal): action literal taken
        """
        last_state = self._last_state
        self._last_state = state
        in_plan = False

        # Continue executing plan?
        if self._plan and (last_state != state):

            in_plan = True

            GOAL_BABBLING_LOGGER.info("CONTINUING PLAN")
            GOAL_BABBLING_LOGGER.info(f"PLAN: {self._plan}")
            if len(self._operators) > 0:
                return in_plan, self._operators.pop(0), self._plan.pop(0)
            else:
                return in_plan, None, self._plan.pop(0)

        SAMPLE_GOAL = goal is None

        # Try to sample a goal for which we can find a plan
        sampling_attempts = planning_attempts = 0
        while (planning_attempts < ac.max_planning_tries and sampling_attempts < ac.max_sampling_tries):
            if SAMPLE_GOAL:
                goal, self._goal_from_llm = self._sample_goal(state)
                GOAL_BABBLING_LOGGER.debug(f"SAMPLED GOAL: {goal}")
            else:
                GOAL_BABBLING_LOGGER.info(f"USING GIVEN GOAL: {goal}")
                
            sampling_attempts += 1

            if not self._goal_is_valid(goal):
                GOAL_BABBLING_LOGGER.info(f"Goal invalid!")
                continue


            # Create a pddl problem file with the goal and current state
            problem_fname = self._create_problem_pddl(
                state, goal, prefix=self._name)

            # GOAL_BABBLING_LOGGER.info(f"Problem file: {problem_fname}")
            # Get a plan
            try:
                self._plan, self._operators = self._planning_module.get_plan(
                    problem_fname, use_cache=False, use_learned_ops=False)
                os.remove(problem_fname)
            except NoPlanFoundException:
                GOAL_BABBLING_LOGGER.info(f"No plan found.")
                os.remove(problem_fname)
                continue
            except PlannerTimeoutException:
                GOAL_BABBLING_LOGGER.info(f"PLANNER TIMED OUT")
                os.remove(problem_fname)
                break
            planning_attempts += 1

            if self._plan_is_good():
                if len(self._plan) != 0:
                    in_plan = True
                    GOAL_BABBLING_LOGGER.debug(f"\tGOAL: {goal}")
                    GOAL_BABBLING_LOGGER.info(f"\tPLAN: {self._plan}")
    
                self._plan = self._finish_plan(self._plan)
                self._goal = goal
                # import ipdb; ipdb.set_trace()
                # Take the first step in the plan
                if len(self._operators) > 0:
                    return in_plan, self._operators.pop(0), self._plan.pop(0)
                else:
                    return in_plan, None, self._plan.pop(0)
            self._plan = []

        # No plan found within budget; take a random action
        # print("falling back to random")
        return in_plan, None, self._get_fallback_action(state)

    def _get_fallback_action(self, state):
        return self._action_space.sample(state)

    def _plan_is_good(self):
        return bool(self._plan)

    def learn(self, itr):
        pass

    def rename_lits_in_operator(self, operator_llm:Operator, operator_learner:Operator) -> Operator:
        """Rename the lits in the learner's operator to match the `operator_llm`. Helper for mix_lifted_operators.

        Args:
            operator_llm : Operator
            operator_learner : Operator

        Returns:
            Operator: the revised `operator_learner`
        """
        # Map variable names by type from LLM to Learner
        llm_arg_names:list[str] = []
        llm_var_types:list[str] = []
        for param in operator_llm.params:
            var_name, v_type = param._str.split(':')
            llm_arg_names.append(var_name)
            llm_var_types.append(v_type)

        next_var_name = f"?x{len(operator_llm.params)}"
        # Rename the learner's operators. Map from old variable name to new.
        names_mapping = {}
        for param in operator_learner.params:
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
        for lit in operator_learner.preconds.literals:
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


    def mix_lifted_preconditions(self, llm_operator, operator_learner) -> list[set[Literal]]:
        """Get a list of preconditions that have a mix of the LLM operator preconditions with the learner preconditions.

        The suggested operator from the LLM, and uses the preconditions P as goals. 
        In addition, take a random operator with the same action, and get its preconditions p. Get all combinations of (p \ P), including [].
        For each c in those combinations, consider P + c equally when sampling a goal. Consider all P + c first before the other grounded goals.

        Args:
            llm_operator (_type_): operator from the LLM
            learner_operator (_type_): operator from the learner with the same action as `llm_operator`

        Returns:
            list[set[Literal]]: lists of precondition sets
        """


        learned_lits_combinations = []
        learner_operator = self.rename_lits_in_operator(llm_operator, operator_learner)
        learned_lits = []
        for lit in learner_operator.preconds.literals:
            if (lit.positive not in llm_operator.preconds.literals) and (lit.negative not in llm_operator.preconds.literals):
                learned_lits.append(lit)
        for i in range(0, len(learned_lits)+1):
            for lits in combinations(learned_lits, i):
                learned_lits_combinations.append(lits)

        combined_preconditions = []
        for lits_to_add in learned_lits_combinations:

            ### Rename the additional lits, assigning variable names by type according to the LLM's operator naming scheme.
            llm_arg_types:list[str] = []
            llm_arg_names:list[str] = []
            for t in llm_operator.params:
                var_name, v_type = t._str.split(':')
                llm_arg_types.append(v_type)   
                llm_arg_names.append(var_name)
            next_var_name = f"?x{len(llm_operator.params)}"


            renamed_additional_lits = []
            for lit in lits_to_add:
                args = []
                for t in lit.predicate.var_types:
                    if t in llm_arg_types:
                        i = llm_arg_types.index(t)
                        llm_arg_types[i] = None
                        args.append(TypedEntity(llm_arg_names[i], Type(t)))
                    else:
                        args.append(TypedEntity(next_var_name, Type(t)))
                        next_var_name = f"?x" + str(int(next_var_name.lstrip("?x")) + 1)
                renamed_additional_lits.append(lit.predicate(*args))

            precond = llm_operator.preconds.literals + renamed_additional_lits
            combined_preconditions.append(precond)
        return combined_preconditions
        
     