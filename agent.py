import math
import traceback 
from planning_modules.base_planner import PlannerTimeoutException, \
    NoPlanFoundException
from collections import defaultdict
import itertools
from curiosity_modules import create_curiosity_module
from operator_learning_modules import create_operator_learning_module
from planning_modules import create_planning_module
from pddlgym.structs import Anti, State, Not, LiteralConjunction, ground_literal, Exists, Literal, Type, TypedEntity, Predicate
from pddlgym.parser import Operator
from settings import LLMConfig as lc
from openai_interface import OpenAI_Model
from settings import EnvConfig as ec
from settings import AgentConfig as ac
from ndr.learn import print_rule_set
from ndr.ndrs import NOISE_OUTCOME
from llm_parsing import GoalParser
import os
import pickle
import time
import numpy as np
from typing import Optional, Tuple
import logging
from pprint import pprint
from copy import deepcopy


class Agent:
    """An agent interacts with an env, learns PDDL operators, and plans.
    This is a simple wrapper around three modules:
    1. a curiosity module
    2. an operator learning module
    3. a planning module
    The curiosity module selects actions to collect training data.
    The operator learning module learns operators from the training data.
    The planning module plans at test time.
    The planning module (and optionally the curiosity module) use the
    learned operators. The operator learning module contributes to them.
    """
    def __init__(self, domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name, log_llm_path:Optional[str]):
        """

        Args:
            domain_name (str): from PDDLGym environment
            action_space : from PDDLGym environment
            observation_space : from PDDLGym environment
            curiosity_module_name (str): 
            operator_learning_name (str): 
            planning_module_name (str): 
            log_llm_path (str or None): Path to log the LLM output.
        """
        self.name = "agent"
        self.curiosity_time = 0.0
        self.domain_name = domain_name
        self.curiosity_module_name = curiosity_module_name
        self.operator_learning_name = operator_learning_name
        self.planning_module_name = planning_module_name
        self._rand_state = np.random.RandomState(seed=ac.seed)
        # The main objective of the agent is to learn good operators
        self.planning_operators = set()
        self.learned_operators = set()

        self.llm = OpenAI_Model()
        self.llm_precondition_goals = dict() # Op from LLM: Op from Learner with the same action predicate (random)

        # The operator learning module learns operators. It should update the
        # agent's learned operators set
        self._operator_learning_module = create_operator_learning_module(
            operator_learning_name, self.planning_operators, self.learned_operators, self.domain_name, self.llm, self.llm_precondition_goals, log_llm_path, self._rand_state)
        # The planning module uses the learned operators to plan at test time.
        self._planning_module = create_planning_module(
            planning_module_name, self.planning_operators, self.learned_operators, domain_name,
            action_space, observation_space)
        # The curiosity module dictates how actions are selected during training
        # It may use the learned operators to select actions
        self._curiosity_module = create_curiosity_module(
            curiosity_module_name, action_space, observation_space,
            self._planning_module, self.planning_operators, self.learned_operators,
            self._operator_learning_module, domain_name, self.llm_precondition_goals, self._rand_state)
        
        # Flag to tell if at the episode start. Unset after observing the first effect.
        self.episode_start = False


    ## Training time methods
    def get_action(self, state, _problem_idx, _precond_targeting_only):
        """Get an exploratory action to collect more training data.
           Not used for testing. Planner is used for testing."""
        if self.domain_name.lower() == 'bakingrealistic':
            obs_literals = set()
            for lit in state.literals:
                if lit.predicate.name not in ('different', 'name-less-than'):
                    obs_literals.add(lit)
            state = State(frozenset(obs_literals), state.objects, state.goal)

        start_time = time.time()
        in_plan, op_name, action = self._curiosity_module.get_action(state)
        logging.info(f"Getting action took {time.time() - start_time}")
        self.curiosity_time += time.time()-start_time

        if in_plan:
            self._action_in_plan = op_name
        else:
            self._action_in_plan = False
        return action

    def observe(self, state, action, next_state, itr):
        """Observe a transition.

        Args:
            state (pddlgym.structs.State): initial state of the transition
            action (Literal): action taken
            effects (set[Literal]): effects of the transition
            itr (int): training iteration #
        """
        if self.domain_name.lower() == 'bakingrealistic':
            obs_literals = set()
            next_obs_literals = set()
            for lit in state.literals:
                if lit.predicate.name not in ('different', 'name-less-than'):
                    obs_literals.add(lit)
            for lit in next_state.literals:
                if lit.predicate.name not in ('different', 'name-less-than'):
                    next_obs_literals.add(lit)
            state = State(frozenset(obs_literals), state.objects, state.goal)
            next_state = State(frozenset(next_obs_literals), next_state.objects, next_state.goal)
        # Get effects
        effects = self._compute_effects(state, next_state)
        logging.info(f"EFFECTS: \n{effects}")
        # Add data
        self._operator_learning_module.observe(state, action, effects, start_episode=self.episode_start, itr=itr)
        # Some curiosity modules might use transition data
        start_time = time.time()
        self._curiosity_module.observe(state, action, effects)
        self.curiosity_time += time.time()-start_time
        self.episode_start = False

        # Set the info about the operator executed in the plan for the learning module.
        # If the action is not in a plan, this is None. Interested when the action is in a plan, and the operator executed has no effects (the operator fails).
        # if (len(effects) == 0) and self._action_in_plan:
        #     self._skill_to_edit = (action.predicate, self._action_in_plan)
        # else:
        #     self._skill_to_edit = None

    def learn(self, itr):
        # Learn
        start = time.time()
        some_learned_operator_changed, some_planning_operator_changed = self._operator_learning_module.learn(itr, skill_to_edit=None) #FIXME skill_to_edit is not able to run like this
        # logging.info(f"Learning took {time.time() - start} s")

        # Used in LLMIterative only
        if self.operator_learning_name in ['LLM+LNDR', 'LLMIterative+LNDR']:
            self._curiosity_module.learn(itr)

        if some_learned_operator_changed:
            start_time = time.time()
            self._curiosity_module.learning_callback()
            # logging.info(f"Resetting curiosity took {time.time() - start_time}")
            self.curiosity_time += time.time()-start_time
            # for pred, dt in self._operator_learning_module.learned_dts.items():
            #     print(pred)
            #     print(dt.print_conditionals())
            # print()
        # for k, v in self._operator_learning_module._ndrs.items():
        #     print(k)
        #     print(str(v))
        return some_learned_operator_changed, some_planning_operator_changed

    def reset_episode(self, state, _):
        obs_literals = set()
        if self.domain_name.lower() == 'bakingrealistic':
            for lit in state.literals:
                if lit.predicate.name not in ('different', 'name-less-than'):
                    obs_literals.add(lit)
            state = State(frozenset(obs_literals), state.objects, state.goal)

        start_time = time.time()
        self._curiosity_module.reset_episode(state)
        logging.info(f"Resetting episode for curiosity took {time.time() - start_time}")
        self.curiosity_time += time.time()-start_time
        self.episode_start = True

    @staticmethod
    def _compute_effects(state, next_state):
        positive_effects = {e for e in next_state.literals - state.literals}
        negative_effects = {Anti(ne) for ne in state.literals - next_state.literals}
        return positive_effects | negative_effects

    ## Test time methods
    def get_policy(self, problem_fname, use_learned_ops=False):
        """Get a plan given the learned operators and a PDDL problem file."""
        return self._planning_module.get_policy(problem_fname, use_learned_ops)

class InteractiveAgentGrounded(Agent):
    """An agent with initial demonstration data to each of the 4 train tasks.

    Must be run with GLIB_G, since goals will be grounded.
    """
    def __init__(self, domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name, log_llm_path:Optional[str]):
        super().__init__(domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name, log_llm_path)
        
        self.name = 'interactive'
        # dict: problem index -> step in the plan to execute next
        self.problem_to_plan_step = {i: 0 for i in range(len(ac.train_env.problems))}

        # dict: problem index -> list of plan steps (ground action predicate strings)
        self.plans = {}
        # self._get_plans()
        self.prev_episode_idx = None
        # Keep track of episodes that have finished at least once
        self.terminated_episodes = set()
        self.action_space = action_space
        self.obs_space = observation_space

        self._plan_to_op_preconds_failed = False

        # Load the demos
        with open(f'demonstrations/{self.domain_name.lower()}_demonstrations.pkl', 'rb') as f:
            transitions = pickle.load(f)
        self._operator_learning_module._transitions = transitions
        # for action_pred in transitions:
        #     self._operator_learning_module._fits_all_data[action_pred] = True
 
        for action_pred in transitions:
            self._operator_learning_module._fits_all_data[action_pred] = False
        
        # Get the subgoals.
        # Keep track of the action seq to get to the last achieved subgoal.
        self.action_seq = []
        self._plan_to_next_subgoal = None
        self._last_plan_to_next_subgoal = None
        self.next_subgoal_idx = 0
        self.subgoals = []
        self._loaded_subgoals = False
        # Keep track if the last action was part of a plan to subgoals.
        self._action_in_plan = False
        # Keep track of executed actions since the last subgoal.
        self.actions_since_last_subgoal = []

        # User inputs
        self.next_action = None
        # Keep track of the actions from the user input
        self.action_seq_reset = []
        self.observe_last_transition = False

        # Planning to preconditions
        self.precondition_targeting = True
        self._preconds_plan = None
        # This is set by the Runner.run() method and also self._get_action_with_preconds_as_goals()
        self.finished_preconds_plan = False
        self._last_preconds_action = None
        self._ops_preconds_executed = set()

        # Keeps track of preconditions already planned to from states
        self._visited_preconds_states = {a: set() for a in action_space.predicates} # Map from action predciate to set

        # # Load visited set
        # with open("ops_visited.pkl", 'rb') as f:
        #     self._ops_preconds_executed = pickle.load(f)
        # with open('visited_preconds.pkl', 'rb') as f:
        #     self._visited_preconds_states = pickle.load(f)
        # with open('rand_state.pkl', 'rb') as f:
        #     self._rand_state.set_state(pickle.load(f))

        # # # Load NDRs
        # with open('ndrs.pkl', 'rb') as f:
        #     self._operator_learning_module._ndrs = pickle.load(f)


        # # Load ops
        # with open('ops.pkl', 'rb') as f:
        #     ops = pickle.load(f)
        # for op in ops:
        #     self.learned_operators.add(op)
        #     self.planning_operators.add(op)


    def reset_episode(self, state, subgoals_path):
        if subgoals_path:
            self._load_subgoals(state, subgoals_path) 
        else:
            self.subgoals = None
        self.next_subgoal_idx = 0
        self.action_seq = []
        obs_literals = set()
        if self.domain_name.lower() == 'bakingrealistic':
            for lit in state.literals:
                if lit.predicate.name not in ('different', 'name-less-than'):
                    obs_literals.add(lit)
            state = State(frozenset(obs_literals), state.objects, state.goal)

        start_time = time.time()
        self._curiosity_module.reset_episode(state)
        logging.info(f"Resetting episode for curiosity took {time.time() - start_time}")
        self.curiosity_time += time.time()-start_time
        self.episode_start = True


    def _load_subgoals(self, state, subgoals_file):
        """Loads the subgoals into grounded goals."""
        with open(subgoals_file, 'r') as f:
            lines = f.readlines()
        subgoals = []
        for goal_line in lines:
            goal_lits = []
            for literal_str in goal_line.split(','):
                literal_str = literal_str.strip()[1:-1]
                if literal_str.startswith('not '):
                    literal_str = literal_str[len('not '):]
                    literal_str = literal_str[1:-1]
                    items = literal_str.split()
                    pred = Not(self._get_obs_predicate(items[0], items[1:], state.objects))
                else:
                    items = literal_str.split()
                    pred = self._get_obs_predicate(items[0], items[1:], state.objects)
                goal_lits.append(pred)
            subgoals.append(LiteralConjunction(goal_lits))
        self.subgoals = subgoals
        self.next_subgoal_idx = 0
        self.actions_since_last_subgoal = []
        logging.info(f"Loaded subgoals from {subgoals_file}: ")
        logging.info(self.subgoals)
    
    def _get_obs_predicate(self, pred_name:str, object_names:list, objects:frozenset):
        pred = [p for p in self.obs_space.predicates if p.name == pred_name][0]
        args = []
        for object_name in object_names:
            for o in objects:
                obj_name, _ = o._str.split(':')
                if obj_name == object_name:
                    args.append(o)
                    break
        return pred(*args)
    
    def _get_action_with_preconds_as_goals(self, state, ops_to_exclude):
        """Returns the action, or None, if the stopping condition is reached.

        Args:
            state: the current state
            ops_to_exclude: don't do precondition targeting for these actions.

        Stopping condition:
            If all of the preconditions are either unreachable from this state or the same preconditions has already had an action tried from it.
        """
        # Have successfully executed the plan to the operator preconds, and will execute the operator next
        if self._preconds_plan is not None and len(self._preconds_plan) == 1:
            self.finished_preconds_plan = True
            self.actions_since_last_subgoal = []
            logging.info(f"FOLLOWING PLAN: {self._preconds_plan}")
            self._ops_preconds_executed.add(self._op_preconds_to_execute)
            self._op_preconds_to_execute = None
            return self._preconds_plan.pop()

        # Follow plan to the operator's preconditions
        elif self._preconds_plan is not None and len(self._preconds_plan) > 0:
            self.finished_preconds_plan = False
            logging.info(f"FOLLOWING PLAN: {self._preconds_plan}")
            return self._preconds_plan.pop(0)

        # set a hyperparameter for how many ground preconditions to try.
        NUM_TRIES = 200

        action_predicates = set(p.name for p in self.action_space.predicates)
        for op in self._rand_state.permutation(sorted(self.learned_operators, key=lambda op: op.name)):
            # since the last time operators were learned, if operator has been successfully executed at the end of the plan, or
            # the plan failed in the middle to the operator preconditions, skip it.
            if op.name in self._ops_preconds_executed:
                continue
            if op.name in ops_to_exclude: continue
            preconds = op.preconds.literals

            logging.info(f"Trying preconds for op: {op.name}: {preconds}")
            ground_preconds_list = self._get_ground_preconds(op, state)
            for i in self._rand_state.permutation(len(ground_preconds_list))[:NUM_TRIES]:
                grounded_precond = ground_preconds_list[i]
                preconds_hash = get_hashable_preconds_action(grounded_precond)
                ground_act = [p for p in grounded_precond if p.predicate.name in action_predicates][0]
                if (preconds_hash, state) in self._visited_preconds_states[ground_act.predicate]:
                    continue
                grounded_precond_no_act = [p for p in grounded_precond if p.predicate.name not in action_predicates]
                plan = self._get_plan_to_preconds(grounded_precond_no_act, state)
                self._visited_preconds_states[ground_act.predicate].add((preconds_hash, state))
                if plan == 'skip':
                    self._ops_preconds_executed.add(op.name)
                    break
                elif plan is not None:
                    self._preconds_plan = plan + [ground_act]
                    logging.info(f"Found plan to preconds: {preconds_hash}")
                    logging.info(f"PLAN: {self._preconds_plan}")
                    self._op_preconds_to_execute = op.name
                    if len(self._preconds_plan) == 1:
                        self.finished_preconds_plan = True
                        self.actions_since_last_subgoal = []
                        self._ops_preconds_executed.add(self._op_preconds_to_execute)
                        self._op_preconds_to_execute = None
                    return self._preconds_plan.pop(0)

        # once done, proceed to the next subgoal in the file.
        return None
    
    def _get_ground_preconds(self, operator, state):
        """Return a list of lists of grounded literals that form the precondition.

        Returns:
        [[ grounded precond literals version 1], [precond grounding version 2], ...]
        """
        preconds = sorted(operator.preconds.literals)
        assignments = self._get_assignments(preconds, state)
        ground_preconds_list = []
        for assignment in assignments:
            ground_preconds = tuple(ground_literal(p, assignment) for p in preconds)
            ground_preconds_list.append(ground_preconds)
        return ground_preconds_list
    
    def _get_assignments(self, precond_literals, state):
        """Return a list of assignments of parameter variable (TypedEntity) to object (TypedEntity)."""
        objects = sorted(state.objects)
        var_names_to_type = {}

        for lit in precond_literals:
            for var in lit.variables:
                t = var._str.split(':')[1]
                var_names_to_type[var] = t.strip()

        var_names_types = sorted([(v, t) for v,t in var_names_to_type.items()], key=lambda x: x[0])

        def recurse(var_names_types, i,  objects, assignment, assignments=[]):
            if i == len(var_names_types):
                assignments.append(deepcopy(assignment))
                return assignments
            var, t = var_names_types[i]
            for obj in objects:
                if obj._str.split(':')[-1] == t and obj not in assignment.values():
                    assignment[var] = obj
                    # recurse
                    assignments = recurse(var_names_types, i+1, objects, assignment, assignments)
                    del assignment[var]
            return assignments
        assignments = recurse(var_names_types, 0, objects, {}, [])
        return assignments

    def _get_plan_to_preconds(self, grounded_precond_lits:list, state):
        """Returns None if no plan found, otherwise a list of action literals."""
        goal = LiteralConjunction(grounded_precond_lits)

        # Create a pddl problem file with the goal and current state
        problem_fname = self._curiosity_module._create_problem_pddl(
            state, goal, prefix='glibg1_preconds')

        # logging.info(problem_fname)
        # Get a plan
        try:
            plan, _ = self._planning_module.get_plan(
                problem_fname, use_cache=False, use_learned_ops=True)
            os.remove(problem_fname)
            return plan
        except NoPlanFoundException:
            logging.info(f"No plan found.")
        except PlannerTimeoutException:
            logging.info(f"PLANNER TIMED OUT")
            if input("skip this preconditions? y or anything").strip() == 'y':
                os.remove(problem_fname)
                return 'skip'

        os.remove(problem_fname)

        return None
        

    def get_action(self, state, _problem_idx, precond_targeting_only: bool):
        """Get an exploratory action to collect more training data.
           Not used for testing. Planner is used for testing.
        
        Args:
            state: The current state.
            precond_targeting_only: True if mode is to target preconditions.
        """
        if self.domain_name.lower() == 'bakingrealistic':
            obs_literals = set()
            for lit in state.literals:
                if lit.predicate.name not in ('different', 'name-less-than'):
                    obs_literals.add(lit)
            state = State(frozenset(obs_literals), state.objects, state.goal)

        # Before getting to a new subgoal, try out all the operator preconditions if they haven't been tried before, to refine incorrect preconditions.
        if self.precondition_targeting:
            # prompt if want to target preconditions or not.
            if not ac.auto_target_preconds and not precond_targeting_only:
                target_preconds = (input(f"Target preconditions (y) or skip precondition targeting? Ops that would be tried: {[o.name for o in self.learned_operators if o.name not in self._ops_preconds_executed]}\n y or anything").strip() == 'y')
                ops_to_exclude = set()
                # select operator names that should be skipped.
                operator_names = set(o.name for o in self.learned_operators)
                if not target_preconds:
                    uip = input("Enter an op name to exclude, or n to quit: ").strip()
                    while uip != 'n':
                        if uip in  operator_names:
                            ops_to_exclude.add(uip)
                        else:
                            logging.info(f"Invalid operator name: {uip}")
                        uip = input("Enter an op name to exclude, or n to quit: ").strip()

            else:
                target_preconds = True 
                ops_to_exclude = set()
                
            if target_preconds:
                self._action_in_plan = False
                logging.info("Getting plan to precondition...")
                action = self._get_action_with_preconds_as_goals(state, ops_to_exclude)
                if action is None:
                    self._action_in_plan_to_preconds = False
                    self.precondition_targeting = False 
                else:
                    self._action_in_plan_to_preconds = True
                    return action
            else:
                self.precondition_targeting = False
                self._action_in_plan_to_preconds = False
        else:
            self._action_in_plan_to_preconds = False


        if precond_targeting_only:
            return None

        if self.next_subgoal_idx == len(self.subgoals):
            return self._prompt_demos_or_subgoals(state)

        logging.info("Getting plan to next subgoal...")
        if self._plan_to_next_subgoal is not None and len(self._plan_to_next_subgoal) > 0:
            logging.info(f"Continuing plan: {self._plan_to_next_subgoal}")
            act = self._plan_to_next_subgoal.pop(0)
            # add to the actions list
            self.actions_since_last_subgoal.append(act)
            return act 
        # Get a plan to the next subgoal
        problem_fname = self._curiosity_module._create_problem_pddl(state, self.subgoals[self.next_subgoal_idx], prefix='glibg1_subgoal')
        plan = None
        try:
            plan, _ = self._planning_module.get_plan(
                problem_fname, use_cache=False, use_learned_ops=True)
            os.remove(problem_fname)
        except NoPlanFoundException:
            logging.info(f"No plan found.")
        except PlannerTimeoutException:
            logging.info(f"Planner timed out.")
        if plan:
            logging.info(f"Found plan: {plan}")
            self._action_in_plan = True
            if self._last_plan_to_next_subgoal == plan:
                logging.info("REPEATED PLAN")
                self._plan_to_next_subgoal = None
                return self._prompt_demos_or_subgoals(state)
            else:
                self._last_plan_to_next_subgoal = plan

            self._plan_to_next_subgoal = plan
            act = self._plan_to_next_subgoal.pop(0)
            # add to the actions list
            self.actions_since_last_subgoal.append(act)
            return act
        else:
            return self._prompt_demos_or_subgoals(state)

    def _prompt_demos_or_subgoals(self, state):
        # print ops, action seq, and current state
        pprint(sorted(state.literals))
        # for o in self._operator_learning_module._learned_operators:
            # logging.info(o.pddl_str())
        print_rule_set(self._operator_learning_module._ndrs)
        logging.info("Action sequence to previous subgoal below. Look at logs to see all the actions from last subgoal to now")
        for act in self.action_seq:
            logging.info(act)
        self.action_seq_reset = []
        if self.next_subgoal_idx < len(self.subgoals):
            logging.info(f"Next subgoal to achieve: {self.subgoals[self.next_subgoal_idx]}")
        else:
            logging.info("No more subgoals to achieve.")
        option_str = \
"""Please pick an option:

*** Demonstration ***
[0] Enter an action. Execute it, and observe the transition. Then, reset to the previous achieved subgoal.
[2] Enter an action sequence. Reset to start, then execute it, and observing all the transitions. Then, reset back to the previous subgoal.
[6] Enter an action sequence. Execute it from here, observing all of them. Reset to previous subgoal.
[4] Execute a sequence of actions from here, observing all of them. Don't reset.

*** Curriculum ***
[7] Abandon this subgoals list, and enter a new curriculum to execute from this state.
[8] Abandon this subgoals list, reset the episode, and try the new curriculum.

*** Precondition Targeting ***
[10] Reset episode and target preconditions in the selected episode.

*** Utils ***
[5] Dump the transitions and operators.
[9] Evaluate operators.
[11] End experiment.

*** MISC ***
[3] Execute a random action, observe it, and reset to the previous achieved subgoal.
"""
        try:
            option = int(input(option_str))
        except:
            option = None
        while option is None and option not in [0,2,3,4,5,6,7,8,9,10,11]:
            try:
                option = int(input(option_str))
            except:
                option = None           
        # 1. Execute the action, and observe that transition. Then, reset.
        self.option = option
        if option == 0:
            action = self._safe_action_input(state)
            self.next_action = action
        #when resetting to the previous subgoal, clear the actions list.
            self.actions_since_last_subgoal = []
        elif option == 2:
            action_str = input("Enter the next action, or q to quit: ")
            while action_str != 'q':
                loop = True
                while loop and action_str != 'q':
                    try:
                        action = self._parse_action_from_string(action_str, state.objects)
                        loop = False
                    except:
                        action_str = input("Error parsing. Re-enter the action, or enter q to quit:")
                if action_str == 'q': break
                self.action_seq_reset.append(action)
                action_str = input("Enter the next action: ")
        #when resetting to the previous subgoal, clear the actions list.
            self.actions_since_last_subgoal = []
        elif option == 3:
            self.next_action = self.action_space.sample(state)
        elif option == 4:
            action_str = input("Enter the next action, or q to quit: ")
            while action_str != 'q':
                loop = True
                while loop and action_str != 'q':
                    try:
                        action = self._parse_action_from_string(action_str, state.objects)
                        loop = False
                    except:
                        action_str = input("Error parsing. Re-enter the action, or enter q to quit:")
                if action_str == 'q': break
                self.action_seq_reset.append(action)
                # when not resetting to the previous subgoal (option 4) add to the actions buffer
                self.actions_since_last_subgoal.append(action)
                action_str = input("Enter the next action: ")
        elif option == 5:
            self.next_action = self.action_space.sample(state)
            dump_intermediate_state(self)
        elif option == 6:
            action_str = input("Enter the next action, or q to quit: ")
            while action_str != 'q':
                loop = True
                while loop and action_str != 'q':
                    try:
                        action = self._parse_action_from_string(action_str, state.objects)
                        loop = False
                    except:
                        action_str = input("Error parsing. Re-enter the action, or enter q to quit:")
                if action_str == 'q': break
                self.action_seq_reset.append(action)
                action_str = input("Enter the next action: ")
        #when resetting to the previous subgoal, clear the actions list.
            self.actions_since_last_subgoal = []
        elif option == 7:
            logging.info(f"Starting new curriculum from current state.")
            subgoals_list_fname = input("Enter the new subgoals list: ")
            while not os.path.exists(subgoals_list_fname):
                subgoals_list_fname = input("Enter the new subgoals list: ")               
            self._load_subgoals(state, subgoals_list_fname)
        elif option == 8:           
            logging.info(f"Starting new curriculum from start of episode.")
            subgoals_list_fname = input("Enter the new subgoals list: ")
            while not os.path.exists(subgoals_list_fname):
                subgoals_list_fname = input("Enter the new subgoals list: ")               
            self._load_subgoals(state, subgoals_list_fname)
            self.action_seq = []
            self.action_seq_reset = []

        self._action_in_plan = False
        return None
    
    def _safe_action_input(self, state):
        loop = True
        action_str = input("Enter the action:")
        while loop:
            try:
                action = self._parse_action_from_string(action_str, state.objects)
                loop = False
            except:
                action_str = input("Error parsing. Re-enter the action:")
        return action
    

    def observe(self, state, action, next_state, itr):
        """Observe a transition.

        Args:
            state (pddlgym.structs.State): initial state of the transition
            action (Literal): action taken
            effects (set[Literal]): effects of the transition
            itr (int): training iteration #
        """
        if self.domain_name.lower() == 'bakingrealistic':
            obs_literals = set()
            next_obs_literals = set()
            for lit in state.literals:
                if lit.predicate.name not in ('different', 'name-less-than'):
                    obs_literals.add(lit)
            for lit in next_state.literals:
                if lit.predicate.name not in ('different', 'name-less-than'):
                    next_obs_literals.add(lit)
            state = State(frozenset(obs_literals), state.objects, state.goal)
            next_state = State(frozenset(next_obs_literals), next_state.objects, next_state.goal)
        # Get effects
        effects = self._compute_effects(state, next_state)
        logging.info(f"EFFECTS: \n{effects}")
        # Add data
        self._operator_learning_module.observe(state, action, effects, start_episode=self.episode_start, itr=itr)
        # Some curiosity modules might use transition data
        start_time = time.time()
        self._curiosity_module.observe(state, action, effects)
        self.curiosity_time += time.time()-start_time
        self.episode_start = False

        # Check if planned to preconditions
        if self._action_in_plan_to_preconds:
            # Stop executing the plan if it failed in the middle.
            if len(effects) == 0:
                self.finished_preconds_plan = True
                # About to reset to the previous subgoal, so clear this list.
                self.actions_since_last_subgoal = []
                self._preconds_plan = None
                # If the operators don't change as a result of adding the NOP, then the op preconds should be added to the visited set.
                self._plan_to_op_preconds_failed = True
            else:
                self._plan_to_op_preconds_failed = False
                
        else:
            if len(effects) == 0:
                self._plan_to_next_subgoal = None

        # Check if planned to the next subgoal
        if self._action_in_plan and self.next_subgoal_idx < len(self.subgoals):
            # for each Not(), assert that the positive version isn't in the literals, and call the helper only on the positive literals in the subgoal
            lits = self.subgoals[self.next_subgoal_idx].literals 
            positive_lits = [l for l in lits if not l.is_negative]
            negative_lits_that_are_negated = [l.positive for l in lits if l.is_negative]
            if any(lit in next_state.literals for lit in negative_lits_that_are_negated):
                # There is no assignment that holds, since a negated lit in the subgoal is positive in the state.
                return
            if all(l in next_state.literals for l in positive_lits):
                # Check that all object names in the state literals match the object names in the goal
                self.precondition_targeting = True
                logging.info(f"ACHIEVED SUBGOAL {self.subgoals[self.next_subgoal_idx]}")
                self.next_subgoal_idx += 1
                self.action_seq.extend(self.actions_since_last_subgoal)
                self.actions_since_last_subgoal = []
                self._plan_to_next_subgoal = None
                self._last_plan_to_next_subgoal = None

        
    def _parse_action_from_string(self, action_string, objects_frozenset):
        """Given action string (pred obj-0 obj-1...), parse the pddlgym action.
        """
        items = action_string.strip()[1:-1].split()
        action_predicate_name = items[0]
        object_names = items[1:]

        action_pred = [p for p in self.action_space.predicates if p.name == action_predicate_name][0]
        objects = [o for o in objects_frozenset]
        args = []
        for object_name in object_names:
            for o in objects:
                obj_name, _ = o._str.split(':')
                if obj_name == object_name:
                    args.append(o)
                    break
        return action_pred(*args)


    def learn(self, itr):
        # Learn
        start = time.time()
        some_learned_operator_changed, updated_action_pred_names = self._operator_learning_module.learn(itr)
        logging.info(f"Learning took {time.time() - start} s")

        # Used in LLMIterative only
        if self.operator_learning_name in ['LLM+LNDR', 'LLMIterative+LNDR']:
            self._curiosity_module.learn(itr)

        if some_learned_operator_changed:
            self._curiosity_module.learning_callback()
            # only remove the operators for the action predicates that have been updated
            removes = set()
            for op_name in self._ops_preconds_executed:
                if op_name.rstrip('0123456789') in updated_action_pred_names:
                    removes.add(op_name)
            for r in removes:
                self._ops_preconds_executed.remove(r)

            # replan to the next subgoal.
            self._plan_to_next_subgoal = None

        else:
            # If the operators don't change as a result of adding the NOP, then the op preconds should be added to the visited set.
            if self._plan_to_op_preconds_failed:
                # This may be None if the plan to the operator preconds was length 1, so the visited set already contains the operator preconditions.
                if self._op_preconds_to_execute is not None:
                    self._ops_preconds_executed.add(self._op_preconds_to_execute)
                    self._op_preconds_to_execute = None
        return some_learned_operator_changed, some_learned_operator_changed
    
class InteractiveAgentLifted(InteractiveAgentGrounded):
    def __init__(self, domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name, log_llm_path:Optional[str]):
        super().__init__(domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name, log_llm_path)
        self._curiosity_module._ignore_mutex = False
        self._curiosity_module._ignore_statics = False
        self._curiosity_module._compute_goals = False
 
    def _get_action_with_preconds_as_goals(self, state, ops_to_exclude):
        # Have successfully executed the plan to the operator preconds, and will execute the operator next
        if self._preconds_plan is not None and len(self._preconds_plan) == 0:
            self.finished_preconds_plan = True
            self.actions_since_last_subgoal = []
            self._ops_preconds_executed.add(self._op_preconds_to_execute)
            self._op_preconds_to_execute = None
            self._preconds_plan = None
            # Ground action
            goal, lifted_act = self._current_goal_action
            ground_act = self._curiosity_module._sample_action_from_goal(goal, lifted_act,state, self._rand_state)
            logging.info(f"GROUNDED ACTION: {ground_act}")
            return ground_act

        # Follow plan to the operator's preconditions
        elif self._preconds_plan is not None and len(self._preconds_plan) > 0:
            self.finished_preconds_plan = False
            logging.info(f"FOLLOWING PLAN: {self._preconds_plan}")
            return self._preconds_plan.pop(0)

        action_predicates = set(p.name for p in self.action_space.predicates)
        for op in self._rand_state.permutation(sorted(self.learned_operators, key=lambda op: op.name)):
            # since the last time operators were learned, if operator has been successfully executed at the end of the plan, or
            # the plan failed in the middle to the operator preconditions, skip it.
            if op.name in self._ops_preconds_executed:
                continue
            if op.name in ops_to_exclude: continue
            preconds = op.preconds.literals

            logging.info(f"Trying preconds for op: {op.name}: {preconds}")

            # plan to lifted preconditions.
            preconds_hash = get_hashable_preconds_action(preconds)
            lifted_act = [p for p in preconds if p.predicate.name in action_predicates][0]
            if (preconds_hash, state) in self._visited_preconds_states[lifted_act.predicate]:
                continue
            lifted_precond_no_act = [p for p in preconds if p.predicate.name not in action_predicates]
            # add differents
            variables = sorted({ v for lit in lifted_precond_no_act for v in lit.variables })
            logging.info(f"variables: {variables}")
            Different = Predicate('different', 2)
            for param1 in variables:
                param1_type = param1._str[param1._str.find(':'):]
                for param2 in variables:
                    if param1._str >= param2._str:
                        continue
                    param2_type = param2._str[param2._str.find(':'):]

                    if param1_type == param2_type:
                        lifted_precond_no_act.append(Different(param1, param2))

            plan = self._get_plan_to_preconds(lifted_precond_no_act, state)
            self._current_goal_action = (tuple(lifted_precond_no_act), lifted_act)
            self._visited_preconds_states[lifted_act.predicate].add((preconds_hash, state))
            if plan == 'skip':
                self._ops_preconds_executed.add(op.name)
                break
            elif plan is not None:
                self._preconds_plan = plan
                logging.info(f"Found plan to preconds: {preconds_hash}")
                logging.info(f"PLAN: {self._preconds_plan}")
                self._op_preconds_to_execute = op.name
                if len(self._preconds_plan) == 0:
                    # ground action
                    goal, lifted_act = self._current_goal_action
                    ground_act = self._curiosity_module._sample_action_from_goal(goal, lifted_act,state, self._rand_state)
                    logging.info(f"GROUNDED ACTION: {ground_act}")
 
                    self.finished_preconds_plan = True
                    self.actions_since_last_subgoal = []
                    self._ops_preconds_executed.add(self._op_preconds_to_execute)
                    self._op_preconds_to_execute = None
                    self._preconds_plan = None
                    return ground_act
                else:
                    return self._preconds_plan.pop(0)

        # once done, proceed to the next subgoal in the file.
        return None

    def _get_plan_to_preconds(self, lifted_precond_lits:list, state):
        """Returns None if no plan found, otherwise a list of action literals."""
        variables = sorted({ v for lit in lifted_precond_lits for v in lit.variables })
        body = LiteralConjunction(lifted_precond_lits)
        goal = Exists(variables, body)
        logging.info(f"Planning to goal: {goal}")

        # Create a pddl problem file with the goal and current state
        problem_fname = self._curiosity_module._create_problem_pddl(
            state, goal, prefix='glibl_preconds')

        # Get a plan
        try:
            plan, _ = self._planning_module.get_plan(
                problem_fname, use_cache=False, use_learned_ops=True)
            os.remove(problem_fname)
            return plan
        except NoPlanFoundException:
            logging.info(f"No plan found.")
        except PlannerTimeoutException:
            logging.info(f"PLANNER TIMED OUT")
            if input("skip this preconditions? y or anything").strip() == 'y':
                os.remove(problem_fname)
                return 'skip'

        os.remove(problem_fname)

        return None
 

class DemonstrationsAgent(Agent):
     def __init__(self, domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name, log_llm_path:Optional[str]):
        super().__init__(domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name, log_llm_path)
        self.name = 'demoagent'   

        # Load the demos
        with open('bakingrealistic_demonstrations.pkl', 'rb') as f:
            transitions = pickle.load(f)
        self._operator_learning_module._transitions = transitions
 
        for action_pred in transitions:
            self._operator_learning_module._fits_all_data[action_pred] = False
 
    
class CreateDemonstrationsAgent(Agent):
    """An agent with initial demonstration data to each of the 4 train tasks."""
    def __init__(self, domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name, log_llm_path:Optional[str]):
        super().__init__(domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name, log_llm_path)
        
        self.name = 'demos'
        # dict: problem index -> step in the plan to execute next
        self.problem_to_plan_step = {i: 0 for i in range(len(ac.train_env.problems))}

        # dict: problem index -> list of plan steps (ground action predicate strings)
        self.plans = {}
        self._get_plans()
        self.prev_episode_idx = None
        # Keep track of episodes that have finished at least once
        self.terminated_episodes = set()
        self.action_space = action_space
        self.finished_preconds_plan = False
        # for action_pred in transitions:
        #     self._operator_learning_module._fits_all_data[action_pred] = True
 
        

    def get_action(self, state, problem_idx, _precond_targeting_only):

        if self.prev_episode_idx is not None and self.prev_episode_idx != problem_idx:
            self.terminated_episodes.add(self.prev_episode_idx)

        # If this is the first time in this episode, execute the plan until the episode terminates or the plan terminates.
        if problem_idx not in self.terminated_episodes and self.problem_to_plan_step[problem_idx] != "DONE":
            plan = self.plans[problem_idx]
            plan_step = self.problem_to_plan_step[problem_idx]
            action =  self._parse_action_from_string(plan[plan_step], state.objects)
            if plan_step + 2 > len(plan):
                self.problem_to_plan_step[problem_idx] = "DONE"
            else:
                self.problem_to_plan_step[problem_idx] += 1
            self.prev_episode_idx = problem_idx
            self._action_in_plan = False
            return action
            

        self.prev_episode_idx = problem_idx

        in_plan, op_name, action = self._curiosity_module.get_action(state)

        if in_plan:
            self._action_in_plan = op_name
        else:
            self._action_in_plan = False
        return action

    def _parse_action_from_string(self, action_string, objects_frozenset):
        """Given action string (pred obj-0 obj-1...), parse the pddlgym action.
        """
        items = action_string.strip()[1:-1].split()
        action_predicate_name = items[0]
        object_names = items[1:]

        action_pred = [p for p in self.action_space.predicates if p.name == action_predicate_name][0]
        objects = [o for o in objects_frozenset]
        args = []
        for object_name in object_names:
            for o in objects:
                obj_name, _ = o._str.split(':')
                if obj_name == object_name:
                    args.append(o)
                    break
        return action_pred(*args)

    def _get_plans(self):
        """Fill in self.plans with the plans from txt files."""
        demos_path = f'/home/catalan/GLIB-Baking-Fails-and-LLMs/demonstrations/{self.domain_name.capitalize()}'
        for i, file in enumerate(sorted(os.listdir(demos_path))):
            filepath = os.path.join(demos_path, file)
            problem_i = int(file[len('problem'):-len('.txt')])
            with open(filepath, 'r') as f:
                self.plans[problem_i] = [l for l in f.readlines() if l.strip() != '']
    
    def reset_episode(self, state, subgoals_path):
        obs_literals = set()
        if self.domain_name.lower() == 'bakingrealistic':
            for lit in state.literals:
                if lit.predicate.name not in ('different', 'name-less-than'):
                    obs_literals.add(lit)
            state = State(frozenset(obs_literals), state.objects, state.goal)

        start_time = time.time()
        self._curiosity_module.reset_episode(state)
        logging.info(f"Resetting episode for curiosity took {time.time() - start_time}")
        self.curiosity_time += time.time()-start_time
        self.episode_start = True
    
class StudentAgent(InteractiveAgentLifted):
    def __init__(self, domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name, log_llm_path:Optional[str]):
        super().__init__(domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name, log_llm_path)
 
        self.name = 'student'
        self._ops_executed = set()
        self._mode = "teacher_subgoals" #'preconds_as_goals'
        self.plan = None
        self._ground_truth_operators = {op for op in ac.train_env.domain.operators.values()}
        obj_types = set()
        for p in (self.action_space.predicates + self.obs_space.predicates):
            for t in p.var_types:
                obj_types.add(t)
        self.parser = GoalParser({p.name: p for p in self.action_space.predicates}, {p.name: p for p in self.obs_space.predicates}, obj_types)
        self._action_in_plan_to_preconds = False       


    def observe(self, state, action, next_state, itr):
        """Observe a transition.

        Args:
            state (pddlgym.structs.State): initial state of the transition
            action (Literal): action taken
            effects (set[Literal]): effects of the transition
            itr (int): training iteration #
        """
        if self.domain_name.lower() == 'bakingrealistic':
            obs_literals = set()
            next_obs_literals = set()
            for lit in state.literals:
                if lit.predicate.name not in ('different', 'name-less-than'):
                    obs_literals.add(lit)
            for lit in next_state.literals:
                if lit.predicate.name not in ('different', 'name-less-than'):
                    next_obs_literals.add(lit)
            state = State(frozenset(obs_literals), state.objects, state.goal)
            next_state = State(frozenset(next_obs_literals), next_state.objects, next_state.goal)
        # Get effects
        effects = self._compute_effects(state, next_state)
        logging.info(f"EFFECTS: \n{effects}")
        # Add data
        self._operator_learning_module.observe(state, action, effects, start_episode=self.episode_start, itr=itr)
        # Some curiosity modules might use transition data
        start_time = time.time()
        self._curiosity_module.observe(state, action, effects)
        self.curiosity_time += time.time()-start_time
        self.episode_start = False

        # Check if planned to preconditions
        if self._action_in_plan_to_preconds:
            # Stop executing the plan if it failed in the middle.
            if len(effects) == 0:
                self.finished_preconds_plan = True
                # About to reset to the previous subgoal, so clear this list.
                self.actions_since_last_subgoal = []
                self._preconds_plan = None
                # If the operators don't change as a result of adding the NOP, then the op preconds should be added to the visited set.
                self._plan_to_op_preconds_failed = True
            else:
                self._plan_to_op_preconds_failed = False
                
        else:
            if len(effects) == 0:
                logging.info(f"Setting plan to none ")
                self.plan = None


        # Check if planned to the next subgoal
 
    def get_action(self, state,  _problem_idx, precond_targeting_only):

        if self.plan is not None:
            return self._execute_plan(self.plan, state)

        if self._mode == 'preconds_as_goals':
            action = self._get_action_with_preconds_as_goals(state, set())
            if action is None:
                self._mode = 'teacher_subgoals' 
                self._action_in_plan_to_preconds = False
            else:
                self._action_in_plan_to_preconds = True
                return action

        if self._mode == 'teacher_subgoals':
            self._action_in_plan_to_preconds = False
            if input("Evaluate? y or anything").strip() == 'y':
                self.option = 9
                return None
            goals_without_plans = []
            operator_names_tried = set()
            all_operator_names = {o.name for o in self.learned_operators}
            while operator_names_tried != all_operator_names:
                goal, op_names = self._get_goal(operator_names_tried)
                operator_names_tried.update(op_names)
                if goal is None:
                    continue
                plan = self._get_plan(goal, state)
                if plan is not None:
                    logging.info(f"FOUND PLAN UNDER LEARNED OPS: {plan}")
                    return self._execute_plan(plan, state)
                else:
                    goals_without_plans.append(goal)
            for goal in goals_without_plans:
                plan =  self._get_ground_truth_plan(goal, state)
                if plan is not None:
                    logging.info(f"FOUND PLAN UNDER GT OPS. Goal: {goal}\nPlan: {plan}")
                    return self._execute_plan(plan, state)
            raise Exception(f"Don't know what to do when get here...")
                
        else:
            raise ValueError(self._mode)

    def _get_goal(self, operators_tried_already) -> Tuple[list,set[str]]:
        """Return the goals to plan to."""

       ### First step: operator matching

        # OP = None
        # for o in self.learned_operators:
        #     if o.name not in operators_tried_already:
        #         OP = o
        #         logging.info(f"Selected op: {OP.pddl_str()}")
        #         break
        # action_pred = [l.predicate for l in OP.preconds.literals if l.predicate in self.action_space.predicates][0]


    #    # Group ops by action predicate.
    #     ops_to_consider = []
    #     for o in self.learned_operators:
    #         if o.name == OP.name: continue
    #         a = [l.predicate for l in o.preconds.literals if l.predicate in self.action_space.predicates][0]           
    #         if a == action_pred:
    #             ops_to_consider.append(o)
           
       # Attempt to join as many operators as possible.
        # new_op_name = OP.name.rstrip('0123456789') + str(len(ops_to_consider) + 1)
        # ops_covered = [OP.name]
        # all_possible_joined = False
        # while not all_possible_joined:
        #     all_possible_joined = True
        #     new_ops_to_consider = []
        #     logging.info(f"Ops to consider: {ops_to_consider}")
        #     for o in ops_to_consider:
        #         logging.info(f"Considering {o.name}")
        #         new_op = join_operators(o, OP, new_op_name)
        #         if new_op is not None:
        #             logging.info(f"JOINED with {o.name}")
        #             ops_covered.append(o.name)
        #             OP = new_op
        #             all_possible_joined = False
        #         else:
        #             new_ops_to_consider.append(o)
        #     ops_to_consider = new_ops_to_consider

        # logging.info(f"Looking for g.t. operator that matches operator: {OP.pddl_str()}")
        # # Compare the joined learned operator effects to the ground truth operators effects.
        # ground_truth_operator = None
        # for op in self._ground_truth_operators:
        #     logging.info(f"Checking if equal: {op.name}")
        #     if effects_equal(op, OP):
        #         ground_truth_operator = op
        #         break
        
        # assert ground_truth_operator is not None, "Unexpected."
        # logging.info(f"Matched with ground truth operator: {ground_truth_operator.pddl_str()}")
        

        ### Testing code
        for OP in self.learned_operators:
            logging.info(f"Selected op: {OP.pddl_str()}")
            action_pred = [l.predicate for l in OP.preconds.literals if l.predicate in self.action_space.predicates][0]
            ops_to_consider = []
            for o in self.learned_operators:
                if o.name == OP.name: continue
                a = [l.predicate for l in o.preconds.literals if l.predicate in self.action_space.predicates][0]           
                if a == action_pred:
                    ops_to_consider.append(o)
            new_op_name = OP.name.rstrip('0123456789') + str(len(ops_to_consider) + 1)
            ops_covered = [OP.name]
            all_possible_joined = False
            while not all_possible_joined:
                all_possible_joined = True
                new_ops_to_consider = []
                logging.info(f"Ops to consider: {ops_to_consider}")
                for o in ops_to_consider:
                    logging.info(f"Considering {o.name}")
                    new_op = join_operators(o, OP, new_op_name)
                    if new_op is not None:
                        logging.info(f"JOINED with {o.name}")
                        ops_covered.append(o.name)
                        OP = new_op
                        logging.info(OP.pddl_str())
                        all_possible_joined = False
                    else:
                        new_ops_to_consider.append(o)
                ops_to_consider = new_ops_to_consider

            ground_truth_operator = None
            for op in self._ground_truth_operators:
                logging.info(f"Checking if equal: {op.name}")
                if effects_equal(op, OP):
                    ground_truth_operator = op
                    break
            assert ground_truth_operator is not None, "Unexpected."
            logging.info(f"Matched with ground truth operator: {ground_truth_operator.pddl_str()}")
            


        ### TODO: test, and then 2nd Step: goal selection.

        param_names = []
        param_types = []
        while True:
            goal_file = input("Enter the lifted goal file:").strip()
            try:
                with open(goal_file, 'r') as f:
                    lines = f.readlines()
                for variable_type in lines[0].split(','):
                    name, v_type = variable_type.split('-')
                    param_names.append(name.strip())
                    param_types.append(v_type.strip())
                goal_str = ''.join(lines[1:])
                body = self.parser._parse_into_cnf(goal_str, param_names, param_types, False)
                body = body[0]
                if isinstance(body, Literal):
                    body = LiteralConjunction([body])
                logging.info(f"parsed: {body}")
                lifted_act = [lit for lit in body.literals if lit.predicate in self.action_space.predicates][0]
                g = [lit for lit in body.literals if lit.predicate not in self.action_space.predicates]
                variables = sorted({ v for lit in body.literals for v in lit.variables })
                # add differents
                Different = Predicate('different', 2)
                for param1 in variables:
                    param1_type = param1._str[param1._str.find(':'):]
                    for param2 in variables:
                        if param1._str>= param2._str:
                            continue
                        param2_type = param2._str[param2._str.find(':'):]
 
                        if param1_type == param2_type:
                            g.append(Different(param1, param2))
                            
                body = LiteralConjunction(g)
                goal = Exists(variables, body)
                self._current_goal_action = (g, lifted_act)
                return goal, set(ops_covered)         
            except Exception as e:
                print(e)
                traceback.print_exc() 
                input("Continue or Ctrl-C to quit:")
                continue

    def _get_plan(self, goal, state):
        # Create a pddl problem file with the goal and current state
        problem_fname = self._curiosity_module._create_problem_pddl(
            state, goal, prefix='glibl_preconds')

        # Get a plan
        try:
            plan, _ = self._planning_module.get_plan(
                problem_fname, use_cache=False, use_learned_ops=True)
            os.remove(problem_fname)
            return plan
        except NoPlanFoundException:
            logging.info(f"No plan found.")
        except PlannerTimeoutException:
            logging.info(f"PLANNER TIMED OUT")

        os.remove(problem_fname)

        return None
 
    
    def _get_ground_truth_plan(self, goal, state):
        problem_fname = self._curiosity_module._create_problem_pddl(
            state, goal, prefix='glibl_preconds')
        # Get a plan
        try:
            plan, _ = self._planning_module.get_plan(
                problem_fname, use_cache=False, use_learned_ops=False, ops=self._ground_truth_operators)
            os.remove(problem_fname)
            return plan
        except NoPlanFoundException:
            logging.info(f"No plan found.")
        except PlannerTimeoutException:
            logging.info(f"PLANNER TIMED OUT")

        os.remove(problem_fname)

        return None


    def _execute_plan(self, plan, state):

        self.plan = plan

        if len(self.plan) == 0:
            goal, lifted_act = self._current_goal_action
            ground_act = self._curiosity_module._sample_action_from_goal(goal, lifted_act,state, self._rand_state)
            self.plan = None
            self.finished_preconds_plan = True
            logging.info(f"Executing grounded action: {ground_act}")
            return ground_act

        return self.plan.pop(0)


def get_hashable_preconds_action(preconds):
    # Sort preconditions by alphabetical order of its string representation.
    strings = []
    for pre in preconds:
        if pre.is_negative:
            pred = f'NOT-{pre.predicate.name}'
        else:
            pred = pre.predicate.name
        strings.append(f'({pred}' + ','.join(pre.pddl_variables()) + ')')
    s = ','.join(sorted(strings))
    return s
    
def dump_intermediate_state(agent:InteractiveAgentGrounded, fname='transitions.pkl'):
    with open(fname, 'wb') as f:
        pickle.dump(agent._operator_learning_module._transitions, f)
    with open('ops.pkl', 'wb') as f:
        pickle.dump(agent.learned_operators, f)
    with open('visited_preconds.pkl', 'wb') as f:
        pickle.dump(agent._visited_preconds_states, f)
    with open('ops_visited.pkl', 'wb') as f:
        pickle.dump(agent._ops_preconds_executed, f)
    with open("ndrs.pkl", 'wb') as f:
        pickle.dump(agent._operator_learning_module._ndrs, f)
    with open('rand_state.pkl', 'wb') as f:
        rand_state = agent._rand_state.get_state()
        pickle.dump(rand_state, f)

# def rename_variables_in_operator(op):
#     """Mutates the operator by renaming variables starting from ?x0."""
#     mapping = {}
#     i = 0
#     for param in op.params:
#         mapping[param] = TypedEntity(f'?x{i}', Type(param._str.split(':')[1]))
#         i += 1
#     for conjunction in [op.preconds.literals, op.effects.literals]:
#         for lit in conjunction:
#             lit.set_variables([mapping[param] for param in lit.variables])
#     op.params = set(mapping.values())
#     return op

    
def rename_variables_in_lits(given_lits:list, conditioned_mapping: dict = {}) -> Tuple[list, dict]:
    literals = deepcopy(sorted(given_lits))
    params = set()
    for lit in literals:
        for v in lit.variables:
            params.add(v) 
    variable_nums_taken = {int(v._str.split(':')[0][len("?x"):]) for v in conditioned_mapping.values()}
    i = 0
    rename_map = {}
    for v in params:
        if v in conditioned_mapping:
            rename_map[v] = conditioned_mapping[v]
        name, v_type = v._str.split(':')
        while i in variable_nums_taken:
            i += 1
        new_var = TypedEntity(f'?x{i}', Type(v_type))
        rename_map[v] = new_var 
        i += 1

    for lit in literals:
        lit.set_variables([rename_map[v] for v in lit.variables])

    return literals, rename_map

def effects_equal(op1, op2):
    """Returns True if the lifted effects of the operators are equal, False otherwise."""

    # renumber the variables in the effects from 0 for both operators.
    op1_preconds, op1_preconds_map = rename_variables_in_lits(op1.preconds.literals)
    op2_preconds, op2_preconds_map = rename_variables_in_lits(op2.preconds.literals)
    op1_effects, _ = rename_variables_in_lits(op1.effects.literals, op1_preconds_map)
    op2_effects, _ = rename_variables_in_lits(op2.effects.literals, op2_preconds_map)

    # # This IS A HEURISTIC that is incorrect for domains in general. covers the Antis in the effects that are already negative in the preconditions
    # for eff_lit in op1_effects:
    #     if eff_lit.is_anti and (eff_lit.inverted_anti not in op2_preconds) and (eff_lit not in op2_effects) and (eff_lit.inverted_anti not in op2_effects):
    #         op2_effects.append(eff_lit)
    # for eff_lit in op2_effects:
    #     if eff_lit.is_anti and (eff_lit.inverted_anti not in op1_preconds) and (eff_lit not in op1_effects) and (eff_lit.inverted_anti not in op1_effects):
    #         op1_effects.append(eff_lit)                       

    op1_effects = sorted(op1_effects)
    op2_effects = sorted(op2_effects)
    logging.info(f"Comparing {op1_effects} to {op2_effects}")
    # Get all parameterizations of the op1 params.
        # get all the variable names in a list, and use itertools.permutations(var_names)
    op1_params_list = []
    for lit in op1_effects:
        for param in lit.variables:
            op1_params_list.append(param._str.split(':')[0])

    # If number of literals aren't equal, return False.
    predicate_name_counts_op1 = defaultdict(lambda: 0)
    predicate_name_counts_op2 = defaultdict(lambda: 0)
    type_to_param_op1_effects = defaultdict(lambda: [])
    type_to_param_op2_effects = defaultdict(lambda: [])
    for lit in op1_effects:
        p_name = f'{lit.predicate.name}-{lit.is_anti}-{lit.is_negative}'
        predicate_name_counts_op1[p_name] += 1
        # for v in lit.variables:
            # type_to_param_op1_effects[v._str.split(':')[1]].append(v)
    for lit in op2_effects:
        p_name = f'{lit.predicate.name}-{lit.is_anti}-{lit.is_negative}'
        predicate_name_counts_op2[p_name] += 1
        # for v in lit.variables:
            # type_to_param_op2_effects[v._str.split(':')[1]].append(v)
    
    if sorted(predicate_name_counts_op1.keys()) != sorted(predicate_name_counts_op2.keys()):
        return False

    for key in predicate_name_counts_op1:
        if predicate_name_counts_op1[key] != predicate_name_counts_op2[key]:
            # logging.info("Returned at 2")
            return False
    
    return True
    # i = 0
    # # restrict the variable types to match the op2_effects and do permutations within each type.
    # d = {}
    # for t in type_to_param_op1_effects:
    #     logging.info(f"Computing length: {math.factorial(len(type_to_param_op2_effects[t]))}")
    #     d[t]= [list(zip(type_to_param_op1_effects[t], perm)) for perm in itertools.permutations(type_to_param_op2_effects[t])]

    # # 'assignment' is a list of lists
    # for assignment in itertools.product(*d.values()):
    #     p = []
    #     for a in assignment:
    #         p.extend(a)
    #     variables = dict(p)
    #     # map from the original variable name list to the permutation
    #     # Change the preconds and effects of op1 to the new arg names
    #     # Change the name from op1 param to the corresponding op2 param in preconditions and effects
    #     effects = []
    #     for l in op1_effects:
    #         args = []
    #         for v in l.variables:
    #             args.append(variables[v])
    #         effects.append(Literal(l.predicate, args))

    #     # Check that the preconditions and effects of the changed op1 are the same as in op2
    #     if set(op2_effects) == set(effects):
    #     # If the effects match, return True
    #         logging.info("Returned at 3")
    #         return True
    #     i += 1
    #     if i % 10000 == 0:
    #         logging.info(f"Checked {i+1} permutations")
 
    # logging.info("Returned at 4")
    # return False

def join_operators(op1, op2, new_op_name):
    """Returns a new operator if these operators can be joined, or None if they can't be joined."""
    # Find a reparameterization where the preconditions can be joined, or return fail if not found.

    # reparameterize the variables in the operators starting from 0
    op1_preconds, op1_preconds_mapping = rename_variables_in_lits(op1.preconds.literals)
    op2_preconds, op2_preconds_mapping = rename_variables_in_lits(op2.preconds.literals)

    # Get all parameterizations of the op1 params.
        # get all the variable names in a list, and use itertools.permutations(var_names)
    op1_preconds_params_list = set()
    for lit in op1_preconds:
        for param in lit.variables:
            op1_preconds_params_list.add(param)
    op1_preconds_params_list = list(op1_preconds_params_list)

    for perm in itertools.permutations(op1_preconds_params_list):
        # map from the original variable name list to the permutation
        variables = dict(zip(op1_preconds_params_list, perm))
        # Change the preconds and effects of op1 to the new arg names
        # Change the name from op1 param to the corresponding op2 param in preconditions
        preconds = []
        for l in op1_preconds:
            args = []
            for v in l.variables:
                args.append(variables[v])
            preconds.append(Literal(l.predicate, args))
        # check if there's a subset that is a complement and then the rest of the preconds are equal.
        common_in_op1 = []
        common_in_op2 = []
        for lit in preconds:
            if lit.negative in op2_preconds:
                common_in_op1.append(lit)
                common_in_op2.append(lit.negative)
        base_preconds = (set(preconds) - set(common_in_op1))
        if base_preconds == (set(op2_preconds) - set(common_in_op2)):

            # Carry over the precondition conditions to the effects if possible for both operators.

            op1_conditioned_mapping = {}
            for lit in op1.preconds.literals:
                for v in lit.variables:
                    op1_conditioned_mapping[v] = variables[op1_preconds_mapping[v]]
            
            op1_effects, op1_operator_mapping = rename_variables_in_lits(op1.effects.literals, op1_conditioned_mapping)
            op2_effects, op2_operator_mapping = rename_variables_in_lits(op2.effects.literals, op2_preconds_mapping)

            # get the variables not covered in the preconditions mapping
            op1_effects_params_list = set() 
            for lit in op1_effects:
                for v in lit.variables:
                    if v not in op1_conditioned_mapping.values():
                        op1_effects_params_list.add(v)
            op1_effects_params_list = list(op1_effects_params_list)

            # Iterate over the permutations of the effects mappings conditioned on the preconditions mapping
            for effects_perm in itertools.permutations(op1_effects_params_list):
                op1_effects_var_mapping = dict(zip(op1_effects_params_list, effects_perm))

                # Change the effects
                new_effects = []
                for lit in op1_effects:
                    args = []
                    for v in lit.variables:
                        if v in op1_effects_var_mapping:
                            args.append(op1_effects_var_mapping[v]) 
                        else:
                            args.append(v)
                    new_effects.append(Literal(lit.predicate, args))

                op1_effects = deepcopy(new_effects)

                # carry over the preconditions
                # This covers the lits in the preconds
                for lit in base_preconds:
                    if lit.predicate not in ac.train_env.action_space.predicates:
                        if lit.negative not in op1_effects: 
                            op1_effects.append(lit)
                        if lit.negative not in op2_effects:
                            op2_effects.append(lit)
                    
                # FIXME: In general, this is incorrect. but it works as an approximately correct alg for our domains. This covers the Antis in the effects that are already negative in the preconditions
                for eff_lit in op1_effects:
                    if eff_lit.is_anti and (eff_lit.inverted_anti not in base_preconds) and (eff_lit.predicate not in [l.predicate for l in op2_effects]):
                        op2_effects.append(eff_lit)
                for eff_lit in op2_effects:
                    if eff_lit.is_anti and (eff_lit.inverted_anti not in base_preconds) and (eff_lit.predicate not in [l.predicate for l in op1_effects]):
                        op1_effects.append(eff_lit)                       
                        new_effects.append(eff_lit)

                # Compare the effects to be equal or not. If equal, create the joined operator and return it.
                logging.info(f"Comparing {sorted(op1_effects)} to {sorted(op2_effects)}")
                if set(op1_effects) == set(op2_effects):
                    params = set()
                    for lits in [op1_effects, base_preconds]:
                        for lit in lits:
                            for v in lit.variables:
                                params.add(v)
                    return Operator(new_op_name, params, LiteralConjunction(list(base_preconds)), LiteralConjunction(new_effects))

    return None

